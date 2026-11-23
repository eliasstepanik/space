//! Client-side prediction and reconciliation.
//!
//! A client cannot wait a round trip for the server to move its own ship, so it
//! runs the *same* integrator the server runs — [`crate::step_ship`] — on the
//! last authoritative state plus the inputs it has sent but not yet seen
//! acknowledged. When the next authoritative snapshot arrives, the client
//! compares what it predicted with what the server says, adopts the server's
//! state and keeps only the inputs the server has not applied yet.
//!
//! ```text
//!   apply_input ─┐                    predict(t)
//!                ▼                        │
//!   pending: [i₁ i₂ i₃ …] ────────────────┤  authoritative ⊗ replay pending
//!                                        ▼
//!   reconcile(authority, t) ◄──── predicted state at t
//!        │  correction = |predicted − authority|
//!        └─ authority ← snapshot, pending ← unacknowledged tail
//! ```
//!
//! # Why the correction is small
//!
//! The correction is bounded by how far the two simulations can diverge between
//! two snapshots, and both sides run bit-identical code (see the determinism
//! policy in [the crate docs](crate)). The only sources of divergence are inputs
//! the server has not received yet (in flight or lost) and cross-platform float
//! differences. The acceptance test [`prediction_correction_bounded` in
//! `tests/acceptance.rs`](../space_sim/tests/acceptance.rs) fuzzes 1 000 ticks
//! with 10 % input loss and asserts the correction stays under **0.10 m**, the
//! handoff H3 budget.
//!
//! [`PredictedShip`] is deliberately free of any network or ECS dependency: it is
//! a state machine over [`ShipState`] and [`PlayerInput`], so the same type works
//! on a native client, in a test, and on a server that wants to keep a predicted
//! view of a neighbouring shard's ship. It predicts **free flight** exactly
//! (the server steps a ship that touches nothing with the same
//! [`crate::step_ship`]), but knows nothing of gravity, springs or contacts.
//!
//! [`PredictedPhysics`] is the replacement for a world with physics: the same
//! pending-input queue and reconciliation rules, but the replay steps the own
//! ship through a private [`SimWorld`] holding the scene's statics, its
//! gravity field and **kinematic copies** (ghosts) of the bodies near the
//! ship, so the predicted ship lands on the floor, bumps into crates and falls
//! in the gravity room exactly as the server's would (to the extent the
//! copies' motion is known: a copy is extrapolated with its last replicated
//! velocity and cannot be pushed, so a collision with a *dynamic* body is
//! predicted as one with an immovable one — reconciliation corrects it). The
//! private world forms its own physics bubble around the ship, so prediction
//! works the same anywhere in the world.

use std::collections::{BTreeMap, VecDeque};

use space_ecs::ComponentId;
use space_protocol::{EntityGuid, PlayerInput};
use space_world::WorldPos;

use crate::params::ShipParams;
use crate::physics::step_ship;
use crate::rigid::PhysicsError;
use crate::sim_world::SimWorld;
use crate::tick::SimTick;
use crate::ShipState;

/// How far past the last authoritative tick [`PredictedShip::predict`] will
/// replay, in ticks — and how many unacknowledged inputs
/// [`PredictedShip::apply_input`] keeps.
///
/// Both numbers come off the network: the tick to predict to is a snapshot's
/// tick (or a clock derived from one), and the queue grows by one per tick for
/// as long as the server acknowledges nothing. Unbounded, a corrupt or hostile
/// `new_tick` (say `u64::MAX`) would make [`PredictedShip::reconcile`] replay
/// forever, and an outage would grow the queue without limit. 300 ticks is 10 s
/// at [`crate::tick::TICK_HZ`] — an order of magnitude more than a healthy
/// prediction window (latency plus the snapshot interval, a handful of ticks),
/// so the bound never changes a prediction that means anything.
pub const MAX_PREDICT_TICKS: SimTick = 300;

/// A predicted ship: the last authoritative state plus the inputs on their way
/// to the server.
///
/// In addition to the three state fields the brief names, it carries the
/// [`ShipParams`] the integrator needs — prediction must integrate with exactly
/// the parameters the server used, so they live next to the state rather than
/// being threaded through every call.
///
/// # Invariant
///
/// [`pending`](Self::pending) is sorted by tick, holds at most one input per
/// tick, at most [`MAX_PREDICT_TICKS`] inputs, and — after every
/// [`apply_input`](Self::apply_input) and [`reconcile`](Self::reconcile) — no
/// input at or before [`last_tick`](Self::last_tick). The last part matters: the
/// server applies an input tagged `T` only in the step that *produces* tick `T`
/// and drops it once that tick has passed, so an input the authority has
/// already moved past is dead, and leaving it in the queue would let
/// [`predict`](Self::predict) "hold" a stick the server never saw.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictedShip {
    /// The newest state the server has confirmed.
    pub authoritative: ShipState,
    /// The tick [`authoritative`](Self::authoritative) describes.
    pub last_tick: SimTick,
    /// Inputs sent to the server and not yet covered by an authoritative state,
    /// oldest first. One entry per simulated tick.
    pub pending: VecDeque<PlayerInput>,
    /// The flight model both sides use.
    pub params: ShipParams,
}

impl PredictedShip {
    /// A ship whose last authority is `state` at `last_tick`.
    #[must_use]
    pub const fn new(state: ShipState, last_tick: SimTick, params: ShipParams) -> Self {
        Self {
            authoritative: state,
            last_tick,
            pending: VecDeque::new(),
            params,
        }
    }

    /// An input that does nothing, used to fill ticks past the end of
    /// [`pending`](Self::pending).
    const fn coast(tick: SimTick) -> PlayerInput {
        PlayerInput {
            entity: space_protocol::EntityGuid::new(space_protocol::ShardId::NONE, 0),
            thrust: [0.0; 3],
            torque: [0.0; 3],
            tick,
        }
    }

    /// Record an input the client is sending this tick.
    ///
    /// Inputs are kept in tick order. An input that repeats a tick already
    /// pending *replaces* it — that is a client correcting itself before the
    /// server saw it, and keeping both would double-step the ship.
    ///
    /// An input for a tick the authority already covers (`tick <= last_tick`) is
    /// **ignored**: the server can no longer apply it (see the
    /// [invariant](Self#invariant)), so it must not enter the replay queue. That
    /// is ordinary network timing — a snapshot overtook the input — not a logic
    /// error, so it is not a panic in any build.
    ///
    /// The queue is bounded at [`MAX_PREDICT_TICKS`] inputs; past that the
    /// oldest is dropped (after ten seconds without an acknowledgement the
    /// server has certainly dropped it too, and the newest inputs are the ones
    /// a late snapshot will still be waiting for).
    pub fn apply_input(&mut self, input: PlayerInput) {
        if input.tick <= self.last_tick {
            return;
        }
        if let Some(slot) = self.pending.iter_mut().find(|p| p.tick == input.tick) {
            *slot = input;
            return;
        }
        // Keep the queue sorted by tick (it already is, unless a late tick was
        // generated out of order, which a client with a background input thread
        // can do).
        match self.pending.iter().rposition(|p| p.tick < input.tick) {
            Some(i) => self.pending.insert(i + 1, input),
            None => self.pending.push_front(input),
        }
        let cap = usize::try_from(MAX_PREDICT_TICKS).unwrap_or(usize::MAX);
        while self.pending.len() > cap {
            self.pending.pop_front();
        }
    }

    /// The state the client believes the ship is in at `tick`.
    ///
    /// Replays [`pending`](Self::pending) from the authoritative state:
    /// `tick − last_tick` steps, the step producing each intermediate tick `t`
    /// using the pending input whose `tick == t` — so the inputs replayed are
    /// those for ticks `last_tick + 1 ..= tick`. That is the server's convention:
    /// [`SimWorld::step`](crate::SimWorld::step) drains `input_buf[tick + 1]`,
    /// i.e. the input tagged `T` is applied by the step that **produces** tick
    /// `T`; a prediction window that started at `last_tick` and used the input
    /// tagged `last_tick` would be one tick behind the server and would coast on
    /// the newest input instead of driving with it.
    ///
    /// Inputs are looked up **by tick**, not by position, so a gap in the queue
    /// (an input the server dropped, or one the client never sent) is stepped as a
    /// **coast** (zero input) rather than shifting every later input one tick
    /// early. Past the last pending tick the client holds the **last** pending
    /// input (a client keeps pushing the stick it is pushing). With an empty queue
    /// the ship coasts. `tick <= last_tick` returns the authoritative state
    /// unchanged.
    ///
    /// The replay is **bounded**: at most [`MAX_PREDICT_TICKS`] steps are taken,
    /// so `predict(tick)` for a `tick` further ahead than that returns the state
    /// at `last_tick + MAX_PREDICT_TICKS`. `tick` ultimately comes off the
    /// network; without the bound a single absurd value is an endless loop.
    ///
    /// [`SimWorld::step`]: crate::SimWorld::step
    #[must_use]
    pub fn predict(&self, tick: SimTick) -> ShipState {
        let steps = tick.saturating_sub(self.last_tick).min(MAX_PREDICT_TICKS);
        let mut state = self.authoritative;
        if steps == 0 {
            return state;
        }
        let coast = Self::coast(tick);
        let last_pending = self.pending.back().unwrap_or(&coast);
        // Index the pending inputs by tick so a gap coasts instead of shifting the
        // tail. `pending` is kept sorted by tick (see `apply_input`).
        for step in 0..steps {
            // The step producing tick `t` consumes the input tagged `t` (the
            // server drains `input_buf[tick + 1]`), so the first step after
            // `last_tick` uses the input tagged `last_tick + 1`. This cannot
            // overflow: `last_tick + steps <= tick`.
            let step_tick = self.last_tick.wrapping_add(1).wrapping_add(step);
            // The first `pending.len()` ticks replay the pending inputs *by tick*;
            // a tick with no pending input coasts (a lost/never-sent input). Past
            // the last pending tick the client holds its last input (or coasts).
            let input = match self.pending.iter().find(|i| i.tick == step_tick) {
                Some(i) => i,
                None if step_tick > self.pending.back().map_or(0, |i| i.tick) => last_pending,
                None => &coast,
            };
            step_ship(&mut state, input, &self.params);
        }
        state
    }

    /// Adopt `new_authority` (the server's state at `new_tick`) and report the
    /// position correction the client has to apply, in metres.
    ///
    /// The correction is measured against the state this client *would have*
    /// shown at `new_tick` — that is the visual jump a reconciliation has to
    /// absorb, so it is the number the H3 budget is about.
    ///
    /// Pending inputs are dropped **by tick**: every input with
    /// `tick <= input_ack` is removed, where `input_ack` is the highest input
    /// tick the server has applied (the `input_ack` field of the arriving
    /// [`space_protocol::Snapshot`] / [`space_protocol::Delta`]). Acknowledging by
    /// *tick* — not by counting `new_tick − last_tick` entries off the front, as
    /// an earlier revision did — is what survives reordering and loss: a client
    /// that sent inputs 10..=14 and gets `input_ack = 12` keeps exactly 13 and
    /// 14 whatever order the server received them in, and a queue with gaps (a
    /// corrected or never-sent tick) cannot shift the drop window onto the wrong
    /// inputs. The rest survive and are replayed on top of the new authority by
    /// the next [`predict`](Self::predict).
    ///
    /// Inputs with `tick <= new_tick` are dropped as well, acknowledged or not.
    /// The server applies an input tagged `T` only in the step that produces
    /// tick `T`; an authority at `new_tick` therefore already contains every
    /// input up to `new_tick` that will *ever* be applied, and one the server
    /// missed is gone for good. Such an input could never be replayed (the
    /// replay starts at `new_tick + 1`), but left in the queue it could be the
    /// *newest* entry — and [`predict`](Self::predict) holds the newest entry
    /// past the end of the queue, so the client would keep predicting thrust
    /// the server never applied, snapshot after snapshot.
    ///
    /// # Stale snapshots
    ///
    /// A snapshot **older than the current authority** (`new_tick < last_tick`)
    /// is ignored and reports a correction of `0.0`: deltas travel as datagrams
    /// and can be reordered, and adopting an older state would rewind the ship
    /// after the inputs between the two ticks have already been dropped — the
    /// replay would coast through ticks the server actually drove. A snapshot
    /// for the *same* tick is adopted as before (it is idempotent).
    ///
    /// "Older" is a statement about **one** tick clock. A ship that is handed to
    /// a shard whose tick counter differs must be given a fresh `PredictedShip`
    /// seeded from that shard's first snapshot: its pending inputs are tagged in
    /// the old clock, and this guard would otherwise sit out every snapshot of a
    /// shard whose clock is behind.
    pub fn reconcile(
        &mut self,
        new_authority: ShipState,
        new_tick: SimTick,
        input_ack: SimTick,
    ) -> f32 {
        if new_tick < self.last_tick {
            return 0.0;
        }
        let predicted = self.predict(new_tick);
        let correction = WorldPos::distance(predicted.pos, new_authority.pos) as f32;
        let covered = input_ack.max(new_tick);
        self.pending.retain(|i| i.tick > covered);
        self.authoritative = new_authority;
        self.last_tick = new_tick;
        correction
    }

    /// How many ticks of prediction are currently unacknowledged.
    #[must_use]
    pub fn pending_ticks(&self) -> usize {
        self.pending.len()
    }
}

/// What [`PredictedPhysics`] knows of a body near the ship: its last
/// replicated state and the tick it is from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Nearby {
    state: ShipState,
    tick: SimTick,
}

/// Client prediction through the full physics (see the [module docs](self)).
///
/// Same contract as [`PredictedShip`] — [`apply_input`](Self::apply_input),
/// [`predict`](Self::predict), [`reconcile`](Self::reconcile), the same
/// pending-queue invariant and the same [`MAX_PREDICT_TICKS`] bound — with
/// two differences: `predict` takes `&mut self` (it steps a world), and the
/// caller keeps the copies of nearby bodies current with
/// [`set_nearby`](Self::set_nearby) / [`remove_nearby`](Self::remove_nearby)
/// from the snapshots it receives.
///
/// Build the private world like the server builds a shard's: the scene's
/// catalogue, statics and gravity and the same [`ShipParams`]
/// (`space_scene::install_environment` does all three).
#[derive(Debug)]
pub struct PredictedPhysics {
    world: SimWorld,
    own: EntityGuid,
    /// The newest state the server has confirmed.
    pub authoritative: ShipState,
    /// The tick [`authoritative`](Self::authoritative) describes.
    pub last_tick: SimTick,
    /// Inputs sent and not yet covered by an authoritative state, oldest first.
    pub pending: VecDeque<PlayerInput>,
    nearby: BTreeMap<EntityGuid, Nearby>,
}

impl PredictedPhysics {
    /// Predict the ship `own` (driven by session `owner`), whose last
    /// authority is `state` at `last_tick`, in `world` (the scene's
    /// environment, see the type docs; it must not contain `own`).
    ///
    /// # Errors
    /// [`PhysicsError::Duplicate`] if `world` already holds `own`.
    pub fn new(
        mut world: SimWorld,
        own: EntityGuid,
        owner: u64,
        state: ShipState,
        last_tick: SimTick,
    ) -> Result<Self, PhysicsError> {
        if world.entity_of(own).is_some() {
            return Err(PhysicsError::Duplicate(own));
        }
        world.reset_clock(last_tick);
        world.spawn_ship(own, owner, state);
        Ok(Self {
            world,
            own,
            authoritative: state,
            last_tick,
            pending: VecDeque::new(),
            nearby: BTreeMap::new(),
        })
    }

    /// The private world (its bubbles, statics and copies).
    #[must_use]
    pub fn world(&self) -> &SimWorld {
        &self.world
    }

    /// The flight model in use.
    #[must_use]
    pub fn params(&self) -> ShipParams {
        self.world.params()
    }

    /// Record an input (same rules as [`PredictedShip::apply_input`]).
    pub fn apply_input(&mut self, input: PlayerInput) {
        if input.tick <= self.last_tick {
            return;
        }
        if let Some(slot) = self.pending.iter_mut().find(|p| p.tick == input.tick) {
            *slot = input;
            return;
        }
        match self.pending.iter().rposition(|p| p.tick < input.tick) {
            Some(i) => self.pending.insert(i + 1, input),
            None => self.pending.push_front(input),
        }
        let cap = usize::try_from(MAX_PREDICT_TICKS).unwrap_or(usize::MAX);
        while self.pending.len() > cap {
            self.pending.pop_front();
        }
    }

    /// Add or refresh the copy of a body near the ship: its replicated
    /// component list (as a snapshot carries it) and its state at `tick`. The
    /// copy is a ghost: it collides, it is never pushed.
    ///
    /// # Errors
    /// [`PhysicsError`] if the components do not decode or are the own ship.
    pub fn set_nearby(
        &mut self,
        guid: EntityGuid,
        components: &[(ComponentId, Vec<u8>)],
        state: ShipState,
        tick: SimTick,
    ) -> Result<(), PhysicsError> {
        if guid == self.own {
            return Err(PhysicsError::Duplicate(guid));
        }
        if !self.nearby.contains_key(&guid) {
            self.world.insert_ghost(guid, components)?;
        }
        self.nearby.insert(guid, Nearby { state, tick });
        Ok(())
    }

    /// Refresh only the state of a copy already known. Returns `false` for an
    /// unknown guid.
    pub fn update_nearby(&mut self, guid: EntityGuid, state: ShipState, tick: SimTick) -> bool {
        match self.nearby.get_mut(&guid) {
            Some(n) => {
                *n = Nearby { state, tick };
                true
            }
            None => false,
        }
    }

    /// Forget a copy (it left the area of interest).
    pub fn remove_nearby(&mut self, guid: EntityGuid) {
        if self.nearby.remove(&guid).is_some() {
            self.world.despawn(guid);
        }
    }

    /// The state the client believes its ship is in at `tick`: the authority
    /// replayed through the physics with the pending inputs (by tick, gaps
    /// coast, the last input is held past the end — exactly as
    /// [`PredictedShip::predict`]), copies extrapolated with their velocity.
    pub fn predict(&mut self, tick: SimTick) -> ShipState {
        let steps = tick.saturating_sub(self.last_tick).min(MAX_PREDICT_TICKS);
        self.world.reset_clock(self.last_tick);
        if let Some(e) = self.world.entity_of(self.own) {
            if let Some(s) = self.world.world_mut().get_mut::<ShipState>(e) {
                *s = self.authoritative;
            }
        }
        let last_pending = self.pending.back().copied();
        let last_tick = self.pending.back().map_or(0, |i| i.tick);
        for step in 0..steps {
            let at = self.last_tick.wrapping_add(step);
            let step_tick = at.wrapping_add(1);
            for (guid, n) in &self.nearby {
                let dt_s = (at as f64 - n.tick as f64) * f64::from(crate::tick::dt());
                let mut s = n.state;
                s.pos = WorldPos::from_f64_metres_around(
                    s.pos,
                    [
                        f64::from(s.vel[0]) * dt_s,
                        f64::from(s.vel[1]) * dt_s,
                        f64::from(s.vel[2]) * dt_s,
                    ],
                );
                self.world.drive_ghost(*guid, s, None);
            }
            let input = match self.pending.iter().find(|i| i.tick == step_tick) {
                Some(i) => Some(*i),
                None if step_tick > last_tick => last_pending,
                None => None,
            };
            if let Some(mut i) = input {
                i.entity = self.own;
                i.tick = step_tick;
                self.world.apply_inputs(self.world.tick(), &[i]);
            }
            self.world.step();
        }
        self.world.state_of(self.own).unwrap_or(self.authoritative)
    }

    /// Adopt the server's state (same rules as [`PredictedShip::reconcile`]);
    /// returns the position correction in metres.
    pub fn reconcile(
        &mut self,
        new_authority: ShipState,
        new_tick: SimTick,
        input_ack: SimTick,
    ) -> f32 {
        if new_tick < self.last_tick {
            return 0.0;
        }
        let predicted = self.predict(new_tick);
        let correction = WorldPos::distance(predicted.pos, new_authority.pos) as f32;
        let covered = input_ack.max(new_tick);
        self.pending.retain(|i| i.tick > covered);
        self.authoritative = new_authority;
        self.last_tick = new_tick;
        correction
    }

    /// How many ticks of prediction are currently unacknowledged.
    #[must_use]
    pub fn pending_ticks(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::PredictedShip;
    use crate::params::ShipParams;
    use crate::sim_world::SimWorld;
    use crate::ShipState;
    use space_protocol::{EntityGuid, PlayerInput, ShardId};
    use space_world::WorldPos;

    fn guid() -> EntityGuid {
        EntityGuid::new(ShardId(1), 1)
    }

    fn input(tick: u64, thrust: [f32; 3]) -> PlayerInput {
        PlayerInput {
            entity: guid(),
            thrust,
            torque: [0.0; 3],
            tick,
        }
    }

    fn predicted() -> PredictedShip {
        PredictedShip::new(ShipState::IDENTITY, 0, ShipParams::PLACEHOLDER)
    }

    /// Independent ground truth: a real [`SimWorld`] server that started at
    /// `start` (tick 0) and was stepped tick-by-tick, feeding the `t`-th entry
    /// of `inputs` to the step that produces tick `t` (the server's own
    /// convention: the input tagged `T` is applied by the step producing `T`).
    /// `None` is a tick with no input, i.e. a coast.
    ///
    /// This is deliberately *not* built from [`PredictedShip`]: a test that
    /// compares `predict` with `predict` proves nothing. The server and the
    /// client share only [`crate::step_ship`], so agreement here is the real
    /// "client prediction equals authority" property.
    fn server_truth(start: ShipState, inputs: &[Option<PlayerInput>]) -> ShipState {
        let mut sim = SimWorld::with_params(ShardId(1), ShipParams::PLACEHOLDER);
        sim.spawn_ship(guid(), 42, start);
        for (i, slot) in inputs.iter().enumerate() {
            let tick = u64::try_from(i + 1).unwrap();
            let batch: &[PlayerInput] = slot.as_slice();
            sim.apply_inputs(tick, batch);
            sim.step();
        }
        sim.ship_state(guid()).expect("ship is alive")
    }

    /// `n` driven ticks (all with `thrust`), tagged `1..=n`, as `Some`.
    fn driven(n: u64, thrust: [f32; 3]) -> Vec<Option<PlayerInput>> {
        (1..=n).map(|t| Some(input(t, thrust))).collect()
    }

    #[test]
    fn predict_from_rest_with_nothing_coasts() {
        let p = predicted();
        assert_eq!(p.predict(0), ShipState::IDENTITY);
        assert_eq!(p.predict(10), ShipState::IDENTITY);
    }

    #[test]
    fn predict_replays_pending_inputs() {
        let mut p = predicted();
        for tick in 1..=5 {
            p.apply_input(input(tick, [1.0, 0.0, 0.0]));
        }
        assert_eq!(p.pending_ticks(), 5);
        let one = p.predict(1);
        let three = p.predict(3);
        let five = p.predict(5);
        assert!(one.vel[0] < three.vel[0] && three.vel[0] < five.vel[0]);
        // Replaying the same inputs twice gives the same answer (pure).
        assert_eq!(five, p.predict(5));

        // Ground truth: a server driven by the same inputs. `predict(1)` must be
        // the state after ONE driven step — before the off-by-one fix it coasted
        // (it used the input tagged `last_tick`, which is tick 0 and does not
        // exist), so `predict(1)` equalled the unchanged authority.
        assert_ne!(one, ShipState::IDENTITY, "predict(1) must drive one step");
        assert_eq!(
            one,
            server_truth(ShipState::IDENTITY, &driven(1, [1.0, 0.0, 0.0]))
        );
        assert_eq!(
            three,
            server_truth(ShipState::IDENTITY, &driven(3, [1.0, 0.0, 0.0]))
        );
        assert_eq!(
            five,
            server_truth(ShipState::IDENTITY, &driven(5, [1.0, 0.0, 0.0]))
        );
    }

    #[test]
    fn apply_input_replaces_a_duplicate_tick() {
        let mut p = predicted();
        p.apply_input(input(1, [1.0, 0.0, 0.0]));
        p.apply_input(input(1, [0.0, 1.0, 0.0]));
        assert_eq!(p.pending_ticks(), 1);
        assert_eq!(p.pending[0].thrust, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn apply_input_keeps_the_queue_sorted() {
        let mut p = predicted();
        p.apply_input(input(3, [1.0, 0.0, 0.0]));
        p.apply_input(input(1, [0.0, 1.0, 0.0]));
        p.apply_input(input(2, [0.0, 0.0, 1.0]));
        let ticks: Vec<u64> = p.pending.iter().map(|i| i.tick).collect();
        assert_eq!(ticks, [1, 2, 3]);
    }

    #[test]
    fn reconcile_drops_consumed_inputs_and_reports_zero() {
        let mut p = predicted();
        let script = driven(8, [1.0, 0.0, 0.0]);
        for slot in &script {
            p.apply_input(slot.unwrap());
        }
        // Ground truth: a server stepped to tick 5 with the same inputs.
        let truth = server_truth(ShipState::IDENTITY, &script[..5]);
        let correction = p.reconcile(truth, 5, 5);
        assert!(
            correction < 1e-6,
            "authority equal to the server's must need no correction, got {correction}"
        );
        assert_eq!(p.last_tick, 5);
        assert_eq!(p.pending_ticks(), 3, "the unacknowledged tail survives");
        assert_eq!(p.authoritative, truth);
    }

    #[test]
    fn reconcile_reports_the_position_correction() {
        let mut p = predicted();
        for tick in 1..=8 {
            p.apply_input(input(tick, [1.0, 0.0, 0.0]));
        }
        let mut truth = p.predict(5);
        truth.pos = truth.pos.offset([0.05, 0.0, 0.0]);
        let correction = p.reconcile(truth, 5, 5);
        assert!(
            (correction - 0.05).abs() < 1e-4,
            "correction was {correction}, expected ≈0.05"
        );
        assert_eq!(p.pending_ticks(), 3);
    }

    /// Reconcile drops by `input_ack` (tick), not by count: a queue with a gap
    /// (a tick the client never sent / the server dropped) keeps exactly the
    /// inputs above the ack, whatever the queue length.
    #[test]
    fn reconcile_drops_by_tick_not_by_count() {
        let mut p = predicted();
        // A gapped queue: ticks 1,2,4,5 (tick 3 missing).
        for tick in [1_u64, 2, 4, 5] {
            p.apply_input(input(tick, [1.0, 0.0, 0.0]));
        }
        assert_eq!(p.pending_ticks(), 4);
        // Ground truth: a server driven on ticks 1,2 (tick 3 never sent → coast).
        let script: Vec<Option<PlayerInput>> = vec![
            Some(input(1, [1.0, 0.0, 0.0])),
            Some(input(2, [1.0, 0.0, 0.0])),
        ];
        let authority = server_truth(ShipState::IDENTITY, &script);
        // input_ack = 2 drops ticks 1 and 2 (by tick), leaving 4 and 5 — a
        // count-based drop of `new_tick - last_tick` would drop the wrong rows.
        let correction = p.reconcile(authority, 2, 2);
        assert!(
            correction < 1e-6,
            "authority from the server must need no correction, got {correction}"
        );
        let ticks: Vec<u64> = p.pending.iter().map(|i| i.tick).collect();
        assert_eq!(ticks, [4, 5]);
    }

    /// `predict` looks inputs up by tick: a gap in the pending queue coasts for
    /// the missing tick instead of shifting the tail one tick early.
    #[test]
    fn predict_coasts_over_a_gap() {
        let mut p = predicted();
        // Inputs on ticks 1 and 3 (tick 2 missing).
        p.apply_input(input(1, [1.0, 0.0, 0.0]));
        p.apply_input(input(3, [1.0, 0.0, 0.0]));
        // Ground truth: a server stepped tick-by-tick with the same inputs —
        // step(1) drives, step(2) coasts (no input arrived), step(3) drives. The
        // tick-2 gap must coast and the tick-3 input must land on tick 3, not on
        // tick 2.
        let script: Vec<Option<PlayerInput>> = vec![
            Some(input(1, [1.0, 0.0, 0.0])),
            None,
            Some(input(3, [1.0, 0.0, 0.0])),
        ];
        assert_eq!(p.predict(3), server_truth(ShipState::IDENTITY, &script));
    }

    /// The regression test for the off-by-one: **lossless, zero-latency,
    /// in-order** delivery. A real [`SimWorld`] server and a [`PredictedShip`]
    /// client are driven with the same inputs, tick for tick. Because nothing is
    /// lost or reordered, the client's replay of its pending inputs is exactly
    /// the server's step sequence, so every reconcile must report a correction of
    /// **exactly `0.0`** (not "small"), and `predict(1)` must equal the server
    /// after one driven step.
    ///
    /// Before the fix `predict` used the input tagged `last_tick + step` (ticks
    /// `last_tick ..= tick - 1`), one tick behind the server, so it coasted on the
    /// newest input and both assertions failed.
    #[test]
    fn predict_matches_the_server_with_lossless_in_order_delivery() {
        const TICKS: u64 = 60;
        const SNAPSHOT_EVERY: u64 = 5;
        const OWNER: u64 = 42;

        let params = ShipParams::PLACEHOLDER;
        let start = ShipState::at_rest(WorldPos::from_metres([500_000, -250_000, 7_000]));
        let mut server = SimWorld::with_params(ShardId(1), params);
        server.spawn_ship(guid(), OWNER, start);
        let mut client = PredictedShip::new(start, 0, params);

        let mut rng = SplitMix(0x1234_5678_9abc_def0);
        for tick in 1..=TICKS {
            let input = PlayerInput {
                entity: guid(),
                thrust: [rng.signed(), rng.signed(), rng.signed()],
                torque: [rng.signed(), rng.signed(), rng.signed()],
                tick,
            };
            // Zero latency, in order, nothing lost: the client predicts with the
            // input and the server receives it before the step that produces it.
            client.apply_input(input);
            server.apply_inputs(tick, &[input]);
            server.step();

            if tick % SNAPSHOT_EVERY == 0 {
                let authority = server.ship_state(guid()).expect("ship is alive");
                let ack = server.input_ack(OWNER);
                let correction = client.reconcile(authority, tick, ack);
                assert_eq!(
                    correction, 0.0,
                    "lossless in-order delivery must reconcile to exactly 0.0 at tick {tick}"
                );
            }
        }

        // `predict(1)` is the state after ONE driven step, not a coast.
        let mut one_step = SimWorld::with_params(ShardId(1), params);
        one_step.spawn_ship(guid(), OWNER, start);
        let first = PlayerInput {
            entity: guid(),
            thrust: [0.5, -0.25, 0.125],
            torque: [0.0, 0.75, -0.5],
            tick: 1,
        };
        one_step.apply_inputs(1, &[first]);
        one_step.step();
        let mut fresh = PredictedShip::new(start, 0, params);
        fresh.apply_input(first);
        assert_eq!(
            fresh.predict(1),
            one_step.ship_state(guid()).unwrap(),
            "predict(1) must equal the server after one driven step"
        );
    }

    /// A snapshot older than the current authority (a reordered datagram) must
    /// not rewind the ship: the inputs between the two ticks were already
    /// dropped by the newer acknowledgement, so adopting it would replay those
    /// ticks as a coast.
    #[test]
    fn reconcile_ignores_a_snapshot_older_than_the_authority() {
        let script = driven(8, [1.0, 0.0, 0.0]);
        let mut p = predicted();
        for slot in &script {
            p.apply_input(slot.unwrap());
        }
        let at_3 = server_truth(ShipState::IDENTITY, &script[..3]);
        let at_5 = server_truth(ShipState::IDENTITY, &script[..5]);
        assert_eq!(p.reconcile(at_5, 5, 5), 0.0);
        let before = p.clone();

        // The tick-3 snapshot arrives late.
        let correction = p.reconcile(at_3, 3, 3);
        assert_eq!(correction, 0.0, "a stale snapshot applies no correction");
        assert_eq!(p, before, "a stale snapshot must change nothing");
        // The prediction is still the server's future, not a rewound one.
        assert_eq!(p.predict(8), server_truth(ShipState::IDENTITY, &script));

        // A repeat of the *current* tick is still adopted (idempotent).
        assert_eq!(p.reconcile(at_5, 5, 5), 0.0);
        assert_eq!(p, before);
    }

    /// The replay window is bounded: a tick absurdly far ahead (it comes off
    /// the network) must not loop forever, in `predict` or in `reconcile`.
    #[test]
    fn predict_is_bounded_for_an_absurd_tick() {
        use super::MAX_PREDICT_TICKS;

        let mut p = predicted();
        p.apply_input(input(1, [1.0, 0.0, 0.0]));
        // Held thrust: the ship is still accelerating at the bound, so every
        // further step would change the state.
        let at_bound = p.predict(MAX_PREDICT_TICKS);
        assert_ne!(at_bound, p.predict(MAX_PREDICT_TICKS - 1));
        // Past the bound nothing more is replayed. (Unbounded, the next line is
        // 2^64 integrator steps.)
        assert_eq!(p.predict(MAX_PREDICT_TICKS + 1), at_bound);
        assert_eq!(p.predict(u64::MAX), at_bound);

        // `reconcile` predicts to the snapshot's tick: it must return too.
        let mut q = p.clone();
        let correction = q.reconcile(ShipState::IDENTITY, u64::MAX, 0);
        assert!(correction.is_finite());
        assert_eq!(q.last_tick, u64::MAX);
        assert_eq!(q.pending_ticks(), 0);
        // Nothing lies beyond the last tick, so nothing more is predicted or
        // queued — and nothing overflows.
        assert_eq!(q.predict(u64::MAX), ShipState::IDENTITY);
        q.apply_input(input(7, [1.0, 0.0, 0.0]));
        assert_eq!(q.pending_ticks(), 0);
    }

    /// With the server silent, the pending queue stops growing at
    /// `MAX_PREDICT_TICKS` and keeps the newest inputs.
    #[test]
    fn pending_queue_is_bounded() {
        use super::MAX_PREDICT_TICKS;

        let mut p = predicted();
        let total = MAX_PREDICT_TICKS * 3 + 17;
        for tick in 1..=total {
            p.apply_input(input(tick, [0.5, 0.0, 0.0]));
        }
        assert_eq!(p.pending_ticks() as u64, MAX_PREDICT_TICKS);
        assert_eq!(p.pending.back().map(|i| i.tick), Some(total));
        assert_eq!(
            p.pending.front().map(|i| i.tick),
            Some(total - MAX_PREDICT_TICKS + 1)
        );
        let ticks: Vec<u64> = p.pending.iter().map(|i| i.tick).collect();
        assert!(ticks.windows(2).all(|w| w[0] < w[1]), "still sorted");
    }

    /// An input the authority has already moved past is dead even when the
    /// server never acknowledged it — the server drops an input once its tick
    /// has been simulated — so `reconcile` must drop it too. Left in the queue
    /// it becomes the "last pending input" and `predict` holds it forever: the
    /// client predicts thrust the server never applied.
    #[test]
    fn reconcile_drops_inputs_the_authority_has_passed() {
        let thrust = [1.0, 0.0, 0.0];
        let mut p = predicted();
        // The client sent ticks 1..=4, then released the stick and sent nothing.
        for tick in 1..=4 {
            p.apply_input(input(tick, thrust));
        }
        // The server got 1 and 2 only; 3 and 4 were lost. Its tick-6 state:
        let mut seen: Vec<Option<PlayerInput>> =
            vec![Some(input(1, thrust)), Some(input(2, thrust))];
        seen.resize(6, None);
        let at_6 = server_truth(ShipState::IDENTITY, &seen);
        // ...arrives with input_ack = 2 (the highest input tick it applied).
        let correction = p.reconcile(at_6, 6, 2);
        assert!(correction > 0.0, "the client had predicted the lost thrust");
        assert_eq!(
            p.pending_ticks(),
            0,
            "inputs 3 and 4 are behind the authority and can never be applied"
        );

        // From here the server coasts, and so must the prediction. (Holding
        // the stale tick-4 input instead would keep thrusting.)
        seen.resize(9, None);
        assert_eq!(p.predict(9), server_truth(ShipState::IDENTITY, &seen));

        // An acknowledgement *ahead* of the snapshot tick still wins: inputs up
        // to it are dropped even though their ticks are in the future.
        let mut p = predicted();
        for tick in 1..=8 {
            p.apply_input(input(tick, thrust));
        }
        p.reconcile(ShipState::IDENTITY, 2, 5);
        let ticks: Vec<u64> = p.pending.iter().map(|i| i.tick).collect();
        assert_eq!(ticks, [6, 7, 8]);
    }

    /// An input for a tick the authority already covers is ignored (in every
    /// build — a snapshot overtaking an input is ordinary timing, not a bug).
    #[test]
    fn apply_input_ignores_a_tick_the_authority_already_covers() {
        let script = driven(8, [1.0, 0.0, 0.0]);
        let mut p = predicted();
        for slot in &script {
            p.apply_input(slot.unwrap());
        }
        let at_5 = server_truth(ShipState::IDENTITY, &script[..5]);
        p.reconcile(at_5, 5, 5);
        let before = p.clone();

        p.apply_input(input(5, [0.0, 1.0, 0.0])); // the authority's own tick
        p.apply_input(input(2, [0.0, 0.0, 1.0])); // long gone
        assert_eq!(p, before, "a dead input must not enter the queue");
        assert_eq!(p.predict(8), server_truth(ShipState::IDENTITY, &script));

        // The next tick is live.
        p.apply_input(input(6, [0.0, 1.0, 0.0]));
        assert_eq!(p.pending[0].thrust, [0.0, 1.0, 0.0]);
    }

    /// A tiny integer-only PRNG so the test's input script is deterministic.
    struct SplitMix(u64);

    impl SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// A float in `[-1, 1]` with 1e-6 resolution, from integers only.
        fn signed(&mut self) -> f32 {
            // The remainder fits u32, so the widening is exact.
            ((self.next_u64() % 2_000_001) as u32 as f64 / 1_000_000.0 - 1.0) as f32
        }
    }
}
