//! The player's own ship, ahead of the server, and the clock everything is
//! drawn by.
//!
//! # The clock
//!
//! A shard steps 30 times a second and tells the client about every tick;
//! the client draws at whatever rate its display runs. [`Clock`] estimates
//! the shard's tick *now*, as a fractional number of ticks, from the newest
//! tick heard and the time since it arrived — smoothed, so the arrival jitter
//! of the frames does not shake the picture, and snapped when it is far off
//! (a stall, or a transition onto a shard that counts other ticks:
//! [`Replica::tick_epoch`]; the shards of a cluster count the same, and the
//! clock simply runs on through a transition between them). Other
//! entities are drawn [`crate::bodies::INTERPOLATION_DELAY`] ticks behind it;
//! the own ship is drawn ahead of it.
//!
//! # Prediction
//!
//! The link tags every input batch with the tick the shard is to apply it at
//! — [`crate::link::INPUT_LEAD`] ticks ahead of the newest tick heard — so
//! the shard's word on the ship arrives a few ticks after the player pressed
//! the key. Shown as it is, the ship would answer late and move in 30 Hz
//! steps. [`OwnShip`] runs it ahead instead, with the same physics the shard
//! runs ([`PredictedPhysics`]: the scene's statics, its gravity and
//! kinematic copies of the bodies and ships nearby, from `space-scene`), and
//! replays the inputs the shard has not acknowledged on top of its last word.
//! Walking into a wall or standing on the floor is predicted as the shard
//! simulates it, so a landed ship does not jitter and a wall does not
//! rubber-band. When the shard's word arrives it is adopted
//! ([`PredictedPhysics::reconcile`]): the difference between what was
//! predicted for that tick and what the shard says is the **correction**
//! ([`OwnShip::correction`]), absorbed over [`CORRECTION_SMOOTHING`] instead
//! of shown as a jump.
//!
//! The ship is drawn at a fractional tick — interpolated between the
//! predicted states of the two ticks around it — so it moves smoothly at any
//! frame rate. A ship at rest is not sent every tick (nothing changes); its
//! state at the newest tick heard is then its last one, and the replay never
//! has to go back further than a few ticks.
//!
//! Past the newest input sent the prediction holds that input on. The link
//! sends inputs on a timer of its own, so the frame being drawn may already
//! be past the newest one; when the next input says something else — fast
//! travel going off at speed is the case that shows — what it changes in
//! that frame is carried in the same visual offset (up to
//! `MAX_INPUT_SMOOTHED`), and an offset that large decays no faster than the
//! ship flies: the drawn ship waits for the predicted one, it never steps
//! back.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use space_ecs::{ComponentDef, ComponentId};
use space_protocol::{EntityGuid, Scene, ShardId};
use space_sim::{
    Articulation, Body, Owner, PartStates, PosResidual, PredictedPhysics, ShipControls, ShipState,
    SimWorld, Springs, Visual,
};
use space_world::WorldPos;

use crate::link::{Replica, INPUT_LEAD};

/// Ticks per second.
fn hz() -> f64 {
    1.0 / f64::from(space_sim::dt())
}

/// Further than this from the newest tick heard (ticks) the clock is snapped
/// rather than steered.
const SNAP_TICKS: f64 = 6.0;

/// How fast the clock is steered towards the newest tick heard: the share of
/// the error removed per second.
const STEER_PER_SECOND: f64 = 4.0;

/// The estimate of the shard's tick, for drawing (see the module docs).
#[derive(Clone, Debug, Default)]
pub struct Clock {
    /// The estimate, ticks.
    estimate: Option<f64>,
    /// When it was last advanced.
    at: Option<Instant>,
    /// The tick timeline the estimate is on ([`Replica::tick_epoch`]): a
    /// transition onto another one starts over.
    domain: u64,
}

impl Clock {
    /// A clock that has heard nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The estimate, ticks (`None` before the first frame).
    #[must_use]
    pub fn now(&self) -> Option<f64> {
        self.estimate
    }

    /// Advance to `now` and steer towards what `replica` heard last.
    pub fn update(&mut self, replica: &Replica, now: Instant) -> Option<f64> {
        let heard = replica.tick_seen?;
        if replica.tick == 0 {
            return self.estimate;
        }
        let since = now.saturating_duration_since(heard).as_secs_f64().min(1.0);
        let target = replica.tick as f64 + since * hz();
        let domain = replica.tick_epoch;
        let elapsed = self
            .at
            .map_or(0.0, |at| now.saturating_duration_since(at).as_secs_f64());
        self.at = Some(now);
        let estimate = match self.estimate {
            Some(e) if domain == self.domain => {
                let advanced = e + elapsed * hz();
                let error = target - advanced;
                if error.abs() > SNAP_TICKS {
                    target
                } else {
                    advanced + error * (elapsed * STEER_PER_SECOND).min(1.0)
                }
            }
            _ => target,
        };
        self.domain = domain;
        self.estimate = Some(estimate);
        self.estimate
    }

    /// Start over (left the shard).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// A correction is absorbed over about this long (the time constant of its
/// decay) instead of being shown as a jump.
pub const CORRECTION_SMOOTHING: Duration = Duration::from_millis(100);

/// A correction larger than this (metres) is shown as it is: it is a
/// teleport, not a misprediction.
const MAX_SMOOTHED: f64 = 2.0;

/// What a late input changes in the frame being drawn is carried in the
/// visual offset up to this (metres); beyond it is shown as it is. An input
/// is at most about a tick late, so this covers leaving fast travel at up to
/// about 1 km/s; faster than that the ship is drawn stepping back by up to a
/// tick of its flight — at most a thirtieth of the distance to the nearest
/// surface, which is what the speed limit of fast travel is.
const MAX_INPUT_SMOOTHED: f64 = 32.0;

/// The most ticks a prediction runs past the shard's last word. The link
/// sends inputs [`INPUT_LEAD`] ticks ahead; more than this means the shard's
/// word is late, and predicting further only costs time.
const MAX_AHEAD: u64 = 20;

/// Bodies (and other ships) whose centre is closer than this to the ship,
/// metres, are copied into the prediction world...
const NEARBY: f64 = 8.0;
/// ...and dropped from it again beyond this.
const FARAWAY: f64 = 12.0;

/// How long corrections are kept for [`OwnShip::corrections`].
const CORRECTION_WINDOW: Duration = Duration::from_secs(5);

/// Gravity weaker than this (m/s²) counts as weightless for "landed".
const WEIGHTLESS: f64 = 0.05;

/// The component ids a prediction world accepts for a copy.
const COPYABLE: [ComponentId; 9] = [
    ShipState::ID,
    ShipControls::ID,
    Owner::ID,
    Body::ID,
    Visual::ID,
    Springs::ID,
    Articulation::ID,
    PartStates::ID,
    PosResidual::ID,
];

/// What a prediction is for: a change of any of these builds it anew. Not
/// the shard and not the session: a ship handed from one shard of a cluster
/// to the next is the same ship on the same tick timeline, and its
/// prediction — the inputs still to be confirmed, the bodies around it, the
/// correction being absorbed — simply goes on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    guid: EntityGuid,
    /// The tick timeline ([`Replica::tick_epoch`]).
    tick_epoch: u64,
    scene: Scene,
    /// Whether the shard has said which scene it runs (a shard nobody gave
    /// one simulates an empty world).
    scene_set: bool,
    anchor: Option<WorldPos>,
}

/// The player's ship, predicted (see the module docs).
#[derive(Debug)]
pub struct OwnShip {
    key: Key,
    predictor: PredictedPhysics,
    /// Predicted states by tick, until something changes.
    cache: BTreeMap<u64, ShipState>,
    /// The bodies copied into the prediction world, with the tick of their
    /// copy.
    nearby: BTreeMap<EntityGuid, u64>,
    /// Entities whose components the prediction world refused.
    refused: BTreeSet<EntityGuid>,
    /// The visual offset still to absorb, metres.
    offset: [f64; 3],
    /// When `offset` was last decayed.
    offset_at: Instant,
    /// Where the ship was drawn last ([`OwnShip::shown`]): a large offset
    /// never decays by more than the picture has come since.
    last_drawn: Option<WorldPos>,
    /// The corrections of the last [`CORRECTION_WINDOW`].
    corrections: VecDeque<(Instant, f32)>,
    /// Whether the ship stood on something in the last predicted tick.
    landed: bool,
    /// The gravity at the ship, m/s².
    gravity: [f64; 3],
}

/// The ship as drawn this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shown {
    /// Its state (interpolated between two predicted ticks), with the
    /// correction still being absorbed added to the position.
    pub state: ShipState,
    /// The fractional tick it is drawn at.
    pub tick: f64,
}

impl OwnShip {
    /// A prediction of the ship `replica` reports, in `scene`; `None` until
    /// the shard has shown the ship.
    #[must_use]
    pub fn new(replica: &Replica, scene: Scene) -> Option<Self> {
        let key = key_of(replica, scene)?;
        let state = replica.state?;
        let mut world = SimWorld::new(replica.shard.unwrap_or(ShardId::NONE));
        if let Some(clock) = replica.scene_clock {
            world.set_scene_clock(clock);
        }
        if let Some(anchor) = key.anchor.filter(|_| key.scene_set) {
            if let Err(e) = space_scene::install_environment(&mut world, scene, anchor) {
                log::warn!("space-client: the prediction has no scene: {e}");
            }
        }
        let mut predictor = PredictedPhysics::new(
            world,
            key.guid,
            replica.session,
            state,
            replica.state_tick(),
        )
        .ok()?;
        // A prediction built in mid-flight (a change of scene or of the tick
        // timeline) starts from the shard's word on fast travel too.
        predictor.authoritative_fast = replica.own_fast();
        Some(Self {
            key,
            predictor,
            cache: BTreeMap::new(),
            nearby: BTreeMap::new(),
            refused: BTreeSet::new(),
            offset: [0.0; 3],
            offset_at: Instant::now(),
            last_drawn: None,
            corrections: VecDeque::new(),
            landed: false,
            gravity: [0.0; 3],
        })
    }

    /// Whether this prediction is still the one for `replica` in `scene`
    /// (same ship, tick timeline, scene and anchor).
    #[must_use]
    pub fn is_for(&self, replica: &Replica, scene: Scene) -> bool {
        key_of(replica, scene) == Some(self.key)
    }

    /// Carry the visual offset of `previous` over (a new prediction — after
    /// a transition onto a shard that counts other ticks, or a change of
    /// scene — should not jump either).
    pub fn inherit(&mut self, previous: &Self, shown_before: Option<WorldPos>, now: Instant) {
        self.corrections = previous.corrections.clone();
        if let Some(before) = shown_before {
            let after = self.raw_position(self.predictor.last_tick as f64);
            let d = WorldPos::delta(after, before);
            if d.iter().map(|c| c * c).sum::<f64>().sqrt() < MAX_SMOOTHED {
                self.offset = d;
                self.offset_at = now;
            }
        }
    }

    /// The ship's state at `tick`: the shard's word, or predicted from it.
    fn state_at(&mut self, tick: u64) -> ShipState {
        let last = self.predictor.last_tick;
        if tick <= last {
            return self.predictor.authoritative;
        }
        let tick = tick.min(last + MAX_AHEAD);
        if let Some(s) = self.cache.get(&tick) {
            return *s;
        }
        let s = self.predictor.predict(tick);
        let world = self.predictor.world();
        // A contact the solver resolved in the last tick — or the ship is
        // asleep: once it stands still the replay ends where it began, the
        // solver puts the body to sleep, and a body asleep is not stepped
        // and reports no contact. In a pull nothing falls asleep but on
        // what it rests on (weightless, `landed()` says no anyway).
        self.landed = world.is_sleeping(self.key.guid) == Some(true)
            || world
                .contacts()
                .iter()
                .any(|c| c.a == self.key.guid || c.b == Some(self.key.guid));
        self.gravity = world.gravity().at(s.pos, tick);
        self.cache.insert(tick, s);
        s
    }

    /// The position at fractional `tick`, without the visual offset.
    fn raw_position(&mut self, tick: f64) -> WorldPos {
        self.interpolated(tick).pos
    }

    fn interpolated(&mut self, tick: f64) -> ShipState {
        let base = tick.floor().max(0.0) as u64;
        let alpha = (tick - base as f64).clamp(0.0, 1.0);
        let a = self.state_at(base);
        let b = self.state_at(base + 1);
        let d = WorldPos::delta(a.pos, b.pos);
        ShipState {
            pos: WorldPos::from_f64_metres_around(a.pos, d.map(|c| c * alpha)),
            vel: [0, 1, 2].map(|k| a.vel[k] + (b.vel[k] - a.vel[k]) * alpha as f32),
            ..b
        }
    }

    /// The tick the ship is drawn at, for a clock reading `now` (ticks): one
    /// tick behind the newest input sent, so two predicted states are always
    /// on hand.
    #[must_use]
    pub fn draw_tick(now: f64) -> f64 {
        now + INPUT_LEAD as f64 - 1.0
    }

    /// Take the frame's news: the shard's newest word on the ship (adopted,
    /// its correction measured and smoothed), the inputs sent since the last
    /// frame (replayed from now on), and the bodies near the ship (copied
    /// into the prediction world). `shown` is the tick the ship is drawn at.
    pub fn update(
        &mut self,
        replica: &Replica,
        sent: &[space_protocol::PlayerInput],
        shown: f64,
        now: Instant,
    ) {
        // The shard's mapping onto scene time (the planets, and what they
        // carry), and its word on fast travel.
        if let Some(clock) = replica.scene_clock {
            if clock != self.predictor.world().scene_clock() {
                self.predictor.set_scene_clock(clock);
                self.cache.clear();
            }
        }
        // The shard's word. A ship at rest is not sent while it stays at rest:
        // then the newest tick heard is the tick of its last state too.
        //
        // Its word on fast travel belongs to the same tick and is taken
        // *after* the reconcile: the replay up to that tick has to start
        // from fast travel as it was at the tick before, or it never sees
        // the ship leave it and skips the drop to the normal top speed — a
        // tick of fast flight of false correction.
        let fast = replica.own_fast();
        if let Some(state) = replica.state {
            let (tick, ack) = if replica.tick > replica.state_tick() && state.vel == [0.0; 3] {
                (replica.tick, replica.frame_ack())
            } else {
                (replica.state_tick(), replica.input_ack())
            };
            if tick > self.predictor.last_tick
                || (tick == self.predictor.last_tick && state != self.predictor.authoritative)
            {
                let before = self.raw_position(shown);
                let correction = self.predictor.reconcile(state, tick, ack);
                self.predictor.authoritative_fast = fast;
                self.cache.clear();
                let after = self.raw_position(shown);
                let d = WorldPos::delta(after, before);
                let jump = d.iter().map(|c| c * c).sum::<f64>().sqrt();
                if jump < MAX_SMOOTHED {
                    for k in 0..3 {
                        self.offset[k] += d[k];
                    }
                } else {
                    self.offset = [0.0; 3];
                }
                self.corrections.push_back((now, correction));
            }
        }
        if fast != self.predictor.authoritative_fast {
            // Without a newer state (a ship at rest).
            self.predictor.authoritative_fast = fast;
            self.cache.clear();
        }
        while self
            .corrections
            .front()
            .is_some_and(|(at, _)| now.saturating_duration_since(*at) > CORRECTION_WINDOW)
        {
            self.corrections.pop_front();
        }
        // The inputs sent. The link sends them on a timer of its own, so one
        // may arrive after frames that already drew its tick with the input
        // before it held on (`PredictedPhysics::predict`). When it changes
        // what those frames showed — fast travel going off at speed is a
        // tick's flight of difference — the difference is carried in the
        // visual offset like a correction, not shown as a step back.
        // (An input that says what the last one said changes nothing: that
        // is what was held on.)
        let guid = self.key.guid;
        let own = |i: &&space_protocol::PlayerInput| i.entity == guid;
        if sent.iter().any(|i| own(&i)) {
            let held = self.predictor.pending.back().copied();
            let news = sent.iter().filter(own).any(|i| {
                held.is_none_or(|h| {
                    h.thrust != i.thrust || h.torque != i.torque || h.fast != i.fast
                })
            });
            let before = news.then(|| self.raw_position(shown));
            for input in sent.iter().filter(own) {
                self.predictor.apply_input(*input);
            }
            self.cache.clear();
            if let Some(before) = before {
                let after = self.raw_position(shown);
                let d = WorldPos::delta(after, before);
                let jump = d.iter().map(|c| c * c).sum::<f64>().sqrt();
                if jump < MAX_INPUT_SMOOTHED {
                    for k in 0..3 {
                        self.offset[k] += d[k];
                    }
                }
            }
        }
        // The bodies nearby.
        self.update_nearby(replica);
    }

    fn update_nearby(&mut self, replica: &Replica) {
        let me = self.predictor.authoritative.pos;
        let mut keep = BTreeSet::new();
        for tracked in replica.tracked() {
            let guid = tracked.ship.guid;
            if guid == self.key.guid || Some(guid) == replica.ship || self.refused.contains(&guid) {
                continue;
            }
            // A far replica is for seeing: it is in no solver, the
            // prediction's included (its samples are sparse and late).
            if tracked.source() == crate::link::Source::Far {
                continue;
            }
            let state = tracked.ship.state;
            let distance = WorldPos::distance(me, state.pos);
            let known = self.nearby.get(&guid).copied();
            let reach = if known.is_some() { FARAWAY } else { NEARBY };
            if distance > reach {
                continue;
            }
            keep.insert(guid);
            match known {
                Some(tick) if tick == tracked.tick => {}
                Some(_) if tracked.physics.as_ref().is_none_or(|p| p.part_count() == 1) => {
                    self.predictor.update_nearby(guid, state, tracked.tick);
                    self.nearby.insert(guid, tracked.tick);
                    self.cache.clear();
                }
                _ => {
                    // New, or articulated (its parts' poses travel only in a
                    // fresh copy).
                    if known.is_some() {
                        self.predictor.remove_nearby(guid);
                    }
                    let components: Vec<(ComponentId, Vec<u8>)> = tracked
                        .components
                        .iter()
                        .filter(|(id, _)| COPYABLE.contains(id))
                        .cloned()
                        .collect();
                    match self
                        .predictor
                        .set_nearby(guid, &components, state, tracked.tick)
                    {
                        Ok(()) => {
                            self.nearby.insert(guid, tracked.tick);
                        }
                        Err(e) => {
                            log::warn!("space-client: cannot predict around {guid}: {e}");
                            self.refused.insert(guid);
                            self.nearby.remove(&guid);
                        }
                    }
                    self.cache.clear();
                }
            }
        }
        let gone: Vec<EntityGuid> = self
            .nearby
            .keys()
            .filter(|g| !keep.contains(g))
            .copied()
            .collect();
        for guid in gone {
            self.predictor.remove_nearby(guid);
            self.nearby.remove(&guid);
            self.cache.clear();
        }
    }

    /// The ship as drawn at fractional `tick` (see [`OwnShip::draw_tick`]),
    /// with the correction still being absorbed.
    pub fn shown(&mut self, tick: f64, now: Instant) -> Shown {
        let elapsed = now.saturating_duration_since(self.offset_at).as_secs_f64();
        self.offset_at = now;
        let decay = (-elapsed / CORRECTION_SMOOTHING.as_secs_f64()).exp();
        let mut state = self.interpolated(tick);
        // The offset decays — but a large one (only a late input makes one
        // over `MAX_SMOOTHED`, see `OwnShip::update`) never by more than the
        // picture has come forward since the last frame: drawn too far
        // ahead, the ship waits until it has caught up with its picture and
        // is never pulled back.
        let mut shrink = self.offset.map(|c| c * (1.0 - decay));
        // How the drawn point moves: along the line between the two ticks it
        // is interpolated between (not the interpolated velocity, which is
        // ahead of the positions when the speed has just dropped).
        let base = tick.floor().max(0.0) as u64;
        let (from, to) = (self.state_at(base).pos, self.state_at(base + 1).pos);
        let v = WorldPos::delta(from, to).map(|c| c * hz());
        let speed = v.iter().map(|c| c * c).sum::<f64>().sqrt();
        let large = self.offset.iter().map(|c| c * c).sum::<f64>().sqrt() >= MAX_SMOOTHED;
        if large && speed > 0.0 {
            let along = (0..3).map(|k| shrink[k] * v[k]).sum::<f64>() / speed;
            // How far the picture comes forward in this frame if the offset
            // stays as it is — measured from where the ship was drawn last,
            // not `speed * elapsed`: the tick drawn does not advance by
            // exactly the wall time that passed (frames are uneven, the draw
            // clock is smoothed), and by that difference the ship crept
            // back, centimetres a frame.
            let undecayed = WorldPos::from_f64_metres_around(state.pos, self.offset);
            let flown = self.last_drawn.map_or(speed * elapsed, |last| {
                let come = WorldPos::delta(last, undecayed);
                ((0..3).map(|k| come[k] * v[k]).sum::<f64>() / speed).max(0.0)
            });
            if along > flown {
                for k in 0..3 {
                    shrink[k] -= (along - flown) * v[k] / speed;
                }
            }
        }
        for k in 0..3 {
            self.offset[k] -= shrink[k];
        }
        state.pos = WorldPos::from_f64_metres_around(state.pos, self.offset);
        self.last_drawn = Some(state.pos);
        Shown { state, tick }
    }

    /// The last correction, metres: how far the shard's word was from what
    /// was predicted for its tick.
    #[must_use]
    pub fn correction(&self) -> f32 {
        self.corrections.back().map_or(0.0, |(_, c)| *c)
    }

    /// The corrections of the last few seconds: (largest, mean, count).
    #[must_use]
    pub fn corrections(&self) -> (f32, f32, usize) {
        let n = self.corrections.len();
        if n == 0 {
            return (0.0, 0.0, 0);
        }
        let max = self.corrections.iter().fold(0.0f32, |m, (_, c)| m.max(*c));
        let mean = self.corrections.iter().map(|(_, c)| c).sum::<f32>() / n as f32;
        (max, mean, n)
    }

    /// The gravity at the ship, m/s², world frame.
    #[must_use]
    pub fn gravity(&self) -> [f64; 3] {
        self.gravity
    }

    /// The scene's gravity field as the prediction has it (its sources, the
    /// shard's scene clock): what moves things with the planets.
    #[must_use]
    pub fn gravity_field(&self) -> &space_sim::GravityField {
        self.predictor.world().gravity()
    }

    /// Whether the ship stands on something in gravity: it touches a static
    /// or a body, there is a pull, and it is not climbing away.
    #[must_use]
    pub fn landed(&self) -> bool {
        let g = self.gravity.iter().map(|c| c * c).sum::<f64>().sqrt();
        self.landed && g > WEIGHTLESS
    }

    /// How many bodies and ships the prediction world holds copies of.
    #[must_use]
    pub fn nearby(&self) -> usize {
        self.nearby.len()
    }
}

fn key_of(replica: &Replica, scene: Scene) -> Option<Key> {
    Some(Key {
        guid: replica.ship?,
        tick_epoch: replica.tick_epoch,
        scene,
        scene_set: replica.scene.is_set(),
        anchor: replica.anchor(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clock runs on through a change of shard on one tick timeline,
    /// and starts over when the timeline changes.
    ///
    /// Mutation this catches: a clock keyed by shard or session (it snaps at
    /// every handoff, and everything drawn by it with it).
    #[test]
    fn the_clock_runs_on_through_a_change_of_shard() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut replica = Replica::default();
        replica.tick = 1_000;
        replica.tick_seen = Some(t0);
        replica.session = 7;
        replica.shard = Some(ShardId::new(1));
        let mut clock = Clock::new();
        let first = clock.update(&replica, t0).expect("a reading");
        assert!((first - 1_000.0).abs() < 1e-9);
        // 100 ms on, the shard changed; its newest tick is one behind what
        // the clock expects (a frame's delivery).
        replica.session = 9;
        replica.shard = Some(ShardId::new(2));
        replica.handoffs = 1;
        replica.tick = 1_002;
        replica.tick_seen = Some(at(100));
        let on = clock.update(&replica, at(100)).expect("a reading");
        assert!(
            (on - 1_003.0).abs() < 0.5 && on > 1_002.5,
            "steered, not snapped back to the newest tick heard: {on}"
        );
        // Onto another timeline: the clock is the new shard's.
        replica.tick_epoch = 1;
        replica.tick = 40;
        replica.tick_seen = Some(at(133));
        let other = clock.update(&replica, at(133)).expect("a reading");
        assert!((other - 40.0).abs() < 1e-9, "{other}");
    }

    #[test]
    fn the_ship_is_drawn_one_tick_behind_the_newest_input() {
        assert_eq!(OwnShip::draw_tick(100.0), 100.0 + INPUT_LEAD as f64 - 1.0);
    }

    /// A ship that has come to rest on the planetoid is `landed` in every
    /// frame for as long as it stands there, and not once it has lifted off.
    ///
    /// Mutation this catches: taking `landed` from the last tick's contacts
    /// alone (`OwnShip::state_at`). A ship at rest is replayed to exactly
    /// where it was, the prediction's solver puts it to sleep within a few
    /// ticks, a body asleep is not stepped and reports no contact: `landed`
    /// showed for a third of a second at touchdown and never again.
    #[test]
    fn a_ship_at_rest_on_the_planetoid_stays_landed() {
        const SESSION: u64 = 7;
        const LIFT: u64 = 330;
        let guid = EntityGuid::new(ShardId::new(1), 1);
        let anchor = WorldPos::from_metres([5_000, 0, 0]);
        let mut server = SimWorld::new(ShardId::new(1));
        space_scene::install_environment(&mut server, Scene::DemoArea, anchor)
            .expect("the demo area");
        server.spawn_ship(guid, SESSION, ShipState::at_rest(anchor));
        let c = space_scene::demo::PLANETOID_CENTER;
        // Half a metre over where its eye is when it stands on the top.
        let over = WorldPos::from_f64_metres_around(
            anchor,
            [c[0], c[1] + space_scene::demo::PLANETOID_RADIUS + 2.2, c[2]],
        );
        let input = |tick: u64| space_protocol::PlayerInput {
            entity: guid,
            thrust: [0.0, if tick >= LIFT { 1.0 } else { 0.0 }, 0.0],
            torque: [0.0; 3],
            tick,
            fast: false,
        };
        let snapshot = |server: &SimWorld| {
            let state = server.state_of(guid).expect("the ship");
            space_protocol::Message::Snapshot(space_protocol::Snapshot {
                tick: server.tick(),
                shard: ShardId::new(1),
                epoch: 1,
                entities: vec![space_protocol::EntitySnapshot::new(
                    guid,
                    vec![
                        (ShipState::ID, space_sim::wire::encode_ship_state(&state)),
                        (Owner::ID, space_protocol::codec::encode_raw(&SESSION)),
                    ],
                )],
                input_ack: server.input_ack(SESSION),
                more: false,
            })
        };
        let t0 = Instant::now();
        let mut replica = Replica::default();
        replica.session = SESSION;
        replica.shard = Some(ShardId::new(1));
        replica.scene = space_protocol::SceneState::new(Scene::DemoArea.id(), 1);
        // The first frame shows the ship on the spawn point (that is the
        // anchor the prediction builds the scene around); then it is over
        // the planetoid.
        server.step();
        replica.apply(&snapshot(&server));
        let mut own = OwnShip::new(&replica, Scene::DemoArea).expect("a prediction");
        server.despawn(guid);
        server.spawn_ship(guid, SESSION, ShipState::at_rest(over));
        let mut said = None;
        let mut sent_up_to = 1u64;
        let (mut touchdown, mut standing, mut missed, mut slept) = (None, 0u32, 0u32, false);
        let mut aloft = false;
        // Two frames a tick.
        for sub in 4..(LIFT + 45) * 2 {
            let clock = sub as f64 / 2.0;
            let now = t0 + Duration::from_secs_f64(clock / hz());
            while server.tick() < clock.floor() as u64 {
                let tick = server.tick() + 1;
                server.apply_inputs(tick, &[input(tick)]);
                server.step();
                // As a shard does: a ship at rest is not sent again.
                let state = server.state_of(guid);
                if state != said {
                    replica.apply(&snapshot(&server));
                    said = state;
                } else {
                    replica.apply(&space_protocol::Message::Delta(space_protocol::Delta {
                        tick,
                        base_tick: tick - 1,
                        shard: ShardId::new(1),
                        epoch: 1,
                        entities: vec![],
                        removed: vec![],
                        input_ack: server.input_ack(SESSION),
                    }));
                }
            }
            let newest = clock.floor() as u64 + INPUT_LEAD;
            let sent: Vec<_> = (sent_up_to + 1..=newest).map(input).collect();
            sent_up_to = sent_up_to.max(newest);
            let at = OwnShip::draw_tick(clock);
            own.update(&replica, &sent, at, now);
            let _ = own.shown(at, now);
            let tick = server.tick();
            if tick + INPUT_LEAD < LIFT {
                if own.landed() {
                    touchdown.get_or_insert(tick);
                    standing += 1;
                } else if touchdown.is_some() {
                    missed += 1;
                }
                slept |= own.predictor.world().is_sleeping(guid) == Some(true);
            } else if tick >= LIFT + 30 {
                aloft |= !own.landed();
            }
        }
        let touchdown = touchdown.expect("the ship landed");
        println!(
            "touchdown at tick {touchdown}; landed in {standing} frames, not in {missed}; the prediction's ship slept: {slept}"
        );
        assert!(touchdown < 60, "touchdown at tick {touchdown}");
        assert!(
            slept,
            "the prediction's ship never fell asleep: the test does not test"
        );
        assert!(
            standing > 500 && missed == 0,
            "landed in {standing} frames at rest, not in {missed}"
        );
        assert!(aloft, "still landed a second after lifting off");
    }

    /// What a flight that leaves fast travel at speed looked like.
    struct FastOff {
        /// The ship's speed when fast travel went off, m/s.
        speed: f64,
        /// The largest correction, metres.
        correction: f32,
        /// The smallest movement of the drawn ship between two frames, along
        /// its flight, metres (negative: it stepped back).
        back: f64,
        /// The largest, metres.
        forward: f64,
    }

    /// Fly a ship in fast travel up to ~775 m/s against a `SimWorld` that
    /// plays the shard, switch fast travel off, and draw it through
    /// [`OwnShip`] four times a tick on injected time. The shard's word on
    /// tick `t` arrives when the clock reads `t`; the input for tick `t + 3`
    /// is handed over `input_phase` of a tick after the clock read `t` (the
    /// link's sender runs on a timer of its own: `0.0` = at once). `wobble`,
    /// in ticks, is how far the draw clock runs ahead of and behind the wall
    /// clock in turn, frame by frame: a real client's frames are not evenly
    /// spaced and its draw clock is smoothed, so the tick drawn advances by
    /// a little more or less than the time that passed.
    fn fly_and_leave_fast_travel(input_phase: f64, wobble: f64) -> FastOff {
        const SESSION: u64 = 7;
        const OFF: u64 = 120;
        const SUB: u64 = 4;
        let guid = EntityGuid::new(ShardId::new(1), 1);
        let mut server = SimWorld::new(ShardId::new(1));
        server.spawn_ship(guid, SESSION, ShipState::at_rest(WorldPos::ZERO));
        // In open space the limit is the cap; the push is scaled so that the
        // velocity fast travel steers towards is 775 m/s.
        let push = (775.0 / space_sim::physics::FAST_MAX_SPEED) as f32;
        let input = |tick: u64| space_protocol::PlayerInput {
            entity: guid,
            thrust: [0.0, 0.0, -push],
            torque: [0.0; 3],
            tick,
            fast: tick < OFF,
        };
        let frame = |server: &SimWorld, fast: bool| {
            let state = server.state_of(guid).expect("the ship");
            let controls = ShipControls {
                thrust: [0.0; 3],
                torque: [0.0; 3],
                fast,
            };
            space_protocol::Message::Snapshot(space_protocol::Snapshot {
                tick: server.tick(),
                shard: ShardId::new(1),
                epoch: 1,
                entities: vec![space_protocol::EntitySnapshot::new(
                    guid,
                    vec![
                        (ShipState::ID, space_sim::wire::encode_ship_state(&state)),
                        (
                            ShipControls::ID,
                            space_protocol::codec::encode_raw(&controls),
                        ),
                        (Owner::ID, space_protocol::codec::encode_raw(&SESSION)),
                    ],
                )],
                input_ack: server.input_ack(SESSION),
                more: false,
            })
        };
        let t0 = Instant::now();
        let mut replica = Replica::default();
        replica.session = SESSION;
        replica.shard = Some(ShardId::new(1));
        // Tick 1, and the prediction built on it.
        server.apply_inputs(1, &[input(1)]);
        server.step();
        replica.apply(&frame(&server, true));
        let mut own = OwnShip::new(&replica, Scene::DemoArea).expect("a prediction");
        let mut sent_up_to = 1u64;
        let mut drawn: Option<WorldPos> = None;
        let mut out = FastOff {
            speed: 0.0,
            correction: 0.0,
            back: f64::INFINITY,
            forward: 0.0,
        };
        for sub in SUB..(OFF + 30) * SUB {
            let clock = sub as f64 / SUB as f64;
            let now = t0 + Duration::from_secs_f64(clock / hz());
            // The shard steps the tick the clock has reached and says so.
            while server.tick() < clock.floor() as u64 {
                let tick = server.tick() + 1;
                server.apply_inputs(tick, &[input(tick)]);
                if tick == OFF {
                    let v = server.state_of(guid).expect("the ship").vel;
                    out.speed = v.iter().map(|c| f64::from(*c).powi(2)).sum::<f64>().sqrt();
                }
                server.step();
                replica.apply(&frame(&server, tick < OFF));
            }
            // The link has sent the inputs up to the lead ahead of its clock.
            let newest = (clock - input_phase).floor().max(1.0) as u64 + INPUT_LEAD;
            let sent: Vec<_> = (sent_up_to + 1..=newest).map(input).collect();
            sent_up_to = sent_up_to.max(newest);
            let ahead = if sub % 2 == 0 { wobble } else { -wobble };
            let at = OwnShip::draw_tick(clock + ahead);
            own.update(&replica, &sent, at, now);
            let shown = own.shown(at, now).state.pos;
            if server.tick() + 6 > OFF {
                out.correction = out.correction.max(own.correction());
                if let Some(before) = drawn {
                    // The ship flies down -Z.
                    let step = -WorldPos::delta(before, shown)[2];
                    out.back = out.back.min(step);
                    out.forward = out.forward.max(step);
                }
                drawn = Some(shown);
            }
        }
        out
    }

    /// Leaving fast travel at speed is predicted on the tick the shard does
    /// it: the shard's word on that tick is no correction, and the drawn ship
    /// goes on without a step.
    ///
    /// Mutation this catches: taking the shard's word on fast travel before
    /// reconciling with the state it came with (`OwnShip::update`): the
    /// replay starts as if fast travel had been off the tick before, never
    /// sees the ship leave it, skips the drop to the normal top speed, and
    /// the prediction is a tick of fast flight ahead of the shard — 21.8 m at
    /// 775 m/s, shown as a step back because it is over the smoothing limit.
    #[test]
    fn leaving_fast_travel_at_speed_is_no_correction() {
        let flight = fly_and_leave_fast_travel(0.0, 0.0);
        println!(
            "fast travel off at {:.0} m/s: correction {:.3} m, drawn steps {:.3} .. {:.3} m",
            flight.speed, flight.correction, flight.back, flight.forward
        );
        assert!(flight.speed > 700.0, "at speed: {} m/s", flight.speed);
        assert!(
            flight.correction < 0.01,
            "correction {} m",
            flight.correction
        );
        assert!(
            flight.back >= -1.0e-6,
            "the drawn ship stepped back {} m",
            flight.back
        );
        assert!(
            flight.forward <= flight.speed / hz() / 4.0 * 1.01,
            "the drawn ship leapt {} m",
            flight.forward
        );
    }

    /// The same when the input that switches fast travel off reaches the
    /// prediction after frames that already drew its tick (the link's sender
    /// runs on a timer of its own): what was drawn too far ahead is kept
    /// until the ship has caught up with it, never taken back.
    ///
    /// Mutation this catches: applying a late input without carrying the
    /// difference in the visual offset, or letting that offset decay faster
    /// than the ship flies (a step back of up to 16 m at 775 m/s for an input
    /// three quarters of a tick late).
    #[test]
    fn a_late_input_that_leaves_fast_travel_is_no_step_back() {
        let flight = fly_and_leave_fast_travel(0.75, 0.0);
        println!(
            "fast travel off at {:.0} m/s, input 0.75 ticks late: correction {:.3} m, drawn steps {:.3} .. {:.3} m",
            flight.speed, flight.correction, flight.back, flight.forward
        );
        assert!(flight.speed > 700.0, "at speed: {} m/s", flight.speed);
        assert!(
            flight.correction < 0.01,
            "correction {} m",
            flight.correction
        );
        assert!(
            flight.back >= -1.0e-6,
            "the drawn ship stepped back {} m",
            flight.back
        );
        assert!(
            flight.forward <= flight.speed / hz() / 4.0 * 1.01,
            "the drawn ship leapt {} m",
            flight.forward
        );
    }

    /// And the same with frames that are not evenly spaced — which is every
    /// real client's: the tick drawn advances by more than the wall time in
    /// one frame and by less in the next. The picture still never goes back.
    ///
    /// Mutation this catches: bounding the decay of a large offset by the
    /// ship's speed times the **wall time** since the last frame
    /// (`speed * elapsed` in `OwnShip::shown`, as it was at first) instead of
    /// by how far the picture has actually come: in a frame whose draw tick
    /// advanced by less than the wall time the offset shrank by more than
    /// the ship flew — a live stack showed the drawn ship creep back 6 to
    /// 8 cm a frame for a few frames after leaving fast travel at 860 to
    /// 990 m/s.
    #[test]
    fn leaving_fast_travel_is_no_step_back_with_uneven_frames() {
        for (phase, wobble) in [(0.0, 0.06), (0.75, 0.06), (0.4, 0.1)] {
            let flight = fly_and_leave_fast_travel(phase, wobble);
            println!(
                "fast travel off at {:.0} m/s, input {phase} ticks late, draw clock +-{wobble} ticks: correction {:.3} m, drawn steps {:.3} .. {:.3} m",
                flight.speed, flight.correction, flight.back, flight.forward
            );
            assert!(flight.speed > 700.0, "at speed: {} m/s", flight.speed);
            assert!(
                flight.correction < 0.01,
                "correction {} m",
                flight.correction
            );
            assert!(
                flight.back >= -1.0e-6,
                "phase {phase}, wobble {wobble}: the drawn ship stepped back {} m",
                flight.back
            );
            assert!(
                flight.forward <= flight.speed / hz() * (0.25 + 2.0 * wobble) * 1.01,
                "phase {phase}, wobble {wobble}: the drawn ship leapt {} m",
                flight.forward
            );
        }
    }
}
