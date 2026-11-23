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
//! (a stall, a shard transition: every shard has its own clock). Other
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
    /// The shard and session the estimate is for: a transition starts over.
    domain: (u64, u64),
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
        let domain = (replica.handoffs, replica.session);
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

/// What a prediction is for: a change of any of these builds it anew.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    guid: EntityGuid,
    session: u64,
    handoffs: u64,
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
        if let Some(anchor) = key.anchor.filter(|_| key.scene_set) {
            if let Err(e) = space_scene::install_environment(&mut world, scene, anchor) {
                log::warn!("space-client: the prediction has no scene: {e}");
            }
        }
        let predictor =
            PredictedPhysics::new(world, key.guid, key.session, state, replica.state_tick())
                .ok()?;
        Some(Self {
            key,
            predictor,
            cache: BTreeMap::new(),
            nearby: BTreeMap::new(),
            refused: BTreeSet::new(),
            offset: [0.0; 3],
            offset_at: Instant::now(),
            corrections: VecDeque::new(),
            landed: false,
            gravity: [0.0; 3],
        })
    }

    /// Whether this prediction is still the one for `replica` in `scene`
    /// (same ship, session, shard clock, scene and anchor).
    #[must_use]
    pub fn is_for(&self, replica: &Replica, scene: Scene) -> bool {
        key_of(replica, scene) == Some(self.key)
    }

    /// Carry the visual offset of `previous` over (a new prediction after a
    /// shard transition should not jump either).
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
        self.landed = world
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
        // The shard's word. A ship at rest is not sent while it stays at rest:
        // then the newest tick heard is the tick of its last state too.
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
        while self
            .corrections
            .front()
            .is_some_and(|(at, _)| now.saturating_duration_since(*at) > CORRECTION_WINDOW)
        {
            self.corrections.pop_front();
        }
        // The inputs sent.
        for input in sent {
            if input.entity == self.key.guid {
                self.predictor.apply_input(*input);
                self.cache.clear();
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
        self.offset = self.offset.map(|c| c * decay);
        let mut state = self.interpolated(tick);
        state.pos = WorldPos::from_f64_metres_around(state.pos, self.offset);
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
        session: replica.session,
        handoffs: replica.handoffs,
        scene,
        scene_set: replica.scene.is_set(),
        anchor: replica.anchor(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ship_is_drawn_one_tick_behind_the_newest_input() {
        assert_eq!(OwnShip::draw_tick(100.0), 100.0 + INPUT_LEAD as f64 - 1.0);
    }
}
