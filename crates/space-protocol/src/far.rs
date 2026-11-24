//! Seeing across shards: the view range, the distance steps and the two
//! shard-to-shard messages that carry **far replicas** (protocol 7,
//! `docs/plan-sharding.md` §6).
//!
//! A player sees every ship and body within [`FAR_RANGE`] of its own ship,
//! whichever shard simulates it. What its own shard simulates, and the ghosts
//! that shard holds of its neighbours' bodies near a border, it is told by that
//! shard as always. Everything else in range it is told by its own shard too —
//! a client keeps one play connection — which learns it from the shard that
//! simulates it: [`FarInterest`] says where a shard's players are,
//! [`FarUpdate`] answers with the entities in their range. A far replica is
//! for **seeing** only: it is in no solver, on a shard or on a client.
//!
//! # Distance steps
//!
//! How often an entity's motion is sent falls with its distance, in the steps
//! of [`STEPS`]: to a client by its distance from that client's ship, between
//! shards by its distance from the receiving shard's nearest player. The same
//! table for both, and for every entity — own, ghost or far. A step is left
//! for the coarser one only [`STEP_HYSTERESIS`] past its bound and for the
//! finer one at the bound, so an entity at a bound does not flap
//! ([`step_after`]).
//!
//! | step | out to | every | |
//! |---|---|---|---|
//! | 0 | 100 m | tick (30 Hz) | the near range: what can be touched |
//! | 1 | 250 m | 3 ticks (10 Hz) | |
//! | 2 | 1 km | 6 ticks (5 Hz) | |
//! | 3 | 10 km | 30 ticks (1 Hz) | |
//!
//! Nothing is sent for an entity that did not change, at any step: a body at
//! rest costs its first sample and nothing more to a client, and to a shard
//! twelve bytes in each refresh ([`FarUpdate::held`]).
//!
//! # Drawing sparse samples
//!
//! A client draws an entity a little in the past, between two samples.
//! [`draw_delay`] says how far, as a function of the distance alone — so it
//! does not step when an entity turns from a far replica into a ghost and then
//! into an entity of the client's own shard — and never less than the entity's
//! step needs to have the next sample in hand.
//!
//! # Where a sample is from
//!
//! An entity a shard does not simulate itself is replicated to its clients
//! with one more component, [`ORIGIN_COMPONENT`]: an [`Origin`] that says what
//! kind of copy it is and, for a far replica, the tick its state is the state
//! **of** — the simulating shard's tick, which is the cluster's
//! (`space-server`'s `clock`), a tick or two before the frame that carries it.

use serde::{Deserialize, Serialize};
use space_ecs::ComponentId;
use space_world::{WorldPos, CELL_HALF};

use crate::guid::EntityGuid;
use crate::messages::{EntitySnapshot, ENTITY_GUID_WIRE_LEN, LENGTH_PREFIX_WIRE_LEN};
use crate::shard::ShardId;

/// How far a player sees, metres: an entity further than this from every
/// player of a shard is not sent to that shard, and one further than this
/// from a client's ship is not sent to that client.
pub const FAR_RANGE: f64 = 10_000.0;

/// The distance steps: `(out to metres, ticks between two samples)`, nearest
/// first (see the [module docs](self)).
pub const STEPS: [(f64, u64); 4] = [(100.0, 1), (250.0, 3), (1_000.0, 6), (FAR_RANGE, 30)];

/// The coarsest step.
pub const COARSEST_STEP: u8 = (STEPS.len() - 1) as u8;

/// An entity moves to the coarser step only once it is this factor past the
/// finer step's bound (and back at the bound itself).
pub const STEP_HYSTERESIS: f64 = 1.1;

/// The step of an entity `distance` metres away, with no history.
#[must_use]
pub fn step_for(distance: f64) -> u8 {
    STEPS
        .iter()
        .position(|(out_to, _)| distance <= *out_to)
        .unwrap_or(STEPS.len() - 1) as u8
}

/// The step of an entity `distance` metres away that was at step `previous`:
/// finer as soon as the distance says so, coarser only [`STEP_HYSTERESIS`]
/// past the bound.
#[must_use]
pub fn step_after(distance: f64, previous: u8) -> u8 {
    let fresh = step_for(distance);
    if fresh > previous {
        step_for(distance / STEP_HYSTERESIS).max(previous)
    } else {
        fresh
    }
}

/// Ticks between two samples at `step`.
#[must_use]
pub fn interval(step: u8) -> u64 {
    STEPS[usize::from(step).min(STEPS.len() - 1)].1
}

/// What the near range is drawn behind the newest tick heard, ticks (the
/// client's interpolation delay as it always was).
pub const NEAR_DELAY: f64 = 2.0;
/// Out to here everything is drawn [`NEAR_DELAY`] behind, metres: what a
/// player can touch within the next second (the client predicts its ship
/// against the bodies within 8 to 12 m). From here the delay rises to a
/// step's own at 50 m — gently, because while the distance to a thing
/// changes, its drawn time runs faster or slower by the slope times the
/// speed (a tenth at 50 m/s).
pub const NEAR_DELAY_RANGE: f64 = 12.0;
/// From here on the delay is a step's own, metres.
const STEP_DELAY_FROM: f64 = 50.0;
/// What a sample of another shard's entity may be late by, ticks, on top of
/// the spacing of its step: a tick from the simulating shard to the player's,
/// a tick to the client, and a tick and a half of slack.
const STEP_DELAY_MARGIN: f64 = 3.5;
/// The fraction of a step's bound from which the delay rises to the next
/// step's (it is there at the bound: the coarser step may be in use from the
/// bound on, see [`step_after`]).
const DELAY_RAMP_FROM: f64 = 0.75;

/// The delay of `step` beyond the near range, ticks.
fn step_delay(step: usize) -> f64 {
    STEPS[step].1 as f64 + STEP_DELAY_MARGIN
}

/// How far behind the newest tick heard a client draws an entity `distance`
/// metres from its ship, ticks: [`NEAR_DELAY`] in the near range, then the
/// spacing of the entity's step plus a margin — continuous in the distance,
/// and at every distance at least what the coarsest step that may be in use
/// there needs.
#[must_use]
pub fn draw_delay(distance: f64) -> f64 {
    let ramp = |from: f64, to: f64, a: f64, b: f64| {
        // Eased at both ends: a corner in the delay is a step in the speed
        // something that comes nearer or goes away is drawn at.
        let t = ((distance - from) / (to - from)).clamp(0.0, 1.0);
        a + (b - a) * t * t * (3.0 - 2.0 * t)
    };
    if distance <= STEP_DELAY_FROM {
        return ramp(NEAR_DELAY_RANGE, STEP_DELAY_FROM, NEAR_DELAY, step_delay(0));
    }
    let step = usize::from(step_for(distance));
    if step + 1 >= STEPS.len() {
        return step_delay(step);
    }
    let bound = STEPS[step].0;
    ramp(
        bound * DELAY_RAMP_FROM,
        bound,
        step_delay(step),
        step_delay(step + 1),
    )
}

/// The component id an [`Origin`] travels under in a replicated component
/// list. Not a component of the simulation: a shard adds it to what it tells
/// its clients about an entity it does not simulate.
pub const ORIGIN_COMPONENT: ComponentId = 11;

/// What kind of copy a replicated entity is on the shard that sent it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginKind {
    /// A ghost: a neighbour's body near the border, in this shard's physics.
    Ghost,
    /// A far replica: seen, and in no physics.
    Far,
}

/// Where a replicated entity's state is from, when not from the shard that
/// sent it (see the [module docs](self), "Where a sample is from").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin {
    /// What kind of copy it is.
    pub kind: OriginKind,
    /// The shard that simulates the entity.
    pub shard: ShardId,
    /// A far replica: the tick its state is the state of. A ghost: 0 (a
    /// ghost's state is of the frame's own tick).
    pub tick: u64,
}

const ORIGIN_WIRE_VERSION: u8 = 1;
const ORIGIN_WIRE_LEN: usize = 14;

impl Origin {
    /// A ghost of `shard`'s entity.
    #[must_use]
    pub const fn ghost(shard: ShardId) -> Self {
        Self {
            kind: OriginKind::Ghost,
            shard,
            tick: 0,
        }
    }

    /// A far replica of `shard`'s entity as it was at `tick`.
    #[must_use]
    pub const fn far(shard: ShardId, tick: u64) -> Self {
        Self {
            kind: OriginKind::Far,
            shard,
            tick,
        }
    }

    /// The wire form: version, kind, shard, tick (big endian, 14 bytes).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(ORIGIN_WIRE_LEN);
        out.push(ORIGIN_WIRE_VERSION);
        out.push(match self.kind {
            OriginKind::Ghost => 1,
            OriginKind::Far => 2,
        });
        out.extend_from_slice(&self.shard.0.to_be_bytes());
        out.extend_from_slice(&self.tick.to_be_bytes());
        out
    }

    /// Read the wire form; `None` for anything else (another version, an
    /// unknown kind, a wrong length).
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != ORIGIN_WIRE_LEN || bytes[0] != ORIGIN_WIRE_VERSION {
            return None;
        }
        let kind = match bytes[1] {
            1 => OriginKind::Ghost,
            2 => OriginKind::Far,
            _ => return None,
        };
        let shard = u32::from_be_bytes(bytes[2..6].try_into().ok()?);
        let tick = u64::from_be_bytes(bytes[6..14].try_into().ok()?);
        Some(Self {
            kind,
            shard: ShardId(shard),
            tick,
        })
    }
}

/// The most players one [`FarInterest`] names; a receiver drops a message
/// that names more (a shard takes 100 players).
pub const MAX_FAR_PLAYERS: usize = 256;
/// The most guids one [`FarInterest`] asks for again.
pub const MAX_FAR_MISSING: usize = 256;
/// The most bytes of entities, removals and names one [`FarUpdate`] carries
/// ([`FarUpdate::wire_len`]); a receiver drops a larger one unread.
pub const MAX_FAR_UPDATE_BYTES: usize = 32 * 1024;

/// Where one player is ([`FarInterest::players`]): a `WorldPos` as two plain
/// arrays (`space-world` carries no serde).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FarPlayer {
    /// The cell.
    pub cell: [i64; 3],
    /// The place in the cell, metres.
    pub local: [f32; 3],
}

impl FarPlayer {
    /// The wire form of `pos`.
    #[must_use]
    pub fn at(pos: WorldPos) -> Self {
        Self {
            cell: pos.cell,
            local: pos.local,
        }
    }

    /// The position, if it is one: finite and inside its cell (what a peer
    /// sends is not trusted to be).
    #[must_use]
    pub fn position(&self) -> Option<WorldPos> {
        self.local
            .iter()
            .all(|c| c.is_finite() && c.abs() <= CELL_HALF)
            .then_some(WorldPos {
                cell: self.cell,
                local: self.local,
            })
    }
}

/// Shard → every other shard of the map, a few times a second: "my players
/// are here" (protocol 7). The receiver sends, in [`FarUpdate`]s, what it
/// simulates within [`FAR_RANGE`] of one of them and does not already send as
/// a ghost. A shard with no players says so with an empty list, and is then
/// sent nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FarInterest {
    /// The shard whose players these are (checked against the connection).
    pub from: ShardId,
    /// The sender's tick.
    pub tick: u64,
    /// Where its players' ships are, at most [`MAX_FAR_PLAYERS`].
    pub players: Vec<FarPlayer>,
    /// Entities of the receiver's the sender was told to hold (a changed
    /// component of one, or its name in a refresh) and does not: to be sent
    /// again in full. At most [`MAX_FAR_MISSING`].
    pub missing: Vec<EntityGuid>,
}

/// Shard → shard: far replicas (protocol 7) — what the sender simulates in
/// view of the receiver's players, beyond the border margin its ghosts cover.
///
/// Soft state: an entity the receiver is not known to hold comes in `new`
/// with everything a client needs to draw it, one it holds in `changed` with
/// its `ShipState` and whatever else changed (an empty payload removes a
/// component), each at most as often as its step allows and only when it
/// changed. `removed` names what left the range, was despawned or is no
/// longer the sender's. A **refresh** (every two seconds) names in `held`
/// every far replica the receiver should hold from the sender; what the
/// receiver holds beyond that goes, and what it lacks it asks for
/// ([`FarInterest::missing`]). A shard that sends nothing for three seconds
/// loses its far replicas on the receiver.
///
/// Bounded: [`FarUpdate::wire_len`] is at most [`MAX_FAR_UPDATE_BYTES`] —
/// what does not fit waits for the next tick, and a refresh that does not fit
/// goes out in chunks (`more`), like a [`crate::Snapshot`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FarUpdate {
    /// The shard that simulates the entities (checked against the
    /// connection).
    pub from: ShardId,
    /// The tick the states are the states of: the sender's, which is the
    /// cluster's.
    pub tick: u64,
    /// Entities the receiver does not hold yet, in full.
    pub new: Vec<EntitySnapshot>,
    /// Entities it holds: what changed.
    pub changed: Vec<EntitySnapshot>,
    /// Entities it must no longer show as the sender's.
    pub removed: Vec<EntityGuid>,
    /// `held` is (a chunk of) a refresh.
    pub refresh: bool,
    /// A refresh: every far replica the receiver should hold from the sender.
    pub held: Vec<EntityGuid>,
    /// Further chunks of this refresh follow.
    pub more: bool,
}

impl FarUpdate {
    /// The bytes of entities and names this update carries, as the codec
    /// writes them (the fixed fields of the message are a few dozen more).
    #[must_use]
    pub fn wire_len(&self) -> usize {
        let entities = |list: &[EntitySnapshot]| {
            LENGTH_PREFIX_WIRE_LEN + list.iter().map(EntitySnapshot::encoded_len).sum::<usize>()
        };
        let guids = |n: usize| LENGTH_PREFIX_WIRE_LEN + n * ENTITY_GUID_WIRE_LEN;
        entities(&self.new)
            + entities(&self.changed)
            + guids(self.removed.len())
            + guids(self.held.len())
    }
}

/// The bytes one [`FarPlayer`] costs on the wire.
pub const FAR_PLAYER_WIRE_LEN: usize = 3 * 8 + 3 * 4;

impl FarInterest {
    /// The bytes of players and names this message carries.
    #[must_use]
    pub fn wire_len(&self) -> usize {
        2 * LENGTH_PREFIX_WIRE_LEN
            + self.players.len() * FAR_PLAYER_WIRE_LEN
            + self.missing.len() * ENTITY_GUID_WIRE_LEN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The steps are the table of the module docs, and a step is left for
    /// the coarser one only past the hysteresis.
    ///
    /// Mutation this catches: `step_after` returning `step_for` (no
    /// hysteresis): an entity 100 m away then changes its step with every
    /// centimetre it moves to and fro.
    #[test]
    fn the_steps_follow_the_distance_with_hysteresis() {
        assert_eq!(step_for(0.0), 0);
        assert_eq!(step_for(100.0), 0);
        assert_eq!(step_for(100.1), 1);
        assert_eq!(step_for(250.1), 2);
        assert_eq!(step_for(1_000.1), 3);
        assert_eq!(step_for(1e9), 3, "the range is cut elsewhere");
        assert_eq!([0, 1, 2, 3].map(interval), [1, 3, 6, 30]);
        // Going out: step 0 holds to 110 m.
        assert_eq!(step_after(105.0, 0), 0);
        assert_eq!(step_after(110.5, 0), 1);
        // Coming in: step 1 holds to the bound itself.
        assert_eq!(step_after(105.0, 1), 1);
        assert_eq!(step_after(99.9, 1), 0);
        // Two steps at once, and never finer than the hysteresis allows.
        assert_eq!(step_after(5_000.0, 0), 3);
        assert_eq!(step_after(260.0, 1), 1);
        assert_eq!(step_after(260.0, 3), 2);
    }

    /// The delay is continuous, never falls with the distance and covers the
    /// spacing of the coarsest step that may be in use at a distance.
    ///
    /// Mutation this catches: ramping up to a step's delay *behind* its bound
    /// instead of before it (the entity then runs out of samples for the
    /// length of the ramp), or a delay that steps (an entity at a bound jumps
    /// by the difference times its speed).
    #[test]
    fn the_draw_delay_is_continuous_and_covers_the_step_in_use() {
        assert_eq!(draw_delay(0.0), NEAR_DELAY);
        assert_eq!(draw_delay(NEAR_DELAY_RANGE), NEAR_DELAY);
        let mut d = 0.0f64;
        let mut last = draw_delay(0.0);
        while d < 12_000.0 {
            d += 0.25;
            let now = draw_delay(d);
            assert!(now >= last, "at {d} m");
            assert!(now - last < 0.1, "a step of {} ticks at {d} m", now - last);
            // The coarsest step that may be in use here: the one of the
            // distance itself (see `step_after`).
            if d > STEP_DELAY_FROM {
                let needs = interval(step_for(d)) as f64 + STEP_DELAY_MARGIN;
                assert!(now >= needs - 1e-9, "{now} < {needs} at {d} m");
            }
            last = now;
        }
        assert_eq!(draw_delay(20_000.0), 33.5);
    }

    /// An origin round-trips, and nothing else decodes as one.
    #[test]
    fn an_origin_round_trips() {
        for origin in [
            Origin::ghost(ShardId(7)),
            Origin::far(ShardId(u32::MAX - 1), u64::MAX - 3),
        ] {
            assert_eq!(Origin::decode(&origin.encode()), Some(origin));
        }
        let mut bytes = Origin::far(ShardId(1), 9).encode();
        bytes[1] = 3;
        assert_eq!(Origin::decode(&bytes), None, "an unknown kind");
        bytes[1] = 2;
        bytes[0] = 2;
        assert_eq!(Origin::decode(&bytes), None, "another version");
        assert_eq!(Origin::decode(&[1, 2, 0]), None, "too short");
    }

    /// A player's position off the wire is one only if it is finite and
    /// inside its cell.
    #[test]
    fn a_far_player_is_validated() {
        let pos = WorldPos::new([3, -2, 9], [10.0, -500.0, 0.25]);
        assert_eq!(FarPlayer::at(pos).position(), Some(pos));
        for bad in [f32::NAN, f32::INFINITY, 1e30, -600.0] {
            let player = FarPlayer {
                cell: [0; 3],
                local: [0.0, bad, 0.0],
            };
            assert_eq!(player.position(), None, "{bad}");
        }
    }

    /// The computed size is what the codec writes for the lists.
    ///
    /// Mutation this catches: leaving a list's length prefix or `changed` out
    /// of `wire_len` — the sender's budget and the receiver's bound would
    /// then both be off.
    #[test]
    fn the_wire_len_of_an_update_is_exact() {
        let guid = |n: u64| EntityGuid::new(ShardId(2), n);
        let entity = |n: u64| EntitySnapshot::new(guid(n), vec![(0, vec![7; 76]), (5, vec![1; 5])]);
        let empty = FarUpdate {
            from: ShardId(2),
            tick: 1,
            new: Vec::new(),
            changed: Vec::new(),
            removed: Vec::new(),
            refresh: false,
            held: Vec::new(),
            more: false,
        };
        let full = FarUpdate {
            new: vec![entity(1), entity(2)],
            changed: vec![entity(3)],
            removed: vec![guid(4)],
            refresh: true,
            held: vec![guid(1), guid(2), guid(3)],
            ..empty.clone()
        };
        let len = |u: &FarUpdate| crate::codec::encoded_len(&crate::Message::FarUpdate(u.clone()));
        assert_eq!(len(&full) - len(&empty), full.wire_len() - empty.wire_len());
        assert_eq!(empty.wire_len(), 4 * LENGTH_PREFIX_WIRE_LEN);
        let interest = FarInterest {
            from: ShardId(2),
            tick: 1,
            players: vec![FarPlayer::at(WorldPos::ZERO); 3],
            missing: vec![guid(1)],
        };
        let none = FarInterest {
            players: Vec::new(),
            missing: Vec::new(),
            ..interest.clone()
        };
        let len =
            |i: &FarInterest| crate::codec::encoded_len(&crate::Message::FarInterest(i.clone()));
        assert_eq!(
            len(&interest) - len(&none),
            interest.wire_len() - none.wire_len()
        );
    }
}
