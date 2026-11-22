//! Per-client replication: what one client is told about, and when.
//!
//! Two passes, both driven by the ECS change tick, which is exactly what
//! [`SimWorld::step`] maintains (see its "Tick ordering" note):
//!
//! * a **snapshot** — every entity inside the client's interest sphere, sent on
//!   join and whenever the client's view has drifted too far for a delta to be
//!   trustworthy;
//! * a **delta** — the entities whose change tick is past the client's
//!   acknowledged tick *and* that are inside interest, plus an explicit
//!   `removed` list for entities that left the sphere or despawned.
//!
//! # Interest is a filter, not a query
//!
//! [`space_world::InterestGrid`] answers "what is near this position" in time
//! proportional to the cells the sphere touches, which is what makes ~100
//! players × ~1k entities per interest set (U9) affordable at 30 Hz. The grid is
//! keyed by a local `u64` derived from the ECS entity, because the grid knows
//! nothing about guids.
//!
//! # The budget, and what happens to what does not fit
//!
//! [`SnapshotBuilder`] bounds one frame. Entities that do not fit are remembered
//! as *owed* guids and re-encoded from the live world on the next tick — not
//! buffered as stale bytes. Re-encoding is both fresher (the entity has moved on)
//! and cheaper (a guid is 12 bytes against ~158), and it keeps the delta honest:
//! a frame always describes the tick it names, whatever it took to get there.

use std::collections::{HashMap, HashSet};

use space_ecs::{ComponentDef, Entity, WithEntity};
use space_protocol::{Delta, EntityGuid, EntitySnapshot, ShardId, Snapshot, SnapshotBuilder};
use space_sim::{Owner, ShipControls, ShipState, SimWorld};
use space_world::{InterestGrid, WorldPos};

/// The local grid key for an ECS entity: index in the low 32 bits, generation in
/// the high 32 — unique for the life of the world, because a recycled index
/// carries a new generation.
fn grid_key(entity: Entity) -> u64 {
    u64::from(entity.index()) | (u64::from(entity.generation()) << 32)
}

/// The guid ↔ entity index a shard keeps for its own ships.
///
/// [`SimWorld`] stores guid→entity for input routing but exposes no reverse
/// direction (it has no reason to). The replication pass needs the reverse — it
/// walks entities — so the shard keeps the pair here, next to the code that uses
/// it, rather than widening `space-sim` for one caller.
#[derive(Debug, Default)]
pub struct GuidIndex {
    to_entity: HashMap<EntityGuid, Entity>,
    to_guid: HashMap<Entity, EntityGuid>,
}

impl GuidIndex {
    /// An empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `guid` ↔ `entity`.
    pub fn insert(&mut self, guid: EntityGuid, entity: Entity) {
        self.to_entity.insert(guid, entity);
        self.to_guid.insert(entity, guid);
    }

    /// Forget `guid` (and its entity side).
    pub fn remove(&mut self, guid: EntityGuid) {
        if let Some(entity) = self.to_entity.remove(&guid) {
            self.to_guid.remove(&entity);
        }
    }

    /// The entity `guid` names here.
    #[must_use]
    pub fn entity_of(&self, guid: EntityGuid) -> Option<Entity> {
        self.to_entity.get(&guid).copied()
    }

    /// The guid `entity` carries.
    #[must_use]
    pub fn guid_of(&self, entity: Entity) -> Option<EntityGuid> {
        self.to_guid.get(&entity).copied()
    }
}

/// One client's replication state.
#[derive(Debug)]
pub struct ClientReplica {
    /// The client's own ship: always in its interest set, and the anchor the
    /// interest sphere is drawn around.
    pub ship: EntityGuid,
    /// The tick this client is caught up to — the `base_tick` of the next delta.
    /// Seeded from the client's own `InputTick::ack_tick` when it is ahead.
    pub ack_tick: u64,
    /// Every entity the client currently holds, so a delta can name removals.
    pub known: HashSet<EntityGuid>,
    /// Entities a previous frame could not fit, retried next tick.
    pub owed: Vec<EntityGuid>,
    /// The remaining entities of a snapshot that did not fit in one frame.
    pub pending_snapshot: Vec<EntityGuid>,
    /// Lower bound on the `input_ack` this client is told, set by an inbound
    /// handoff (see [`crate::handoff`], guarantee 5).
    pub input_ack_floor: u64,
}

impl ClientReplica {
    /// A replica that knows nothing yet and is anchored on `ship`.
    #[must_use]
    pub fn new(ship: EntityGuid) -> Self {
        Self {
            ship,
            ack_tick: 0,
            known: HashSet::new(),
            owed: Vec::new(),
            pending_snapshot: Vec::new(),
            input_ack_floor: 0,
        }
    }
}

/// The interest grid plus the guid↔key maps needed to translate between it and
/// the simulation.
#[derive(Debug, Default)]
pub struct Interest {
    grid: InterestGrid,
    by_key: HashMap<u64, EntityGuid>,
    by_guid: HashMap<EntityGuid, u64>,
}

impl Interest {
    /// An empty grid.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of tracked entities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_guid.len()
    }

    /// Whether nothing is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_guid.is_empty()
    }

    /// (Re-)place `guid` at `pos`. Called every tick for every live ship, so the
    /// grid never holds a stale position.
    pub fn upsert(&mut self, guid: EntityGuid, entity: Entity, pos: WorldPos) {
        let key = grid_key(entity);
        self.by_key.entry(key).or_insert(guid);
        self.by_guid.insert(guid, key);
        self.grid.insert(key, pos, 0.0);
    }

    /// Stop tracking `guid`.
    pub fn remove(&mut self, guid: EntityGuid) {
        if let Some(key) = self.by_guid.remove(&guid) {
            self.by_key.remove(&key);
            self.grid.remove(key);
        }
    }

    /// The guids within `radius` of `center`, nearest first.
    #[must_use]
    pub fn visible(&self, center: WorldPos, radius: f32) -> Vec<EntityGuid> {
        self.grid
            .query_sphere(center, radius, usize::MAX)
            .into_iter()
            .filter_map(|(key, _)| self.by_key.get(&key).copied())
            .collect()
    }
}

/// The replicated [`EntitySnapshot`] of one entity: [`ShipState`] + [`Owner`].
///
/// [`ShipControls`] is marked replicated in the registry but is *consumed* by
/// [`SimWorld::step`] (reset to zero every tick), so shipping it would put 24
/// bytes of guaranteed zeros on the wire per entity per frame. This mirrors
/// [`SimWorld::guid_state_hash`], which excludes it for the same reason: the two
/// functions must agree on what "the replicated state" is, because a handoff's
/// state and a snapshot's state have to be interchangeable.
#[must_use]
pub fn snapshot_of(sim: &SimWorld, guids: &GuidIndex, entity: Entity) -> Option<EntitySnapshot> {
    let guid = guids.guid_of(entity)?;
    let state = sim.world().get::<ShipState>(entity)?;
    let controls = sim
        .world()
        .get::<ShipControls>(entity)
        .copied()
        .unwrap_or_default();
    let owner = sim
        .world()
        .get::<Owner>(entity)
        .copied()
        .unwrap_or_default();
    // `EntitySnapshot::new`, **not** `entity_snapshot`: the payloads below are
    // already encoded (the wire codec for `ShipState`, `encode_raw` for the
    // others), and `entity_snapshot` would bincode-encode them a second time,
    // wrapping each in a length prefix. A handoff commit built that way cannot be
    // decoded back into a `ShipState` — the receiving shard silently falls back to
    // `ShipState::IDENTITY` and the ship teleports to the origin.
    //
    // All three replicated components are sent, per `Delta`'s contract that "a
    // component list is the entity's full replicated set". `ShipControls` is
    // consumed by the step and excluded from `guid_state_hash`, but a client
    // rendering the ship wants to know whether the engines are lit, and it is what
    // `docs/plan.md`'s 158 B/entity measurement counts.
    Some(EntitySnapshot::new(
        guid,
        vec![
            (
                <ShipState as ComponentDef>::ID,
                space_sim::wire::encode_ship_state(state),
            ),
            (
                <ShipControls as ComponentDef>::ID,
                space_protocol::codec::encode_raw(&controls),
            ),
            (
                <Owner as ComponentDef>::ID,
                space_protocol::codec::encode_raw(&owner.0),
            ),
        ],
    ))
}

/// The entities inside `center`/`radius` whose replicated state changed since
/// `base_tick`.
///
/// The change filter runs first (a tick compare per row) and the interest filter
/// second, so an idle fleet costs nothing here — the whole point of the change
/// tick model.
#[must_use]
pub fn changed_in_interest(
    sim: &mut SimWorld,
    guids: &GuidIndex,
    interest: &Interest,
    center: WorldPos,
    radius: f32,
    base_tick: u64,
) -> Vec<EntityGuid> {
    let visible: HashSet<EntityGuid> = interest.visible(center, radius).into_iter().collect();
    let mut out = Vec::new();
    {
        let query = sim.world_mut().query::<WithEntity<(&ShipState, &Owner)>>();
        query.changed_since(base_tick).for_each(|((_, _), entity)| {
            if let Some(guid) = guids.guid_of(entity) {
                if visible.contains(&guid) {
                    out.push(guid);
                }
            }
        });
    }
    out
}

/// What one replication pass produced.
#[derive(Debug)]
pub enum Frame {
    /// A chunk of a full state dump.
    Snapshot(Snapshot),
    /// A delta against its `base_tick`.
    Delta(Delta),
    /// Nothing to say this tick.
    Idle,
}

/// The shard-wide facts a frame is built from.
///
/// Grouped into one value because they are the same for every client in a tick
/// and because it keeps [`build_frame`] under clippy's argument limit.
#[derive(Debug, Clone, Copy)]
pub struct FrameContext<'a> {
    /// This shard's id, stamped on every frame.
    pub shard: ShardId,
    /// The shard map epoch, stamped on every frame.
    pub epoch: u64,
    /// Per-frame byte budget.
    pub budget: usize,
    /// The guid index used to name entities.
    pub guids: &'a GuidIndex,
}

/// Build the frame one client receives this tick, updating its replica state.
///
/// `visible` is the client's current interest set; `changed` the entities inside
/// interest that changed since `replica.ack_tick`.
pub fn build_frame(
    sim: &mut SimWorld,
    cx: FrameContext<'_>,
    replica: &mut ClientReplica,
    visible: &[EntityGuid],
    changed: &[EntityGuid],
) -> Frame {
    let tick = sim.tick();
    let input_ack = sim
        .input_ack(session_of(sim, cx.guids, replica.ship))
        .max(replica.input_ack_floor);
    let visible_set: HashSet<EntityGuid> = visible.iter().copied().collect();

    // A client that knows nothing gets a snapshot of its whole interest set.
    if replica.known.is_empty() && replica.pending_snapshot.is_empty() {
        replica.pending_snapshot = visible.to_vec();
    }
    if !replica.pending_snapshot.is_empty() {
        let mut builder = SnapshotBuilder::with_budget(cx.budget, tick, cx.shard, cx.epoch);
        let mut sent = Vec::new();
        while let Some(guid) = replica.pending_snapshot.first().copied() {
            replica.pending_snapshot.remove(0);
            let Some(entity) = sim.entity_of(guid) else {
                continue; // despawned between planning and sending
            };
            let Some(snap) = snapshot_of(sim, cx.guids, entity) else {
                continue;
            };
            if !builder.push(&snap) {
                replica.pending_snapshot.insert(0, guid);
                break;
            }
            sent.push(guid);
        }
        replica.known.extend(sent.iter().copied());
        replica.ack_tick = tick;
        return if builder.is_empty() {
            Frame::Idle
        } else {
            Frame::Snapshot(builder.finish_snapshot(input_ack))
        };
    }

    // Removals: everything the client holds that is no longer visible. The
    // client's own ship is never removed (it is the interest anchor).
    let mut removed: Vec<EntityGuid> = replica
        .known
        .iter()
        .filter(|g| !visible_set.contains(g) && **g != replica.ship)
        .copied()
        .collect();
    removed.sort_unstable();
    for guid in &removed {
        replica.known.remove(guid);
    }

    // Debt first (it is the oldest), then this tick's changes, deduplicated.
    let mut candidates: Vec<EntityGuid> = Vec::new();
    let mut seen: HashSet<EntityGuid> = HashSet::new();
    let owed = std::mem::take(&mut replica.owed);
    for guid in owed.iter().chain(changed).copied() {
        if visible_set.contains(&guid) && seen.insert(guid) {
            candidates.push(guid);
        }
    }

    let mut builder = SnapshotBuilder::with_budget(cx.budget, tick, cx.shard, cx.epoch);
    builder.reserve_removed(removed.len());
    let mut pushed = 0usize;
    for guid in &candidates {
        let Some(entity) = sim.entity_of(*guid) else {
            pushed += 1; // gone: not owed, not sent
            continue;
        };
        let Some(snap) = snapshot_of(sim, cx.guids, entity) else {
            pushed += 1;
            continue;
        };
        if !builder.push(&snap) {
            break;
        }
        pushed += 1;
        replica.known.insert(*guid);
    }
    replica.owed.extend(candidates.into_iter().skip(pushed));

    if builder.is_empty() && removed.is_empty() {
        return Frame::Idle;
    }
    let base_tick = replica.ack_tick;
    replica.ack_tick = tick;
    Frame::Delta(builder.finish_delta(base_tick, removed, input_ack))
}

/// The session driving `ship`, whose `input_ack` a frame carries.
fn session_of(sim: &SimWorld, guids: &GuidIndex, ship: EntityGuid) -> u64 {
    guids
        .entity_of(ship)
        .and_then(|e| sim.world().get::<Owner>(e))
        .map_or(0, |o| o.0)
}
