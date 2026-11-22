//! Per-client replication: what one client is told about, and when.
//!
//! Two passes, both driven by the ECS change tick, which is exactly what
//! [`SimWorld::step`] maintains (see its "Tick ordering" note):
//!
//! * a **snapshot** — every entity inside the client's interest sphere, sent on
//!   join and whenever the client's view has drifted too far for a delta to be
//!   trustworthy;
//! * a **delta** — the entities whose change tick is past the tick the client was
//!   last examined at *and* that are inside interest, the entities that have just
//!   entered interest, plus an explicit `removed` list for entities that left the
//!   sphere or despawned.
//!
//! # A frame every tick
//!
//! A client that has a ship is sent a frame on **every** tick, even when nothing
//! in its interest set changed: an empty [`Delta`] still carries the tick and the
//! `input_ack`. That is not padding. A client has no other source for the
//! shard's clock, and it tags its inputs relative to the last tick it heard; a
//! shard that goes silent while a ship is at rest therefore stops the client's
//! clock, every later input arrives tagged with a tick the shard has already
//! simulated, [`SimWorld::apply_inputs`] drops it as stale, nothing changes, and
//! the silence is permanent. One lost datagram was enough to start it.
//!
//! # Interest is a filter, not a query
//!
//! [`space_world::InterestGrid`] answers "what is near this position" in time
//! proportional to the cells the sphere touches, which is what makes ~100
//! players × ~1k entities per interest set (U9) affordable at 30 Hz. The grid is
//! keyed by a local `u64` derived from the ECS entity, because the grid knows
//! nothing about guids. The sphere is queried **once per client per tick** and
//! the result is shared by every step that needs it.
//!
//! # The budget, and what happens to what does not fit
//!
//! [`SnapshotBuilder`] bounds one frame. Entities that do not fit are remembered
//! as *owed* guids and re-encoded from the live world on the next tick — not
//! buffered as stale bytes. Re-encoding is both fresher (the entity has moved on)
//! and cheaper (a guid is 12 bytes against ~158), and it keeps the delta honest:
//! a frame always describes the tick it names, whatever it took to get there.
//!
//! # Frames get lost, and what is done about it
//!
//! A small delta rides an unreliable datagram, and the front end drops a frame
//! outright when a client's write queue is full. The replica is advanced when a
//! frame is *built*, so a frame that never arrives would otherwise take its
//! contents with it for good: a removal is sent once, and an entity that stops
//! changing after its last update is never sent again.
//!
//! So every frame that carried something is remembered ([`SentFrame`]) until the
//! client's `InputTick::ack_tick` settles it. The client acknowledges the highest
//! tick it has applied, which leaves three cases ([`ClientReplica::acknowledge`]):
//! a frame whose tick is acknowledged **exactly** arrived; a frame *older* than
//! the acknowledged tick that was never acknowledged itself is **presumed lost**
//! and its entities and removals are owed again (minus whatever the frame that
//! did arrive carried as well); a newer frame is still in flight. Presuming loss
//! can be wrong (two frames arrived between two input batches) and then costs
//! one redundant re-send — of current state, so it is harmless. A client that
//! never acknowledges anything (it sends no input) simply keeps the old
//! send-and-forget behaviour.
//!
//! That leaves the one frame no acknowledgement can vouch for: the first. A
//! client learns which ship is its own from it, and says nothing — no input, so
//! no acknowledgement — until it has. So until the first acknowledgement arrives
//! every frame repeats the client's own ship ([`UNACKNOWLEDGED_SHIP_TICKS`]),
//! and whichever frame gets there first is enough to start the conversation.
//!
//! # Encoding once
//!
//! An entity is encoded once per tick ([`PassCache`]) however many clients can
//! see it; each client's frame then takes a copy of those bytes. Its encoded
//! size is computed, not measured ([`EntitySnapshot::encoded_len`]).

use std::collections::{HashMap, HashSet, VecDeque};

use space_ecs::{ComponentDef, Entity, WithEntity};
use space_protocol::{Delta, EntityGuid, EntitySnapshot, ShardId, Snapshot, SnapshotBuilder};
use space_sim::{Owner, ShipControls, ShipState, SimWorld};
use space_world::{InterestGrid, WorldPos};

/// How many unacknowledged frames are remembered per client.
///
/// About a second at 30 Hz: an acknowledgement that has not arrived by then is
/// not coming (the client sends no input, or is gone), and the frame is taken to
/// have arrived — which is what the shard assumed for every frame before it
/// tracked acknowledgements at all. It also bounds the bookkeeping for a client
/// that never acknowledges anything.
pub const MAX_IN_FLIGHT_FRAMES: usize = 32;

/// For how many ticks after a client's first frame its own ship is repeated in
/// every frame, while the client has not acknowledged anything yet.
///
/// The frame that names a client's own ship is the one frame it cannot do
/// without: until it knows its ship it sends no input, and a client that sends
/// no input sends no acknowledgement either — so a lost first snapshot could
/// never be noticed, let alone repaired. It is also the frame most likely to be
/// overtaken: the snapshot rides a reliable stream, the next tick's delta rides
/// a datagram, and a client that applies the delta first discards the "older"
/// snapshot. Repeating the ship (≈158 B) until the first acknowledgement makes
/// whichever frame arrives first sufficient; the acknowledgement then shows
/// whether the snapshot itself arrived. The bound is for a client that never
/// acknowledges at all: it is not sent its parked ship forever.
pub const UNACKNOWLEDGED_SHIP_TICKS: u64 = 30;

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

/// What one frame carried, kept until the client's acknowledgement settles it
/// (see the module docs, "Frames get lost").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentFrame {
    /// The tick the frame described.
    pub tick: u64,
    /// The entities it carried.
    pub entities: Vec<EntityGuid>,
    /// The removals it carried.
    pub removed: Vec<EntityGuid>,
}

/// One client's replication state.
#[derive(Debug)]
pub struct ClientReplica {
    /// The client's own ship: always in its interest set, and the anchor the
    /// interest sphere is drawn around.
    pub ship: EntityGuid,
    /// The tick of the last frame built for this client — the `base_tick` of the
    /// next delta.
    pub ack_tick: u64,
    /// The tick up to which this client's interest set has been examined for
    /// changes: the next delta carries what changed after it.
    ///
    /// Equal to [`ClientReplica::ack_tick`] except while a snapshot is going out
    /// in several chunks. A chunk carries no changes, so the base stays at the
    /// *first* chunk's tick until the snapshot is complete; the delta that
    /// follows then picks up everything that moved while the chunks went out.
    /// (Advancing it with every chunk lost the changes to an entity that was in
    /// an early chunk and stopped moving before the last one.)
    pub change_base: u64,
    /// The highest tick the client itself has confirmed applying
    /// (`InputTick::ack_tick`, clamped to the shard's own tick).
    pub client_ack: u64,
    /// The tick of the first frame built for this client, once there is one
    /// (see [`UNACKNOWLEDGED_SHIP_TICKS`]).
    pub introduced_at: Option<u64>,
    /// Every entity the client has been sent and not been told to remove, so a
    /// delta can name removals.
    pub known: HashSet<EntityGuid>,
    /// Entities that must be (re-)sent whether or not they change again: the
    /// ones a previous frame could not fit, and the ones a frame that is
    /// presumed lost was carrying.
    pub owed: Vec<EntityGuid>,
    /// Removals a frame that is presumed lost was carrying, to be sent again.
    pub owed_removals: Vec<EntityGuid>,
    /// The remaining entities of a snapshot that did not fit in one frame.
    pub pending_snapshot: VecDeque<EntityGuid>,
    /// Frames sent and not yet settled by an acknowledgement, oldest first.
    pub in_flight: VecDeque<SentFrame>,
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
            change_base: 0,
            client_ack: 0,
            introduced_at: None,
            known: HashSet::new(),
            owed: Vec::new(),
            owed_removals: Vec::new(),
            pending_snapshot: VecDeque::new(),
            in_flight: VecDeque::new(),
            input_ack_floor: 0,
        }
    }

    /// The client reports having applied the frame for tick `ack` (its highest).
    ///
    /// Settles every remembered frame up to `ack`: the one whose tick is exactly
    /// `ack` arrived; an older one that was never acknowledged itself is presumed
    /// lost, and what it carried is owed again — except what the frame that
    /// *did* arrive carried too, since that is newer and the client has it.
    /// Frames newer than `ack` stay in flight. An acknowledgement that does not
    /// move forward (a reordered or repeated input batch) changes nothing.
    pub fn acknowledge(&mut self, ack: u64) {
        if ack <= self.client_ack {
            return;
        }
        self.client_ack = ack;
        let mut delivered: Option<SentFrame> = None;
        let mut lost_entities: Vec<EntityGuid> = Vec::new();
        let mut lost_removals: Vec<EntityGuid> = Vec::new();
        while self.in_flight.front().is_some_and(|f| f.tick <= ack) {
            let Some(frame) = self.in_flight.pop_front() else {
                break;
            };
            if frame.tick == ack {
                delivered = Some(frame);
            } else {
                lost_entities.extend(frame.entities);
                lost_removals.extend(frame.removed);
            }
        }
        if lost_entities.is_empty() && lost_removals.is_empty() {
            return;
        }
        if let Some(delivered) = delivered {
            // A ship that moves every tick is in every frame: without this, each
            // acknowledgement that skips a frame would owe the whole moving
            // fleet again although the newer frame already delivered it.
            let entities: HashSet<EntityGuid> = delivered.entities.into_iter().collect();
            let removals: HashSet<EntityGuid> = delivered.removed.into_iter().collect();
            lost_entities.retain(|guid| !entities.contains(guid));
            lost_removals.retain(|guid| !removals.contains(guid));
        }
        self.owed.extend(lost_entities);
        self.owed_removals.extend(lost_removals);
    }

    /// Whether the client may still not know which ship is its own: it has not
    /// acknowledged a single frame, and its first frame is recent (see
    /// [`UNACKNOWLEDGED_SHIP_TICKS`]).
    fn ship_is_unconfirmed(&self, tick: u64) -> bool {
        self.client_ack == 0
            && self
                .introduced_at
                .is_some_and(|first| tick.saturating_sub(first) <= UNACKNOWLEDGED_SHIP_TICKS)
    }

    /// Remember what the frame for `tick` carried until it is acknowledged.
    ///
    /// A frame that carried nothing (a heartbeat) has nothing to lose and is not
    /// remembered. The oldest entry is dropped — taken as delivered — once more
    /// than [`MAX_IN_FLIGHT_FRAMES`] are waiting.
    fn remember(&mut self, tick: u64, entities: Vec<EntityGuid>, removed: Vec<EntityGuid>) {
        if entities.is_empty() && removed.is_empty() {
            return;
        }
        self.in_flight.push_back(SentFrame {
            tick,
            entities,
            removed,
        });
        while self.in_flight.len() > MAX_IN_FLIGHT_FRAMES {
            self.in_flight.pop_front();
        }
    }
}

/// The interest grid plus the guid↔key maps needed to translate between it and
/// the simulation.
#[derive(Debug, Default)]
pub struct Interest {
    grid: InterestGrid,
    by_key: HashMap<u64, EntityGuid>,
    /// guid → its grid key and the position it was last placed at.
    by_guid: HashMap<EntityGuid, (u64, WorldPos)>,
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
    ///
    /// A ship that has not moved since it was last placed costs one lookup: the
    /// grid is only touched when the position (or the entity behind the guid)
    /// actually changed, so a parked fleet does not churn the grid's cells every
    /// tick.
    pub fn upsert(&mut self, guid: EntityGuid, entity: Entity, pos: WorldPos) {
        let key = grid_key(entity);
        if self.by_guid.get(&guid) == Some(&(key, pos)) {
            return;
        }
        if let Some((old_key, _)) = self.by_guid.insert(guid, (key, pos)) {
            if old_key != key {
                // The guid moved to a different entity without a `remove` in
                // between: drop the stale grid entry rather than leave a ghost.
                self.by_key.remove(&old_key);
                self.grid.remove(old_key);
            }
        }
        self.by_key.insert(key, guid);
        self.grid.insert(key, pos, 0.0);
    }

    /// Stop tracking `guid`.
    pub fn remove(&mut self, guid: EntityGuid) {
        if let Some((key, _)) = self.by_guid.remove(&guid) {
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

/// Every entity whose replicated state changed since `base_tick`, in the ECS's
/// own row order.
///
/// The change filter is a tick compare per row, so an idle fleet costs nothing
/// here — the whole point of the change tick model. The result is the same for
/// every client examined at the same base, which is why [`PassCache`] computes
/// it once per base and each client then filters it by its own interest set.
#[must_use]
pub fn changed_since(sim: &mut SimWorld, guids: &GuidIndex, base_tick: u64) -> Vec<EntityGuid> {
    let mut out = Vec::new();
    let query = sim.world_mut().query::<WithEntity<(&ShipState, &Owner)>>();
    query.changed_since(base_tick).for_each(|((_, _), entity)| {
        if let Some(guid) = guids.guid_of(entity) {
            out.push(guid);
        }
    });
    out
}

/// What one tick's replication pass shares between clients.
///
/// Two things are the same for every client in a tick and were being recomputed
/// for each one: the set of entities that changed since a given tick, and each
/// entity's encoded snapshot. Both are computed on first use and reused.
#[derive(Debug, Default)]
pub struct PassCache {
    /// guid → its encoded snapshot this tick (`None`: not alive here).
    snapshots: HashMap<EntityGuid, Option<EntitySnapshot>>,
    /// change base → every guid changed since it.
    changed: HashMap<u64, Vec<EntityGuid>>,
}

impl PassCache {
    /// An empty cache, for one tick.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `guid`'s snapshot as of this tick, encoded on first request.
    fn snapshot(
        &mut self,
        sim: &SimWorld,
        guids: &GuidIndex,
        guid: EntityGuid,
    ) -> Option<&EntitySnapshot> {
        self.snapshots
            .entry(guid)
            .or_insert_with(|| {
                sim.entity_of(guid)
                    .and_then(|entity| snapshot_of(sim, guids, entity))
            })
            .as_ref()
    }

    /// The guids changed since `base_tick`, scanned on first request.
    fn changed_since(
        &mut self,
        sim: &mut SimWorld,
        guids: &GuidIndex,
        base_tick: u64,
    ) -> &[EntityGuid] {
        self.changed
            .entry(base_tick)
            .or_insert_with(|| changed_since(sim, guids, base_tick))
    }
}

/// What one replication pass produced for one client.
#[derive(Debug)]
pub enum Frame {
    /// A chunk of a full state dump.
    Snapshot(Snapshot),
    /// A delta against its `base_tick` — possibly an empty one (see the module
    /// docs, "A frame every tick").
    Delta(Delta),
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

/// One client's interest set this tick, in both shapes the frame needs.
#[derive(Debug, Clone, Copy)]
pub struct Visible<'a> {
    /// The guids inside the interest sphere, nearest first.
    pub ordered: &'a [EntityGuid],
    /// The same guids, for membership tests.
    pub set: &'a HashSet<EntityGuid>,
}

/// Build the frame one client receives this tick, updating its replica state.
///
/// `visible` is the client's current interest set, queried once by the caller;
/// `cache` is the tick's shared [`PassCache`].
///
/// Always returns a frame: a client that is being served is told the tick and
/// its `input_ack` every tick, even when nothing else changed.
pub fn build_frame(
    sim: &mut SimWorld,
    cx: FrameContext<'_>,
    cache: &mut PassCache,
    replica: &mut ClientReplica,
    visible: Visible<'_>,
) -> Frame {
    let tick = sim.tick();
    let input_ack = sim
        .input_ack(session_of(sim, cx.guids, replica.ship))
        .max(replica.input_ack_floor);
    replica.introduced_at.get_or_insert(tick);

    // A client that knows nothing gets a snapshot of its whole interest set.
    if replica.known.is_empty() && replica.pending_snapshot.is_empty() {
        replica.pending_snapshot = visible.ordered.iter().copied().collect();
        // Everything the snapshot carries is at least as new as this tick, so
        // this is where the change scan resumes once the snapshot is complete.
        replica.change_base = tick;
    }
    if !replica.pending_snapshot.is_empty() {
        let mut builder = SnapshotBuilder::with_budget(cx.budget, tick, cx.shard, cx.epoch);
        let mut sent = Vec::new();
        while let Some(&guid) = replica.pending_snapshot.front() {
            let Some(snap) = cache.snapshot(sim, cx.guids, guid) else {
                // Despawned between planning and sending.
                replica.pending_snapshot.pop_front();
                continue;
            };
            if !builder.push(snap) {
                break; // stays queued for the next chunk
            }
            replica.pending_snapshot.pop_front();
            sent.push(guid);
        }
        if !builder.is_empty() {
            replica.known.extend(sent.iter().copied());
            replica.ack_tick = tick;
            replica.remember(tick, sent, Vec::new());
            return Frame::Snapshot(builder.finish_snapshot(input_ack));
        }
        // Everything that was planned has despawned: there is nothing to dump,
        // so this tick is an ordinary delta.
    }

    // Removals: everything the client holds that is no longer visible. The
    // client's own ship is never removed (it is the interest anchor). A removal
    // that went out on a frame presumed lost is said again, unless the entity
    // has come back in the meantime.
    let mut removed: Vec<EntityGuid> = replica
        .known
        .iter()
        .filter(|g| !visible.set.contains(g) && **g != replica.ship)
        .copied()
        .collect();
    for guid in std::mem::take(&mut replica.owed_removals) {
        if guid != replica.ship && !visible.set.contains(&guid) && !replica.known.contains(&guid) {
            removed.push(guid);
        }
    }
    removed.sort_unstable();
    removed.dedup();
    for guid in &removed {
        replica.known.remove(guid);
    }

    // The client's own ship first, for as long as it may not have been told
    // which ship that is (see [`UNACKNOWLEDGED_SHIP_TICKS`]). Then debt (it is
    // the oldest), then this tick's changes, then whatever has just come into
    // interest without changing — an entity at rest never shows up in the change
    // scan, so entering the sphere has to be a reason to send it in its own
    // right. Deduplicated, and restricted to what is visible now.
    let mut candidates: Vec<EntityGuid> = Vec::new();
    let mut seen: HashSet<EntityGuid> = HashSet::new();
    let own_ship = replica.ship_is_unconfirmed(tick).then_some(replica.ship);
    let owed = std::mem::take(&mut replica.owed);
    let changed = cache.changed_since(sim, cx.guids, replica.change_base);
    let entered = visible
        .ordered
        .iter()
        .filter(|g| !replica.known.contains(g));
    for guid in own_ship
        .iter()
        .chain(owed.iter())
        .chain(changed)
        .chain(entered)
        .copied()
    {
        if visible.set.contains(&guid) && seen.insert(guid) {
            candidates.push(guid);
        }
    }

    let mut builder = SnapshotBuilder::with_budget(cx.budget, tick, cx.shard, cx.epoch);
    builder.reserve_removed(removed.len());
    let mut sent = Vec::new();
    let mut taken = 0usize;
    for guid in &candidates {
        let Some(snap) = cache.snapshot(sim, cx.guids, *guid) else {
            taken += 1; // gone: not owed, not sent
            continue;
        };
        if !builder.push(snap) {
            break;
        }
        taken += 1;
        replica.known.insert(*guid);
        sent.push(*guid);
    }
    replica.owed.extend(candidates.into_iter().skip(taken));

    let base_tick = replica.ack_tick;
    replica.ack_tick = tick;
    replica.change_base = tick;
    replica.remember(tick, sent, removed.clone());
    Frame::Delta(builder.finish_delta(base_tick, removed, input_ack))
}

/// The session driving `ship`, whose `input_ack` a frame carries.
fn session_of(sim: &SimWorld, guids: &GuidIndex, ship: EntityGuid) -> u64 {
    guids
        .entity_of(ship)
        .and_then(|e| sim.world().get::<Owner>(e))
        .map_or(0, |o| o.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guid(serial: u64) -> EntityGuid {
        EntityGuid::new(ShardId::new(1), serial)
    }

    fn frame(tick: u64, entities: &[u64], removed: &[u64]) -> SentFrame {
        SentFrame {
            tick,
            entities: entities.iter().copied().map(guid).collect(),
            removed: removed.iter().copied().map(guid).collect(),
        }
    }

    /// The three outcomes of an acknowledgement: the exact frame is delivered,
    /// an older unacknowledged one is presumed lost and owed again, a newer one
    /// stays in flight.
    ///
    /// Mutation this catches: treating every frame at or below the acknowledged
    /// tick as delivered (`continue` for `<=` instead of `==`) — the lost frame's
    /// removal is then never said again.
    #[test]
    fn an_acknowledgement_settles_exactly_one_frame_and_presumes_older_ones_lost() {
        let mut replica = ClientReplica::new(guid(1));
        replica.remember(10, vec![guid(2)], vec![guid(7)]);
        replica.remember(11, vec![guid(3)], Vec::new());
        replica.remember(12, vec![guid(4)], vec![guid(8)]);

        replica.acknowledge(11);
        assert_eq!(replica.client_ack, 11);
        assert_eq!(replica.owed, vec![guid(2)], "frame 10 is presumed lost");
        assert_eq!(replica.owed_removals, vec![guid(7)]);
        assert_eq!(
            replica.in_flight,
            VecDeque::from(vec![frame(12, &[4], &[8])]),
            "frame 11 is delivered, frame 12 is still in flight"
        );

        // A stale or repeated acknowledgement changes nothing.
        replica.acknowledge(11);
        replica.acknowledge(5);
        assert_eq!(replica.owed, vec![guid(2)]);
        assert_eq!(replica.in_flight.len(), 1);

        replica.acknowledge(12);
        assert!(replica.in_flight.is_empty());
        assert_eq!(replica.owed, vec![guid(2)], "frame 12 was delivered");
        assert_eq!(replica.owed_removals, vec![guid(7)]);
    }

    /// What a presumed-lost frame carried is not owed again if the frame that
    /// did arrive carried it too: the client has the newer copy.
    ///
    /// Mutation this catches: owing everything a skipped frame carried — a fleet
    /// that moves every tick would then be owed in full on every acknowledgement
    /// that skips a frame, and a client that acknowledges only every other frame
    /// would be sent everything twice.
    #[test]
    fn a_lost_frame_owes_only_what_the_delivered_frame_did_not_carry() {
        let mut replica = ClientReplica::new(guid(1));
        replica.remember(20, vec![guid(5), guid(6)], vec![guid(9), guid(10)]);
        replica.remember(21, vec![guid(6)], vec![guid(9)]);

        replica.acknowledge(21);
        assert_eq!(
            replica.owed,
            vec![guid(5)],
            "6 arrived with frame 21; only 5 was lost"
        );
        assert_eq!(
            replica.owed_removals,
            vec![guid(10)],
            "the removal of 9 arrived with frame 21; only that of 10 was lost"
        );
        assert!(replica.in_flight.is_empty());
    }

    /// A client that never acknowledges cannot grow the bookkeeping, and a
    /// heartbeat (a frame that carried nothing) is not remembered at all.
    #[test]
    fn the_in_flight_list_is_bounded_and_ignores_empty_frames() {
        let mut replica = ClientReplica::new(guid(1));
        replica.remember(1, Vec::new(), Vec::new());
        assert!(
            replica.in_flight.is_empty(),
            "nothing to lose, nothing kept"
        );
        for tick in 0..(MAX_IN_FLIGHT_FRAMES as u64 + 10) {
            replica.remember(tick, vec![guid(tick)], Vec::new());
        }
        assert_eq!(replica.in_flight.len(), MAX_IN_FLIGHT_FRAMES);
        assert_eq!(
            replica.in_flight.front().map(|f| f.tick),
            Some(10),
            "the oldest frames are the ones let go"
        );
        assert!(replica.owed.is_empty(), "aged-out frames are not re-sent");
    }

    /// `n` ships at rest, one metre apart along +x, all owned by distinct
    /// sessions. Returns the guids nearest-first as seen from the first ship.
    fn parked_fleet(n: u64) -> (SimWorld, GuidIndex, Vec<EntityGuid>) {
        let mut sim = SimWorld::new(ShardId::new(1));
        let mut guids = GuidIndex::new();
        let mut all = Vec::new();
        for i in 0..n {
            let g = guid(i + 1);
            let pos = WorldPos::ZERO.offset([i as f32, 0.0, 0.0]);
            let entity = sim.spawn_ship(g, i + 1, ShipState::at_rest(pos));
            guids.insert(g, entity);
            all.push(g);
        }
        (sim, guids, all)
    }

    /// One frame for `replica`, with a fresh per-tick cache.
    fn frame_for(
        sim: &mut SimWorld,
        guids: &GuidIndex,
        replica: &mut ClientReplica,
        visible: &[EntityGuid],
        budget: usize,
    ) -> Frame {
        let set: HashSet<EntityGuid> = visible.iter().copied().collect();
        let cx = FrameContext {
            shard: ShardId::new(1),
            epoch: 1,
            budget,
            guids,
        };
        build_frame(
            sim,
            cx,
            &mut PassCache::new(),
            replica,
            Visible {
                ordered: visible,
                set: &set,
            },
        )
    }

    fn guids_in(frame: &Frame) -> Vec<EntityGuid> {
        match frame {
            Frame::Snapshot(s) => s.entities.iter().map(|e| e.guid).collect(),
            Frame::Delta(d) => d.entities.iter().map(|e| e.guid).collect(),
        }
    }

    /// A ship that was in an early chunk of a multi-frame snapshot, changed while
    /// the later chunks went out and then stopped, is corrected by the first
    /// delta after the snapshot.
    ///
    /// Mutation this catches: advancing `change_base` with every chunk (setting
    /// it next to `ack_tick` in the snapshot branch of `build_frame`) — the
    /// change is then older than the base the first delta scans from, and the
    /// client keeps the stale state for as long as the ship stays parked.
    #[test]
    fn a_change_made_while_a_snapshot_is_going_out_is_not_lost() {
        // 158 B per ship; a 400 B budget carries two per frame.
        const BUDGET: usize = 400;
        let (mut sim, guids, all) = parked_fleet(6);
        let mut replica = ClientReplica::new(all[0]);

        sim.step();
        let first = frame_for(&mut sim, &guids, &mut replica, &all, BUDGET);
        assert!(matches!(first, Frame::Snapshot(_)));
        assert_eq!(guids_in(&first), vec![all[0], all[1]], "chunk 1");

        // The ship that went out in chunk 1 jumps, once, and is parked again.
        let moved_to = WorldPos::ZERO.offset([0.0, 25.0, 0.0]);
        let entity = guids.entity_of(all[0]).expect("alive");
        *sim.world_mut()
            .get_mut::<ShipState>(entity)
            .expect("ship state") = ShipState::at_rest(moved_to);

        sim.step();
        let second = frame_for(&mut sim, &guids, &mut replica, &all, BUDGET);
        assert!(matches!(second, Frame::Snapshot(_)));
        assert_eq!(guids_in(&second), vec![all[2], all[3]], "chunk 2");
        sim.step();
        let third = frame_for(&mut sim, &guids, &mut replica, &all, BUDGET);
        assert!(matches!(third, Frame::Snapshot(_)));
        assert_eq!(guids_in(&third), vec![all[4], all[5]], "chunk 3");

        // The snapshot is complete. Nothing has changed since chunk 2 — but the
        // client still has the first ship where chunk 1 put it.
        sim.step();
        let mut carried = None;
        for _ in 0..4 {
            let frame = frame_for(&mut sim, &guids, &mut replica, &all, BUDGET);
            let Frame::Delta(delta) = frame else {
                panic!("the snapshot is complete, so this must be a delta");
            };
            if let Some(entity) = delta.entities.iter().find(|e| e.guid == all[0]) {
                let payload = entity
                    .components
                    .iter()
                    .find(|(id, _)| *id == <ShipState as ComponentDef>::ID)
                    .map(|(_, p)| p.as_slice())
                    .expect("a ship state");
                carried = Some(space_sim::wire::decode_ship_state(payload).expect("decodes"));
                break;
            }
            sim.step();
        }
        let carried = carried.expect("the change made during the snapshot must be sent");
        assert_eq!(carried.pos, moved_to);
    }

    /// The interest index follows a ship that moves and is not disturbed by one
    /// that does not.
    #[test]
    fn interest_follows_a_moving_ship_and_tolerates_repeated_placement() {
        let (_sim, guids, all) = parked_fleet(1);
        let entity = guids.entity_of(all[0]).expect("alive");
        let here = WorldPos::ZERO;
        let far = WorldPos::ZERO.offset([50_000.0, 0.0, 0.0]);

        let mut interest = Interest::new();
        interest.upsert(all[0], entity, here);
        interest.upsert(all[0], entity, here);
        assert_eq!(interest.len(), 1);
        assert_eq!(interest.visible(here, 100.0), vec![all[0]]);

        interest.upsert(all[0], entity, far);
        assert_eq!(interest.len(), 1);
        assert!(interest.visible(here, 100.0).is_empty(), "it left");
        assert_eq!(interest.visible(far, 100.0), vec![all[0]], "and arrived");

        interest.remove(all[0]);
        assert!(interest.is_empty());
        assert!(interest.visible(far, 100.0).is_empty());
    }
}
