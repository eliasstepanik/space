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
//! can be wrong (two frames arrived between two input batches — on a live
//! stack that happens often) and then costs one redundant re-send, of
//! current state. A client that never acknowledges anything (it sends no
//! input) has its frames taken as arrived after [`MAX_IN_FLIGHT_FRAMES`]
//! ticks: the old send-and-forget behaviour.
//!
//! Two things make that sound. **An entry lists what changed since the last
//! frame known to have arrived** ([`Sent::base`]: acknowledged, or aged out),
//! not since the last frame *sent* — so the frame after a lost one says
//! everything the lost one said, removals included, and "the frame that did
//! arrive carried it as well" really means the client has it all. (Listing
//! what changed since the last send lost a component that changed once, or
//! went away, in the lost frame: fast travel that stayed on in a client's
//! prediction, an `Origin` that stayed on a ship.) An entity no frame is
//! known to have brought goes out in full, with what it is remembered to
//! have lost ([`ComponentLog::full_with_removals`]): the client merges a
//! delta's entry into whatever copy it may still hold. **And a re-send does
//! not move an entity's cadence** — see "How often" below.
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
//!
//! # Every body, and only what changed
//!
//! Every physics entity in a client's interest is replicated — ships, bodies
//! and the ghosts of a neighbour's bodies near the border — with its whole
//! replicated component set ([`replicated_components`]: `ShipState` and, for
//! a ship, `ShipControls` and `Owner` — and, for a ship a client flies, the
//! player model as its `Visual` (`space_scene::visual::PLAYER`), so a client
//! draws a character for it even where its `Owner` names nobody it knows,
//! as on a ghost; for a body its `Body`, `Visual`, `Springs`,
//! `Articulation`, `PartStates` and `PosResidual`).
//!
//! The [`ComponentLog`] remembers, per entity and component, the bytes last
//! seen and the tick they changed. A snapshot, and a delta entry for an entity
//! no frame is known to have brought the client, carries the full set; any
//! other delta entry carries the `ShipState` and the `Owner` (always: they
//! name the entity's motion and its player) plus only the components whose
//! bytes changed since the last frame with the entity that is known to have
//! arrived — a few ticks back, the round trip of an acknowledgement — so a
//! crate's shape, a chain's joints and a pendulum's spring go out once (a few
//! times while the first frame is unacknowledged), and a moving body costs
//! its state, its residual and its part states. A component that went away is
//! listed with an empty payload (protocol 5, [`Delta::entities`]).
//!
//! # How often: distance steps
//!
//! An entity is not sent to a client more often than its **step** allows
//! ([`space_protocol::far`]: every tick within 100 m of the client's ship,
//! then 10, 5 and 1 times a second out to the 10 km of the view range), and
//! only when it changed. So every client holds, per entity, the tick it was
//! last sent it ([`ClientReplica::sent`]) in place of one change base for
//! all: an entity that was skipped for a while and has come to rest by the
//! time its turn comes is still brought up to date, with whatever changed
//! since *its* last word. An entity that has been at rest for longer than its
//! step's spacing and moves again is due at once, so its first sample is the
//! start of the motion, not a point somewhere along it.
//!
//! The spacing is counted **between the samples** — the ticks the states are
//! the states of ([`Sent::sample`], [`ComponentLog::sample_of`]) — and only
//! for an entity that changed and then stopped from the last send
//! (`step_is_due`). For what this shard simulates the two are the same. A
//! far replica's samples are taken by another shard, exactly the spacing
//! apart, and are here a tick later, give or take one: each is relayed in
//! the frame of the tick it arrives, so a far sample reaches a client a
//! constant tick or two after the tick it is of. (Counted from the last
//! send, every re-send of a frame presumed lost moved the gate later for
//! good and the stored sample aged in the shard — up to a whole spacing: far
//! moons that stood and leapt, on a live stack.)
//!
//! The bound that follows: a client is sent at most
//! `sum over the moving entities in its view of (bytes of one sample x the
//! rate of its step)` a second, plus 56 bytes a tick for the frame itself,
//! and never more than the frame budget a tick. A sample of a body that only
//! moves is its `ShipState` and `PosResidual` — about 130 bytes with the
//! entry's own framing — so a hundred bodies tumbling a kilometre away cost what
//! three and a half cost at arm's length.
//!
//! # Far replicas
//!
//! What another shard simulates beyond the reach of its ghosts is in a
//! client's interest set like anything else, and is not in the simulation at
//! all: the shard keeps such an entity in the interest grid under a key of
//! its own ([`Interest::upsert_far`]) and records its components in the
//! [`ComponentLog`] when the other shard sends them (`far.rs`).

use std::collections::{HashMap, HashSet, VecDeque};

use space_ecs::{ComponentDef, ComponentId, Entity, WithEntity};
use space_protocol::{far, Delta, EntityGuid, EntitySnapshot, ShardId, Snapshot, SnapshotBuilder};
use space_sim::{Body, Owner, ShipControls, ShipState, SimWorld};
use space_world::{InterestGrid, WorldPos};

/// A replicated component list: `(id, encoded payload)` in id order.
pub type Components = Vec<(ComponentId, Vec<u8>)>;

/// How long a component removal is remembered for deltas, ticks. A client is
/// sent a frame every tick, so its change base is never older than its frames
/// in flight ([`MAX_IN_FLIGHT_FRAMES`]); a client further behind is re-sent in
/// full anyway (presumed lost frames owe their entities).
const TOMBSTONE_TICKS: u64 = 4 * MAX_IN_FLIGHT_FRAMES as u64;

/// The replicated component list of the entity `guid` this world simulates:
/// [`SimWorld::export_entity`], except that a ship's `Body` is left out — it
/// is always [`Body::ship`] of the shard's parameters, which the receiver
/// already knows (`SimWorld::import_entity` gives a ship without one exactly
/// that), and leaving it out keeps a ship at its 158 bytes.
#[must_use]
pub fn replicated_components(sim: &SimWorld, guid: EntityGuid) -> Option<Components> {
    let mut list = sim.export_entity(guid)?;
    if list
        .iter()
        .any(|(id, _)| *id == <ShipControls as ComponentDef>::ID)
    {
        list.retain(|(id, _)| *id != <Body as ComponentDef>::ID);
    }
    Some(list)
}

/// Per entity, the bytes of each replicated component and the tick they last
/// changed (see the module docs, "Every body, and only what changed").
#[derive(Debug, Default)]
pub struct ComponentLog {
    entries: HashMap<EntityGuid, LogEntry>,
    /// The sim tick of the last [`ComponentLog::refresh`].
    scanned: u64,
}

#[derive(Debug, Default)]
struct LogEntry {
    /// `(id, bytes, tick the bytes last changed)`, in id order.
    components: Vec<(ComponentId, Vec<u8>, u64)>,
    /// Components that went away, and when.
    removed: Vec<(ComponentId, u64)>,
    /// The tick of the last [`ComponentLog::record`]: the entity changed
    /// then (its row did, or a far replica's sample arrived), whether or not
    /// any byte differs.
    touched: u64,
    /// The tick the recorded state is the state **of**: `touched` for an
    /// entity of this shard's world, the simulating shard's tick for a far
    /// replica (a tick or two earlier). What an entity's distance step is
    /// counted in ([`build_frame`]).
    sample: u64,
    /// The entity left this shard's view ([`ComponentLog::forget`]): nothing
    /// is logged for it, but what it had is remembered as removed for a
    /// while, in case it is back under another kind of copy before a client
    /// was told it had gone.
    gone: bool,
}

impl ComponentLog {
    /// An empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entities logged.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.values().filter(|e| !e.gone).count()
    }

    /// Whether nothing is logged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Record `guid`'s current components at `tick`: a component whose bytes
    /// differ from the last record (or is new) is stamped `tick`; one that is
    /// gone is remembered as removed at `tick`.
    pub fn record(&mut self, guid: EntityGuid, components: Components, tick: u64) {
        self.record_sampled(guid, components, tick, tick);
    }

    /// [`ComponentLog::record`] for a state that is the state of tick
    /// `sample`, recorded at `tick`: a far replica's, which another shard
    /// sampled a tick or two before it got here.
    pub fn record_sampled(
        &mut self,
        guid: EntityGuid,
        components: Components,
        tick: u64,
        sample: u64,
    ) {
        let entry = self.entries.entry(guid).or_default();
        entry.touched = tick;
        entry.sample = sample;
        entry.gone = false;
        let mut next: Vec<(ComponentId, Vec<u8>, u64)> = Vec::with_capacity(components.len());
        for (id, bytes) in components {
            let changed_at = match entry.components.iter().find(|(old, _, _)| *old == id) {
                Some((_, old, at)) if *old == bytes => *at,
                _ => tick,
            };
            entry.removed.retain(|(gone, _)| *gone != id);
            next.push((id, bytes, changed_at));
        }
        for (id, _, _) in &entry.components {
            if !next.iter().any(|(n, _, _)| n == id) {
                entry.removed.push((*id, tick));
            }
        }
        entry
            .removed
            .retain(|(_, at)| tick.saturating_sub(*at) <= TOMBSTONE_TICKS);
        entry.components = next;
    }

    /// Forget `guid` (it left this shard's world).
    ///
    /// Its components are kept as **removed** for [`TOMBSTONE_TICKS`]. An
    /// entity that leaves the world may be back the same tick as another
    /// kind of copy — a ghost whose ship has just been handed here and is
    /// installed as this shard's own, a ghost that is withdrawn and shown as
    /// a far replica — before any client was told it had gone. Such a client
    /// goes on holding it, and is sent what changed: that has to include what
    /// the new copy no longer has (its `Origin`, above all: the ship would
    /// stay "a ghost" on that client for good, as a first live run showed).
    pub fn forget(&mut self, guid: EntityGuid) {
        let at = self.scanned + 1;
        if let Some(entry) = self.entries.get_mut(&guid) {
            for (id, _, _) in entry.components.drain(..) {
                entry.removed.retain(|(gone, _)| *gone != id);
                entry.removed.push((id, at));
            }
            entry.touched = at;
            entry.gone = true;
        }
    }

    /// Record every entity whose row changed since the last refresh, with
    /// `components` naming each one's replicated list (`None`: not
    /// replicated — it is forgotten).
    pub fn refresh(
        &mut self,
        sim: &mut SimWorld,
        guids: &GuidIndex,
        mut components: impl FnMut(&SimWorld, EntityGuid) -> Option<Components>,
    ) {
        let tick = sim.tick();
        let changed = changed_since(sim, guids, self.scanned);
        for guid in changed {
            match components(sim, guid) {
                Some(list) => self.record(guid, list, tick),
                None => self.forget(guid),
            }
        }
        self.scanned = tick;
        self.entries
            .retain(|_, e| !e.gone || tick.saturating_sub(e.touched) <= TOMBSTONE_TICKS);
    }

    /// Whether `guid` was recorded after tick `base`: something about it
    /// changed since a client was sent it at `base`.
    #[must_use]
    pub fn touched_after(&self, guid: EntityGuid, base: u64) -> bool {
        self.entries
            .get(&guid)
            .is_some_and(|e| !e.gone && e.touched > base)
    }

    /// The tick `guid`'s recorded state is the state of (see
    /// [`ComponentLog::record_sampled`]); 0 for an entity that is not logged.
    #[must_use]
    pub fn sample_of(&self, guid: EntityGuid) -> u64 {
        self.entries
            .get(&guid)
            .filter(|e| !e.gone)
            .map_or(0, |e| e.sample)
    }

    /// The full component set of `guid` **and** an empty payload for every
    /// component it is remembered to have lost ([`TOMBSTONE_TICKS`]): what a
    /// delta carries for an entity the client is not known to hold as it is.
    /// The client may hold an older copy all the same — the frame that
    /// removed it was lost, or it was another kind of copy — and merges a
    /// delta's entry into what it holds: without the removals a component the
    /// new copy lacks (a ghost's `Origin` on a ship that is now this shard's
    /// own) would stay for good.
    #[must_use]
    pub fn full_with_removals(&self, guid: EntityGuid) -> Option<EntitySnapshot> {
        let mut entity = self.full(guid)?;
        let entry = self.entries.get(&guid)?;
        entity
            .components
            .extend(entry.removed.iter().map(|(id, _)| (*id, Vec::new())));
        entity.components.sort_by_key(|(id, _)| *id);
        Some(entity)
    }

    /// The full component set of `guid`, as a snapshot entity.
    #[must_use]
    pub fn full(&self, guid: EntityGuid) -> Option<EntitySnapshot> {
        let entry = self.entries.get(&guid).filter(|e| !e.gone)?;
        Some(EntitySnapshot::new(
            guid,
            entry
                .components
                .iter()
                .map(|(id, bytes, _)| (*id, bytes.clone()))
                .collect(),
        ))
    }

    /// What changed about `guid` after tick `base`: its `ShipState` and
    /// `Owner` always, every other component stamped after `base`, and an
    /// empty payload for each component removed after `base`.
    #[must_use]
    pub fn since(&self, guid: EntityGuid, base: u64) -> Option<EntitySnapshot> {
        let entry = self.entries.get(&guid).filter(|e| !e.gone)?;
        let always = |id: ComponentId| {
            id == <ShipState as ComponentDef>::ID || id == <Owner as ComponentDef>::ID
        };
        let mut list: Components = entry
            .components
            .iter()
            .filter(|(id, _, at)| always(*id) || *at > base)
            .map(|(id, bytes, _)| (*id, bytes.clone()))
            .collect();
        for (id, at) in &entry.removed {
            if *at > base {
                list.push((*id, Vec::new()));
            }
        }
        list.sort_by_key(|(id, _)| *id);
        Some(EntitySnapshot::new(guid, list))
    }
}

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

    /// Record `guid` ↔ `entity` (replacing what `guid` named before).
    pub fn insert(&mut self, guid: EntityGuid, entity: Entity) {
        if let Some(old) = self.to_entity.insert(guid, entity) {
            if old != entity && self.to_guid.get(&old) == Some(&guid) {
                self.to_guid.remove(&old);
            }
        }
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

/// What a client was last sent of one entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sent {
    /// The tick of the frame that last carried it: an entity is due again
    /// only if something about it changed after that.
    pub at: u64,
    /// Its distance step for this client ([`space_protocol::far`]).
    pub step: u8,
    /// The tick the state it was last sent is the state **of**
    /// ([`ComponentLog::sample_of`]): what its step's spacing is counted
    /// from. Not `at` — a far replica's sample is a tick or two older than
    /// the frame, and a frame may carry the same sample twice (the re-send
    /// of a frame presumed lost).
    pub sample: u64,
    /// The tick of the newest frame that carried it and is **known to have
    /// arrived** (acknowledged, or aged out of the in-flight list); `None`
    /// until one has. The next entry for it lists what changed after this
    /// tick — not after `at`: a frame that is lost then takes nothing with
    /// it, because the next one says the same again and more.
    pub base: Option<u64>,
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
    /// The highest tick the client itself has confirmed applying
    /// (`InputTick::ack_tick`, clamped to the shard's own tick).
    pub client_ack: u64,
    /// The tick of the first frame built for this client, once there is one
    /// (see [`UNACKNOWLEDGED_SHIP_TICKS`]).
    pub introduced_at: Option<u64>,
    /// Every entity the client has been sent and not been told to remove
    /// (so a delta can name removals), with the tick it was last sent at and
    /// its distance step: an entity is sent again when it changed after that
    /// tick and its step's spacing has passed (see the module docs, "How
    /// often"). Per entity, not one base for all: an entity in an early chunk
    /// of a snapshot that changed while the later chunks went out, or one
    /// that was skipped at a coarse step and then stopped, is still brought
    /// up to date. (One base that advanced with every frame lost both.)
    pub sent: HashMap<EntityGuid, Sent>,
    /// Entities that must be (re-)sent whether or not they change again: the
    /// ones a previous frame could not fit, and the ones a frame that is
    /// presumed lost was carrying.
    pub owed: Vec<EntityGuid>,
    /// Removals a frame that is presumed lost was carrying, to be sent again.
    pub owed_removals: Vec<EntityGuid>,
    /// The remaining entities of a snapshot that did not fit in one frame.
    pub pending_snapshot: VecDeque<EntityGuid>,
    /// A chunk of a snapshot went out saying "more follow"
    /// ([`Snapshot::more`]) and the chunk that says "that was all" has not:
    /// the client is collecting, and has to be told the end even if nothing
    /// is left to send (the rest despawned).
    pub snapshot_open: bool,
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
            client_ack: 0,
            introduced_at: None,
            sent: HashMap::new(),
            owed: Vec::new(),
            owed_removals: Vec::new(),
            pending_snapshot: VecDeque::new(),
            snapshot_open: false,
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
                self.arrived(&frame);
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
            // (Delivered it **all**: an entry lists what changed since the
            // last frame known to have arrived — `Sent::base` — so the
            // delivered frame's entry says everything the lost one said.
            // While entries listed what changed since the last frame *sent*,
            // this dropped the lost frame's own changes for good: a
            // component that changed once — `ShipControls::fast` — or went
            // away — an `Origin`.)
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
            if let Some(frame) = self.in_flight.pop_front() {
                self.arrived(&frame);
            }
        }
    }

    /// `frame` arrived (it was acknowledged, or is taken to have): what it
    /// carried is what the next entries are differences to.
    fn arrived(&mut self, frame: &SentFrame) {
        for guid in &frame.entities {
            if let Some(sent) = self.sent.get_mut(guid) {
                sent.base = sent.base.max(Some(frame.tick));
            }
        }
    }

    /// Take the frames no acknowledgement has settled within
    /// [`MAX_IN_FLIGHT_FRAMES`] **ticks** as arrived (a client that
    /// acknowledges nothing: see the constant). By age as well as by count:
    /// far things are in a frame once a second, and a base that lagged
    /// thirty-two such frames would be older than the log remembers what a
    /// component was removed ([`TOMBSTONE_TICKS`]).
    fn settle_aged(&mut self, tick: u64) {
        while self
            .in_flight
            .front()
            .is_some_and(|f| tick.saturating_sub(f.tick) > MAX_IN_FLIGHT_FRAMES as u64)
        {
            if let Some(frame) = self.in_flight.pop_front() {
                self.arrived(&frame);
            }
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
    /// The last key handed to a far replica ([`Interest::upsert_far`]).
    far_keys: u64,
}

/// The bit that marks the grid key of a far replica: an entity's key carries
/// its generation in the high half ([`grid_key`]), which does not get there.
const FAR_KEY_BIT: u64 = 1 << 63;

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

    /// (Re-)place the far replica `guid` — an entity that is in nobody's
    /// world here, so it has no ECS entity to take a key from — at `pos`.
    /// A guid that is tracked as an entity of the world is left alone: the
    /// world's copy is the one that counts.
    pub fn upsert_far(&mut self, guid: EntityGuid, pos: WorldPos) {
        let key = match self.by_guid.get(&guid) {
            Some((key, _)) if key & FAR_KEY_BIT == 0 => return,
            Some((_, at)) if *at == pos => return,
            Some((key, _)) => *key,
            None => {
                self.far_keys += 1;
                FAR_KEY_BIT | self.far_keys
            }
        };
        self.by_guid.insert(guid, (key, pos));
        self.by_key.insert(key, guid);
        self.grid.insert(key, pos, 0.0);
    }

    /// Whether `guid` is tracked as a far replica.
    #[must_use]
    pub fn is_far(&self, guid: EntityGuid) -> bool {
        self.by_guid
            .get(&guid)
            .is_some_and(|(key, _)| key & FAR_KEY_BIT != 0)
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
        self.visible_at(center, radius).0
    }

    /// [`Interest::visible`] with each guid's distance from `center`, metres
    /// (two lists of one length).
    #[must_use]
    pub fn visible_at(&self, center: WorldPos, radius: f32) -> (Vec<EntityGuid>, Vec<f64>) {
        self.grid
            .query_sphere(center, radius, usize::MAX)
            .into_iter()
            .filter_map(|(key, distance)| Some((self.by_key.get(&key).copied()?, distance)))
            .unzip()
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

/// Every physics entity (ship, body or ghost) whose row changed since
/// `base_tick`, in the ECS's own row order.
///
/// The change filter is a tick compare per row, so an idle fleet costs nothing
/// here — the whole point of the change tick model. The result is the same for
/// every client examined at the same base, which is why [`PassCache`] computes
/// it once per base and each client then filters it by its own interest set.
#[must_use]
pub fn changed_since(sim: &mut SimWorld, guids: &GuidIndex, base_tick: u64) -> Vec<EntityGuid> {
    let mut out = Vec::new();
    let query = sim.world_mut().query::<WithEntity<&ShipState>>();
    query.changed_since(base_tick).for_each(|(_, entity)| {
        if let Some(guid) = guids.guid_of(entity) {
            out.push(guid);
        }
    });
    out
}

/// What one tick's replication pass shares between clients.
///
/// An entity's encoded entry is the same for every client that was last
/// sent it on the same tick — everybody near it, at the finest step — and is
/// built on first use and reused.
#[derive(Debug, Default)]
pub struct PassCache {
    /// `(guid, what of it)` → its encoded entry this tick (`None`: not
    /// logged).
    snapshots: HashMap<(EntityGuid, Base), Option<EntitySnapshot>>,
}

/// What of an entity an entry carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Base {
    /// Its full set: an entity of a snapshot.
    Whole,
    /// Its full set and what it is remembered to have lost: a delta's entry
    /// for an entity no frame is known to have brought the client.
    Entry,
    /// What changed after this tick.
    Since(u64),
}

impl PassCache {
    /// An empty cache, for one tick.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `guid`'s entry as of this tick, built on first request.
    fn snapshot(
        &mut self,
        log: &ComponentLog,
        guid: EntityGuid,
        base: Base,
    ) -> Option<&EntitySnapshot> {
        self.snapshots
            .entry((guid, base))
            .or_insert_with(|| match base {
                Base::Whole => log.full(guid),
                Base::Entry => log.full_with_removals(guid),
                Base::Since(base) => log.since(guid, base),
            })
            .as_ref()
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

impl Frame {
    /// The bytes of this frame's payload as the codec writes them (computed,
    /// like the budget: the fixed fields of a delta, the entities, the
    /// removals).
    #[must_use]
    pub fn wire_len(&self) -> usize {
        let (entities, removed) = match self {
            Frame::Snapshot(s) => (&s.entities, 0),
            Frame::Delta(d) => (&d.entities, d.removed.len()),
        };
        56 + entities
            .iter()
            .map(EntitySnapshot::encoded_len)
            .sum::<usize>()
            + removed * 12
    }
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
    /// What every entity's components are (refreshed before the pass).
    pub log: &'a ComponentLog,
}

/// One client's interest set this tick, in both shapes the frame needs.
#[derive(Debug, Clone, Copy)]
pub struct Visible<'a> {
    /// The guids inside the interest sphere, nearest first.
    pub ordered: &'a [EntityGuid],
    /// The same guids, for membership tests.
    pub set: &'a HashSet<EntityGuid>,
    /// How far each guid of `ordered` is from the client's ship, metres: what
    /// its distance step is taken from. Empty (or shorter): distance 0, the
    /// finest step.
    pub distances: &'a [f64],
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
    if replica.sent.is_empty() && replica.pending_snapshot.is_empty() {
        replica.pending_snapshot = visible.ordered.iter().copied().collect();
    }
    if !replica.pending_snapshot.is_empty() || replica.snapshot_open {
        let mut builder = SnapshotBuilder::with_budget(cx.budget, tick, cx.shard, cx.epoch);
        let mut sent = Vec::new();
        while let Some(&guid) = replica.pending_snapshot.front() {
            let Some(snap) = cache.snapshot(cx.log, guid, Base::Whole) else {
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
        // Every chunk but the last says that more follow: the client takes
        // "what the snapshot does not name is gone" from the whole of it.
        // A snapshot whose last entities despawned before their chunk still
        // ends with a chunk that says so — an empty one.
        if !builder.is_empty() || replica.snapshot_open {
            let more = !replica.pending_snapshot.is_empty();
            replica.snapshot_open = more;
            // Each entity's own base is the tick of the chunk it went
            // out in: what it does afterwards, while the other chunks go
            // out, is after that.
            for guid in &sent {
                replica.sent.insert(
                    *guid,
                    Sent {
                        at: tick,
                        step: 0,
                        sample: cx.log.sample_of(*guid),
                        base: None,
                    },
                );
            }
            replica.ack_tick = tick;
            replica.remember(tick, sent, Vec::new());
            return Frame::Snapshot(builder.finish_snapshot_chunk(input_ack, more));
        }
        // Everything that was planned has despawned before a first chunk went
        // out: there is nothing to dump, so this tick is an ordinary delta.
    }

    // Removals: everything the client holds that is no longer visible. The
    // client's own ship is never removed (it is the interest anchor). A removal
    // that went out on a frame presumed lost is said again, unless the entity
    // has come back in the meantime.
    let mut removed: Vec<EntityGuid> = replica
        .sent
        .keys()
        .filter(|g| !visible.set.contains(g) && **g != replica.ship)
        .copied()
        .collect();
    for guid in std::mem::take(&mut replica.owed_removals) {
        if guid != replica.ship && !visible.set.contains(&guid) && !replica.sent.contains_key(&guid)
        {
            removed.push(guid);
        }
    }
    removed.sort_unstable();
    removed.dedup();
    for guid in &removed {
        replica.sent.remove(guid);
        // What a frame still in flight carried of it is no word on the copy
        // the client may be sent later.
        for frame in &mut replica.in_flight {
            frame.entities.retain(|carried| carried != guid);
        }
    }
    replica.settle_aged(tick);

    // The client's own ship first, for as long as it may not have been told
    // which ship that is (see [`UNACKNOWLEDGED_SHIP_TICKS`]). Then debt (it is
    // the oldest), then what is **due** — it changed since this client was
    // last sent it, and its distance step's spacing has passed (the module
    // docs, "How often") — then whatever has just come into interest without
    // changing: an entity at rest is never due, so entering the sphere has to
    // be a reason to send it in its own right. Deduplicated, and restricted
    // to what is visible now.
    //
    // What goes out in full, with what it is remembered to have lost
    // ([`ComponentLog::full_with_removals`]): the unconfirmed own ship and
    // anything no frame is known to have brought the client. Everything else
    // — debt too — goes out as what changed since the last frame that is
    // known to have arrived ([`Sent::base`]), removals included.
    let mut candidates: Vec<EntityGuid> = Vec::new();
    let mut seen: HashSet<EntityGuid> = HashSet::new();
    let own_ship = replica.ship_is_unconfirmed(tick).then_some(replica.ship);
    let owed = std::mem::take(&mut replica.owed);
    let mut due: Vec<EntityGuid> = Vec::new();
    let mut entered: Vec<EntityGuid> = Vec::new();
    for (i, guid) in visible.ordered.iter().enumerate() {
        let Some(sent) = replica.sent.get_mut(guid) else {
            entered.push(*guid);
            continue;
        };
        let distance = visible.distances.get(i).copied().unwrap_or(0.0);
        sent.step = far::step_after(distance, sent.step);
        if cx.log.touched_after(*guid, sent.at) && step_is_due(sent, cx.log.sample_of(*guid), tick)
        {
            due.push(*guid);
        }
    }
    for guid in own_ship
        .iter()
        .chain(owed.iter())
        .chain(due.iter())
        .chain(entered.iter())
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
        let held = replica.sent.get(guid).copied();
        let base = held
            .filter(|_| own_ship != Some(*guid))
            .and_then(|held| held.base)
            .map_or(Base::Entry, Base::Since);
        let Some(snap) = cache.snapshot(cx.log, *guid, base) else {
            taken += 1; // gone: not owed, not sent
            continue;
        };
        if !builder.push(snap) {
            break;
        }
        taken += 1;
        replica.sent.insert(
            *guid,
            Sent {
                at: tick,
                step: held.map_or(0, |held| held.step),
                sample: cx.log.sample_of(*guid),
                base: held.and_then(|held| held.base),
            },
        );
        sent.push(*guid);
    }
    replica.owed.extend(candidates.into_iter().skip(taken));

    let base_tick = replica.ack_tick;
    replica.ack_tick = tick;
    replica.remember(tick, sent, removed.clone());
    Frame::Delta(builder.finish_delta(base_tick, removed, input_ack))
}

/// Whether an entity that changed since `sent` is due again at `tick`, its
/// state now being the state of tick `sample`: its step's spacing has passed
/// **between the two samples** — or since it was last sent, for one that
/// changed in between and then stopped (its last state is still owed, and
/// nothing newer is coming).
///
/// The first half is what keeps a far replica's delay constant. Its samples
/// are taken by another shard, exactly the spacing apart, and arrive here a
/// tick later give or take one; counted from the last *send*, a sample that
/// arrives a tick early would wait, and every re-send of a frame presumed
/// lost would move the gate later for good — the stored sample ages in the
/// shard, the next one more, until the client has nothing to draw between
/// (`tests/far.rs`, `a_far_sample_is_relayed_on_the_tick_it_arrives_...`).
/// Counted from the sample, each is relayed in the frame of the tick it
/// arrives. (A sample from before the last one: its owner changed. Due.)
fn step_is_due(sent: &Sent, sample: u64, tick: u64) -> bool {
    let spacing = far::interval(sent.step);
    sample < sent.sample
        || sample - sent.sample >= spacing
        || tick.saturating_sub(sent.at) >= spacing
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

    /// One frame for `replica`, with a fresh per-tick cache and the log
    /// refreshed from the world first (what the shard's pass does).
    fn frame_for(
        sim: &mut SimWorld,
        guids: &GuidIndex,
        log: &mut ComponentLog,
        replica: &mut ClientReplica,
        visible: &[EntityGuid],
        budget: usize,
    ) -> Frame {
        log.refresh(sim, guids, replicated_components);
        let set: HashSet<EntityGuid> = visible.iter().copied().collect();
        let cx = FrameContext {
            shard: ShardId::new(1),
            epoch: 1,
            budget,
            guids,
            log,
        };
        build_frame(
            sim,
            cx,
            &mut PassCache::new(),
            replica,
            Visible {
                ordered: visible,
                set: &set,
                distances: &[],
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
    /// Mutation this catches: one base for every entity, advanced with every
    /// chunk (giving each entity of a chunk the tick of the *last* chunk as
    /// its `Sent::at`) — the change is then older than the base the first
    /// delta looks from, and the client keeps the stale state for as long as
    /// the ship stays parked.
    #[test]
    fn a_change_made_while_a_snapshot_is_going_out_is_not_lost() {
        // 158 B per ship; a 400 B budget carries two per frame.
        const BUDGET: usize = 400;
        let (mut sim, guids, all) = parked_fleet(6);
        let mut replica = ClientReplica::new(all[0]);
        let mut log = ComponentLog::new();

        sim.step();
        let first = frame_for(&mut sim, &guids, &mut log, &mut replica, &all, BUDGET);
        assert!(
            matches!(&first, Frame::Snapshot(s) if s.more),
            "more follow"
        );
        assert_eq!(guids_in(&first), vec![all[0], all[1]], "chunk 1");

        // A ship that went out in chunk 1 jumps, once, and is parked again.
        // (Not the client's own ship: that one is repeated in every frame
        // until the client's first acknowledgement, whatever the bases say —
        // which is what this test moved until phase 5, and why it could not
        // fail.)
        let moved_to = WorldPos::ZERO.offset([0.0, 25.0, 0.0]);
        let entity = guids.entity_of(all[1]).expect("alive");
        *sim.world_mut()
            .get_mut::<ShipState>(entity)
            .expect("ship state") = ShipState::at_rest(moved_to);

        sim.step();
        let second = frame_for(&mut sim, &guids, &mut log, &mut replica, &all, BUDGET);
        assert!(
            matches!(&second, Frame::Snapshot(s) if s.more),
            "more follow"
        );
        assert_eq!(guids_in(&second), vec![all[2], all[3]], "chunk 2");
        sim.step();
        let third = frame_for(&mut sim, &guids, &mut log, &mut replica, &all, BUDGET);
        assert!(matches!(&third, Frame::Snapshot(s) if !s.more), "the last");
        assert_eq!(guids_in(&third), vec![all[4], all[5]], "chunk 3");

        // The snapshot is complete. Nothing has changed since chunk 2 — but the
        // client still has the second ship where chunk 1 put it.
        sim.step();
        let mut carried = None;
        for _ in 0..4 {
            let frame = frame_for(&mut sim, &guids, &mut log, &mut replica, &all, BUDGET);
            let Frame::Delta(delta) = frame else {
                panic!("the snapshot is complete, so this must be a delta");
            };
            if let Some(entity) = delta.entities.iter().find(|e| e.guid == all[1]) {
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

    /// An entity that leaves the log and is back before a client was told
    /// — a ghost installed as the shard's own ship, in one tick — is sent
    /// with what it no longer has listed as removed.
    ///
    /// Mutation this catches: `forget` dropping the entry outright: the
    /// client is then sent the new copy's components and keeps the old
    /// copy's `Origin` — it shows the shard's own ship as a ghost for good.
    #[test]
    fn a_copy_that_is_replaced_within_a_tick_loses_what_the_new_one_lacks() {
        let mut log = ComponentLog::new();
        let it = guid(9);
        let origin = space_protocol::ORIGIN_COMPONENT;
        log.record(it, vec![(0, vec![1]), (10, vec![7]), (origin, vec![2])], 5);
        log.scanned = 5;
        assert_eq!(log.len(), 1);
        log.forget(it);
        assert_eq!(log.len(), 0);
        assert!(log.full(it).is_none() && log.since(it, 0).is_none());
        assert!(
            !log.touched_after(it, 0),
            "nothing to send while it is gone"
        );
        log.record(it, vec![(0, vec![3]), (2, vec![4])], 6);
        let sent = log.since(it, 5).expect("logged again");
        assert_eq!(
            sent.components,
            vec![
                (0, vec![3]),
                (2, vec![4]),
                (10, Vec::new()),
                (origin, Vec::new())
            ],
            "the new copy, and what the old one had beyond it as removed"
        );
        assert_eq!(
            log.full(it).expect("logged").components,
            vec![(0, vec![3]), (2, vec![4])],
            "a client that never held it gets the new copy alone"
        );
        // A forgotten entity that does not come back is dropped for good.
        log.forget(it);
        let (mut sim, guids, _) = parked_fleet(1);
        for _ in 0..(TOMBSTONE_TICKS + 10) {
            sim.step();
        }
        log.refresh(&mut sim, &guids, replicated_components);
        assert!(!log.entries.contains_key(&it));
    }

    /// What a frame says about `guid`, if anything.
    fn entry_of(frame: &Frame, guid: EntityGuid) -> Option<Components> {
        let entities = match frame {
            Frame::Snapshot(s) => &s.entities,
            Frame::Delta(d) => &d.entities,
        };
        entities
            .iter()
            .find(|e| e.guid == guid)
            .map(|e| e.components.clone())
    }

    /// A frame that is lost takes nothing with it: what it carried of an
    /// entity — a component that changed once, one that went away — is in
    /// the next frame that names the entity, whether that is the entity's
    /// next change or the re-send of what the lost frame owed.
    ///
    /// The client merges a delta's entry into what it holds
    /// (`space-client`, `link.rs`), so a change it is not sent it never has,
    /// and a component it is not told has gone it keeps: fast travel that
    /// stays on in its prediction, a ship of its own shard drawn as a ghost.
    ///
    /// Mutations this catches (each run): listing what changed since the
    /// entity was last **sent** (`Base::Since(held.at)` in `build_frame`) —
    /// the first half fails: the frame after the lost one says the motion
    /// and nothing of the lost frame's change, and the acknowledgement then
    /// takes the entity off the debt because "the newer frame carried it";
    /// sending debt as the full set without removals (`Base::Whole` for an
    /// owed entity) — the second half fails: the component that went away
    /// in the lost frame stays on the client for good.
    #[test]
    fn a_change_carried_by_a_lost_frame_is_sent_again() {
        let (mut sim, guids, all) = parked_fleet(1);
        let mut replica = ClientReplica::new(all[0]);
        let mut log = ComponentLog::new();
        // An entity of another shard, as the log holds one: its state, its
        // controls and where it is from.
        let it = guid(77);
        let (state, controls, origin) = (0, 1, space_protocol::ORIGIN_COMPONENT);
        let visible = [all[0], it];
        let budget = 64 * 1024;

        // Tick 1: the snapshot, and it arrives.
        sim.step();
        log.record(
            it,
            vec![(state, vec![1]), (controls, vec![0]), (origin, vec![9])],
            sim.tick(),
        );
        let first = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert!(matches!(first, Frame::Snapshot(_)));
        replica.acknowledge(sim.tick());

        // Tick 2: fast travel goes on and the origin goes away — once. The
        // frame that says so is lost.
        sim.step();
        log.record(it, vec![(state, vec![2]), (controls, vec![1])], sim.tick());
        let lost = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert_eq!(
            entry_of(&lost, it),
            Some(vec![
                (state, vec![2]),
                (controls, vec![1]),
                (origin, Vec::new())
            ])
        );
        // Tick 3: it only moves. This frame arrives.
        sim.step();
        log.record(it, vec![(state, vec![3]), (controls, vec![1])], sim.tick());
        let next = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert_eq!(
            entry_of(&next, it),
            Some(vec![
                (state, vec![3]),
                (controls, vec![1]),
                (origin, Vec::new())
            ]),
            "the frame after a lost one says what the lost one said"
        );
        replica.acknowledge(sim.tick());
        assert!(replica.owed.is_empty(), "frame 3 carried all of it");

        // Tick 4: the controls go away, in a frame that is lost, and then
        // nothing changes any more.
        sim.step();
        log.record(it, vec![(state, vec![4])], sim.tick());
        let lost = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert_eq!(
            entry_of(&lost, it),
            Some(vec![(state, vec![4]), (controls, Vec::new())])
        );
        sim.step();
        let quiet = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert_eq!(entry_of(&quiet, it), None, "nothing changed");
        // The client acknowledges tick 5: frame 4 never came.
        replica.acknowledge(sim.tick());
        assert_eq!(replica.owed, vec![it]);
        sim.step();
        let again = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert_eq!(
            entry_of(&again, it),
            Some(vec![(state, vec![4]), (controls, Vec::new())]),
            "the debt is paid with the removal the lost frame carried"
        );
    }

    /// An entity no frame is **known** to have brought the client goes out
    /// whole — and with an empty payload for every component it is
    /// remembered to have lost. The client may hold a copy all the same (the
    /// frame that brought it arrived and its acknowledgement is still on its
    /// way; or it was another kind of copy a tick ago), and it merges a
    /// delta's entry into what it holds.
    ///
    /// Mutation this catches (run): the full set alone for such an entity
    /// (`Base::Entry => log.full(guid)` in `PassCache::snapshot`): the
    /// `Origin` that went away while the first frame was unacknowledged is
    /// never taken off the client's copy — a ship of its own shard drawn as
    /// a ghost of another's for good.
    #[test]
    fn an_entity_no_frame_is_known_to_have_brought_is_sent_whole_with_what_it_lost() {
        let (mut sim, guids, all) = parked_fleet(1);
        let mut replica = ClientReplica::new(all[0]);
        let mut log = ComponentLog::new();
        let it = guid(77);
        let (state, shape, origin) = (0, 5, space_protocol::ORIGIN_COMPONENT);
        let visible = [all[0], it];
        let budget = 64 * 1024;

        // Tick 1: the snapshot. It arrives, but the shard does not know yet.
        sim.step();
        log.record(
            it,
            vec![(state, vec![1]), (shape, vec![7, 7]), (origin, vec![9])],
            sim.tick(),
        );
        let first = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert!(matches!(first, Frame::Snapshot(_)));
        assert_eq!(replica.sent[&it].base, None, "not known to have arrived");

        // Tick 2: the entity is this shard's own now — no origin any more.
        sim.step();
        log.record(it, vec![(state, vec![2]), (shape, vec![7, 7])], sim.tick());
        let next = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert_eq!(
            entry_of(&next, it),
            Some(vec![
                (state, vec![2]),
                (shape, vec![7, 7]),
                (origin, Vec::new())
            ]),
            "whole, and with what it lost"
        );

        // Once a frame with it is known to have arrived, an entry is what
        // changed since that frame and no more.
        replica.acknowledge(sim.tick());
        sim.step();
        log.record(it, vec![(state, vec![3]), (shape, vec![7, 7])], sim.tick());
        let later = frame_for(&mut sim, &guids, &mut log, &mut replica, &visible, budget);
        assert_eq!(entry_of(&later, it), Some(vec![(state, vec![3])]));
    }

    /// One frame with each visible guid at `distance` metres from the ship.
    fn frame_at(
        sim: &mut SimWorld,
        guids: &GuidIndex,
        log: &mut ComponentLog,
        replica: &mut ClientReplica,
        visible: &[(EntityGuid, f64)],
    ) -> Frame {
        log.refresh(sim, guids, replicated_components);
        let ordered: Vec<EntityGuid> = visible.iter().map(|(g, _)| *g).collect();
        let distances: Vec<f64> = visible.iter().map(|(_, d)| *d).collect();
        let set: HashSet<EntityGuid> = ordered.iter().copied().collect();
        let cx = FrameContext {
            shard: ShardId::new(1),
            epoch: 1,
            budget: 64 * 1024,
            guids,
            log,
        };
        build_frame(
            sim,
            cx,
            &mut PassCache::new(),
            replica,
            Visible {
                ordered: &ordered,
                set: &set,
                distances: &distances,
            },
        )
    }

    /// A ship far from the client is sent at its distance step's rate, an
    /// entity that moved while it was not due and then stopped is still
    /// brought up to date, and one that moves after a long rest is sent at
    /// once.
    ///
    /// Mutations this catches: one change base for all entities, advanced
    /// with every frame (the ship that jumped between two of its turns and
    /// then parked is never corrected); `>=` the step's spacing replaced by
    /// "always" (thirty samples a second at any distance); making an entity
    /// wait for a slot of its step after a rest (its first sample is then up
    /// to a second into its motion).
    #[test]
    fn an_entity_is_sent_at_its_steps_rate_and_never_left_stale() {
        let (mut sim, guids, all) = parked_fleet(3);
        let mut replica = ClientReplica::new(all[0]);
        let mut log = ComponentLog::new();
        // The client's own ship, one ship 600 m off (five times a second)
        // and one 3 km off (once a second).
        let visible = [(all[0], 0.0), (all[1], 600.0), (all[2], 3_000.0)];
        let set = |sim: &mut SimWorld, guid: EntityGuid, y: f32| {
            let entity = guids.entity_of(guid).expect("alive");
            *sim.world_mut().get_mut::<ShipState>(entity).expect("state") =
                ShipState::at_rest(WorldPos::ZERO.offset([0.0, y, 0.0]));
        };
        sim.step();
        let first = frame_at(&mut sim, &guids, &mut log, &mut replica, &visible);
        assert!(matches!(first, Frame::Snapshot(_)));
        // Both far ships move every tick for two seconds.
        let mut sent = [0usize; 3];
        for i in 0..60 {
            set(&mut sim, all[1], i as f32);
            set(&mut sim, all[2], i as f32);
            sim.step();
            let frame = frame_at(&mut sim, &guids, &mut log, &mut replica, &visible);
            for guid in guids_in(&frame) {
                sent[all.iter().position(|g| *g == guid).expect("ours")] += 1;
            }
        }
        assert_eq!(sent[1], 10, "600 m: every sixth tick");
        assert_eq!(sent[2], 2, "3 km: every thirtieth");
        assert_eq!(replica.sent[&all[1]].step, 2);
        assert_eq!(replica.sent[&all[2]].step, 3);
        // The far one jumps once more, right after its turn, and parks.
        set(&mut sim, all[2], 500.0);
        let mut carried = None;
        for _ in 0..31 {
            sim.step();
            let frame = frame_at(&mut sim, &guids, &mut log, &mut replica, &visible);
            let Frame::Delta(delta) = frame else {
                panic!("a delta")
            };
            if let Some(e) = delta.entities.iter().find(|e| e.guid == all[2]) {
                let state = &e.components[0].1;
                carried = Some(space_sim::wire::decode_ship_state(state).expect("decodes"));
            }
        }
        let carried = carried.expect("the change is sent when the ship's turn comes");
        assert_eq!(carried.pos, WorldPos::ZERO.offset([0.0, 500.0, 0.0]));
        // A long rest, then it moves: in the very next frame.
        for _ in 0..90 {
            sim.step();
            let frame = frame_at(&mut sim, &guids, &mut log, &mut replica, &visible);
            assert!(
                !guids_in(&frame).contains(&all[2]),
                "it does not move: it is not sent"
            );
        }
        set(&mut sim, all[2], 501.0);
        sim.step();
        let frame = frame_at(&mut sim, &guids, &mut log, &mut replica, &visible);
        assert!(guids_in(&frame).contains(&all[2]), "at once");
    }

    /// A snapshot whose last entities despawn before their chunk went out
    /// still ends: the client is sent a chunk that says "that was all".
    ///
    /// Mutation this catches: ending the snapshot only with a chunk that
    /// carries something (the client would wait for the end for ever, and
    /// never drop what the snapshot did not name).
    #[test]
    fn a_snapshot_whose_rest_despawned_still_ends() {
        const BUDGET: usize = 400;
        let (mut sim, mut guids, all) = parked_fleet(4);
        let mut replica = ClientReplica::new(all[0]);
        let mut log = ComponentLog::new();
        sim.step();
        let first = frame_for(&mut sim, &guids, &mut log, &mut replica, &all, BUDGET);
        assert!(matches!(&first, Frame::Snapshot(s) if s.more));
        assert!(replica.snapshot_open);
        // The two that were still to come are gone (as a shard removes an
        // entity: from the world, its index and what it tells clients).
        for guid in &all[2..] {
            sim.despawn(*guid);
            guids.remove(*guid);
            log.forget(*guid);
        }
        sim.step();
        let second = frame_for(&mut sim, &guids, &mut log, &mut replica, &all[..2], BUDGET);
        let Frame::Snapshot(last) = second else {
            panic!("the snapshot has to be ended: {second:?}");
        };
        assert!(!last.more && last.entities.is_empty(), "{last:?}");
        assert!(!replica.snapshot_open);
        sim.step();
        let third = frame_for(&mut sim, &guids, &mut log, &mut replica, &all[..2], BUDGET);
        assert!(matches!(third, Frame::Delta(_)), "then deltas: {third:?}");
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
