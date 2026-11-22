//! The per-client byte budget.
//!
//! A shard must not hand one client an unbounded frame: a busy tick can change
//! thousands of entities, and a 40 MiB `Delta` both blows the 1 MiB frame limit
//! and turns one slow client into a bandwidth incident for the whole shard.
//! [`SnapshotBuilder`] is the enforcement point: the replication pass pushes
//! changed entities into it, and it hands back a finished [`Snapshot`] as soon
//! as the next entity would cross the budget. Whatever did not fit is pushed
//! again next tick — the change-tick model makes that free, because an entity
//! stays "changed" until a client has been told about it.
//!
//! The budget is measured on the **encoded payload of the message the builder
//! produces**, which is what actually costs bandwidth and what the codec limits
//! are about: the fixed overhead of the enclosing `Snapshot` is reserved up
//! front, so a finished snapshot always encodes within [`SnapshotBuilder::budget`].

use crate::codec::{encoded_len, MAX_FRAME_SIZE};
use crate::guid::EntityGuid;
use crate::messages::{Delta, EntitySnapshot, Message, Snapshot};
use crate::shard::ShardId;

/// Default per-frame payload budget, in bytes (64 KiB).
///
/// Comfortably below [`MAX_FRAME_SIZE`], and large enough that a full interest
/// set (~1k entities per `docs/plan.md` U9) is a handful of frames.
pub const DEFAULT_BUDGET: usize = 64 * 1024;

/// Accumulates [`EntitySnapshot`]s up to a byte budget.
#[derive(Debug, Clone)]
pub struct SnapshotBuilder {
    budget: usize,
    /// Bytes the enclosing message costs with an empty `entities` list.
    overhead: usize,
    /// Bytes reserved for the `removed` list a finished `Delta` will carry
    /// (`removed_reserve` guids × [`GUID_WIRE_LEN`]). Subtracted from the entity
    /// budget so a `finish_delta` handed that many removals still encodes within
    /// [`SnapshotBuilder::budget`]. Set with [`SnapshotBuilder::reserve_removed`].
    removed_reserve: usize,
    tick: u64,
    shard: ShardId,
    epoch: u64,
    entities: Vec<EntitySnapshot>,
    used: usize,
}

/// Encoded size of one [`EntityGuid`] under the pinned config: `shard u32 (4) +
/// serial u64 (8)` = 12 bytes (fixed-width, big endian).
const GUID_WIRE_LEN: usize = 12;

impl SnapshotBuilder {
    /// A builder for `shard` at `tick`/`epoch` with the default budget.
    #[must_use]
    pub fn new(tick: u64, shard: ShardId, epoch: u64) -> Self {
        Self::with_budget(DEFAULT_BUDGET, tick, shard, epoch)
    }

    /// A builder with an explicit payload budget.
    ///
    /// The budget is clamped to [`MAX_FRAME_SIZE`]: a builder can never be
    /// configured to produce a frame the codec would reject.
    #[must_use]
    pub fn with_budget(budget: usize, tick: u64, shard: ShardId, epoch: u64) -> Self {
        let budget = budget.clamp(1, MAX_FRAME_SIZE);
        // A pathological overhead (it is a few tens of bytes in practice) must
        // never make the entity budget negative.
        Self {
            budget,
            overhead: Self::overhead(tick, shard, epoch).min(budget / 2),
            removed_reserve: 0,
            tick,
            shard,
            epoch,
            entities: Vec::new(),
            used: 0,
        }
    }

    /// Reserve room for `n` removals in the entity budget.
    ///
    /// A `Delta`'s `removed` list is not known until `finish_delta` is called,
    /// but it costs `n × `[`GUID_WIRE_LEN`] bytes on the wire. A replication pass
    /// that knows it has `n` removals to report calls this first, so the budget
    /// check in [`SnapshotBuilder::push`] leaves room for them and the finished
    /// delta still encodes within [`SnapshotBuilder::budget`].
    pub fn reserve_removed(&mut self, n: usize) {
        self.removed_reserve = n.saturating_mul(GUID_WIRE_LEN);
    }

    /// Bytes the enclosing message costs with an empty entity list.
    ///
    /// One helper, used by both [`SnapshotBuilder::with_budget`] and
    /// [`SnapshotBuilder::set_tick`], and defined as the **larger** of the empty
    /// `Snapshot` and the empty `Delta` (a `Delta` carries `base_tick`,
    /// `removed` and so costs more). Computing the overhead from one message
    /// shape only — as an earlier revision did in `set_tick` — let a
    /// `finish_delta` after `set_tick` overrun the budget by the difference.
    fn overhead(tick: u64, shard: ShardId, epoch: u64) -> usize {
        encoded_len(&Message::Snapshot(empty_snapshot(tick, shard, epoch))).max(encoded_len(
            &Message::Delta(empty_delta(tick, shard, epoch)),
        ))
    }

    /// The payload budget in force.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Bytes of entity payload accumulated so far.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used
    }

    /// Entities accumulated so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether nothing has been accumulated yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Whether `entity` still fits in the budget.
    #[must_use]
    pub fn fits(&self, entity: &EntitySnapshot) -> bool {
        self.entities.is_empty() || self.used + entity.encoded_len() <= self.entity_budget()
    }

    /// Try to add one entity.
    ///
    /// Returns `false` (and stores nothing) when the entity would cross the
    /// budget *and* at least one entity is already buffered — the first entity
    /// is always accepted, otherwise a single oversized entity could never be
    /// sent at all. The caller keeps the rejected entity and retries next tick.
    pub fn push(&mut self, entity: &EntitySnapshot) -> bool {
        let size = entity.encoded_len();
        if !self.entities.is_empty() && self.used + size > self.entity_budget() {
            return false;
        }
        self.entities.push(entity.clone());
        self.used += size;
        true
    }

    /// Finish: return the buffered entities as a [`Snapshot`] and reset for the
    /// next batch.
    ///
    /// `input_ack` is the highest input tick the server has applied from the
    /// recipient (see [`Snapshot::input_ack`]); it used to be hard-coded to `0`,
    /// which silently told every client that nothing it sent had been applied and
    /// made it replay its whole pending queue. The `removed`/`input_ack` fields
    /// are real now.
    pub fn finish_snapshot(&mut self, input_ack: u64) -> Snapshot {
        let entities = std::mem::take(&mut self.entities);
        self.used = 0;
        Snapshot {
            tick: self.tick,
            shard: self.shard,
            epoch: self.epoch,
            entities,
            input_ack,
        }
    }

    /// Finish the same batch as a [`Delta`] against `base_tick`.
    ///
    /// `input_ack` is the server's input acknowledgement (see
    /// [`Snapshot::input_ack`]) and `removed` the guids the recipient must drop
    /// (see [`Delta::removed`]); both used to be hard-coded (`input_ack: 0`,
    /// `removed: vec![]`). The caller must have reserved room for `removed.len()`
    /// with [`SnapshotBuilder::reserve_removed`] before pushing entities, so the
    /// finished delta encodes within [`SnapshotBuilder::budget`].
    pub fn finish_delta(
        &mut self,
        base_tick: u64,
        removed: Vec<EntityGuid>,
        input_ack: u64,
    ) -> Delta {
        let entities = std::mem::take(&mut self.entities);
        self.used = 0;
        self.removed_reserve = 0;
        Delta {
            tick: self.tick,
            base_tick,
            shard: self.shard,
            epoch: self.epoch,
            entities,
            removed,
            input_ack,
        }
    }

    /// Build a snapshot directly from a slice of entities, taking as many as
    /// fit. Returns the snapshot and how many entities were taken.
    pub fn from_entities(
        tick: u64,
        shard: ShardId,
        epoch: u64,
        budget: usize,
        entities: &[EntitySnapshot],
    ) -> (Snapshot, usize) {
        let mut builder = Self::with_budget(budget, tick, shard, epoch);
        for entity in entities {
            if !builder.push(entity) {
                break;
            }
        }
        let taken = builder.len();
        (builder.finish_snapshot(0), taken)
    }

    /// Point the builder at a new tick/epoch between batches.
    pub fn set_tick(&mut self, tick: u64, epoch: u64) {
        self.tick = tick;
        self.epoch = epoch;
        self.overhead = Self::overhead(tick, self.shard, epoch).min(self.budget / 2);
    }

    /// Bytes available for entity payloads.
    fn entity_budget(&self) -> usize {
        // The removal reserve is charged against the entity budget so a
        // `finish_delta` carrying `reserve_removed` guids still fits `budget`.
        self.budget
            .saturating_sub(self.overhead)
            .saturating_sub(self.removed_reserve)
    }
}

fn empty_snapshot(tick: u64, shard: ShardId, epoch: u64) -> Snapshot {
    Snapshot {
        tick,
        shard,
        epoch,
        entities: Vec::new(),
        input_ack: 0,
    }
}

fn empty_delta(tick: u64, shard: ShardId, epoch: u64) -> Delta {
    Delta {
        tick,
        base_tick: tick,
        shard,
        epoch,
        entities: Vec::new(),
        removed: Vec::new(),
        input_ack: 0,
    }
}

/// Build an [`EntitySnapshot`] from values, encoding each component payload
/// with the pinned config — the shape a shard's replication pass produces.
#[must_use]
pub fn entity_snapshot<T: serde::Serialize>(
    guid: EntityGuid,
    components: &[(u16, &T)],
) -> EntitySnapshot {
    EntitySnapshot {
        guid,
        components: components
            .iter()
            .map(|(id, value)| (*id, crate::codec::encode_raw(value)))
            .collect(),
    }
}
