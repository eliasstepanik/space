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

use crate::codec::MAX_FRAME_SIZE;
use crate::guid::EntityGuid;
use crate::messages::{Delta, EntitySnapshot, Snapshot, ENTITY_GUID_WIRE_LEN};
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
const GUID_WIRE_LEN: usize = ENTITY_GUID_WIRE_LEN;

/// Bytes the enclosing message costs with an empty entity list: the **larger**
/// of the empty `Snapshot` and the empty `Delta` (a `Delta` carries `base_tick`
/// and `removed`, so it costs more).
///
/// ```text
/// Delta: tag u32 (4) + tick (8) + base_tick (8) + shard (4) + epoch (8)
///        + entities len (8) + removed len (8) + input_ack (8) = 56
/// ```
///
/// A constant, not a measurement: the pinned config is fixed-width, so the
/// overhead does not depend on the tick, shard or epoch being stamped. It used
/// to be measured by encoding two empty messages every time a builder was
/// created — once per client per tick. Budgeting against one message shape only
/// (as an earlier revision did in `set_tick`) let a `finish_delta` overrun the
/// budget by the Snapshot/Delta difference; the `message_overhead_is_exact`
/// test pins this value to both shapes.
pub(crate) const MESSAGE_OVERHEAD: usize = 56;

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
            overhead: MESSAGE_OVERHEAD.min(budget / 2),
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
        self.finish_snapshot_chunk(input_ack, false)
    }

    /// Finish the batch as one chunk of a snapshot that goes out in several:
    /// `more` says whether further chunks follow ([`Snapshot::more`]).
    pub fn finish_snapshot_chunk(&mut self, input_ack: u64, more: bool) -> Snapshot {
        let entities = std::mem::take(&mut self.entities);
        self.used = 0;
        Snapshot {
            tick: self.tick,
            shard: self.shard,
            epoch: self.epoch,
            entities,
            input_ack,
            more,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encoded_len;
    use crate::messages::Message;

    /// The constant overhead is what the codec really writes for an empty frame
    /// of either shape, whatever tick, shard and epoch it is stamped with.
    ///
    /// Mutation this catches: a field added to `Snapshot`/`Delta` (or a config
    /// change away from fixed-width integers) without updating
    /// [`MESSAGE_OVERHEAD`] — the builder would then overrun its budget.
    #[test]
    fn message_overhead_is_exact() {
        for (tick, shard, epoch) in [
            (0u64, ShardId::new(0), 0u64),
            (1, ShardId::new(7), 3),
            (u64::MAX, ShardId::NONE, u64::MAX),
        ] {
            let snapshot = Message::Snapshot(Snapshot {
                tick,
                shard,
                epoch,
                entities: Vec::new(),
                input_ack: tick,
                more: false,
            });
            let delta = Message::Delta(Delta {
                tick,
                base_tick: tick,
                shard,
                epoch,
                entities: Vec::new(),
                removed: Vec::new(),
                input_ack: tick,
            });
            assert_eq!(
                encoded_len(&snapshot).max(encoded_len(&delta)),
                MESSAGE_OVERHEAD,
                "empty frame at tick {tick}"
            );
        }
    }

    /// The computed entity size is the size the codec writes, for every shape a
    /// replication pass produces (no components, empty payloads, large ones).
    ///
    /// Mutation this catches: dropping the per-component length prefix from
    /// `EntitySnapshot::encoded_len` — the builder would then pack more entities
    /// than the budget allows.
    #[test]
    fn entity_snapshot_encoded_len_is_exact() {
        let guid = EntityGuid::new(ShardId::new(3), 0x0123_4567_89ab_cdef);
        let shapes: Vec<Vec<(u16, Vec<u8>)>> = vec![
            Vec::new(),
            vec![(0, Vec::new())],
            vec![(0, vec![1; 76]), (1, vec![2; 24]), (2, vec![3; 8])],
            vec![(u16::MAX, vec![9; 4096]), (7, vec![0; 1])],
        ];
        for components in shapes {
            let entity = EntitySnapshot::new(guid, components);
            assert_eq!(
                entity.encoded_len(),
                encoded_len(&entity),
                "{} component(s)",
                entity.components.len()
            );
        }
    }
}
