//! Globally unique, shard-stable entity identity.

use serde::{Deserialize, Serialize};

use crate::shard::ShardId;

/// A globally unique entity id: the shard that *minted* the entity plus a
/// per-shard serial.
///
/// The pair is the identity for the entity's whole life: when the entity is
/// handed off to another shard the guid does **not** change — `shard` records
/// the minting shard forever, and ownership is tracked separately by the shard
/// map and by handoff messages. That is what makes a guid safe to embed in
/// client state, ghost replicas and log lines: two peers that saw the same guid
/// always mean the same entity, and no shard ever gets to alias its own serial
/// space onto another's.
///
/// Serials are handed out per shard from a monotonic counter, so `(shard,
/// serial)` is unique across the world without a central allocator.
///
/// # ShardId reuse (Phase 2 / Phase 3 decision)
///
/// Guid uniqueness rests on `(shard, serial)` never colliding, which requires a
/// [`ShardId`] never to be handed to two different shard processes over the life
/// of a world — the orchestrator creates and removes shards as it splits and
/// merges. The contract is therefore: **ShardIds are monotonic and never
/// reused.** A shard id counter that wrapped, or an orchestrator that recycled
/// an id after a shard died, would let a new shard mint a guid equal to a dead
/// shard's, silently aliasing two entities. The orchestrator (Phase 3) owns the
/// counter and must keep it monotonic for the world's lifetime (persisted across
/// orchestrator restarts). If monotonicity ever proves impractical, the
/// alternative is an incarnation field on the guid (bumping it whenever an id is
/// reused) — deferred to Phase 3; Phase 1 keeps the guid at `(shard, serial)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityGuid {
    /// Shard that minted the entity. Never rewritten on handoff.
    pub shard: ShardId,
    /// Serial allocated by that shard.
    pub serial: u64,
}

impl EntityGuid {
    /// Build a guid from its parts.
    #[must_use]
    pub const fn new(shard: ShardId, serial: u64) -> Self {
        Self { shard, serial }
    }
}

impl core::fmt::Display for EntityGuid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "shard {} #{:016x}", self.shard.0, self.serial)
    }
}
