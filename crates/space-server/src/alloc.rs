//! Monotonic [`ShardId`] allocation.
//!
//! `docs/plan.md` (Phase 3, FIX 9) and [`space_protocol::ShardId`]'s own docs
//! make this a correctness rule, not a policy: an [`EntityGuid`] is unique only
//! as the pair `(shard, serial)`, and every shard mints `serial`s from its own
//! counter. Hand a shard id back to a new shard and the new shard's serials
//! collide with the old shard's still-live guids — two distinct entities, one
//! guid, and every guid-keyed structure (input routing, the interest table, the
//! handoff ledger) silently merges them.
//!
//! Phase 3 moves the counter into the orchestrator (and persists it across
//! restarts). Until then this is the allocator the shard processes use, and it
//! enforces the rule structurally: the counter only ever increases, and there is
//! **no** `free`/`release` function to get wrong.

use std::sync::atomic::{AtomicU64, Ordering};

use space_protocol::ShardId;

/// A process-wide, monotonic shard-id counter.
///
/// Cloning shares the counter (it is `Arc`-free by design: hand it by reference).
#[derive(Debug, Default)]
pub struct ShardIdAllocator {
    next: AtomicU64,
}

impl ShardIdAllocator {
    /// An allocator whose first id is `first`.
    ///
    /// A real deployment starts from the persisted counter, which is why the
    /// start is not hard-coded.
    #[must_use]
    pub const fn new(first: u64) -> Self {
        Self {
            next: AtomicU64::new(first),
        }
    }

    /// Allocate the next id.
    ///
    /// # Panics
    ///
    /// Panics if the counter would pass [`u32::MAX`] or reach
    /// [`ShardId::NONE`], the reserved "no shard" id. Both are world-lifetime
    /// limits, not runtime conditions; wrapping would reuse ids.
    #[must_use]
    pub fn allocate(&self) -> ShardId {
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        assert!(
            id <= u64::from(u32::MAX) && u32::try_from(id).unwrap_or(u32::MAX) != u32::MAX,
            "shard id space exhausted at {id}: ids are never reused, so the world cannot continue"
        );
        ShardId::new(u32::try_from(id).expect("checked above"))
    }

    /// The id the next [`allocate`](Self::allocate) call returns.
    #[must_use]
    pub fn peek(&self) -> u64 {
        self.next.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::ShardIdAllocator;
    use space_protocol::ShardId;

    #[test]
    fn ids_are_monotonic_and_never_reused() {
        let alloc = ShardIdAllocator::new(1);
        let mut seen = Vec::new();
        for _ in 0..1_000 {
            seen.push(alloc.allocate());
        }
        // Strictly increasing, no duplicates — including after the "holder" of
        // an id is gone, because there is nothing that hands an id back.
        for pair in seen.windows(2) {
            assert!(pair[0] < pair[1], "{pair:?} is not strictly increasing");
        }
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 1_000, "an id was handed out twice");
        assert_eq!(seen.first().copied(), Some(ShardId::new(1)));
    }

    #[test]
    fn allocation_resumes_from_a_persisted_counter() {
        // A restart resumes from the persisted value: the first id after the
        // restart is strictly greater than the last one before it.
        let before = ShardIdAllocator::new(10);
        let last = before.allocate();
        assert_eq!(last, ShardId::new(10));
        let after = ShardIdAllocator::new(before.peek());
        assert_eq!(after.allocate(), ShardId::new(11));
    }

    #[test]
    fn the_reserved_none_id_is_never_handed_out() {
        let alloc = ShardIdAllocator::new(u64::from(u32::MAX - 1));
        assert_eq!(alloc.allocate(), ShardId::new(u32::MAX - 1));
        assert!(
            std::panic::catch_unwind(|| alloc.allocate()).is_err(),
            "allocating past u32::MAX must fail loudly, not wrap"
        );
    }
}
