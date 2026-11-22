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

/// The shard-id space is used up: ids are never reused, so there is nothing to
/// hand out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdSpaceExhausted {
    /// The counter value that could not be allocated.
    pub next: u64,
}

impl core::fmt::Display for IdSpaceExhausted {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "shard id space exhausted at {}: ids are never reused",
            self.next
        )
    }
}

impl std::error::Error for IdSpaceExhausted {}

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
    /// # Errors
    ///
    /// [`IdSpaceExhausted`] if the counter would pass [`u32::MAX`] or reach
    /// [`ShardId::NONE`], the reserved "no shard" id. Wrapping would reuse ids,
    /// so the allocator refuses instead; the counter is left unchanged. (It used
    /// to panic, which let one poisoned counter crash the orchestrator.)
    pub fn allocate(&self) -> Result<ShardId, IdSpaceExhausted> {
        let mut current = self.next.load(Ordering::SeqCst);
        loop {
            // Exhausted: past u32::MAX, or on it (that is `ShardId::NONE`).
            let id = match u32::try_from(current) {
                Ok(v) if v != u32::MAX => v,
                _ => return Err(IdSpaceExhausted { next: current }),
            };
            match self.next.compare_exchange(
                current,
                current + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return Ok(ShardId::new(id)),
                Err(actual) => current = actual,
            }
        }
    }

    /// The id the next [`allocate`](Self::allocate) call returns.
    #[must_use]
    pub fn peek(&self) -> u64 {
        self.next.load(Ordering::SeqCst)
    }

    /// Record that `id` is in use by a shard this allocator did not hand out
    /// (a relaunch or a shard claiming a known id), so [`allocate`](Self::allocate)
    /// never returns it later. The counter only ever moves up
    /// (`fetch_max(id + 1)`); [`ShardId::NONE`] is not an id and is ignored.
    pub fn reserve(&self, id: ShardId) {
        if id != ShardId::NONE {
            self.next.fetch_max(u64::from(id.0) + 1, Ordering::SeqCst);
        }
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
            seen.push(alloc.allocate().expect("id space not exhausted"));
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
        let last = before.allocate().expect("allocate");
        assert_eq!(last, ShardId::new(10));
        let after = ShardIdAllocator::new(before.peek());
        assert_eq!(after.allocate(), Ok(ShardId::new(11)));
    }

    #[test]
    fn the_reserved_none_id_is_never_handed_out() {
        let alloc = ShardIdAllocator::new(u64::from(u32::MAX - 1));
        assert_eq!(alloc.allocate(), Ok(ShardId::new(u32::MAX - 1)));
        // Past the limit: an Err (no panic, no wrap), and the counter stays put.
        let at = alloc.peek();
        assert!(alloc.allocate().is_err(), "must fail, not wrap");
        assert!(alloc.allocate().is_err(), "and keep failing");
        assert_eq!(alloc.peek(), at, "a failed allocation does not advance");
    }

    #[test]
    fn a_poisoned_counter_is_an_error_not_a_panic() {
        let alloc = ShardIdAllocator::new(u64::from(u32::MAX));
        assert!(alloc.allocate().is_err());
        let alloc = ShardIdAllocator::new(u64::MAX);
        assert!(alloc.allocate().is_err());
    }
}
