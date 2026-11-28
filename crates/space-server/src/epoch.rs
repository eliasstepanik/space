//! Shard-side epoch protocol participant (Phase 3, slice 2).
//!
//! When the orchestrator bumps the epoch and publishes a new shard map, the
//! shard must actively participate: receive [`EpochPrepare`], answer with
//! [`EpochAck`], then on [`EpochCommit`] swap in the new map (or on
//! [`EpochAbort`] discard the staged change).
//!
//! This module is **additive**: it does not change Phase-2 handoff semantics.
//! The existing epoch-stale abort path in `handoff_pass` continues to work
//! unchanged — a handoff that is in-flight when the epoch bumps still aborts
//! with `EpochChanged` because the handoff's epoch no longer matches the map
//! epoch after commit.
//!
//! # What a prepare carries, and what the shard can reconstruct
//!
//! [`EpochPrepare`] on the wire carries `changes: Vec<(NodeRef, ShardId)>` — the
//! nodes that change owner and their new owner — not a full [`ShardMap`] (see
//! `space_protocol::shard::EpochPrepare`). A split is expressed *additively*
//! (plan F5): the carved child node is appended with its new owner and the
//! parent keeps the old owner, so applying a change to the shard's current map
//! is exactly "append one region". The shard therefore rebuilds the new map
//! itself from the changes rather than receiving it whole.
//!
//! A change names a new owner but not its endpoint. For a region whose owner
//! already appears in the map (a rebalance or a merge back to a known shard) the
//! endpoint is copied from that owner's existing region. For a brand-new owner
//! the endpoint is not on the wire, so the region gets the un-dialable
//! [`ShardRegion::UNRESOLVED_ENDPOINT`] sentinel (never a parent's address, which
//! would misroute peer traffic); the orchestrator follows every commit with an
//! authoritative map push (a `DirectoryReply`, handled by
//! `Shard::orchestrator_message`; plan F9) to fill it in. The change-apply
//! path here keeps *ownership* correct (the routing function `lookup` flips only
//! inside the child, which is H11's structural half); endpoint fidelity for a
//! never-before-seen shard is the map push's job.

use space_protocol::shard::{
    EpochAbort, EpochAck, EpochChange, EpochCommit, EpochPrepare, ShardId, ShardMap, ShardRegion,
};

/// The shard's staged epoch: what a prepare recorded, awaiting commit or abort.
#[derive(Debug, Clone)]
pub struct StagedEpoch {
    /// The epoch being prepared.
    pub epoch: u64,
    /// The tick at which the new map activates.
    pub activation_tick: u64,
    /// The ownership changes to apply on commit.
    pub changes: Vec<EpochChange>,
}

/// Tracks the shard's epoch-protocol participation.
///
/// One outstanding prepare at a time (plan H10/H12/H14 all depend on it): a
/// second prepare while one is staged is rejected, matching the publisher's
/// "one change at a time" rule on the orchestrator side.
#[derive(Debug, Default)]
pub struct EpochParticipant {
    staged: Option<StagedEpoch>,
}

impl EpochParticipant {
    /// A participant with nothing staged.
    #[must_use]
    pub const fn new() -> Self {
        Self { staged: None }
    }

    /// Handle an incoming [`EpochPrepare`].
    ///
    /// Returns the [`EpochAck`] the shard should send back, or `None` when the
    /// prepare is not acceptable: an epoch that is not exactly one above the
    /// current one (stale or skipping ahead), or a prepare that arrived while
    /// another is still outstanding.
    #[must_use]
    pub fn handle_prepare(
        &mut self,
        prepare: &EpochPrepare,
        current_epoch: u64,
        from: ShardId,
    ) -> Option<EpochAck> {
        // One outstanding prepare at a time.
        if self.staged.is_some() {
            return None;
        }
        // An epoch bump is exactly +1 over the current map (the publisher
        // asserts the same invariant on its side).
        if prepare.epoch != current_epoch + 1 {
            return None;
        }
        self.staged = Some(StagedEpoch {
            epoch: prepare.epoch,
            activation_tick: prepare.activation_tick,
            changes: prepare.changes.clone(),
        });
        Some(EpochAck {
            epoch: prepare.epoch,
            shard: from,
        })
    }

    /// Handle an incoming [`EpochCommit`].
    ///
    /// Returns the new map to install, built by applying the staged changes to
    /// `current`, or `None` when no matching prepare is staged (the commit is
    /// stale, was already applied, or names a different epoch — in which case a
    /// staged prepare for another epoch is left untouched).
    #[must_use]
    pub fn handle_commit(&mut self, commit: &EpochCommit, current: &ShardMap) -> Option<ShardMap> {
        let matches = self
            .staged
            .as_ref()
            .is_some_and(|s| s.epoch == commit.epoch);
        if !matches {
            return None;
        }
        // `staged` matched, so `take` yields the staged epoch.
        let staged = self.staged.take().expect("staged epoch present");
        Some(apply_changes(current, staged.epoch, &staged.changes))
    }

    /// Handle an incoming [`EpochAbort`].
    ///
    /// Returns `true` when a matching staged prepare was discarded.
    #[must_use]
    pub fn handle_abort(&mut self, abort: &EpochAbort) -> bool {
        if self.staged.as_ref().is_some_and(|s| s.epoch == abort.epoch) {
            self.staged = None;
            true
        } else {
            false
        }
    }

    /// Discard a staged prepare that installing `epoch` has superseded.
    ///
    /// Called when a map for `epoch` is installed by any route (a commit, or a
    /// `DirectoryReply` push covering a commit this shard missed). A staged
    /// prepare for an epoch `<= epoch` can never be committed any more, and left
    /// in place it would make [`EpochParticipant::handle_prepare`] refuse every
    /// later prepare. A staged prepare for a *later* epoch is untouched.
    ///
    /// Returns `true` when a staged prepare was discarded.
    #[must_use]
    pub fn discard_through(&mut self, epoch: u64) -> bool {
        if self.staged.as_ref().is_some_and(|s| s.epoch <= epoch) {
            self.staged = None;
            true
        } else {
            false
        }
    }

    /// The staged epoch, if one is outstanding.
    #[must_use]
    pub fn staged(&self) -> Option<&StagedEpoch> {
        self.staged.as_ref()
    }

    /// Whether a prepare is outstanding.
    #[must_use]
    pub fn has_staged(&self) -> bool {
        self.staged.is_some()
    }
}

/// Build the map that activates at commit: `current` with every change applied
/// and the epoch bumped to `epoch`.
///
/// A change `(node, new_owner)` for a node the map does not name yet (a split
/// carving a child) is appended as a region owned by `new_owner`. A change for a
/// node the map **already** names (a merge handing a child back to its parent's
/// owner, a rebalance) *replaces* that region's owner: appending a second
/// region at the same node would leave the old owner in charge, because
/// [`ShardMap::lookup`] keeps the first of two regions of equal depth.
///
/// The endpoints — the internal one and the one players dial — are copied from
/// an existing region already owned by `new_owner` (the common case for a
/// rebalance or merge); when the owner is new to the map both are the
/// un-dialable [`ShardRegion::UNRESOLVED_ENDPOINT`] sentinel (never a parent's
/// address, which would misroute peer traffic — or a joining player — to the
/// wrong shard); the orchestrator's map push (plan F9) replaces them with the
/// real addresses once the owner registers. Ownership, which is what `lookup`
/// and the handoff boundary test depend on, is exact either way.
fn apply_changes(current: &ShardMap, epoch: u64, changes: &[EpochChange]) -> ShardMap {
    let mut regions = current.regions.clone();
    for (node, new_owner) in changes {
        let unresolved = ShardRegion::UNRESOLVED_ENDPOINT;
        let (endpoint, client_endpoint) = regions
            .iter()
            .find(|r| r.shard == *new_owner)
            .map_or((unresolved, unresolved), |r| {
                (r.endpoint, r.client_endpoint)
            });
        // One region per node. A node the map already names changes hands in
        // place (and any duplicate a malformed map carried goes with it).
        regions.retain(|r| r.node != *node);
        regions.push(
            ShardRegion::new(*node, *new_owner, endpoint).with_client_endpoint(client_endpoint),
        );
    }
    ShardMap::new(epoch, regions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::shard::AbortReason;
    use space_world::NodeRef;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn addr(n: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), n)
    }

    fn region(depth: u8, index: [i64; 3], owner: u32, port: u16) -> ShardRegion {
        ShardRegion::new(NodeRef::new(depth, index), ShardId::new(owner), addr(port))
    }

    /// Root (depth 40) owned by 1, with one child octant already carved for 2.
    fn map_with_two() -> ShardMap {
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[0];
        ShardMap::new(
            1,
            vec![
                region(40, [0, 0, 0], 1, 7_001),
                region(41, child.index, 2, 7_002),
            ],
        )
    }

    fn prepare(epoch: u64, changes: Vec<EpochChange>) -> EpochPrepare {
        EpochPrepare {
            epoch,
            activation_tick: 100,
            changes,
        }
    }

    #[test]
    fn prepare_then_commit_applies_the_change_and_bumps_the_epoch() {
        let mut ep = EpochParticipant::new();
        let current = map_with_two();
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[1];
        let changes = vec![(child, ShardId::new(2))];
        let ack = ep
            .handle_prepare(&prepare(2, changes.clone()), 1, ShardId::new(1))
            .expect("a +1 prepare is acked");
        assert_eq!(ack.epoch, 2);
        assert!(ep.has_staged());

        let new_map = ep
            .handle_commit(&EpochCommit { epoch: 2 }, &current)
            .expect("the commit installs");
        assert_eq!(new_map.epoch, 2);
        assert_eq!(new_map.regions.len(), 3, "the child region was appended");
        assert!(!ep.has_staged());
    }

    #[test]
    fn prepare_then_abort_discards_the_staged_change() {
        let mut ep = EpochParticipant::new();
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[1];
        ep.handle_prepare(
            &prepare(2, vec![(child, ShardId::new(2))]),
            1,
            ShardId::new(1),
        )
        .expect("acked");
        let abort = EpochAbort {
            epoch: 2,
            reason: AbortReason::Timeout,
        };
        assert!(ep.handle_abort(&abort));
        assert!(!ep.has_staged());
        // A commit after the abort installs nothing.
        assert!(ep
            .handle_commit(&EpochCommit { epoch: 2 }, &map_with_two())
            .is_none());
    }

    #[test]
    fn a_stale_or_skipped_prepare_is_rejected() {
        let mut ep = EpochParticipant::new();
        // Same epoch as the map: stale.
        assert!(ep
            .handle_prepare(&prepare(1, vec![]), 1, ShardId::new(1))
            .is_none());
        // Two ahead: a skipped epoch is not accepted.
        assert!(ep
            .handle_prepare(&prepare(3, vec![]), 1, ShardId::new(1))
            .is_none());
        assert!(!ep.has_staged());
    }

    #[test]
    fn a_second_prepare_while_one_is_staged_is_rejected() {
        let mut ep = EpochParticipant::new();
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[1];
        ep.handle_prepare(
            &prepare(2, vec![(child, ShardId::new(2))]),
            1,
            ShardId::new(1),
        )
        .expect("first acked");
        // A prepare for epoch 3 while epoch 2 is staged: refused (one at a time).
        assert!(ep
            .handle_prepare(&prepare(3, vec![]), 1, ShardId::new(1))
            .is_none());
        assert!(ep.has_staged(), "the first prepare survives the refusal");
    }

    #[test]
    fn a_commit_for_an_unstaged_epoch_installs_nothing() {
        let mut ep = EpochParticipant::new();
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[1];
        ep.handle_prepare(
            &prepare(3, vec![(child, ShardId::new(2))]),
            2,
            ShardId::new(1),
        )
        .expect("acked");
        // Commit for epoch 2 while epoch 3 is staged: ignored, staging kept.
        assert!(ep
            .handle_commit(&EpochCommit { epoch: 2 }, &map_with_two())
            .is_none());
        assert!(ep.has_staged());
    }

    #[test]
    fn a_change_for_a_new_owner_gets_the_unresolved_sentinel_endpoint() {
        let mut ep = EpochParticipant::new();
        let current = map_with_two();
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[2];
        // Shard 9 is brand new: no region in the map, so no endpoint on the wire.
        ep.handle_prepare(
            &prepare(2, vec![(child, ShardId::new(9))]),
            1,
            ShardId::new(1),
        )
        .expect("acked");
        let new_map = ep
            .handle_commit(&EpochCommit { epoch: 2 }, &current)
            .expect("commit installs");
        // The child region is present and owned by 9 (ownership is exact), and its
        // endpoint is the un-dialable sentinel — never the parent's address.
        let region = new_map
            .regions
            .iter()
            .find(|r| r.node == child && r.shard == ShardId::new(9))
            .expect("the child region is present");
        assert_eq!(region.endpoint, ShardRegion::UNRESOLVED_ENDPOINT);
        assert_eq!(region.endpoint.port(), 0);
        assert_eq!(new_map.epoch, 2);
    }

    #[test]
    fn a_change_for_a_known_owner_copies_that_owners_endpoint() {
        let mut ep = EpochParticipant::new();
        let current = map_with_two();
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[3];
        ep.handle_prepare(
            &prepare(2, vec![(child, ShardId::new(2))]),
            1,
            ShardId::new(1),
        )
        .expect("acked");
        let new_map = ep
            .handle_commit(&EpochCommit { epoch: 2 }, &current)
            .expect("commit installs");
        let region = new_map
            .regions
            .iter()
            .find(|r| r.node == child)
            .expect("present");
        assert_eq!(region.endpoint, addr(7_002));
    }

    /// A merge hands a child node back to its parent's owner: the change names a
    /// node the map already has, and the owner must actually change.
    ///
    /// Mutation this catches: appending the change as a second region at the
    /// same node — `lookup` keeps the first of two regions of equal depth, so
    /// the departed shard would still own the node.
    #[test]
    fn a_change_for_a_node_the_map_already_names_replaces_its_owner() {
        let mut ep = EpochParticipant::new();
        let current = map_with_two();
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[0];
        assert_eq!(
            current.lookup(&child.center()),
            Some(ShardId::new(2)),
            "precondition: shard 2 owns the child"
        );
        ep.handle_prepare(
            &prepare(2, vec![(child, ShardId::new(1))]),
            1,
            ShardId::new(1),
        )
        .expect("acked");
        let new_map = ep
            .handle_commit(&EpochCommit { epoch: 2 }, &current)
            .expect("commit installs");
        assert_eq!(
            new_map.lookup(&child.center()),
            Some(ShardId::new(1)),
            "the child's new owner must win the lookup"
        );
        let regions: Vec<&ShardRegion> =
            new_map.regions.iter().filter(|r| r.node == child).collect();
        assert_eq!(regions.len(), 1, "one region per node");
        assert_eq!(regions[0].shard, ShardId::new(1));
        assert_eq!(
            regions[0].endpoint,
            addr(7_001),
            "with the endpoint its new owner is already known by"
        );
        assert_eq!(new_map.regions.len(), 2, "nothing was appended");
    }

    #[test]
    fn discard_through_clears_a_staged_prepare_at_or_below_the_installed_epoch() {
        let mut ep = EpochParticipant::new();
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[1];
        ep.handle_prepare(
            &prepare(3, vec![(child, ShardId::new(2))]),
            2,
            ShardId::new(1),
        )
        .expect("acked");
        // Installing an older epoch leaves the staged prepare for 3 alone.
        assert!(!ep.discard_through(2));
        assert!(ep.has_staged());
        // Installing epoch 3 supersedes it; a later prepare is accepted again.
        assert!(ep.discard_through(3));
        assert!(!ep.has_staged());
        assert!(ep
            .handle_prepare(&prepare(4, vec![]), 3, ShardId::new(1))
            .is_some());
        // Installing a newer epoch than the staged one also clears it.
        assert!(ep.discard_through(7));
        assert!(!ep.has_staged());
        assert!(!ep.discard_through(7), "nothing left to discard");
    }
}
