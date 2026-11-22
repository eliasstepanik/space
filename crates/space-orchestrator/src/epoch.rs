//! The two-phase epoch publication state machine (H10, H12).
//!
//! ```text
//! Idle --propose(changes)--> Preparing{epoch, activation_tick, awaiting}
//! Preparing --all acked-----> Committed{epoch}   (published map := new map)
//! Preparing --deadline------> Aborted            (published map unchanged)
//! ```
//!
//! Pure: it takes the map plus the shards that must ack and produces the next
//! [`Message`]s and a deadline tick. The async layer only moves messages.
//!
//! The rules it enforces (plan §3.3):
//! * the published map is **never mutated in place** — a commit swaps in the new
//!   map, an abort leaves the published map byte-identical;
//! * recipients are every shard named in `changes` **plus** every shard whose
//!   region is adjacent to a changed node (a shard never told the map moved
//!   keeps offering handoffs under the old epoch and eats `StaleEpoch` forever);
//! * `activation_tick = max(observer ticks) + EPOCH_LEAD_TICKS`;
//! * **one outstanding prepare at a time** — H10, H12 and H14 all depend on it.

use std::collections::BTreeSet;

use space_protocol::shard::{AbortReason, EpochChange, ShardId, ShardMap};
use space_protocol::Message;

use crate::config::{EPOCH_ACK_TIMEOUT_TICKS, EPOCH_LEAD_TICKS};

/// What the state machine wants the async layer to send, and to whom.
#[derive(Debug, Clone, PartialEq)]
pub struct Outbound {
    /// Recipients.
    pub to: Vec<ShardId>,
    /// The message.
    pub message: Message,
}

/// The publisher's state.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// Nothing in flight; a `propose` is accepted.
    Idle,
    /// A prepare is outstanding; `deadline` is the tick it aborts at.
    Preparing {
        /// Epoch being prepared.
        epoch: u64,
        /// Tick the new map activates everywhere.
        activation_tick: u64,
        /// Ticks the deadline fires at.
        deadline: u64,
        /// Acks still missing.
        awaiting: BTreeSet<ShardId>,
        /// Everyone the prepare was sent to (the abort goes to those who acked,
        /// i.e. `recipients - awaiting`).
        recipients: BTreeSet<ShardId>,
    },
    /// A commit landed; the published map is the new one.
    Committed {
        /// Epoch now in effect.
        epoch: u64,
    },
    /// A prepare timed out; the published map is untouched.
    Aborted {
        /// The epoch that was rolled back.
        epoch: u64,
    },
}

/// The epoch publisher.
#[derive(Debug)]
pub struct EpochPublisher {
    /// The map every shard is currently allowed to observe.
    published: ShardMap,
    phase: Phase,
    /// The prepare currently staged (kept so a commit can swap it in).
    staged: Option<ShardMap>,
    /// The last tick a commit landed at (enforces `MIN_EPOCH_INTERVAL_TICKS`).
    last_commit_tick: Option<u64>,
}

impl EpochPublisher {
    /// A publisher whose published map is `map` (epoch 1 in a fresh cluster).
    #[must_use]
    pub fn new(map: ShardMap) -> Self {
        Self {
            published: map,
            phase: Phase::Idle,
            staged: None,
            last_commit_tick: None,
        }
    }

    /// The map shards must be running: the only map they ever observe.
    #[must_use]
    pub fn published(&self) -> &ShardMap {
        &self.published
    }

    /// Fill in the real endpoint of every published region whose owner is in
    /// `endpoints`.
    ///
    /// A committed epoch's ownership is exact, but a split names its child before
    /// the child's address is known (the region carries a placeholder borrowed
    /// from its parent). The orchestrator learns the real address from the
    /// child's `ShardHello`; this applies it to the published map so the map it
    /// pushes carries endpoint fidelity (plan F9). Ownership and the epoch are
    /// untouched, so H10's "map changes only through an epoch" still holds.
    pub fn set_endpoints(
        &mut self,
        endpoints: &std::collections::BTreeMap<ShardId, std::net::SocketAddr>,
    ) {
        for region in &mut self.published.regions {
            if let Some(endpoint) = endpoints.get(&region.shard) {
                region.endpoint = *endpoint;
            }
        }
    }

    /// The current phase (tests assert on it; the loop reads it to gate work).
    #[must_use]
    pub fn phase(&self) -> &Phase {
        &self.phase
    }

    /// Whether a new change may be proposed right now (H12/H14 serialisation).
    #[must_use]
    pub fn can_propose(&self, tick: u64, min_interval: u64) -> bool {
        matches!(self.phase, Phase::Idle)
            && self
                .last_commit_tick
                .is_none_or(|t| tick.saturating_sub(t) >= min_interval)
    }

    /// Every shard that must hear about `changes`: the shards named in the
    /// changes, the owners of the changed nodes' parents, and every shard whose
    /// region is adjacent to a changed node (a sibling octant of the changed
    /// node or its parent — the shards a handoff could be offered to).
    #[must_use]
    pub fn recipients(&self, changes: &[EpochChange]) -> BTreeSet<ShardId> {
        use space_world::octree;
        let mut out: BTreeSet<ShardId> = BTreeSet::new();
        for (node, owner) in changes {
            out.insert(*owner);
            // The old owner of the region this node falls under.
            if let Some(r) = crate::topology::region_of(&self.published, *owner) {
                out.insert(r.shard);
            }
            if let Some(parent) = octree::parent(*node) {
                for region in &self.published.regions {
                    if region.node == parent {
                        out.insert(region.shard);
                    }
                }
            }
            // Neighbours: owners of the changed node's siblings, and of the
            // parent's siblings one level up.
            let mut siblings = Vec::new();
            if let Some(p) = octree::parent(*node) {
                siblings.extend(octree::children(p).into_iter().flatten());
            }
            if let Some(gp) = octree::parent(*node).and_then(octree::parent) {
                siblings.extend(octree::children(gp).into_iter().flatten());
            }
            for region in &self.published.regions {
                if siblings.contains(&region.node) {
                    out.insert(region.shard);
                }
            }
        }
        out
    }

    /// Propose `changes` as a new epoch.
    ///
    /// `new_map` is the map to publish on commit (epoch = current + 1);
    /// `observer_ticks` are the current ticks of every shard that must switch —
    /// the activation tick is their maximum plus the lead, because the shards
    /// run independent loops and drift.
    ///
    /// Returns the `EpochPrepare` to send, or `None` if a prepare is already
    /// outstanding (one change at a time — the caller must wait).
    #[must_use]
    pub fn propose(
        &mut self,
        new_map: ShardMap,
        changes: Vec<EpochChange>,
        observer_ticks: &[(ShardId, u64)],
        tick: u64,
    ) -> Option<Outbound> {
        if !matches!(self.phase, Phase::Idle) {
            return None;
        }
        assert_eq!(
            new_map.epoch,
            self.published.epoch + 1,
            "an epoch bump is exactly +1 over the published map"
        );
        let max_tick = observer_ticks.iter().map(|(_, t)| *t).max().unwrap_or(tick);
        let activation_tick = max_tick + EPOCH_LEAD_TICKS;
        let awaiting: BTreeSet<ShardId> = self.recipients(&changes);
        let recipients = awaiting.clone();
        let epoch = new_map.epoch;
        let outbound = Outbound {
            to: awaiting.iter().copied().collect(),
            message: Message::EpochPrepare(space_protocol::EpochPrepare {
                epoch,
                activation_tick,
                changes,
            }),
        };
        self.staged = Some(new_map);
        self.phase = Phase::Preparing {
            epoch,
            activation_tick,
            deadline: tick + EPOCH_ACK_TIMEOUT_TICKS,
            awaiting,
            recipients,
        };
        Some(outbound)
    }

    /// Record an `EpochAck`. When the last one lands, returns the `EpochCommit`
    /// to broadcast and swaps the staged map in.
    ///
    /// An ack only counts toward the epoch it acknowledges: while preparing, an
    /// ack whose `ack_epoch` differs from the staged epoch is ignored (returns
    /// `None`, `awaiting` untouched).
    ///
    /// Known limit (accepted): a late ack carrying the *same* epoch number
    /// issued after an abort is indistinguishable from a fresh one — the epoch
    /// number is the only discriminator — and still counts.
    #[must_use]
    pub fn ack(&mut self, shard: ShardId, ack_epoch: u64, tick: u64) -> Option<Outbound> {
        let (epoch, commit_now) = match &mut self.phase {
            Phase::Preparing {
                epoch, awaiting, ..
            } => {
                if ack_epoch != *epoch {
                    return None;
                }
                awaiting.remove(&shard);
                (*epoch, awaiting.is_empty())
            }
            _ => return None,
        };
        if !commit_now {
            return None;
        }
        let new_map = self.staged.take().expect("a prepare stages a map");
        self.published = new_map;
        self.phase = Phase::Committed { epoch };
        self.last_commit_tick = Some(tick);
        Some(Outbound {
            // The commit goes to *every* shard in the map, not just the changed
            // ones: every shard's `shard_map()` must carry the new epoch so its
            // next Welcome/Redirect does too (plan §3.3).
            to: self
                .published
                .regions
                .iter()
                .map(|r| r.shard)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            message: Message::EpochCommit(space_protocol::EpochCommit { epoch }),
        })
    }

    /// The deadline check: at `tick >= deadline` an un-acked prepare aborts.
    ///
    /// Returns the `EpochAbort` to send to everyone who *did* ack. The published
    /// map is byte-identical to what it was before the propose.
    #[must_use]
    pub fn timeout(&mut self, tick: u64) -> Option<Outbound> {
        let Phase::Preparing {
            epoch,
            deadline,
            awaiting,
            recipients,
            ..
        } = &self.phase
        else {
            return None;
        };
        if tick < *deadline {
            return None;
        }
        let epoch = *epoch;
        // The abort goes to everyone who acked: recipients minus the stragglers
        // (the straggler never staged the change, so it has nothing to roll back).
        let acked: Vec<ShardId> = recipients.difference(awaiting).copied().collect();
        self.staged = None;
        self.phase = Phase::Aborted { epoch };
        Some(Outbound {
            to: acked,
            message: Message::EpochAbort(space_protocol::EpochAbort {
                epoch,
                reason: AbortReason::Timeout,
            }),
        })
    }

    /// Return to `Idle` after a commit or abort was delivered, so the next
    /// change can be proposed.
    pub fn reset(&mut self) {
        self.phase = Phase::Idle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::shard::ShardRegion;
    use space_world::NodeRef;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn addr(n: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), n)
    }

    fn shard(n: u32) -> ShardId {
        ShardId::new(n)
    }

    fn region(depth: u8, index: [i64; 3], owner: u32) -> ShardRegion {
        ShardRegion::new(
            NodeRef::new(depth, index),
            shard(owner),
            addr(7_000 + owner as u16),
        )
    }

    /// Root (depth 40) owned by 1, with one child octant already carved for 2.
    fn map_with_two() -> ShardMap {
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[0];
        ShardMap::new(
            1,
            vec![region(40, [0, 0, 0], 1), region(41, child.index, 2)],
        )
    }

    #[test]
    fn epochs_are_monotonic_and_the_map_is_versioned() {
        let mut pub_ = EpochPublisher::new(map_with_two());
        assert_eq!(pub_.published().epoch, 1);
        // Propose a split of the root: child octant 1 -> shard 3, epoch 2.
        let mut next = pub_.published().clone();
        next.epoch = 2;
        let child = space_world::octree::children(NodeRef::new(40, [0, 0, 0])).unwrap()[1];
        next.regions.push(region(41, child.index, 3));
        let prepare = pub_
            .propose(
                next.clone(),
                vec![(child, shard(3))],
                &[(shard(1), 100), (shard(3), 90)],
                100,
            )
            .expect("first propose accepted");
        assert!(matches!(prepare.message, Message::EpochPrepare(_)));
        // A second propose while one is outstanding is refused (one at a time).
        assert!(pub_.propose(next.clone(), vec![], &[], 101).is_none());
        // Every recipient acks -> commit, and the published map is the staged one.
        let Outbound { to, .. } = &prepare;
        let mut last = None;
        for s in to.clone() {
            last = pub_.ack(s, 2, 110);
        }
        let commit = last.expect("the final ack commits");
        assert!(matches!(commit.message, Message::EpochCommit(_)));
        assert_eq!(pub_.published().epoch, 2);
        assert_eq!(
            pub_.published(),
            &next,
            "the published map is the staged map"
        );
        pub_.reset();
        // The next epoch is exactly +1 again.
        let mut next2 = pub_.published().clone();
        next2.epoch = 3;
        assert!(pub_
            .propose(next2, vec![(child, shard(4))], &[(shard(1), 200)], 200)
            .is_some());
    }

    #[test]
    fn an_unacked_prepare_aborts_and_the_published_map_is_byte_identical() {
        let mut pub_ = EpochPublisher::new(map_with_two());
        let before = pub_.published().clone();
        let mut next = before.clone();
        next.epoch = 2;
        let child = space_world::octree::children(NodeRef::new(40, [0, 0, 0])).unwrap()[2];
        next.regions.push(region(41, child.index, 3));
        let prepare = pub_
            .propose(next, vec![(child, shard(3))], &[(shard(1), 500)], 500)
            .expect("propose");
        // One shard acks, one never does.
        let straggler = *prepare.to.iter().max().expect("at least one recipient");
        let acker = *prepare.to.iter().min().expect("at least two recipients");
        assert_ne!(straggler, acker, "test needs two recipients");
        assert!(pub_.ack(acker, 510).is_none(), "one ack must not commit");
        // Before the deadline: nothing.
        assert!(pub_.timeout(500 + EPOCH_ACK_TIMEOUT_TICKS - 1).is_none());
        // At the deadline: abort, and the published map is untouched.
        let abort = pub_
            .timeout(500 + EPOCH_ACK_TIMEOUT_TICKS)
            .expect("deadline aborts");
        assert!(matches!(abort.message, Message::EpochAbort(_)));
        assert_eq!(
            abort.to,
            vec![acker],
            "the abort goes to the shard that acked"
        );
        assert_eq!(pub_.published(), &before, "byte-identical after an abort");
        pub_.reset();
        assert!(pub_.can_propose(10_000, 0), "idle again after the abort");
    }

    #[test]
    fn activation_uses_the_max_observer_tick_plus_the_lead() {
        let mut pub_ = EpochPublisher::new(map_with_two());
        let mut next = pub_.published().clone();
        next.epoch = 2;
        let child = space_world::octree::children(NodeRef::new(40, [0, 0, 0])).unwrap()[3];
        next.regions.push(region(41, child.index, 3));
        let prepare = pub_
            .propose(
                next,
                vec![(child, shard(3))],
                &[(shard(1), 1_000), (shard(3), 40)],
                40,
            )
            .expect("propose");
        match prepare.message {
            Message::EpochPrepare(p) => {
                assert_eq!(
                    p.activation_tick,
                    1_000 + EPOCH_LEAD_TICKS,
                    "max tick + lead"
                );
            }
            other => panic!("expected EpochPrepare, got {other:?}"),
        }
    }

    #[test]
    fn every_shard_that_must_switch_is_a_prepare_recipient() {
        // H10's recipient rule: the changed node's owner, the parent's owner,
        // and the owners of adjacent (sibling) regions.
        let mut pub_ = EpochPublisher::new(map_with_two());
        let root = NodeRef::new(40, [0, 0, 0]);
        let kids = space_world::octree::children(root).unwrap();
        // Change: carve kids[1] for shard 3. Shard 2 owns kids[0] — adjacent —
        // and shard 1 owns the root — the parent. Both must be recipients.
        let mut next = pub_.published().clone();
        next.epoch = 2;
        next.regions.push(region(41, kids[1].index, 3));
        let prepare = pub_
            .propose(next, vec![(kids[1], shard(3))], &[(shard(1), 10)], 10)
            .expect("propose");
        let recipients: BTreeSet<ShardId> = prepare.to.into_iter().collect();
        assert!(
            recipients.contains(&shard(1)),
            "the parent's owner must be told"
        );
        assert!(
            recipients.contains(&shard(2)),
            "the adjacent sibling's owner must be told"
        );
        assert!(recipients.contains(&shard(3)), "the new owner must be told");
    }

    #[test]
    fn commits_are_spaced_by_the_minimum_epoch_interval() {
        let mut pub_ = EpochPublisher::new(map_with_two());
        let mut next = pub_.published().clone();
        next.epoch = 2;
        let child = space_world::octree::children(NodeRef::new(40, [0, 0, 0])).unwrap()[4];
        next.regions.push(region(41, child.index, 3));
        let prepare = pub_
            .propose(next, vec![(child, shard(3))], &[(shard(1), 10)], 10)
            .expect("propose");
        for s in prepare.to {
            let _ = pub_.ack(s, 10);
        }
        pub_.reset();
        // Immediately after a commit at tick 10, a new propose is gated until
        // the minimum interval has passed.
        assert!(!pub_.can_propose(10, 90), "no commit inside the interval");
        assert!(pub_.can_propose(100, 90), "the interval has passed");
    }

    #[test]
    fn set_endpoints_fills_addresses_without_touching_ownership_or_epoch() {
        let mut pub_ = EpochPublisher::new(map_with_two());
        let before = pub_.published().clone();
        let mut endpoints = std::collections::BTreeMap::new();
        endpoints.insert(shard(2), addr(9_002));
        pub_.set_endpoints(&endpoints);
        let after = pub_.published();
        assert_eq!(after.epoch, before.epoch);
        assert_eq!(after.regions.len(), before.regions.len());
        for (a, b) in after.regions.iter().zip(&before.regions) {
            assert_eq!((a.node, a.shard), (b.node, b.shard));
            if a.shard == shard(2) {
                assert_eq!(a.endpoint, addr(9_002));
            } else {
                assert_eq!(a.endpoint, b.endpoint, "an unknown shard keeps its address");
            }
        }
    }
}
