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
//! * recipients are **every shard in the published map** plus every shard
//!   named in `changes` (a shard never told the map moved keeps offering
//!   handoffs under the old epoch and eats `StaleEpoch` forever, and with
//!   regions carved at any depth there is no cheap rule for who "borders" a
//!   change — so everybody hears it, and everybody must ack);
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
        /// Everyone the prepare was sent to. An abort goes to **all** of them:
        /// a shard whose ack is merely late has staged the change too.
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
    /// The `EpochPrepare` currently outstanding, kept so it can be sent again
    /// to a recipient that only registers after it was first published (a
    /// split's child, which does not exist yet when its parent is told).
    outstanding: Option<Message>,
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
            outstanding: None,
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

    /// The tick the last commit landed at, if any since this publisher was
    /// built. Read-only, for an operator view.
    #[must_use]
    pub fn last_commit_tick(&self) -> Option<u64> {
        self.last_commit_tick
    }

    /// The ownership changes of the prepare currently outstanding (empty when
    /// nothing is in flight). Read-only, for an operator view.
    #[must_use]
    pub fn outstanding_changes(&self) -> &[EpochChange] {
        match &self.outstanding {
            Some(Message::EpochPrepare(prepare)) => &prepare.changes,
            _ => &[],
        }
    }

    /// Whether a new change may be proposed right now (H12/H14 serialisation).
    #[must_use]
    pub fn can_propose(&self, tick: u64, min_interval: u64) -> bool {
        matches!(self.phase, Phase::Idle)
            && self
                .last_commit_tick
                .is_none_or(|t| tick.saturating_sub(t) >= min_interval)
    }

    /// Every shard that must hear about `changes` — and ack them before they
    /// commit: **every shard in the published map**, and every owner the
    /// changes name (the new shard of a carve is not in the map yet).
    ///
    /// It used to be the changed node's direct parent's owner and the owners
    /// of sibling and grand-sibling octants: the shards a handoff could be
    /// offered to when a split only ever carved a direct child. A cell carved
    /// twenty-four levels below the root has neither a region for a parent nor
    /// one for a sibling, so that rule reached nobody but the new shard — not
    /// even the shard losing the cell — and everyone else learned of the
    /// change from the map push that follows a commit, which is offered to a
    /// queue and may be dropped. A cluster is a handful of shards
    /// (`docs/plan-sharding.md` §1), so telling all of them costs nothing and
    /// makes the prepare/ack/commit exchange the one way a map reaches a shard.
    ///
    /// The price: a shard of the map that cannot ack — it is gone, or the
    /// reserved root owner nobody has claimed yet — holds up every change
    /// until it is replaced.
    #[must_use]
    pub fn recipients(&self, changes: &[EpochChange]) -> BTreeSet<ShardId> {
        self.published
            .regions
            .iter()
            .map(|region| region.shard)
            .chain(changes.iter().map(|(_, owner)| *owner))
            .collect()
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
        self.propose_without(new_map, changes, observer_ticks, tick, &BTreeSet::new())
    }

    /// [`Self::propose`], with `absent` left out of the recipients: shards the
    /// orchestrator **knows to be dead**. They are neither sent the prepare nor
    /// waited for.
    ///
    /// This exists for one change only: the one that gives a dead shard's
    /// regions to its replacement. Every shard of the map must ack a change
    /// ([`Self::recipients`]), a dead one never will — so without this the
    /// change that removes it from the map could never commit, and neither
    /// could any other. Leaving a *live* shard out would break H10 (it would
    /// run a map nobody told it was replaced), which is why the callers pass
    /// nothing here for any other change.
    #[must_use]
    pub fn propose_without(
        &mut self,
        new_map: ShardMap,
        changes: Vec<EpochChange>,
        observer_ticks: &[(ShardId, u64)],
        tick: u64,
        absent: &BTreeSet<ShardId>,
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
        let awaiting: BTreeSet<ShardId> = self
            .recipients(&changes)
            .difference(absent)
            .copied()
            .collect();
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
        self.outstanding = Some(outbound.message.clone());
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
        self.outstanding = None;
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
    /// Returns the `EpochAbort` to send to **every recipient of the prepare**,
    /// acked or not. The published map is byte-identical to what it was before
    /// the propose.
    ///
    /// The abort used to go only to the shards that had acked, on the theory
    /// that a straggler "never staged the change". But a shard stages the change
    /// the moment the prepare *arrives*; its ack can simply be late (it was
    /// paused, its link hiccupped) or lost. Such a shard was never told to roll
    /// back, kept the prepare staged for ever, and — a shard holds one staged
    /// prepare at a time — refused every later prepare, so no epoch that needed
    /// its ack could commit again. A shard that really staged nothing ignores an
    /// abort for an epoch it does not know.
    #[must_use]
    pub fn timeout(&mut self, tick: u64) -> Option<Outbound> {
        let Phase::Preparing {
            epoch,
            deadline,
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
        // Everyone the prepare went to, stragglers included (see above).
        let to: Vec<ShardId> = recipients.iter().copied().collect();
        self.staged = None;
        self.outstanding = None;
        self.phase = Phase::Aborted { epoch };
        Some(Outbound {
            to,
            message: Message::EpochAbort(space_protocol::EpochAbort {
                epoch,
                reason: AbortReason::Timeout,
            }),
        })
    }

    /// Abort the outstanding prepare **now**, whatever its deadline: a
    /// recipient is known to be dead, so the acks can never be complete.
    /// Returns the `EpochAbort` for every recipient, exactly as
    /// [`Self::timeout`] does at the deadline; `None` when nothing is in
    /// flight. The published map is untouched.
    #[must_use]
    pub fn abort(&mut self) -> Option<Outbound> {
        self.timeout(u64::MAX)
    }

    /// Return to `Idle` after a commit or abort was delivered, so the next
    /// change can be proposed.
    pub fn reset(&mut self) {
        self.phase = Phase::Idle;
    }

    /// The outstanding `EpochPrepare`, if `shard` is a recipient whose ack is
    /// still missing.
    ///
    /// A split names its child as a recipient before the child exists: the
    /// prepare is published, the child is launched, and only then does it
    /// register. Whoever registers it asks here whether there is a prepare it
    /// still owes an ack for, and sends it — without this the child could never
    /// ack and no split could ever commit.
    #[must_use]
    pub fn outstanding_prepare_for(&self, shard: ShardId) -> Option<Message> {
        match &self.phase {
            Phase::Preparing { awaiting, .. } if awaiting.contains(&shard) => {
                self.outstanding.clone()
            }
            _ => None,
        }
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
        assert!(pub_.ack(acker, 2, 510).is_none(), "one ack must not commit");
        // Before the deadline: nothing.
        assert!(pub_.timeout(500 + EPOCH_ACK_TIMEOUT_TICKS - 1).is_none());
        // At the deadline: abort, and the published map is untouched.
        let abort = pub_
            .timeout(500 + EPOCH_ACK_TIMEOUT_TICKS)
            .expect("deadline aborts");
        assert!(matches!(abort.message, Message::EpochAbort(_)));
        // Mutation this catches: aborting only the shards that acked. The
        // straggler has the prepare staged too (its ack is merely missing), and
        // a shard that is never told to roll back refuses every later prepare.
        let mut told = abort.to.clone();
        told.sort_unstable();
        let mut everyone = prepare.to.clone();
        everyone.sort_unstable();
        assert_eq!(
            told, everyone,
            "the abort goes to every recipient of the prepare"
        );
        assert!(abort.to.contains(&acker) && abort.to.contains(&straggler));
        assert_eq!(pub_.published(), &before, "byte-identical after an abort");
        pub_.reset();
        assert!(pub_.can_propose(10_000, 0), "idle again after the abort");
    }

    #[test]
    fn the_outstanding_prepare_is_available_to_a_recipient_that_has_not_acked() {
        let mut pub_ = EpochPublisher::new(map_with_two());
        let root = NodeRef::new(40, [0, 0, 0]);
        let child = space_world::octree::children(root).unwrap()[6];
        assert!(pub_.outstanding_prepare_for(shard(3)).is_none(), "idle");
        let mut next = pub_.published().clone();
        next.epoch = 2;
        next.regions.push(region(41, child.index, 3));
        let prepare = pub_
            .propose(next, vec![(child, shard(3))], &[(shard(1), 10)], 10)
            .expect("propose");
        // The child (3) has not acked: it can be handed the same prepare again.
        assert_eq!(
            pub_.outstanding_prepare_for(shard(3)),
            Some(prepare.message.clone())
        );
        // A shard that is not a recipient gets nothing.
        assert!(pub_.outstanding_prepare_for(shard(99)).is_none());
        // Once it has acked there is nothing left to send it.
        let _ = pub_.ack(shard(3), 2, 11);
        assert!(pub_.outstanding_prepare_for(shard(3)).is_none());
        // And after the abort nothing is outstanding for anyone.
        let _ = pub_.timeout(10 + EPOCH_ACK_TIMEOUT_TICKS).expect("abort");
        for s in prepare.to {
            assert!(pub_.outstanding_prepare_for(s).is_none());
        }
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
        // H10's recipient rule: the changed node's owner and every shard of
        // the published map — here the parent's owner and an adjacent
        // sibling's.
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

    /// A cell carved twenty-four levels below the root has no region for a
    /// parent and none for a sibling, and the shards around it are cells and
    /// octants at all sorts of depths. Everyone in the map hears about the
    /// change and must ack it, plus the new owner; the same for the merge that
    /// takes the cell away again, the shard being merged included.
    ///
    /// Mutation this catches: the old rule (the direct parent's owner and the
    /// owners of sibling and grand-sibling octants), which reaches only the
    /// new shard here — the root's owner, who loses the cell, would never
    /// stage the change.
    #[test]
    fn a_deep_carve_reaches_every_shard_in_the_map() {
        let root = NodeRef::new(40, [0, 0, 0]);
        let k = root.center().cell[0];
        let octant = space_world::octree::children(root).unwrap()[0];
        let map = ShardMap::new(
            7,
            vec![
                region(40, [0, 0, 0], 1),
                region(64, [k, k, k - 1], 2),
                region(64, [k - 1, k, k], 3),
                region(41, octant.index, 4),
                // A cell inside that octant, far from the others.
                region(64, [5, 5, 5], 5),
            ],
        );
        let everyone: BTreeSet<ShardId> = (1..=5).map(shard).collect();
        let mut pub_ = EpochPublisher::new(map.clone());

        // The carve of the fourth quadrant's cell for a new shard 6.
        let cell = NodeRef::new(64, [k - 1, k, k - 1]);
        let mut with_six = everyone.clone();
        with_six.insert(shard(6));
        assert_eq!(pub_.recipients(&[(cell, shard(6))]), with_six);
        let mut next = map.clone();
        next.epoch = 8;
        next.regions.push(region(64, cell.index, 6));
        let prepare = pub_
            .propose(next.clone(), vec![(cell, shard(6))], &[], 10)
            .expect("propose");
        assert_eq!(
            prepare.to.iter().copied().collect::<BTreeSet<_>>(),
            with_six,
            "the prepare goes to all five owners and the new shard"
        );
        // It commits only when every one of them has acked — the far cell's
        // owner (5) as much as the root's.
        let mut commit = None;
        for (i, s) in prepare.to.iter().enumerate() {
            assert!(commit.is_none(), "not before the last ack");
            assert_eq!(pub_.published().epoch, 7);
            commit = pub_.ack(*s, 8, 11);
            assert_eq!(commit.is_some(), i + 1 == prepare.to.len());
        }
        let commit = commit.expect("the last ack commits");
        assert_eq!(
            commit.to.iter().copied().collect::<BTreeSet<_>>(),
            with_six,
            "and the commit goes to everyone in the new map"
        );
        assert_eq!(pub_.published(), &next);
        pub_.reset();

        // The merge of shard 3's cell back into the root's owner: the change
        // names shard 1, and everyone in the map — shard 3 too, which still
        // owns its cell until the commit — is a recipient.
        let gone = NodeRef::new(64, [k - 1, k, k]);
        assert_eq!(pub_.recipients(&[(gone, shard(1))]), with_six);
        // No change at all still reaches the whole map (a map push by epoch).
        assert_eq!(pub_.recipients(&[]), with_six);
    }

    /// The recipients are a set: an owner of several regions, or one named by
    /// several changes, is told once.
    #[test]
    fn a_recipient_is_listed_once() {
        let map = ShardMap::new(
            1,
            vec![
                region(40, [0, 0, 0], 1),
                region(64, [1, 1, 1], 1),
                region(64, [2, 2, 2], 2),
            ],
        );
        let pub_ = EpochPublisher::new(map);
        let a = NodeRef::new(64, [3, 3, 3]);
        let b = NodeRef::new(64, [4, 4, 4]);
        let recipients = pub_.recipients(&[(a, shard(9)), (b, shard(9)), (b, shard(2))]);
        assert_eq!(
            recipients.into_iter().collect::<Vec<_>>(),
            vec![shard(1), shard(2), shard(9)]
        );
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
            let _ = pub_.ack(s, 2, 10);
        }
        pub_.reset();
        // Immediately after a commit at tick 10, a new propose is gated until
        // the minimum interval has passed.
        assert!(!pub_.can_propose(10, 90), "no commit inside the interval");
        assert!(pub_.can_propose(100, 90), "the interval has passed");
    }

    #[test]
    fn an_ack_for_a_different_epoch_cannot_commit_the_staged_epoch() {
        let mut pub_ = EpochPublisher::new(map_with_two());
        let mut next = pub_.published().clone();
        next.epoch = 2;
        let child = space_world::octree::children(NodeRef::new(40, [0, 0, 0])).unwrap()[5];
        next.regions.push(region(41, child.index, 3));
        let prepare = pub_
            .propose(next, vec![(child, shard(3))], &[(shard(1), 10)], 10)
            .expect("propose");
        assert!(prepare.to.len() >= 2, "test needs two awaited shards");
        let mut shards = prepare.to.clone();
        let stale = shards.remove(0);

        // Every other shard acks the right epoch; only `stale` is outstanding.
        for s in shards {
            assert!(pub_.ack(s, 2, 11).is_none(), "not the last ack yet");
        }
        // `stale` acks epoch N-1: ignored, still preparing, still awaited.
        assert!(
            pub_.ack(stale, 1, 12).is_none(),
            "a mismatched ack must not commit"
        );
        match pub_.phase() {
            Phase::Preparing { awaiting, .. } => {
                assert!(awaiting.contains(&stale), "the shard is still awaited");
            }
            other => panic!("expected Preparing, got {other:?}"),
        }
        assert_eq!(pub_.published().epoch, 1, "nothing was published");

        // The same shard acking the staged epoch commits it.
        let commit = pub_.ack(stale, 2, 13).expect("the matching ack commits");
        assert!(matches!(commit.message, Message::EpochCommit(_)));
        assert_eq!(pub_.published().epoch, 2);
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
