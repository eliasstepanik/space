//! [`Orchestrator`]: the tie-in point (plan §3.1, §3.6).
//!
//! The orchestrator is a **synchronous state machine** driven by a tick, exactly
//! like [`space_server::shard::Shard`]: every decision — id allocation, topology,
//! epoch, autoscale — is a pure call into [`crate::control`] / [`crate::topology`]
//! / [`crate::epoch`] / [`crate::autoscale`], and the async layer only moves
//! messages. That split is what makes the whole control loop testable without a
//! wall clock: a test calls [`Orchestrator::tick`] and injects reports/acks
//! directly, so dwell timers advance by ticks, not by sleeping (plan §3.6, H14).
//!
//! # The loop (plan §3.6)
//!
//! ```text
//! loop { select! {
//!     report = load_reports()      => orch.handle_load_report(report),
//!     ack    = control.acks()      => orch.handle_epoch_ack(shard, epoch),
//!     hello  = control.hellos()    => orch.handle_shard_hello(hello),
//!     _      = clock.tick()        => { orch.tick(); drain outbox; launcher.poll(); }
//! }}
//! ```
//!
//! [`Orchestrator::tick`] is the synchronous heart of that loop: it advances the
//! orchestrator's own clock, runs the epoch deadline, feeds the autoscaler the
//! latest report per shard, and on a `Split`/`Merge` computes a new topology,
//! bumps the epoch through [`crate::epoch::EpochPublisher`] and stages the
//! launcher work ([`Orchestrator::take_pending_spawn`] /
//! [`Orchestrator::take_pending_kill`]) for the async layer to execute. A
//! split's child is staged for launch when the split is proposed (it has to
//! register and ack before the epoch can commit); a
//! merged shard is staged for killing only when the merge **commits** — until
//! then it still owns its region — and a child whose split aborts is staged for
//! killing at the abort. The async [`Orchestrator::run`] wires those pieces
//! together over real QUIC; the tests drive the synchronous half directly.
//!
//! # What the orchestrator believes about load
//!
//! Only a [`LoadReport`] from a shard that owns a region in the published map is
//! kept, and only while it is fresh
//! ([`crate::config::REPORT_EXPIRY_TICKS`]): a shard that was merged away,
//! killed, or simply stopped reporting is forgotten — by the autoscaler too —
//! instead of being re-split or re-merged for ever on its last words.

use std::collections::BTreeMap;
use std::sync::Arc;

use space_protocol::control::{AdminCommand, AdminReply, LoadReport};
use space_protocol::shard::{EpochAck, ShardId, ShardMap};
use space_protocol::{Message, ShardHello, ShardWelcome};

use crate::autoscale::ScaleAction;
use crate::config::{OrchestratorConfig, MIN_EPOCH_INTERVAL_TICKS, REPORT_EXPIRY_TICKS};
use crate::control::ControlPlane;
use crate::epoch::Outbound;
use crate::launcher::{Launcher, Poll, SpawnSpec};
use crate::topology;

/// The orchestrator: control plane + the launcher seam.
///
/// `L` is generic rather than `dyn` because [`Launcher`] uses associated-return-
/// position `impl Future` (not dyn-compatible) — which is also what lets the
/// acceptance tests inject a `MockLauncher` as a plain struct.
#[derive(Debug)]
pub struct Orchestrator<L: Launcher> {
    /// The pure control state (ids, epoch publisher, autoscaler).
    pub control: ControlPlane,
    /// The process layer. `Arc` so a task can hold one while the loop keeps it.
    pub launcher: Arc<L>,
    /// The orchestrator's own clock, in ticks. Advanced only by [`Self::tick`].
    tick: u64,
    /// The most recent [`LoadReport`] per shard, fed to the autoscaler each tick.
    latest: BTreeMap<ShardId, LoadReport>,
    /// The most recent tick each shard reported at (drives `activation_tick`).
    shard_ticks: BTreeMap<ShardId, u64>,
    /// A child id whose shard the loop must spawn (staged by a split).
    pending_spawn: Vec<SpawnSpec>,
    /// A shard the loop must kill (staged by a committed merge, an aborted
    /// split's child, or the admin verb).
    pending_kill: Vec<ShardId>,
    /// The shard the merge **in flight** removes. It is killed when that epoch
    /// commits and left alone when it aborts: until the commit it still owns its
    /// region, so killing it any earlier leaves the map naming a dead owner.
    pending_merge: Option<ShardId>,
    /// The child the split **in flight** launched. If that epoch aborts the
    /// child owns nothing and is killed, instead of running on unowned.
    pending_split: Option<ShardId>,
    /// Messages the async layer must deliver, keyed by destination shard.
    outbox: Vec<(ShardId, Message)>,
    /// The endpoints learned from each shard's `ShardHello` (for spawn specs).
    endpoints: BTreeMap<ShardId, std::net::SocketAddr>,
}

impl<L: Launcher> Orchestrator<L> {
    /// Assemble an orchestrator from its parts.
    #[must_use]
    pub fn new(control: ControlPlane, launcher: Arc<L>) -> Self {
        Self {
            control,
            launcher,
            tick: 0,
            latest: BTreeMap::new(),
            shard_ticks: BTreeMap::new(),
            pending_spawn: Vec::new(),
            pending_kill: Vec::new(),
            pending_merge: None,
            pending_split: None,
            outbox: Vec::new(),
            endpoints: BTreeMap::new(),
        }
    }

    /// The config the orchestrator was built with.
    #[must_use]
    pub fn config(&self) -> &OrchestratorConfig {
        &self.control.config
    }

    /// The orchestrator's current tick.
    #[must_use]
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// The published map (the only map shards may observe).
    #[must_use]
    pub fn map(&self) -> &ShardMap {
        self.control.map()
    }

    /// Drain the messages the async layer must send, in order.
    pub fn drain_outbox(&mut self) -> Vec<(ShardId, Message)> {
        std::mem::take(&mut self.outbox)
    }

    /// Drain the spawn specs the loop staged this tick (a split's child).
    pub fn take_pending_spawn(&mut self) -> Vec<SpawnSpec> {
        std::mem::take(&mut self.pending_spawn)
    }

    /// Drain the shard ids the loop staged for killing (a committed merge's
    /// child, the child of a split that aborted, an admin kill).
    pub fn take_pending_kill(&mut self) -> Vec<ShardId> {
        std::mem::take(&mut self.pending_kill)
    }

    /// The shard the merge in flight will remove once its epoch commits.
    #[must_use]
    pub fn pending_merge(&self) -> Option<ShardId> {
        self.pending_merge
    }

    /// The latest load report held for `shard`, if it is still fresh.
    #[must_use]
    pub fn load_report(&self, shard: ShardId) -> Option<&LoadReport> {
        self.latest.get(&shard)
    }

    // ------------------------------------------------------------- ingestion

    /// Handle a `ShardHello`: allocate/confirm the id and region, remember the
    /// endpoint, and return the `ShardWelcome` to send back.
    ///
    /// This is the orchestrator-owns-allocation path (closes Phase-2 MAJOR 3):
    /// the id comes from [`crate::ids`], never from argv.
    pub fn handle_shard_hello(&mut self, hello: &ShardHello) -> Result<ShardWelcome, String> {
        let welcome = self.control.handle_shard_hello(hello)?;
        self.endpoints.insert(welcome.shard, hello.endpoint);
        Ok(welcome)
    }

    /// Ingest a `LoadReport`: remember it as the shard's latest and record the
    /// orchestrator tick it arrived at (its freshness, and the observer tick an
    /// epoch's `activation_tick` is derived from).
    ///
    /// A report naming a shard that owns no region in the published map is
    /// dropped: it is from a shard that was merged away or killed (its last
    /// reports can still be in flight), or from nobody the cluster knows. Kept,
    /// it would be replayed to the autoscaler every tick and acted on.
    pub fn handle_load_report(&mut self, report: &LoadReport) {
        if !self.owns_a_region(report.shard) {
            return;
        }
        self.latest.insert(report.shard, *report);
        self.shard_ticks.insert(report.shard, self.tick);
    }

    /// Whether `shard` owns a region in the published map.
    fn owns_a_region(&self, shard: ShardId) -> bool {
        self.control.map().regions.iter().any(|r| r.shard == shard)
    }

    /// Stop believing anything about `shard`'s load: drop its report and its
    /// autoscaler state. Its endpoint is kept — it may still own a region.
    fn forget_reports(&mut self, shard: ShardId) {
        self.latest.remove(&shard);
        self.shard_ticks.remove(&shard);
        self.control.autoscaler.forget(shard);
    }

    /// `shard` has left the cluster (merged away, or a child whose split
    /// aborted): forget its load and where it was reachable.
    fn forget_shard(&mut self, shard: ShardId) {
        self.forget_reports(shard);
        self.endpoints.remove(&shard);
    }

    /// Drop every report that has not been refreshed for
    /// [`REPORT_EXPIRY_TICKS`]. A shard reports every
    /// [`crate::config::LOAD_REPORT_INTERVAL_TICKS`]; one that has been silent
    /// for several beats is not "still as loaded as it last said" — it is gone
    /// or stuck, and neither is a reason to split or merge it.
    fn expire_stale_reports(&mut self) {
        let now = self.tick;
        let stale: Vec<ShardId> = self
            .shard_ticks
            .iter()
            .filter(|(_, seen)| now.saturating_sub(**seen) > REPORT_EXPIRY_TICKS)
            .map(|(shard, _)| *shard)
            .collect();
        for shard in stale {
            self.forget_reports(shard);
        }
    }

    /// Record an `EpochAck` from `shard`. When the last required ack lands the
    /// publisher commits; the commit message is queued for every shard and the
    /// autoscaler is told an action completed (cooldown).
    pub fn handle_epoch_ack(&mut self, shard: ShardId, epoch: u64) {
        if let Some(commit) = self.control.publisher.ack(shard, epoch, self.tick) {
            // The epoch in flight settled as a commit: a split's child is now a
            // region owner like any other, and a merge's child owns nothing.
            self.pending_split = None;
            let merged = self.pending_merge.take();
            self.queue_outbound(&commit);
            // Then push the committed map with every known shard's real endpoint
            // (plan §3.3, F9). The commit names ownership only; this is what lets
            // an existing shard dial a child it has never heard of, and what makes
            // the `Redirect` it hands a client point at the child's real address.
            // It rides the same per-shard FIFO as the commit, so a shard always
            // sees the commit first.
            self.control.publisher.set_endpoints(&self.endpoints);
            let push = Outbound {
                to: commit.to.clone(),
                message: Message::DirectoryReply(space_protocol::DirectoryReply {
                    shard_map: self.control.publisher.published().clone(),
                    tick: self.tick,
                }),
            };
            self.queue_outbound(&push);
            if let Some(gone) = merged {
                // The merged shard is not in the new map, so the fan-out above
                // skipped it. Tell it too (it had the prepare staged; now it
                // learns it owns nothing), and only *then* stage its kill: the
                // loop delivers the outbox before it runs the launcher work.
                self.outbox.push((gone, commit.message.clone()));
                self.outbox.push((gone, push.message.clone()));
                self.pending_kill.push(gone);
                self.forget_shard(gone);
            }
            // The epoch settled: the publisher returns to Idle so the next
            // change can be proposed after the minimum interval.
            self.control.publisher.reset();
        }
    }

    /// Record an `EpochAck` carried on the wire (the shard fills `shard`).
    pub fn handle_epoch_ack_message(&mut self, ack: &EpochAck) {
        self.handle_epoch_ack(ack.shard, ack.epoch);
    }

    /// Handle an `AdminCommand`, returning the reply to send.
    ///
    /// `ForceSplit`/`ForceMerge`/`KillShard` are driven through the same code
    /// paths the autoscaler uses, so an admin verb and an autoscale decision go
    /// through the same plan, epoch and launcher steps. A verb that cannot be
    /// carried out (an epoch is in flight, the region cannot split, the shard
    /// still has players) is answered `Refused` with the reason — it used to be
    /// `Accepted` and silently dropped.
    ///
    /// `ForceSplit` names no octant, so it carves the first free one counting
    /// from octant 0 (the autoscaler starts at the region's centre octant
    /// instead, because that is where joining ships spawn).
    pub fn handle_admin(&mut self, command: &AdminCommand) -> AdminReply {
        let outcome = match command {
            AdminCommand::ForceSplit(shard) => self.begin_split(*shard, Some(0)),
            AdminCommand::ForceMerge(shard) => self.begin_merge(*shard),
            AdminCommand::KillShard(shard) => {
                self.pending_kill.push(*shard);
                // A killed shard reports nothing more; its last report must not
                // go on driving the autoscaler.
                self.forget_reports(*shard);
                Ok(())
            }
            other => return self.control.handle_admin(other),
        };
        match outcome {
            Ok(()) => AdminReply::Accepted {
                command: command.clone(),
            },
            Err(reason) => AdminReply::Refused {
                command: command.clone(),
                reason,
            },
        }
    }

    // ----------------------------------------------------------------- tick

    /// Advance the orchestrator's clock by one tick and run the control beat:
    /// the epoch deadline, then the autoscaler over the latest report per shard.
    ///
    /// This is the synchronous heart of the loop (plan §3.6). It never touches a
    /// socket or a wall clock, so a test drives it with a `for` loop.
    pub fn tick(&mut self) {
        self.tick += 1;
        // 1. Epoch deadline: an un-acked prepare aborts and the map is untouched.
        if let Some(abort) = self.control.publisher.timeout(self.tick) {
            self.queue_outbound(&abort);
            self.control.publisher.reset();
            // Nothing the aborted epoch planned happened. A merge's shard still
            // owns its region, so it is *not* killed; a split's child owns
            // nothing, so it is — after the abort above has been delivered.
            self.pending_merge = None;
            if let Some(child) = self.pending_split.take() {
                self.pending_kill.push(child);
                self.forget_shard(child);
            }
        }
        // 2. Autoscaler: feed the latest *fresh* report per shard, get at most
        // one action.
        self.expire_stale_reports();
        let reports: Vec<LoadReport> = self.latest.values().copied().collect();
        if reports.is_empty() {
            return;
        }
        if let Some(action) = self.control.autoscaler.on_reports(&reports, self.tick) {
            let outcome = match action {
                // The autoscaler's octant is a placeholder (it sees load, not
                // positions); the split starts at the region's centre octant.
                ScaleAction::Split { shard, .. } => self.begin_split(shard, None),
                ScaleAction::Merge { shard } => self.begin_merge(shard),
            };
            if let Err(why) = outcome {
                eprintln!("space-orchestrator: {action} was not carried out: {why}");
            }
        }
    }

    // ------------------------------------------------- autoscale -> topology

    /// Turn a split decision into a topology plan + epoch proposal + a staged
    /// child spawn.
    ///
    /// `first_octant` is where the search for an octant to carve starts: `None`
    /// means the region's centre octant (where a shard spawns joining ships, so
    /// the first split of a fresh region moves the players it was meant to
    /// relieve), `Some(n)` an explicit preference. Either way the octant carved
    /// is the first one, counting round from there, whose plan validates — an
    /// octant already carved out for another shard is skipped, so a region can
    /// be split again and again until none is left
    /// ([`topology::pick_split_octant`]).
    ///
    /// The child's id is allocated only once a valid plan exists: a split that
    /// cannot happen (region too small, nothing left to carve) burns no id and
    /// writes nothing to the counter file.
    ///
    /// # Errors
    ///
    /// Why the split was not started. Apart from "an epoch is in flight" the
    /// shard is put on cooldown first (a shard that cannot be split is not
    /// helped by retrying every tick); the caller logs or reports the reason.
    fn begin_split(&mut self, shard: ShardId, first_octant: Option<usize>) -> Result<(), String> {
        if !self
            .control
            .publisher
            .can_propose(self.tick, MIN_EPOCH_INTERVAL_TICKS)
        {
            return Err("an epoch change is in flight or was committed too recently".into());
        }
        let first = first_octant.unwrap_or_else(|| {
            topology::region_of(self.control.map(), shard)
                .map_or(0, |region| topology::centre_octant(region.node))
        });
        let octant = match topology::pick_split_octant(self.control.map(), shard, first) {
            Ok(octant) => octant,
            Err(err) => {
                self.control.autoscaler.note_action(shard, self.tick);
                return Err(err.to_string());
            }
        };
        let new_id = match self.control.ids.allocate() {
            Ok(id) => id,
            Err(err) => {
                // The id space is exhausted or the counter cannot be written:
                // nothing can be spawned. Cool down so we do not retry every tick.
                self.control.autoscaler.note_action(shard, self.tick);
                return Err(err.to_string());
            }
        };
        let plan = match topology::split_plan(self.control.map(), shard, octant, new_id) {
            Ok(plan) => plan,
            Err(err) => {
                self.control.autoscaler.note_action(shard, self.tick);
                return Err(err.to_string());
            }
        };
        let observer_ticks: Vec<(ShardId, u64)> =
            self.shard_ticks.iter().map(|(s, t)| (*s, *t)).collect();
        let Some(prepare) = self.control.publisher.propose(
            plan.map.clone(),
            vec![plan.change],
            &observer_ticks,
            self.tick,
        ) else {
            // Lost the race to an in-flight epoch (can_propose said Idle but the
            // publisher disagreed); cool down so we do not spin.
            self.control.autoscaler.note_action(shard, self.tick);
            return Err("the publisher refused the proposal".into());
        };
        self.queue_outbound(&prepare);
        self.control.autoscaler.note_action(shard, self.tick);
        // Stage the child spawn: the async layer runs the launcher and the child
        // then registers with its own ShardHello (and is handed the prepare it
        // still owes an ack for).
        let parent_endpoint = self
            .endpoints
            .get(&shard)
            .copied()
            .unwrap_or(plan.map.regions[0].endpoint);
        self.pending_spawn.push(SpawnSpec {
            shard: new_id,
            region: space_protocol::shard::ShardRegion::new(plan.child, new_id, parent_endpoint),
            cluster: self.control.config.cluster.clone(),
            control_addr: self.control.config.control_addr(),
            host_port: 0,
        });
        self.pending_split = Some(new_id);
        Ok(())
    }

    /// Turn a merge decision into a topology plan + epoch proposal. The merged
    /// shard is killed when (and only if) that epoch commits.
    ///
    /// **Only a shard with no players is merged away.** A merge removes the
    /// shard's region from the map and then the shard itself; there is no step
    /// in which its players are handed to the parent (once the merge commits
    /// the parent no longer even knows the shard, so it would refuse the
    /// offers). "Cold" to the autoscaler means a load at or below the merge
    /// threshold — up to a third of a full shard — so acting on that alone
    /// dropped every one of those players. The shard's own fresh report has to
    /// say `players == 0`; a shard that has not reported is not known to be
    /// empty and is left alone.
    ///
    /// # Errors
    ///
    /// Why the merge was not started (see [`Self::begin_split`] for the
    /// cooldown rule).
    fn begin_merge(&mut self, shard: ShardId) -> Result<(), String> {
        if !self
            .control
            .publisher
            .can_propose(self.tick, MIN_EPOCH_INTERVAL_TICKS)
        {
            return Err("an epoch change is in flight or was committed too recently".into());
        }
        let players = self.latest.get(&shard).map(|report| report.players);
        if players != Some(0) {
            self.control.autoscaler.note_action(shard, self.tick);
            return Err(match players {
                Some(n) => {
                    format!("{shard} still has {n} player(s); only an empty shard is merged")
                }
                None => {
                    format!("no fresh load report from {shard}, so it is not known to be empty")
                }
            });
        }
        let plan = match topology::merge_plan(self.control.map(), shard) {
            Ok(plan) => plan,
            Err(err) => {
                self.control.autoscaler.note_action(shard, self.tick);
                return Err(err.to_string());
            }
        };
        let observer_ticks: Vec<(ShardId, u64)> =
            self.shard_ticks.iter().map(|(s, t)| (*s, *t)).collect();
        let Some(prepare) = self.control.publisher.propose(
            plan.map.clone(),
            vec![plan.change],
            &observer_ticks,
            self.tick,
        ) else {
            self.control.autoscaler.note_action(shard, self.tick);
            return Err("the publisher refused the proposal".into());
        };
        self.queue_outbound(&prepare);
        self.control.autoscaler.note_action(shard, self.tick);
        // The merged child's region folds back into the parent — when the epoch
        // commits. Until then the shard owns its region and keeps running; it
        // is staged for killing by the commit (`handle_epoch_ack`), and left
        // alone if the epoch aborts (`tick`).
        self.pending_merge = Some(shard);
        Ok(())
    }

    /// Fan an [`Outbound`] out to its recipients, tagging each with the shard it
    /// goes to so the async layer can route it.
    fn queue_outbound(&mut self, outbound: &Outbound) {
        for to in &outbound.to {
            self.outbox.push((*to, outbound.message.clone()));
        }
    }

    // ------------------------------------------------------------- async loop

    /// Bind the control endpoint on the configured address. Split from
    /// [`Self::run_with`] so a caller (a test binding port 0) can read the real
    /// address from [`crate::control::ControlServer::local_addr`] before the loop
    /// starts.
    ///
    /// # Errors
    ///
    /// [`std::io::Error`] if the control endpoint cannot be bound.
    pub fn bind(&self) -> Result<Arc<crate::control::ControlServer>, std::io::Error> {
        Ok(Arc::new(crate::control::ControlServer::bind(
            self.control.config.bind,
        )?))
    }

    /// A liveness probe over the launcher (H15's launcher half): report whether
    /// `shard` is still running. Exposed so the loop (and tests) can reconcile.
    ///
    /// # Errors
    ///
    /// A launcher-specific failure.
    pub async fn poll(&self, shard: ShardId) -> Result<Poll, crate::launcher::LaunchError> {
        self.launcher.poll(shard).await
    }
}

/// The async half. `L: 'static` because a launch runs in its own task (see
/// [`Orchestrator::run_with`]).
impl<L: Launcher + 'static> Orchestrator<L> {
    /// The event loop (plan §3.6): bind the control endpoint, accept shard
    /// connections, and on each tick beat run the synchronous [`Self::tick`],
    /// deliver the outbox, and execute the staged launcher work.
    ///
    /// The loop only moves messages and drives the launcher; every decision is
    /// made by the synchronous methods above. The clock is a wall-clock ticker
    /// here (the production beat); tests drive [`Self::tick`] directly instead.
    ///
    /// # Errors
    ///
    /// [`std::io::Error`] if the control endpoint cannot be bound.
    pub async fn run(&mut self) -> Result<(), std::io::Error> {
        let server = self.bind()?;
        self.run_with(server).await
    }

    /// Run the event loop over an already-bound `server` until it closes.
    ///
    /// Nothing in the loop waits on a peer: messages are *offered* to a shard's
    /// queue ([`crate::control::ControlServer::send_to`]), and a launch runs in
    /// its own task. The second matters for more than latency — a launched
    /// shard registers by sending a `ShardHello` that only **this loop** can
    /// answer, so awaiting the launch here would wait on ourselves.
    ///
    /// # Errors
    ///
    /// Currently infallible; the `Result` keeps [`Self::run`]'s signature.
    pub async fn run_with(
        &mut self,
        server: Arc<crate::control::ControlServer>,
    ) -> Result<(), std::io::Error> {
        use crate::control::Inbound;
        use std::time::Duration;

        let accept = Arc::clone(&server).spawn_accept();
        let mut clock = tokio::time::interval(Duration::from_secs_f64(
            1.0 / f64::from(crate::config::TICK_HZ as u32),
        ));
        clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                inbound = server.next_inbound() => {
                    match inbound {
                        Some(Inbound::Hello { hello, reply }) => {
                            self.answer_hello(&server, &hello, &reply).await;
                        }
                        Some(Inbound::Report(report)) => self.handle_load_report(&report),
                        Some(Inbound::Ack(ack)) => self.handle_epoch_ack_message(&ack),
                        None => break, // every sender gone
                    }
                }
                _ = clock.tick() => self.beat(&server).await,
            }
        }
        accept.abort();
        server.close();
        Ok(())
    }

    /// Answer one `ShardHello` on the connection it arrived on.
    ///
    /// `reply` is that connection's own outbound queue, created empty a moment
    /// ago, so offering the answer to it never waits and never fails for lack
    /// of room.
    async fn answer_hello(
        &mut self,
        server: &crate::control::ControlServer,
        hello: &ShardHello,
        reply: &tokio::sync::mpsc::Sender<Message>,
    ) {
        match self.handle_shard_hello(hello) {
            Ok(welcome) => {
                let shard = welcome.shard;
                server.register(shard, reply.clone()).await;
                let _ = reply.try_send(Message::ShardWelcome(welcome));
                // A split names its child as a prepare recipient before the
                // child exists, so the prepare published at propose time never
                // reached it. Now that it has registered, hand it the prepare it
                // still owes an ack for (behind the welcome, on the same FIFO) —
                // without this no split could ever collect its last ack.
                if let Some(prepare) = self.control.publisher.outstanding_prepare_for(shard) {
                    let _ = reply.try_send(prepare);
                }
            }
            Err(reason) => {
                // Say why, on the hello connection, with the existing `Reject`
                // message (no protocol change): the shard fails fast with the
                // reason instead of timing out.
                eprintln!(
                    "space-orchestrator: rejected hello from {}: {reason}",
                    hello.endpoint
                );
                let _ = reply.try_send(Message::Reject(space_protocol::Reject {
                    reason: space_protocol::RejectReason::Other(reason),
                }));
            }
        }
    }

    /// One tick beat: the synchronous [`Self::tick`], then its consequences in
    /// order — epoch messages out, launches started, kills carried out.
    async fn beat(&mut self, server: &crate::control::ControlServer) {
        self.tick();
        // Deliver the epoch messages the tick (and the inbound handlers since
        // the last beat) produced. Before the kills below: a shard that is about
        // to be killed is told the outcome of its epoch first.
        for (to, message) in self.drain_outbox() {
            server.send_to(to, message).await;
        }
        // Launch the children a split staged, each in its own task (see
        // `run_with`). The child boots, registers with its own ShardHello, is
        // handed the outstanding prepare and acks it; a launch that fails
        // leaves the prepare un-acked, so the epoch aborts at its deadline.
        for mut spec in self.take_pending_spawn() {
            if spec.control_addr.port() == 0 {
                // The control endpoint was bound on port 0 ("any"): the child
                // has to dial the port that was actually bound.
                if let Ok(bound) = server.local_addr() {
                    spec.control_addr.set_port(bound.port());
                }
            }
            let launcher = Arc::clone(&self.launcher);
            tokio::spawn(async move {
                let shard = spec.shard;
                if let Err(err) = launcher.spawn(spec).await {
                    eprintln!("space-orchestrator: spawn {shard} failed: {err}");
                }
            });
        }
        // Kill the shards a committed merge, an aborted split or the admin verb
        // staged.
        for shard in self.take_pending_kill() {
            if let Err(err) = self.launcher.kill(shard).await {
                eprintln!("space-orchestrator: kill {shard} failed: {err}");
            }
            server.unregister(shard).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::ControlPlane;
    use crate::epoch::EpochPublisher;
    use crate::ids::{MemoryCounterSink, PersistentShardIdAllocator};
    use crate::launcher::{LaunchError, Poll, ShardHandle, SpawnSpec};
    use space_protocol::shard::{ShardId, ShardMap, ShardRegion};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn addr(n: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), n)
    }

    /// A launcher that records calls and never touches a process.
    #[derive(Debug, Default)]
    struct MockLauncher {
        spawned: std::sync::Mutex<Vec<ShardId>>,
    }

    impl Launcher for MockLauncher {
        // `async fn` implementing an AFIT method: rustc checks the generated
        // future against the trait's `+ Send` bound.
        async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
            // The lock is taken and released with no await point in between, so
            // the future stays `Send`.
            self.spawned.lock().unwrap().push(spec.shard);
            Ok(ShardHandle {
                shard: spec.shard,
                endpoint: spec.region.endpoint,
                process: "mock".into(),
            })
        }

        async fn kill(&self, _shard: ShardId) -> Result<(), LaunchError> {
            Ok(())
        }

        async fn poll(&self, _shard: ShardId) -> Result<Poll, LaunchError> {
            Ok(Poll::Alive)
        }
    }

    /// Drive a future that is ready without I/O to completion. The mock
    /// launcher's futures never yield, so a no-op waker suffices — this keeps
    /// the test dependency-free (no tokio/futures-lite in the offline cache) and
    /// safe (an `Arc`-backed `Wake`, so no `unsafe` in a `forbid(unsafe_code)`
    /// crate).
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        use std::task::{Context, Poll, Waker};
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut fut = std::pin::pin!(fut);
        loop {
            match std::future::Future::poll(fut.as_mut(), &mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::hint::spin_loop(),
            }
        }
    }

    fn orchestrator() -> Orchestrator<MockLauncher> {
        orchestrator_with_first_id(1)
    }

    fn orchestrator_with_first_id(first_id: u64) -> Orchestrator<MockLauncher> {
        let config = OrchestratorConfig::default();
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(
                config.root_node(),
                ShardId::new(1),
                addr(7_001),
            )],
        );
        let ids =
            PersistentShardIdAllocator::with_sink(first_id, Box::new(MemoryCounterSink::default()));
        let control = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
        Orchestrator::new(control, std::sync::Arc::new(MockLauncher::default()))
    }

    /// An orchestrator over an explicit map and id sink.
    fn orchestrator_over(
        config: OrchestratorConfig,
        regions: Vec<ShardRegion>,
        ids: PersistentShardIdAllocator,
    ) -> Orchestrator<MockLauncher> {
        let map = ShardMap::new(1, regions);
        let control = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
        Orchestrator::new(control, std::sync::Arc::new(MockLauncher::default()))
    }

    /// Root (depth 40) owned by shard 1 with octant 0 carved out for shard 2.
    fn orchestrator_with_a_child() -> Orchestrator<MockLauncher> {
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        let child = space_world::octree::children(root).expect("root splits")[0];
        let ids =
            PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
        orchestrator_over(
            config,
            vec![
                ShardRegion::new(root, ShardId::new(1), addr(7_001)),
                ShardRegion::new(child, ShardId::new(2), addr(7_002)),
            ],
            ids,
        )
    }

    fn report(shard: u32, players: u32) -> LoadReport {
        LoadReport {
            shard: ShardId::new(shard),
            players,
            tick_p95_ms: 1.0,
            load: 0.0,
        }
    }

    /// The `(recipient, epoch)` of every `EpochPrepare` in `outbox`.
    fn prepares(outbox: &[(ShardId, Message)]) -> Vec<(ShardId, u64)> {
        outbox
            .iter()
            .filter_map(|(to, m)| match m {
                Message::EpochPrepare(p) => Some((*to, p.epoch)),
                _ => None,
            })
            .collect()
    }

    /// Drain the outbox and ack the prepare in it on behalf of every recipient.
    /// Returns how many prepares were acked.
    fn ack_the_prepare(orch: &mut Orchestrator<MockLauncher>) -> usize {
        let recipients = prepares(&orch.drain_outbox());
        for (shard, epoch) in &recipients {
            orch.handle_epoch_ack(*shard, *epoch);
        }
        recipients.len()
    }

    fn accepted(reply: &AdminReply) -> bool {
        matches!(reply, AdminReply::Accepted { .. })
    }

    fn refusal(reply: AdminReply) -> String {
        match reply {
            AdminReply::Refused { reason, .. } => reason,
            other => panic!("expected Refused, got {other:?}"),
        }
    }

    // ------------------------------------------------------------- finding 7

    /// Mutation this catches: always carving the same octant. The second split
    /// of a region then proposes a node the map already has, fails validation,
    /// and the hot shard can never be relieved again.
    #[test]
    fn a_region_can_be_split_again_and_each_split_carves_a_new_octant() {
        let mut orch = orchestrator_with_first_id(100);
        let root = orch.config().root_node();
        let kids = space_world::octree::children(root).expect("root splits");
        let mut carved = Vec::new();
        for round in 0..3u64 {
            let reply = orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)));
            assert!(accepted(&reply), "split {round} is accepted: {reply:?}");
            let spec = orch.take_pending_spawn().pop().expect("a child is staged");
            assert!(ack_the_prepare(&mut orch) > 0, "a prepare went out");
            assert_eq!(orch.map().epoch, 2 + round, "split {round} committed");
            carved.push(spec.region.node);
            for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
                orch.tick();
            }
            let _ = orch.drain_outbox();
        }
        // The admin verb counts from octant 0: 0, then 1, then 2.
        assert_eq!(carved, vec![kids[0], kids[1], kids[2]]);
        assert_eq!(orch.map().regions.len(), 4);
        crate::topology::validate(orch.map()).expect("the map stays valid");
    }

    /// The autoscaler's own split starts at the region's centre octant — where
    /// joining ships spawn — not at octant 0, the far corner.
    #[test]
    fn an_autoscaler_split_carves_the_centre_octant_first() {
        let mut orch = orchestrator_with_first_id(100);
        let root = orch.config().root_node();
        let mut staged = None;
        for _ in 0..=(crate::autoscale::SPLIT_DWELL_TICKS + 1) {
            orch.handle_load_report(&report(1, 90));
            orch.tick();
            if let Some(spec) = orch.take_pending_spawn().pop() {
                staged = Some(spec);
                break;
            }
        }
        let spec = staged.expect("a shard hot for the dwell is split");
        let kids = space_world::octree::children(root).expect("root splits");
        assert_eq!(
            spec.region.node, kids[7],
            "the carved octant is the one holding the region centre"
        );
        assert!(space_world::octree::contains(
            spec.region.node,
            root.center()
        ));
    }

    /// Mutation this catches: allocating the child's id before the plan is
    /// known to work. Every refused split then burned an id (and a counter
    /// write) per cooldown, for as long as the shard stayed hot.
    #[test]
    fn a_split_that_cannot_happen_burns_no_id_and_says_why() {
        // A depth-55 root is below the split floor.
        let config = OrchestratorConfig {
            root_depth: 55,
            ..OrchestratorConfig::default()
        };
        let root = config.root_node();
        let ids =
            PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
        let mut orch = orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        );
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("split floor"), "{why}");
        assert_eq!(orch.control.ids.peek(), 100, "no id was allocated");
        assert!(orch.take_pending_spawn().is_empty());
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");

        // The autoscaler path refuses the same way: hot for several dwells, and
        // still not a single id gone.
        for _ in 0..(4 * crate::autoscale::SCALE_COOLDOWN_TICKS) {
            orch.handle_load_report(&report(1, 95));
            orch.tick();
        }
        assert_eq!(orch.control.ids.peek(), 100, "a refused split burns no id");
        assert_eq!(orch.map().epoch, 1);

        // A shard the map does not know is refused too, again without an id.
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(9))));
        assert!(why.contains("no region owned"), "{why}");
        assert_eq!(orch.control.ids.peek(), 100);
    }

    // ------------------------------------------------------------ finding 11

    #[derive(Debug)]
    struct BrokenSink;

    impl crate::ids::CounterSink for BrokenSink {
        fn persist(&mut self, _next: u64) -> std::io::Result<()> {
            Err(std::io::Error::other("disk full"))
        }
    }

    /// Mutation this catches: the `expect` on the counter write. A split with a
    /// full disk took the whole orchestrator down; it must be a refused split.
    #[test]
    fn a_split_with_an_unwritable_counter_is_refused_not_a_panic() {
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        let ids = PersistentShardIdAllocator::with_sink(100, Box::new(BrokenSink));
        let mut orch = orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        );
        let before = orch.map().clone();
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("could not be persisted"), "{why}");
        assert!(orch.take_pending_spawn().is_empty(), "nothing is launched");
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");
        assert_eq!(orch.map(), &before);
        // The same through the autoscaler: it logs and carries on ticking.
        for _ in 0..=(crate::autoscale::SPLIT_DWELL_TICKS + 1) {
            orch.handle_load_report(&report(1, 90));
            orch.tick();
        }
        assert!(orch.take_pending_spawn().is_empty());
        assert_eq!(orch.map(), &before);
    }

    // ------------------------------------------------------------- finding 6

    /// Mutation this catches: keeping a report from a shard the published map
    /// does not name. It was replayed to the autoscaler every tick and acted on.
    #[test]
    fn a_report_from_a_shard_outside_the_map_is_ignored() {
        let mut orch = orchestrator_with_first_id(100);
        for _ in 0..(2 * crate::autoscale::SPLIT_DWELL_TICKS) {
            orch.handle_load_report(&report(77, 95));
            orch.tick();
        }
        assert!(orch.load_report(ShardId::new(77)).is_none());
        assert!(!orch.control.autoscaler.tracks(ShardId::new(77)));
        assert!(
            orch.drain_outbox().is_empty(),
            "nothing was proposed for it"
        );
        assert_eq!(orch.control.ids.peek(), 100, "and no id was spent on it");
    }

    /// Mutation this catches: replaying a shard's last report for ever. A shard
    /// that stopped reporting two thirds of the way through its dwell was split
    /// anyway, a hundred ticks after it went silent.
    #[test]
    fn a_stale_report_expires_and_never_completes_a_dwell() {
        let mut orch = orchestrator_with_first_id(100);
        let dwell = crate::autoscale::SPLIT_DWELL_TICKS;
        for _ in 0..(dwell * 2 / 3) {
            orch.handle_load_report(&report(1, 95));
            orch.tick();
        }
        assert!(orch.load_report(ShardId::new(1)).is_some());
        assert!(orch.control.autoscaler.tracks(ShardId::new(1)));
        // The shard goes silent.
        for _ in 0..(2 * dwell) {
            orch.tick();
        }
        assert!(
            orch.load_report(ShardId::new(1)).is_none(),
            "a report older than REPORT_EXPIRY_TICKS is no longer believed"
        );
        assert!(
            !orch.control.autoscaler.tracks(ShardId::new(1)),
            "and the autoscaler has forgotten the shard"
        );
        assert!(orch.take_pending_spawn().is_empty(), "nothing was split");
        assert!(orch.drain_outbox().is_empty());
        assert_eq!(orch.map().epoch, 1);
        // A report that keeps arriving within the expiry is still believed: one
        // every LOAD_REPORT_INTERVAL_TICKS is enough to complete a dwell.
        let mut split = false;
        for t in 0..=(dwell + crate::config::LOAD_REPORT_INTERVAL_TICKS) {
            if t % crate::config::LOAD_REPORT_INTERVAL_TICKS == 0 {
                orch.handle_load_report(&report(1, 95));
            }
            orch.tick();
            split |= !orch.take_pending_spawn().is_empty();
        }
        assert!(
            split,
            "a shard reporting once a beat is split after its dwell"
        );
    }

    // ------------------------------------------------------------- finding 3

    /// Mutation this catches: staging the kill when the merge is *proposed*.
    /// The loop then killed the shard in the same tick it sent the prepare: the
    /// shard never acked, the epoch aborted, and the map kept naming a dead
    /// owner for its region.
    #[test]
    fn a_merge_kills_the_child_only_once_the_epoch_commits() {
        let mut orch = orchestrator_with_a_child();
        let child = ShardId::new(2);
        orch.handle_load_report(&report(2, 0));
        let reply = orch.handle_admin(&AdminCommand::ForceMerge(child));
        assert!(accepted(&reply), "{reply:?}");
        assert_eq!(orch.pending_merge(), Some(child));
        assert!(
            orch.take_pending_kill().is_empty(),
            "nothing is killed while the prepare is outstanding"
        );
        // A tick passes with the prepare still un-acked: still nobody to kill.
        let outbox = orch.drain_outbox();
        orch.tick();
        assert!(orch.take_pending_kill().is_empty());
        assert_eq!(
            orch.map().regions.len(),
            2,
            "the child still owns its region"
        );

        // Every recipient acks: the merge commits, and only now is the kill staged.
        let recipients = prepares(&outbox);
        assert!(recipients.iter().any(|(to, _)| *to == child));
        for (shard, epoch) in recipients {
            orch.handle_epoch_ack(shard, epoch);
        }
        assert_eq!(orch.map().epoch, 2);
        assert_eq!(orch.map().regions.len(), 1, "the child's region is gone");
        assert_eq!(orch.take_pending_kill(), vec![child]);
        assert_eq!(orch.pending_merge(), None);

        // The departing shard is told the outcome (commit, then the map) before
        // it is killed: the loop delivers the outbox ahead of the kills.
        let to_child: Vec<Message> = orch
            .drain_outbox()
            .into_iter()
            .filter(|(to, _)| *to == child)
            .map(|(_, m)| m)
            .collect();
        assert!(matches!(to_child.first(), Some(Message::EpochCommit(_))));
        assert!(matches!(to_child.get(1), Some(Message::DirectoryReply(_))));
        // And nothing about it is remembered: no report, no autoscaler state.
        assert!(orch.load_report(child).is_none());
        assert!(!orch.control.autoscaler.tracks(child));
        // Its last reports may still arrive; they are dropped (it is off the map).
        orch.handle_load_report(&report(2, 0));
        assert!(orch.load_report(child).is_none());
    }

    /// Mutation this catches: not clearing the pending merge when its epoch
    /// times out — the next commit of anything would kill a shard that still
    /// owns its region.
    #[test]
    fn a_merge_that_aborts_kills_nobody_and_is_forgotten() {
        let mut orch = orchestrator_with_a_child();
        let child = ShardId::new(2);
        orch.handle_load_report(&report(2, 0));
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceMerge(child))
        ));
        let _ = orch.drain_outbox();
        // Nobody acks; the deadline passes.
        for _ in 0..=crate::config::EPOCH_ACK_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 1, "the merge aborted");
        assert_eq!(
            orch.map().regions.len(),
            2,
            "the child still owns its region"
        );
        assert_eq!(orch.pending_merge(), None, "the aborted merge is forgotten");
        assert!(orch.take_pending_kill().is_empty(), "an abort kills nobody");
        let _ = orch.drain_outbox();

        // A later, unrelated epoch commits: the child must survive it.
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(child))
        ));
        let grandchild = orch.take_pending_spawn().pop().expect("staged").shard;
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2, "the split committed");
        assert!(orch.map().regions.iter().any(|r| r.shard == grandchild));
        assert!(
            orch.take_pending_kill().is_empty(),
            "committing another epoch must not kill the shard whose merge aborted"
        );
    }

    /// Mutation this catches: merging on "cold" alone. Cold is a load at or
    /// below a third of a full shard; a merge removes the shard, and there is no
    /// step that hands its players to anyone.
    #[test]
    fn only_a_shard_known_to_be_empty_is_merged() {
        let mut orch = orchestrator_with_a_child();
        let child = ShardId::new(2);
        // No report at all: not known to be empty.
        let why = refusal(orch.handle_admin(&AdminCommand::ForceMerge(child)));
        assert!(why.contains("not known to be empty"), "{why}");
        // Three players: cold by the load metric, but not empty.
        orch.handle_load_report(&report(2, 3));
        let why = refusal(orch.handle_admin(&AdminCommand::ForceMerge(child)));
        assert!(why.contains("3 player"), "{why}");
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");
        assert_eq!(orch.pending_merge(), None);

        // The autoscaler path: cold-but-populated for well past the merge dwell
        // (and its cooldowns) never merges...
        for _ in 0..(3 * crate::autoscale::MERGE_DWELL_TICKS) {
            orch.handle_load_report(&report(2, 3));
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 1, "a populated shard is never merged");
        assert!(orch.take_pending_kill().is_empty());
        assert!(orch.drain_outbox().is_empty());
        // ...and once it reports empty for the dwell, it is.
        let mut proposed = false;
        for _ in 0..(2 * crate::autoscale::MERGE_DWELL_TICKS) {
            orch.handle_load_report(&report(2, 0));
            orch.tick();
            if orch.pending_merge() == Some(child) {
                proposed = true;
                break;
            }
        }
        assert!(proposed, "an empty shard, cold for the dwell, is merged");
    }

    // ------------------------------------------------- finding 2 (the abort)

    /// A split whose epoch aborts leaves a launched child that owns nothing: it
    /// is staged for killing (it used to run on, unowned, for ever), and the
    /// abort reaches every recipient of the prepare.
    #[test]
    fn a_split_that_aborts_stages_the_kill_of_its_child() {
        let mut orch = orchestrator_with_first_id(100);
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
        let child = orch.take_pending_spawn().pop().expect("staged").shard;
        let told: Vec<ShardId> = prepares(&orch.drain_outbox())
            .into_iter()
            .map(|(to, _)| to)
            .collect();
        assert!(orch.take_pending_kill().is_empty());
        for _ in 0..=crate::config::EPOCH_ACK_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 1, "the split aborted");
        assert_eq!(orch.take_pending_kill(), vec![child]);
        let aborted: Vec<ShardId> = orch
            .drain_outbox()
            .into_iter()
            .filter(|(_, m)| matches!(m, Message::EpochAbort(_)))
            .map(|(to, _)| to)
            .collect();
        assert_eq!(aborted, told, "everyone who got the prepare gets the abort");
        // A committed split, by contrast, kills nobody.
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
        let _ = orch.take_pending_spawn();
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2);
        assert!(orch.take_pending_kill().is_empty());
    }

    #[test]
    fn the_orchestrator_is_constructible_without_a_runtime() {
        // The whole point of this slice: the type builds in a plain #[test].
        let orch = orchestrator();
        assert_eq!(orch.config().cluster, OrchestratorConfig::default().cluster);
        assert_eq!(orch.control.map().epoch, 1);
    }

    #[test]
    fn a_commit_is_followed_by_a_map_push_carrying_every_real_endpoint() {
        // Ids start above the root's so the child is a distinct shard.
        let mut orch = orchestrator_with_first_id(100);
        let hello = |shard: ShardId, port: u16, region| ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard,
            endpoint: addr(port),
            region,
        };
        orch.handle_shard_hello(&hello(ShardId::new(1), 9_001, None))
            .expect("root shard registers");
        orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)));
        let spec = orch.take_pending_spawn().pop().expect("a child is staged");
        orch.handle_shard_hello(&hello(spec.shard, 9_002, Some(spec.region.clone())))
            .expect("child registers");
        let outbox = orch.drain_outbox();
        let epoch = outbox
            .iter()
            .find_map(|(_, m)| match m {
                Message::EpochPrepare(p) => Some(p.epoch),
                _ => None,
            })
            .expect("the prepare went out");
        let recipients: Vec<ShardId> = outbox.into_iter().map(|(to, _)| to).collect();
        assert!(!recipients.is_empty(), "the prepare went out");
        for shard in recipients {
            orch.handle_epoch_ack(shard, epoch);
        }

        let out = orch.drain_outbox();
        for shard in [ShardId::new(1), spec.shard] {
            let kinds: Vec<&Message> = out
                .iter()
                .filter(|(to, _)| *to == shard)
                .map(|(_, m)| m)
                .collect();
            assert!(
                matches!(kinds.first(), Some(Message::EpochCommit(_))),
                "the commit reaches {shard} first"
            );
            let Some(Message::DirectoryReply(push)) = kinds.get(1) else {
                panic!("the commit is followed by a map push for {shard}");
            };
            assert_eq!(push.shard_map.epoch, 2);
            let endpoint_of = |s: ShardId| {
                push.shard_map
                    .regions
                    .iter()
                    .find(|r| r.shard == s)
                    .map(|r| r.endpoint)
            };
            assert_eq!(endpoint_of(ShardId::new(1)), Some(addr(9_001)));
            assert_eq!(endpoint_of(spec.shard), Some(addr(9_002)));
        }
        // The orchestrator's own published map carries the same addresses.
        assert!(orch
            .map()
            .regions
            .iter()
            .any(|r| r.shard == spec.shard && r.endpoint == addr(9_002)));
    }

    #[test]
    fn a_child_that_never_registered_stays_at_the_sentinel_in_the_pushed_map() {
        let mut orch = orchestrator_with_first_id(100);
        orch.handle_shard_hello(&ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::new(1),
            endpoint: addr(9_001),
            region: None,
        })
        .expect("root shard registers");
        orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)));
        let spec = orch.take_pending_spawn().pop().expect("a child is staged");
        // The child is never spawned/registered: nobody tells the orchestrator
        // its address. Ack the prepare from whoever it went to.
        let outbox = orch.drain_outbox();
        let epoch = outbox
            .iter()
            .find_map(|(_, m)| match m {
                Message::EpochPrepare(p) => Some(p.epoch),
                _ => None,
            })
            .expect("the prepare went out");
        let recipients: Vec<ShardId> = outbox.into_iter().map(|(to, _)| to).collect();
        assert!(!recipients.is_empty(), "the prepare went out");
        for shard in recipients {
            orch.handle_epoch_ack(shard, epoch);
        }

        let out = orch.drain_outbox();
        let pushes: Vec<&space_protocol::DirectoryReply> = out
            .iter()
            .filter_map(|(_, m)| match m {
                Message::DirectoryReply(push) => Some(push),
                _ => None,
            })
            .collect();
        assert!(!pushes.is_empty(), "the commit is followed by a map push");
        for push in pushes {
            let child = push
                .shard_map
                .regions
                .iter()
                .find(|r| r.shard == spec.shard)
                .expect("the child's region is in the pushed map");
            assert_eq!(
                child.endpoint,
                ShardRegion::UNRESOLVED_ENDPOINT,
                "an unregistered child must not inherit the parent's address"
            );
            let parent = push
                .shard_map
                .regions
                .iter()
                .find(|r| r.shard == ShardId::new(1))
                .expect("parent region");
            assert_eq!(parent.endpoint, addr(9_001));
        }
    }

    #[test]
    fn the_mock_launcher_satisfies_the_trait() {
        // Proves the Launcher trait is implementable as a plain struct (the
        // acceptance tests rely on this).
        let launcher = MockLauncher::default();
        let spec = SpawnSpec {
            shard: ShardId::new(1),
            region: ShardRegion::new(
                OrchestratorConfig::default().root_node(),
                ShardId::new(1),
                addr(7_001),
            ),
            cluster: "space-local".into(),
            control_addr: addr(7_900),
            host_port: 7_870,
        };
        let handle = block_on(launcher.spawn(spec)).expect("spawn");
        assert_eq!(handle.shard, ShardId::new(1));
        assert_eq!(
            block_on(launcher.poll(ShardId::new(1))).unwrap(),
            Poll::Alive
        );
    }
}
