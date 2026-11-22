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
//!     ack    = control.acks()      => orch.handle_epoch_ack(shard),
//!     hello  = control.hellos()    => orch.handle_shard_hello(hello),
//!     _      = clock.tick()        => { orch.tick(); drain outbox; launcher.poll(); }
//! }}
//! ```
//!
//! [`Orchestrator::tick`] is the synchronous heart of that loop: it advances the
//! orchestrator's own clock, runs the epoch deadline, feeds the autoscaler the
//! latest report per shard, and on a `Split`/`Merge` computes a new topology,
//! bumps the epoch through [`crate::epoch::EpochPublisher`] and stages the
//! launcher work (spawn the child / kill the merged shard) in
//! [`Orchestrator::pending_spawn`] / [`Orchestrator::pending_kill`] for the async
//! layer to execute. The async [`Orchestrator::run`] wires those pieces together
//! over real QUIC; the tests drive the synchronous half directly.

use std::collections::BTreeMap;
use std::sync::Arc;

use space_protocol::control::{AdminCommand, AdminReply, LoadReport};
use space_protocol::shard::{EpochAck, ShardId, ShardMap};
use space_protocol::{Message, ShardHello, ShardWelcome};

use crate::autoscale::ScaleAction;
use crate::config::{OrchestratorConfig, MIN_EPOCH_INTERVAL_TICKS};
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
    /// A shard the loop must kill (staged by a merge).
    pending_kill: Vec<ShardId>,
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

    /// Drain the shard ids the loop staged for killing (a merge's child).
    pub fn take_pending_kill(&mut self) -> Vec<ShardId> {
        std::mem::take(&mut self.pending_kill)
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
    /// tick it arrived at (the shard's clock, used for `activation_tick`).
    pub fn handle_load_report(&mut self, report: &LoadReport) {
        self.latest.insert(report.shard, *report);
        self.shard_ticks.insert(report.shard, self.tick);
    }

    /// Record an `EpochAck` from `shard`. When the last required ack lands the
    /// publisher commits; the commit message is queued for every shard and the
    /// autoscaler is told an action completed (cooldown).
    pub fn handle_epoch_ack(&mut self, shard: ShardId, epoch: u64) {
        if let Some(commit) = self.control.publisher.ack(shard, epoch, self.tick) {
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
    /// paths the autoscaler uses, so an admin verb and an autoscale decision
    /// produce identical topology + epoch effects.
    pub fn handle_admin(&mut self, command: &AdminCommand) -> AdminReply {
        match command {
            AdminCommand::ForceSplit(shard) => {
                let octant = 0;
                self.begin_split(*shard, octant);
                AdminReply::Accepted {
                    command: command.clone(),
                }
            }
            AdminCommand::ForceMerge(shard) => {
                self.begin_merge(*shard);
                AdminReply::Accepted {
                    command: command.clone(),
                }
            }
            AdminCommand::KillShard(shard) => {
                self.pending_kill.push(*shard);
                AdminReply::Accepted {
                    command: command.clone(),
                }
            }
            other => self.control.handle_admin(other),
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
        }
        // 2. Autoscaler: feed the latest report per shard, get at most one action.
        let reports: Vec<LoadReport> = self.latest.values().copied().collect();
        if reports.is_empty() {
            return;
        }
        if let Some(action) = self.control.autoscaler.on_reports(&reports, self.tick) {
            match action {
                ScaleAction::Split { shard, octant } => self.begin_split(shard, octant),
                ScaleAction::Merge { shard } => self.begin_merge(shard),
            }
        }
    }

    // ------------------------------------------------- autoscale -> topology

    /// Turn a split decision into a topology plan + epoch proposal + a staged
    /// child spawn. No-op if an epoch is in flight, the region cannot split, or
    /// the plan is invalid (the autoscaler's cooldown still applies: a shard
    /// that cannot complete an epoch is not helped by retrying every tick).
    fn begin_split(&mut self, shard: ShardId, octant: usize) {
        if !self
            .control
            .publisher
            .can_propose(self.tick, MIN_EPOCH_INTERVAL_TICKS)
        {
            return;
        }
        let new_id = self.control.ids.allocate();
        let plan = match topology::split_plan(self.control.map(), shard, octant, new_id) {
            Ok(plan) => plan,
            Err(_) => {
                self.control.autoscaler.note_action(shard, self.tick);
                return;
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
            return;
        };
        self.queue_outbound(&prepare);
        self.control.autoscaler.note_action(shard, self.tick);
        // Stage the child spawn: the async layer runs the launcher and the child
        // then registers with its own ShardHello.
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
    }

    /// Turn a merge decision into a topology plan + epoch proposal + a staged
    /// kill of the merged child.
    fn begin_merge(&mut self, shard: ShardId) {
        if !self
            .control
            .publisher
            .can_propose(self.tick, MIN_EPOCH_INTERVAL_TICKS)
        {
            return;
        }
        let plan = match topology::merge_plan(self.control.map(), shard) {
            Ok(plan) => plan,
            Err(_) => {
                self.control.autoscaler.note_action(shard, self.tick);
                return;
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
            return;
        };
        self.queue_outbound(&prepare);
        self.control.autoscaler.note_action(shard, self.tick);
        // The merged child's region folds back into the parent; kill its shard.
        self.pending_kill.push(shard);
    }

    /// Fan an [`Outbound`] out to its recipients, tagging each with the shard it
    /// goes to so the async layer can route it.
    fn queue_outbound(&mut self, outbound: &Outbound) {
        for to in &outbound.to {
            self.outbox.push((*to, outbound.message.clone()));
        }
    }

    // ------------------------------------------------------------- async loop

    /// The event loop (plan §3.6): bind the control endpoint, accept shard
    /// connections, and on each tick beat run the synchronous [`Self::tick`],
    /// deliver the outbox, and execute the staged launcher work.
    ///
    /// The loop only moves messages and drives the launcher; every decision is
    /// made by the synchronous methods above. The clock is a wall-clock ticker
    /// here (the production beat); tests drive [`Self::tick`] directly instead
    /// and never call this.
    ///
    /// # Errors
    ///
    /// [`std::io::Error`] if the control endpoint cannot be bound.
    pub async fn run(&mut self) -> Result<(), std::io::Error> {
        let server = self.bind()?;
        self.run_with(server).await
    }

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

    /// Run the event loop over an already-bound `server` until it closes.
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
                            if let Ok(welcome) = self.handle_shard_hello(&hello) {
                                server.register(welcome.shard, reply.clone()).await;
                                let _ = reply
                                    .send(Message::ShardWelcome(welcome))
                                    .await;
                            }
                        }
                        Some(Inbound::Report(report)) => self.handle_load_report(&report),
                        Some(Inbound::Ack(ack)) => self.handle_epoch_ack_message(&ack),
                        None => break, // every sender gone
                    }
                }
                _ = clock.tick() => {
                    self.tick();
                    // Deliver the epoch messages the tick produced.
                    for (to, message) in self.drain_outbox() {
                        server.send_to(to, message).await;
                    }
                    // Spawn the children a split staged. The child boots and
                    // registers with its own ShardHello; the welcome gives it the
                    // committed map.
                    for spec in self.take_pending_spawn() {
                        if let Err(err) = self.launcher.spawn(spec.clone()).await {
                            eprintln!("space-orchestrator: spawn {} failed: {err}", spec.shard);
                        }
                    }
                    // Kill the shards a merge (or admin) staged.
                    for shard in self.take_pending_kill() {
                        let _ = self.launcher.kill(shard).await;
                        server.unregister(shard).await;
                    }
                }
            }
        }
        accept.abort();
        server.close();
        Ok(())
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
        let recipients: Vec<ShardId> = orch.drain_outbox().into_iter().map(|(to, _)| to).collect();
        assert!(!recipients.is_empty(), "the prepare went out");
        for shard in recipients {
            orch.handle_epoch_ack(shard);
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
        let recipients: Vec<ShardId> = orch.drain_outbox().into_iter().map(|(to, _)| to).collect();
        assert!(!recipients.is_empty(), "the prepare went out");
        for shard in recipients {
            orch.handle_epoch_ack(shard);
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
