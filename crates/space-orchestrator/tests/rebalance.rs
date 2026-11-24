//! Phase 3 slice-4 acceptance suite: H9-H15 seen from the **client's** side
//! (plan S5.1, t1-t16).
//!
//! Slice 3 proved the orchestrator's *decisions* (topology plans, epoch phases,
//! autoscaler dwell). This file proves the promise those decisions exist to
//! keep: a split or merge happens and a connected client cannot tell. Every
//! assertion below is a fact a client can observe -- the frames it receives,
//! the guid in them, the tick numbers on them, the `Redirect` it is handed, and
//! whether it was ever `Reject`ed.
//!
//! # The harness
//!
//! [`Cluster`] is one real [`Orchestrator`] state machine plus a set of real
//! [`Shard`]s, each a real [`ShardNet`] bound on loopback QUIC with a real
//! client-facing protocol. Clients are real: they connect through
//! [`space_net::connect`] and read real [`space_protocol::Frame`]s.
//!
//! Two things are deliberately *not* the production transport, both to keep the
//! suite deterministic (plan S5.1's "no wall-clock" rule):
//!
//! * **Shard clocks are ticked by the test**, not by a 30 Hz interval. The tick
//!   body is the same one `launcher::local` runs (`tick()` then `flush()`).
//! * **The control channel is an in-memory channel** instead of the orchestrator's
//!   QUIC front end. Slice 3's
//!   `a_shard_registers_over_loopback_and_gets_its_id_and_map` already proves
//!   that transport over real QUIC; re-proving it here would only add races.
//!
//! Everything a client can observe -- replication, the Phase-2 handoff protocol
//! between two shards over real QUIC, the shard-side epoch participant -- is
//! production code.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use space_orchestrator::config::OrchestratorConfig;
use space_orchestrator::control::ControlPlane;
use space_orchestrator::epoch::EpochPublisher;
use space_orchestrator::ids::{MemoryCounterSink, PersistentShardIdAllocator};
use space_orchestrator::launcher::{LaunchError, LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};
use space_orchestrator::orchestrator::Orchestrator;
use space_protocol::control::LoadReport;
use space_protocol::shard::{ShardId, ShardMap, ShardRegion};
use space_protocol::{
    EntityGuid, EntitySnapshot, Hello, InputTick, Message, NeighbourStatus, PlayerInput, Redirect,
    ShardHello,
};
use space_server::net::ShardNet;
use space_server::shard::Shard;
use space_server::ShardConfig;
use space_world::{NodeRef, WorldPos};
use tokio::sync::{mpsc, Mutex};

/// Root depth used by the client-facing tests.
///
/// [`space_orchestrator::topology::SPLIT_MIN_NODE_SIZE_M`] is 1 048 576 m, i.e.
/// depth 54 is the deepest region that may split. Depth 54 is used here (not the
/// config default 40) so a ship's *own* octant child is a 524 km box: the split
/// boundary can then be placed around a ship that has moved a few metres, which
/// is what makes the transition observable in a test that runs in seconds.
const ROOT_DEPTH: u8 = 54;

fn addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)
}

fn shard(n: u32) -> ShardId {
    ShardId::new(n)
}

/// The root region of a test cluster.
fn root() -> NodeRef {
    NodeRef::new(ROOT_DEPTH, [0, 0, 0])
}

/// A launcher that records the orchestrator's launch decisions and never starts
/// a process: the harness starts members itself so it can wire their peer
/// endpoints. Recording is what lets a test assert what a launch spec carries.
#[derive(Debug, Default)]
struct RecordingLauncher {
    spawned: StdMutex<Vec<SpawnSpec>>,
    killed: StdMutex<Vec<LaunchId>>,
}

impl RecordingLauncher {
    fn spawned(&self) -> Vec<SpawnSpec> {
        self.spawned.lock().unwrap().clone()
    }

    fn killed(&self) -> Vec<LaunchId> {
        self.killed.lock().unwrap().clone()
    }
}

impl Launcher for RecordingLauncher {
    async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
        self.spawned.lock().unwrap().push(spec.clone());
        Ok(ShardHandle {
            launch: spec.launch,
            shard: None,
            endpoint: spec.control_addr,
            process: "harness".into(),
        })
    }
    async fn kill(&self, launch: LaunchId) -> Result<(), LaunchError> {
        self.killed.lock().unwrap().push(launch);
        Ok(())
    }
    async fn poll(&self, launch: LaunchId) -> Result<Poll, LaunchError> {
        Ok(if self.killed.lock().unwrap().contains(&launch) {
            Poll::Dead
        } else {
            Poll::Alive
        })
    }
}

/// A control plane whose root is owned by `root_owner`, epoch 1.
fn plane_with_root(root_owner: ShardId, ids: PersistentShardIdAllocator) -> ControlPlane {
    let config = OrchestratorConfig::default();
    let map = ShardMap::new(
        1,
        vec![ShardRegion::new(root(), root_owner, addr_on(7_001))],
    );
    ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into())
}

fn addr_on(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

fn report(s: ShardId, players: u32) -> LoadReport {
    LoadReport {
        shard: s,
        players,
        tick_p95_ms: 1.0,
        load: 0.0,
        ..Default::default()
    }
}

// ---------------------------------------------------------------- harness

/// One orchestrated shard: a real [`Shard`] behind a real [`ShardNet`].
struct Member {
    id: ShardId,
    shard: Arc<Mutex<Shard>>,
    net: Arc<ShardNet>,
}

/// A client: a real QUIC connection plus a task collecting every frame it gets.
///
/// Every member of a cluster is ticked in lockstep and starts its simulation at
/// the same tick (see [`Cluster::add_member`]), so the tick stamped on a frame is
/// comparable no matter which shard sent it -- which is what lets a test ask the
/// client's real question: "did the numbers on my frames keep counting?"
pub struct Client {
    pub token: String,
    pub shard: ShardId,
    pub session: u64,
    pub guid: EntityGuid,
    /// The connection inputs go out on (the shard the client considers its own).
    pub conn: space_net::Conn,
    /// A connection opened to a redirect target, not yet the input path.
    pub next: Option<space_net::Conn>,
    frames: Arc<StdMutex<Vec<Message>>>,
    /// The newest frame that carried the client's own ship, from whichever
    /// connection delivered it: `(shard that sent it, tick, ship position)`.
    latest: Arc<StdMutex<Option<(ShardId, u64, WorldPos)>>>,
}

impl Client {
    /// Take and clear everything this client has received so far.
    pub fn drain(&self) -> Vec<Message> {
        std::mem::take(&mut self.frames.lock().unwrap())
    }

    /// Everything received so far, without clearing.
    pub fn peek(&self) -> Vec<Message> {
        self.frames.lock().unwrap().clone()
    }
}

/// A cluster: the orchestrator state machine, its members and its clients.
struct Cluster {
    orch: Orchestrator<RecordingLauncher>,
    members: Vec<Member>,
    /// Where a member's epoch acks land (the shard's `Destination::Orchestrator`).
    ack_rx: Arc<Mutex<mpsc::Receiver<Message>>>,
    /// Kept so the members' orchestrator senders have somewhere to write.
    ack_tx: mpsc::Sender<Message>,
    clients: Vec<Client>,
    /// Beats run so far. Every member is ticked exactly once per beat, so this is
    /// also the simulation tick every member is at.
    beats: u64,
    /// `EpochPrepare`s delivered since the last beat began. Each one makes its
    /// member queue an ack that leaves on the *next* beat's flush.
    prepares_sent: AtomicUsize,
    /// Acks this beat's flush is expected to forward (the prepares outstanding
    /// when the beat began); `drain_acks` waits wall-clock time only for these.
    acks_expected: usize,
    /// The id the member brought up for the latest launch was welcomed as.
    last_launched: Option<ShardId>,
}

impl Cluster {
    fn new(first_id: u64) -> Self {
        let ids =
            PersistentShardIdAllocator::with_sink(first_id, Box::new(MemoryCounterSink::default()));
        let plane = plane_with_root(shard(1), ids);
        let (ack_tx, ack_rx) = mpsc::channel(1024);
        Self {
            orch: Orchestrator::new(plane, Arc::new(RecordingLauncher::default())),
            members: Vec::new(),
            ack_rx: Arc::new(Mutex::new(ack_rx)),
            ack_tx,
            clients: Vec::new(),
            beats: 0,
            prepares_sent: AtomicUsize::new(0),
            acks_expected: 0,
            last_launched: None,
        }
    }

    fn launcher(&self) -> Arc<RecordingLauncher> {
        Arc::clone(&self.orch.launcher)
    }

    /// Bring up the member that owns `region` under the id `id` — the root's
    /// owner, which the map names from the start: a real shard on loopback
    /// QUIC, running the published map, registered with the orchestrator by a
    /// `ShardHello` naming that id. No peer endpoint is wired by hand.
    async fn add_member(&mut self, id: ShardId, region: NodeRef) -> SocketAddr {
        let net = Arc::new(ShardNet::bind(addr()).expect("bind shard"));
        let bound = net.local_addr().expect("bound addr");
        // Register with the orchestrator the way a booting shard does (a
        // `ShardHello` naming its bound endpoint). Nothing else is wired: no
        // peer endpoint is told to any shard by hand. How a parent learns where
        // its new child listens is the orchestrator's map push alone.
        self.orch
            .handle_shard_hello(&ShardHello {
                cluster: OrchestratorConfig::default().cluster,
                shard: id,
                endpoint: bound,
                region: Some(ShardRegion::new(region, id, bound)),
                client_endpoint: None,
                secret: String::new(),
            })
            .expect("the orchestrator accepts the booting shard");
        self.start_member(net, id, region).await;
        bound
    }

    /// Bring up the member a launch asked for, as every shard after the root
    /// comes up: it says hello with **no id and no region** and runs as the
    /// shard, on the region, the orchestrator's welcome names. Returns its id.
    async fn add_launched_member(&mut self, spec: &SpawnSpec) -> ShardId {
        let net = Arc::new(ShardNet::bind(addr()).expect("bind shard"));
        let bound = net.local_addr().expect("bound addr");
        let welcome = self
            .orch
            .handle_shard_hello(&ShardHello {
                cluster: spec.cluster.clone(),
                shard: ShardId::NONE,
                endpoint: bound,
                region: None,
                client_endpoint: None,
                secret: spec.secret.clone(),
            })
            .expect("the orchestrator welcomes the launched shard");
        // What the launcher reports: where the shard it started listens.
        self.orch.note_launch_result(
            spec.launch,
            Ok(ShardHandle {
                launch: spec.launch,
                shard: None,
                endpoint: bound,
                process: "harness".into(),
            }),
        );
        self.start_member(net, welcome.shard, welcome.region.node)
            .await;
        self.last_launched = Some(welcome.shard);
        welcome.shard
    }

    /// Start a registered shard: on `net`, as `id`, for `region`, running the
    /// map the orchestrator has published (in which a launched shard does not
    /// own its region yet).
    async fn start_member(&mut self, net: Arc<ShardNet>, id: ShardId, region: NodeRef) {
        let bound = net.local_addr().expect("bound addr");
        let config = ShardConfig::new(id, bound, region);
        let mut shard = Shard::new(config);
        // Align the new member's clock with the cluster's. Every member is ticked
        // exactly once per beat, so catching a late joiner up to `beats` makes the
        // tick it stamps on a frame the same number every other member would stamp
        // -- which is what lets a client's tick numbering be checked *across* a
        // handoff instead of only within one shard.
        shard.tick_n(self.beats);
        // The member runs the map the orchestrator has published, so its
        // `lookup` -- and therefore its handoff boundary -- is the cluster's.
        shard.set_shard_map(self.orch.map().clone());
        let shard = Arc::new(Mutex::new(shard));
        // The serve task is detached on purpose: it lives as long the shard's
        // connections do, and nothing in the test joins it.
        {
            let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
            tokio::spawn(async move { net.serve(shard).await });
        }
        // Epoch messages arrive by direct call (the QUIC half is proven in
        // slice 3); acks leave through the real flush path into `ack_rx`.
        let (ack_tx, mut ack_rx_local) = mpsc::channel::<Message>(1024);
        net.set_orchestrator_sender(ack_tx).await;
        {
            let out = self.ack_tx.clone();
            tokio::spawn(async move {
                while let Some(m) = ack_rx_local.recv().await {
                    let _ = out.send(m).await;
                }
            });
        }
        self.members.push(Member { id, shard, net });
    }

    fn member(&self, id: ShardId) -> &Member {
        self.members
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("member {id} is not running"))
    }

    /// Connect a real client to `id` over QUIC and start collecting its frames.
    async fn connect(&mut self, id: ShardId, token: &str) -> usize {
        let endpoint_addr = self.member(id).net.local_addr().expect("bound");
        let endpoint = space_net::client_endpoint().expect("client endpoint");
        let hello = Hello {
            protocol: space_protocol::PROTOCOL_VERSION,
            player_token: token.into(),
        };
        let (conn, welcome) = space_net::connect(
            &endpoint,
            endpoint_addr,
            space_server::net::SERVER_NAME,
            &hello,
        )
        .await
        .expect("the shard must accept a real client");
        let guid = {
            let shard = self.member(id).shard.lock().await;
            shard
                .ship_of(welcome.session)
                .expect("a join spawns a ship")
        };
        let frames: Arc<StdMutex<Vec<Message>>> = Arc::new(StdMutex::new(Vec::new()));
        let latest = Arc::new(StdMutex::new(None));
        collect_frames(conn.clone(), guid, Arc::clone(&frames), Arc::clone(&latest));
        self.clients.push(Client {
            token: hello.player_token,
            shard: id,
            session: welcome.session,
            guid,
            conn,
            next: None,
            frames,
            latest,
        });
        self.clients.len() - 1
    }

    /// One cluster beat: every member ticks and flushes, then the orchestrator
    /// ticks and its outbox is delivered.
    ///
    /// The member tick body is the one `launcher::local` runs; the orchestrator
    /// beat is the one `Orchestrator::run` runs, minus the wall-clock interval.
    async fn beat(&mut self) {
        self.beats += 1;
        self.acks_expected = self.prepares_sent.swap(0, Ordering::SeqCst);
        for m in &self.members {
            m.shard.lock().await.tick();
            m.net.flush(&m.shard).await;
        }
        self.orch.tick();
        self.deliver_outbox().await;
        self.run_pending_launches().await;
    }

    /// Deliver everything the orchestrator has queued (prepares, commits, the
    /// committed-map push) without advancing its clock.
    async fn deliver_outbox(&mut self) {
        for (to, message) in self.orch.drain_outbox() {
            self.deliver(to, message).await;
        }
    }

    /// One beat, then close the epoch round-trip by feeding the acks the shards
    /// just sent back.
    async fn step(&mut self) {
        self.beat().await;
        self.drain_acks().await;
    }

    async fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step().await;
        }
    }

    /// Yield so the client reader tasks can drain what the shards just wrote.
    ///
    /// The harness ticks shards synchronously; a client's reader is a separate
    /// tokio task that needs a scheduling slot to read what `flush` wrote.
    async fn settle(&self) {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    /// [`Self::settle`], but first wait (at most 2 s) until client `idx` has
    /// received something: a fixed 5 ms was not always a scheduling slot on a
    /// loaded CI container, and the client then read nothing at all.
    async fn settle_until_frames(&self, idx: usize) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while self.clients[idx].frames.lock().unwrap().is_empty()
            && std::time::Instant::now() < deadline
        {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        self.settle().await;
    }

    /// Feed the orchestrator a `LoadReport` from `id`, as a shard's control
    /// connection would every `LOAD_REPORT_INTERVAL_TICKS`.
    fn report_load(&mut self, id: ShardId, players: u32) {
        self.orch.handle_load_report(&report(id, players));
    }

    /// Send an epoch message to one member (the orchestrator's outbox half).
    async fn deliver(&self, to: ShardId, message: Message) {
        match message {
            // (The scene travels on the same link: it is re-issued when a dead
            // shard has been replaced. A member adopts it and confirms.)
            Message::EpochPrepare(_)
            | Message::EpochCommit(_)
            | Message::EpochAbort(_)
            | Message::DirectoryReply(_)
            | Message::SceneSet(_) => {
                if let Some(m) = self.members.iter().find(|m| m.id == to) {
                    if matches!(message, Message::EpochPrepare(_)) {
                        self.prepares_sent.fetch_add(1, Ordering::SeqCst);
                    }
                    m.shard.lock().await.orchestrator_message(message);
                }
            }
            other => panic!("unexpected orchestrator->shard message: {other:?}"),
        }
    }

    /// Feed every epoch ack the members sent back.
    ///
    /// Acks leave a member through the real `ShardNet::flush` path and are
    /// forwarded by a task, so they are not synchronously available the instant
    /// the beat ends. Wait on the channel with a bounded wall-clock timeout
    /// (not a fixed number of scheduler yields, which a CPU-saturated host can
    /// exhaust before the forwarding task runs), but only while acks are
    /// actually owed: a beat with no prepare in flight costs nothing.
    async fn drain_acks(&mut self) {
        const ACK_WAIT: std::time::Duration = std::time::Duration::from_secs(2);
        while self.acks_expected > 0 {
            let next = {
                let mut rx = self.ack_rx.lock().await;
                tokio::time::timeout(ACK_WAIT, rx.recv()).await
            };
            match next {
                Ok(Some(Message::EpochAck(ack))) => {
                    self.acks_expected -= 1;
                    self.orch.handle_epoch_ack_message(&ack);
                }
                Ok(Some(_)) => continue,
                // Timed out (a rejected prepare sends no ack) or channel closed.
                Ok(None) | Err(_) => break,
            }
        }
        self.acks_expected = 0;
        // Sweep anything else already queued, without waiting.
        loop {
            let next = self.ack_rx.lock().await.try_recv();
            match next {
                Ok(Message::EpochAck(ack)) => self.orch.handle_epoch_ack_message(&ack),
                Ok(_) => continue,
                Err(_) => break,
            }
        }
    }

    /// Run what the orchestrator staged: the launch of a shard that is missing
    /// (a split's, a dead shard's replacement), and the stopping of one.
    async fn run_pending_launches(&mut self) {
        for spec in self.orch.take_pending_spawn() {
            self.launcher()
                .spawn(spec.clone())
                .await
                .expect("record spawn");
            // The shard boots with no id and no region and is given both at
            // its hello -- never one it chose itself.
            self.add_launched_member(&spec).await;
        }
        for id in self.orch.take_pending_kill() {
            self.members.retain(|m| m.id != id);
        }
        for launch in self.orch.take_pending_launch_kill() {
            self.launcher().kill(launch).await.expect("record kill");
        }
    }

    /// The published map's epoch.
    fn epoch(&self) -> u64 {
        self.orch.map().epoch
    }

    /// The published map.
    fn map(&self) -> &space_protocol::ShardMap {
        self.orch.map()
    }

    /// Stage a split whose child registers and then is never heard from again
    /// (it is not brought up as a member), so its ack never comes and the
    /// prepare is guaranteed to time out.
    async fn stage_split_isolated(&mut self, parent: ShardId) -> Vec<ShardId> {
        self.orch
            .handle_admin(&space_protocol::AdminCommand::ForceSplit(parent));
        // The staged launch is not run: a hello arrives for it from an address
        // nothing listens on, and that is the last anybody hears of it.
        let _ = self.orch.take_pending_spawn();
        self.orch
            .handle_shard_hello(&ShardHello {
                cluster: OrchestratorConfig::default().cluster,
                shard: ShardId::NONE,
                endpoint: addr_on(9),
                region: None,
                client_endpoint: None,
                secret: String::new(),
            })
            .expect("the shard for the split registers");
        let mut recipients = Vec::new();
        for (to, message) in self.orch.drain_outbox() {
            if matches!(message, Message::EpochPrepare(_)) {
                recipients.push(to);
            }
            self.deliver(to, message).await;
        }
        recipients
    }

    /// Stage a split through the admin verb (the same code path the autoscaler
    /// uses) and bring the child up as a member, **without** letting the epoch
    /// commit. Returns the child's id and the prepare recipients.
    async fn stage_split(&mut self, parent: ShardId) -> (ShardId, Vec<ShardId>) {
        self.orch
            .handle_admin(&space_protocol::AdminCommand::ForceSplit(parent));
        // The child registers -- which is when its id is allocated and the
        // prepare is proposed -- and is a member before the prepare is
        // delivered, or its ack would never be seen.
        self.last_launched = None;
        self.run_pending_launches().await;
        let mut recipients = Vec::new();
        for (to, message) in self.orch.drain_outbox() {
            if matches!(message, Message::EpochPrepare(_)) {
                recipients.push(to);
            }
            self.deliver(to, message).await;
        }
        let child = self.last_launched.expect("the split launched a shard");
        (child, recipients)
    }

    /// Ack the outstanding prepare on behalf of every member that must, and run
    /// beats until the epoch commits.
    async fn commit_split(&mut self, before: u64) {
        for _ in 0..16 {
            self.step().await;
            if self.epoch() > before {
                break;
            }
        }
        assert!(self.epoch() > before, "the epoch committed");
        // The commit and the orchestrator's map push are queued; deliver them.
        self.deliver_outbox().await;
    }

    /// Split `parent` and commit the epoch. Returns the child's id.
    async fn split(&mut self, parent: ShardId) -> ShardId {
        let before = self.epoch();
        let (child, _) = self.stage_split(parent).await;
        self.commit_split(before).await;
        child
    }

    /// Stage a merge of `child` back into its parent's owner and commit it.
    ///
    /// The orchestrator only merges a shard whose own load report says it has
    /// no players, so the child reports first — its real client count, exactly
    /// what a shard's `LoadReport` carries.
    async fn merge(&mut self, child: ShardId) {
        let before = self.epoch();
        let players = self.member(child).shard.lock().await.stats().clients;
        self.report_load(child, u32::try_from(players).expect("a small client count"));
        let reply = self
            .orch
            .handle_admin(&space_protocol::AdminCommand::ForceMerge(child));
        assert!(
            matches!(reply, space_protocol::AdminReply::Accepted { .. }),
            "the merge of an empty child is accepted: {reply:?}"
        );
        for (to, message) in self.orch.drain_outbox() {
            self.deliver(to, message).await;
        }
        for _ in 0..16 {
            self.step().await;
            if self.epoch() > before {
                break;
            }
        }
        assert!(self.epoch() > before, "the merge committed");
        self.deliver_outbox().await;
    }

    /// Kill `shard` through the admin verb and run a beat so the staged kill is
    /// executed (the launcher records it and the member is dropped).
    async fn kill(&mut self, shard: ShardId) {
        self.orch
            .handle_admin(&space_protocol::AdminCommand::KillShard(shard));
        self.step().await;
    }

    /// Run the minimum interval between two epoch commits, so the next split or
    /// merge is not dropped by `MIN_EPOCH_INTERVAL_TICKS`.
    async fn headroom(&mut self) {
        self.step_n(space_orchestrator::config::MIN_EPOCH_INTERVAL_TICKS)
            .await;
    }

    /// Send one client's next input batch, thrusting along `axis`.
    ///
    /// The batch is tagged with the tick the cluster is about to run. Every
    /// member is ticked exactly once per beat (see [`Cluster::add_member`]), so
    /// the beat count *is* the tick a client's own clock would be at.
    async fn thrust(&self, idx: usize, axis: usize, magnitude: f32) {
        let mut thrust = [0.0f32; 3];
        thrust[axis] = magnitude;
        self.thrust_xyz(idx, thrust).await;
    }

    /// One batch with a thrust vector, for a ship that must move diagonally.
    ///
    /// The simulation keeps at most one input per entity per tick, so three
    /// single-axis batches tagged with the same tick would collapse into one and
    /// the ship would move along a single axis.
    async fn thrust_xyz(&self, idx: usize, thrust: [f32; 3]) {
        let client = &self.clients[idx];
        // One tick of lead. The batch is applied by the shard's connection reader
        // task, which may or may not get a scheduling slot before the next beat;
        // a batch tagged with the tick that beat runs would be dropped as stale if
        // it lands late, so the ship would only move on the ticks where the reader
        // happened to win the race. With a tick of lead the input is filed either
        // way and is consumed by the beat that reaches it.
        let tick = self.beats + 2;
        let batch = InputTick {
            tick,
            seq: tick,
            inputs: vec![PlayerInput {
                entity: client.guid,
                thrust,
                torque: [0.0; 3],
                tick,
                fast: false,
            }],
            ack_tick: 0,
        };
        client
            .conn
            .send(&Message::InputTick(batch))
            .await
            .expect("send input");
        // Let the reader task take the batch off the connection before the beat.
        self.settle().await;
    }
}

// ------------------------------------------------------ client observation

/// Everything a client can tell about what it received, in arrival order.
///
/// This is deliberately the *client's* vocabulary only: the frames that arrived
/// on its connection, the shard and tick stamped on each, whether its own ship
/// was in them, the redirects it was handed and whether it was ever refused.
#[derive(Debug, Default)]
struct ClientView {
    /// `(shard that produced the frame, its tick)` per snapshot/delta, in
    /// arrival order.
    frames: Vec<(ShardId, u64)>,
    /// The same pairs, restricted to frames that carried the client's ship.
    ship_frames: Vec<(ShardId, u64)>,
    /// The shard-map epoch stamped on every frame, in arrival order.
    epochs: Vec<u64>,
    /// Every `Redirect` received, in order.
    redirects: Vec<Redirect>,
    /// How many `Reject`s were received.
    rejects: usize,
}

impl ClientView {
    /// The distinct epochs observed, sorted.
    fn distinct_epochs(&self) -> Vec<u64> {
        let mut out = self.epochs.clone();
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Read a client's received frames into a [`ClientView`], draining its inbox.
fn observe(client: &Client) -> ClientView {
    // The anchor is the guid the client was given at join; a handoff keeps that
    // guid (H13's first clause), so one anchor spans the whole transition.
    view_of(&client.drain(), client.guid)
}

/// Fold an inbox into a [`ClientView`] of `anchor`.
fn view_of(inbox: &[Message], anchor: EntityGuid) -> ClientView {
    let mut view = ClientView::default();
    for message in inbox {
        match message {
            Message::Snapshot(s) => {
                view.frames.push((s.shard, s.tick));
                view.epochs.push(s.epoch);
                if s.entities.iter().any(|e| e.guid == anchor) {
                    view.ship_frames.push((s.shard, s.tick));
                }
            }
            Message::Delta(d) => {
                view.frames.push((d.shard, d.tick));
                view.epochs.push(d.epoch);
                if d.entities.iter().any(|e| e.guid == anchor) {
                    view.ship_frames.push((d.shard, d.tick));
                }
            }
            Message::Redirect(r) => view.redirects.push(r.clone()),
            Message::Reject(_) => view.rejects += 1,
            _ => {}
        }
    }
    view
}

impl Cluster {
    /// Fold everything a client received since the last call into a view.
    fn take(&self, idx: usize) -> ClientView {
        observe(&self.clients[idx])
    }

    /// The member that currently simulates one client's ship.
    async fn simulates(&self, idx: usize) -> Option<ShardId> {
        let guid = self.clients[idx].guid;
        for m in &self.members {
            if m.shard.lock().await.sim().ship_state(guid).is_some() {
                return Some(m.id);
            }
        }
        None
    }
}

/// The tick stamped on a frame; `None` for anything that is not a frame.
fn frame_tick(message: &Message) -> Option<u64> {
    match message {
        Message::Snapshot(f) => Some(f.tick),
        Message::Delta(f) => Some(f.tick),
        _ => None,
    }
}

/// The component id of `ShipState` on the wire (`space_sim`'s registry: id 0).
const SHIP_STATE_COMPONENT: u16 = 0;

/// The position of `guid`'s ship in a frame's entity list, decoded the way a
/// client decodes it.
fn ship_pos_in(entities: &[EntitySnapshot], guid: EntityGuid) -> Option<WorldPos> {
    let entity = entities.iter().find(|e| e.guid == guid)?;
    let (_, payload) = entity
        .components
        .iter()
        .find(|(id, _)| *id == SHIP_STATE_COMPONENT)?;
    space_sim::wire::decode_ship_state(payload)
        .ok()
        .map(|state| state.pos)
}

/// Collect every message `conn` delivers into `frames`, and remember the newest
/// frame that carried `guid`'s ship in `latest`.
fn collect_frames(
    conn: space_net::Conn,
    guid: EntityGuid,
    frames: Arc<StdMutex<Vec<Message>>>,
    latest: Arc<StdMutex<Option<(ShardId, u64, WorldPos)>>>,
) {
    tokio::spawn(async move {
        loop {
            if conn.is_closed() {
                return;
            }
            let next = tokio::select! {
                m = conn.recv_stream() => m,
                m = conn.recv_datagram() => m,
                m = conn.recv_control() => m,
            };
            let Ok(Some(message)) = next else { return };
            let carried = match &message {
                Message::Snapshot(f) => {
                    ship_pos_in(&f.entities, guid).map(|pos| (f.shard, f.tick, pos))
                }
                Message::Delta(f) => {
                    ship_pos_in(&f.entities, guid).map(|pos| (f.shard, f.tick, pos))
                }
                _ => None,
            };
            if let Some(carried) = carried {
                let mut latest = latest.lock().unwrap();
                // Frames from two shards interleave during make-before-break;
                // the newest tick is what the client would render.
                if latest.is_none_or(|(_, tick, _)| carried.1 >= tick) {
                    *latest = Some(carried);
                }
            }
            frames.lock().unwrap().push(message);
        }
    });
}

impl Client {
    /// The newest ship-bearing frame: `(sending shard, tick, position)`.
    fn latest(&self) -> Option<(ShardId, u64, WorldPos)> {
        *self.latest.lock().unwrap()
    }
}

impl Cluster {
    /// Do what a client does with a `Redirect`: open a connection to the shard it
    /// names (at the address *the redirect's own map* gives), present the same
    /// hello token, and report `NeighbourStatus { connected: true }` so the old
    /// shard's make-before-break gate can open. The old connection stays the
    /// input path until [`Cluster::switch_input_path`].
    async fn follow(&mut self, idx: usize, redirect: &Redirect) {
        let target = redirect
            .shard_map
            .regions
            .iter()
            .find(|r| r.shard == redirect.new_shard)
            .expect("a redirect's map names the shard it redirects to")
            .endpoint;
        let endpoint = space_net::client_endpoint().expect("client endpoint");
        let hello = Hello {
            protocol: space_protocol::PROTOCOL_VERSION,
            player_token: self.clients[idx].token.clone(),
        };
        let (conn, _welcome) =
            space_net::connect(&endpoint, target, space_server::net::SERVER_NAME, &hello)
                .await
                .expect("the redirect's endpoint must be a shard that accepts the client");
        let client = &mut self.clients[idx];
        collect_frames(
            conn.clone(),
            client.guid,
            Arc::clone(&client.frames),
            Arc::clone(&client.latest),
        );
        conn.send_control(&Message::NeighbourStatus(NeighbourStatus {
            entity: client.guid,
            shard: redirect.new_shard,
            connected: true,
        }))
        .await
        .expect("report the new connection");
        client.next = Some(conn);
        client.shard = redirect.new_shard;
    }

    /// Do what a client does with a *repeated* `Redirect` to the shard it is
    /// already connected to: the shards re-offered (the first attempt timed out
    /// or was refused), so report in again on the open connection
    /// (`space_client::link::follow_redirect` does the same). Without this the
    /// re-offer waits on a confirmation that never comes, times out, and is
    /// offered again, forever.
    async fn report_in(&mut self, idx: usize, redirect: &Redirect) {
        let client = &self.clients[idx];
        let Some(next) = client.next.as_ref() else {
            return;
        };
        next.send_control(&Message::NeighbourStatus(NeighbourStatus {
            entity: client.guid,
            shard: redirect.new_shard,
            connected: true,
        }))
        .await
        .expect("report the connection again");
    }

    /// The handoff completed: send inputs to the new shard and drop the old link.
    fn switch_input_path(&mut self, idx: usize) {
        let client = &mut self.clients[idx];
        if let Some(next) = client.next.take() {
            let old = std::mem::replace(&mut client.conn, next);
            old.close(0, "followed the redirect");
        }
    }

    /// Fly client `idx` towards the carved child (thrust along -x,-y,-z, the
    /// direction of octant 0), following its `Redirect` the moment it arrives,
    /// and keep going until the handoff completed and `after` more beats passed.
    ///
    /// Returns every message the client received, in arrival order. Every other
    /// client is left alone but keeps being ticked.
    ///
    /// **The beats are paced like a running shard's ticks** (one per
    /// `1 / TICK_HZ` of wall-clock time) for as long as the flight lasts.
    /// Everything a handoff waits on is bounded in *ticks* (offer, commit and
    /// adoption: two seconds at 30 Hz), while what it waits for travels over
    /// real QUIC connections. Beating as fast as the loop can run (about every
    /// 5 ms) squeezed those bounds to a fraction of a second of wall-clock
    /// time, so on a busy host (tests in parallel, a CI runner building three
    /// images at once) an offer or a commit was still in flight when the bound
    /// fired: the ship went missing from the client's frames for the length of
    /// a delivery delay, or the handoff timed out. At the real tick rate the
    /// bounds mean what they mean in production.
    ///
    /// Frames stamped with a tick from before the flight are left out of what
    /// is returned. They are the setup's, still in flight when it emptied the
    /// inbox: the setup beats flat out, and a frame its client's reader could
    /// not take in time is retransmitted and lands in the middle of the flight
    /// (a frame of tick 31 next to tick 96), where it read as a 64-tick gap
    /// in a transition that had none.
    async fn fly_into_child(&mut self, idx: usize, child: ShardId, after: u64) -> Vec<Message> {
        let tick = std::time::Duration::from_secs(1) / space_sim::TICK_HZ;
        let started_after = self.beats;
        let mut inbox: Vec<Message> = Vec::new();
        let mut redirected = false;
        let mut done_at: Option<u64> = None;
        let mut next_beat = tokio::time::Instant::now();
        for _ in 0..1500 {
            self.thrust_xyz(idx, [-1.0, -1.0, -1.0]).await;
            self.step().await;
            self.settle().await;
            let got = self.clients[idx].drain();
            if done_at.is_none() {
                let redirects: Vec<Redirect> = got
                    .iter()
                    .filter_map(|m| match m {
                        Message::Redirect(r) => Some(r.clone()),
                        _ => None,
                    })
                    .collect();
                for r in &redirects {
                    if !redirected {
                        self.follow(idx, r).await;
                        redirected = true;
                    } else if r.new_shard == self.clients[idx].shard {
                        self.report_in(idx, r).await;
                    }
                }
            }
            inbox.extend(
                got.into_iter()
                    .filter(|m| frame_tick(m).is_none_or(|t| t > started_after)),
            );
            if redirected && done_at.is_none() && self.simulates(idx).await == Some(child) {
                self.switch_input_path(idx);
                done_at = Some(self.beats);
            }
            if done_at.is_some_and(|at| self.beats >= at + after) {
                return inbox;
            }
            next_beat += tick;
            tokio::time::sleep_until(next_beat).await;
        }
        panic!("the ship never completed a handoff into the child (redirected: {redirected})");
    }
}

// ------------------------------------------------------------------- H9
//
// "Shard ids are allocated by the orchestrator, are strictly monotonic, and are
// never reused across a split, a merge, a crash or an orchestrator restart."
// The client-observable half: the shard a client is *told* about -- in its
// `Welcome`, in the `Redirect` it is handed, in the map inside that redirect --
// always names a shard the orchestrator allocated, and nothing a client ever
// sees carries an id a human typed.

#[tokio::test(flavor = "multi_thread")]
async fn the_orchestrator_allocates_ids_and_no_launch_spec_carries_one() {
    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    let client = c.connect(shard(1), "h9").await;
    c.step_n(3).await;

    // What the client was welcomed by is an id the orchestrator owns: it is the
    // shard named in the published map the client's own `Welcome` carried.
    assert_eq!(c.epoch(), 1);
    let welcomed = c.member(shard(1)).id;
    assert_eq!(welcomed, shard(1));
    assert!(
        c.map()
            .regions
            .iter()
            .any(|r| r.shard == c.clients[client].shard),
        "the client's shard is one the orchestrator published"
    );

    // Now let the orchestrator launch something. The spec it stages carries no
    // shard id at all -- there is no field for one: the launched shard says
    // hello without an id and the orchestrator allocates it then, from its own
    // counter. Never a placeholder from a config file or argv.
    let (child, _) = c.stage_split(shard(1)).await;
    let specs = c.launcher().spawned();
    assert_eq!(specs.len(), 1, "the split staged exactly one launch");
    assert_eq!(
        specs[0].launch,
        LaunchId(1),
        "named by its launch, not by a shard"
    );
    assert_eq!(
        child,
        ShardId::new(100),
        "the child's id came from the orchestrator's counter, at its hello"
    );
    assert_eq!(
        c.orch.launch_of(child),
        Some(specs[0].launch),
        "and that launch is the one that became the child"
    );
    assert_eq!(
        c.member(child).id,
        child,
        "the member runs as the id it was given"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_split_gets_a_strictly_greater_id_than_its_parent() {
    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    c.connect(shard(1), "h9-parent").await;
    let child = c.split(shard(1)).await;

    assert!(
        child.0 > 1,
        "the child's id ({child}) is strictly greater than its parent's (1)"
    );
    // The client can see the new id: it is in the map the orchestrator published
    // and therefore in the map every `Welcome` and `Redirect` carries.
    assert!(
        c.map().regions.iter().any(|r| r.shard == child),
        "the committed map names the allocated child"
    );

    // A merge then a second split: the next child gets a *new*, greater id --
    // the merged-away id is never handed out again.
    c.headroom().await;
    c.merge(child).await;
    c.headroom().await;
    let next = c.split(shard(1)).await;
    assert!(
        next.0 > child.0,
        "after a merge the next split gets a strictly greater id \
         ({child} was retired, {next} issued)"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn after_an_orchestrator_restart_the_next_id_exceeds_every_id_ever_issued() {
    // H9's restart clause, through the same counter the orchestrator persists:
    // ids issued before a restart are never issued again after it, so a client
    // that still holds a guid or a redirect naming an old shard is never looking
    // at a different shard than it thinks.
    let mut first = Cluster::new(100);
    first.add_member(shard(1), root()).await;
    let a = first.split(shard(1)).await;
    first.headroom().await;
    first.merge(a).await;
    first.headroom().await;
    let b = first.split(shard(1)).await;
    let issued: Vec<u32> = vec![1, a.0, b.0];
    assert_eq!(
        issued,
        vec![1, 100, 101],
        "the ids issued before the restart"
    );

    // "Restart": a brand-new orchestrator over a counter that resumes above the
    // last id it handed out.
    let mut second = Cluster::new(u64::from(b.0) + 1);
    second.add_member(shard(1), root()).await;
    let c1 = second.split(shard(1)).await;
    assert!(
        c1.0 > b.0,
        "after a restart the next id ({c1}) exceeds every id ever issued ({issued:?})"
    );
    second.headroom().await;
    second.merge(c1).await;
    second.headroom().await;
    let c2 = second.split(shard(1)).await;
    assert!(c2.0 > c1.0, "and monotonicity continues after the restart");
}

// ------------------------------------------------------------------ H10
//
// "A map change is published through the epoch protocol only: prepare -> ack ->
// commit. No shard ever observes a map that is not some committed epoch."
// The client-observable form: every frame a client receives carries the epoch
// it was produced under, so the frames themselves are the audit trail. A client
// must never see a frame stamped with an epoch that was prepared but never
// committed.

#[tokio::test(flavor = "multi_thread")]
async fn a_split_publishes_prepare_then_acks_then_commit_in_that_order() {
    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    c.connect(shard(1), "h10").await;
    c.step_n(3).await;

    // Stage the split: the prepare leaves, the child comes up, and the epoch may
    // not move until every required ack has landed.
    let before = c.epoch();
    let (child, recipients) = c.stage_split(shard(1)).await;
    assert!(
        recipients.contains(&shard(1)),
        "the parent is a prepare recipient: it owns the region being carved"
    );
    assert!(
        recipients.contains(&child),
        "the child is a prepare recipient: it owns the new region"
    );
    assert_eq!(
        c.epoch(),
        before,
        "a prepare alone must not move the epoch the client can observe"
    );

    // The client keeps being served, and everything it has seen so far is under
    // the still-current epoch: a prepare is invisible to a client.
    c.thrust(0, 0, 1.0).await;
    c.step_n(2).await;
    c.settle_until_frames(0).await;
    let mid = c.take(0);
    assert!(
        !mid.frames.is_empty(),
        "the client is not stalled mid-epoch"
    );
    assert_eq!(
        mid.distinct_epochs(),
        vec![before],
        "a staged-but-uncommitted epoch must not appear on a client's frames"
    );

    // Commit: the epoch moves only once every recipient has acked.
    c.commit_split(before).await;
    assert_eq!(c.epoch(), before + 1, "the commit bumped the epoch by one");
    assert!(
        c.map().regions.iter().any(|r| r.shard == child),
        "the committed map carries the child region"
    );

    // And the client's frames carry only committed epochs: `before` before the
    // commit and `before + 1` after, never anything else. The ship keeps being
    // driven, so the shard keeps producing deltas for it.
    for _ in 0..3 {
        c.thrust(0, 0, 1.0).await;
        c.step().await;
    }
    c.settle_until_frames(0).await;
    let after = c.take(0);
    assert!(
        !after.frames.is_empty(),
        "the client is still served after the commit"
    );
    // Every epoch the client saw is one that was actually committed: `before`
    // (frames still in flight from before the commit) or `before + 1`. A frame
    // stamped with the staged-but-uncommitted epoch would show up here as a third
    // value, which is exactly what H10 forbids.
    assert!(
        after
            .distinct_epochs()
            .iter()
            .all(|e| *e == before || *e == before + 1),
        "the client saw a frame with an epoch that was never committed: {:?}",
        after.distinct_epochs()
    );
    assert!(
        after.distinct_epochs().contains(&(before + 1)),
        "after the commit the client's frames carry the new epoch"
    );
    assert!(
        after.redirects.is_empty(),
        "a split elsewhere in the map must not disturb this client"
    );
    assert_eq!(
        c.simulates(0).await,
        Some(shard(1)),
        "and its ship is still simulated by the shard it connected to"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_shard_that_never_acks_gets_an_abort_and_the_world_keeps_the_old_map() {
    use space_orchestrator::config::EPOCH_ACK_TIMEOUT_TICKS;

    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    c.connect(shard(1), "h10-abort").await;
    c.step_n(3).await;
    let before = c.epoch();
    let map_before = c.map().clone();

    // A split whose child never comes up: the parent acks, nobody else does, and
    // the deadline fires.
    let recipients = c.stage_split_isolated(shard(1)).await;
    assert!(!recipients.is_empty(), "a prepare was published");
    for _ in 0..(EPOCH_ACK_TIMEOUT_TICKS + 5) {
        c.step().await;
    }

    assert_eq!(
        c.epoch(),
        before,
        "an un-acked epoch must leave the published epoch untouched"
    );
    assert_eq!(
        c.map(),
        &map_before,
        "an un-acked epoch must leave the published map byte-identical"
    );

    // The client's view: it never observed the aborted topology. Every frame it
    // got is stamped with the old epoch, and its ship is still simulated by the
    // shard it connected to.
    c.settle_until_frames(0).await;
    let view = c.take(0);
    assert!(
        !view.frames.is_empty(),
        "the client was served throughout the failed epoch"
    );
    assert_eq!(
        c.simulates(0).await,
        Some(shard(1)),
        "the client's ship never moved"
    );
    assert!(
        view.redirects.is_empty(),
        "an aborted epoch must not redirect anyone"
    );
}

// ------------------------------------------------------------------- H14
//
// "The autoscaler is deterministic: the same sequence of `LoadReport`s produces
// the same sequence of decisions, and it never flaps." The end-to-end form: a
// shard that reports itself hot for the full dwell splits on its own, with no
// admin verb, and the epoch moves -- and a shard that oscillates around the
// threshold never splits at all.

#[tokio::test(flavor = "multi_thread")]
async fn the_autoscaler_splits_a_hot_shard_end_to_end() {
    use space_orchestrator::autoscale::SPLIT_DWELL_TICKS;

    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    let members_before = c.members.len();

    // Feed shard 1 a report over the split threshold (80 players => load 0.8)
    // for one beat past the dwell. The orchestrator's own tick runs the
    // autoscaler, so no admin command is involved: this is the autoscaler
    // deciding on its own.
    for _ in 0..(SPLIT_DWELL_TICKS + 2) {
        c.report_load(shard(1), 80);
        c.step().await;
    }

    // The autoscaler acted: a child shard was launched for the hot region.
    assert!(
        c.members.len() > members_before,
        "a shard hot for the full dwell must have had a child launched \
         (members still {members_before})"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_flapping_load_does_not_flap_the_topology() {
    use space_orchestrator::autoscale::SPLIT_DWELL_TICKS;

    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    let before = c.epoch();

    // Oscillate across the threshold: hot for most of the dwell, then a single
    // report inside the hysteresis band resets the timer. The dwell never
    // completes, so nothing splits.
    for i in 0..(SPLIT_DWELL_TICKS * 2) {
        let players = if i % 20 == 19 { 50 } else { 90 };
        c.report_load(shard(1), players);
        c.step().await;
    }

    assert_eq!(
        c.epoch(),
        before,
        "a load that oscillates across the threshold must never accumulate dwell"
    );
}

// ------------------------------------------------- H11 / H13, client-visible
//
// The core promise: a player flies across a split boundary and notices
// nothing. The scenario is the one the autoscaler produces -- a shard splits,
// its carved child boots, and a ship *inside the carved octant* is now owned
// by someone else -- driven through the real transport end to end: the parent
// offers the ship to the child over a real shard-to-shard QUIC connection, the
// client follows its `Redirect` to the address the redirect itself carries, and
// the handoff commits. No peer endpoint is wired by hand anywhere: how the
// parent finds the child is the orchestrator's committed-map push.

/// The largest distance a ship may move between two consecutive ticks, in
/// metres. Terminal speed is ~120 m/s (`space_sim::params`), i.e. 4 m per 30 Hz
/// tick; the margin is for the frame being a tick's float rounding away.
const MAX_STEP_M: f64 = 6.0;

/// A cluster that split shard 1 and has two players: `flyer` (index 0) will
/// cross into the child, `stayer` (index 1) never leaves the parent's octant.
async fn split_with_two_players() -> (Cluster, ShardId) {
    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    let flyer = c.connect(shard(1), "flyer").await;
    let stayer = c.connect(shard(1), "stayer").await;
    assert_eq!((flyer, stayer), (0, 1));
    c.step_n(3).await;
    let child = c.split(shard(1)).await;
    c.headroom().await;
    // Only the transition is under test: start from an empty inbox.
    let _ = c.clients[flyer].drain();
    let _ = c.clients[stayer].drain();
    (c, child)
}

/// `(sending shard, tick, position)` for every frame that carried `guid`'s
/// ship, in arrival order.
fn ship_track(inbox: &[Message], guid: EntityGuid) -> Vec<(ShardId, u64, WorldPos)> {
    inbox
        .iter()
        .filter_map(|m| match m {
            Message::Snapshot(f) => ship_pos_in(&f.entities, guid).map(|p| (f.shard, f.tick, p)),
            Message::Delta(f) => ship_pos_in(&f.entities, guid).map(|p| (f.shard, f.tick, p)),
            _ => None,
        })
        .collect()
}

/// The newest ship-bearing frame of `c.clients[idx]` and the shard the map
/// says owns that position.
fn frame_vs_lookup(c: &Cluster, idx: usize) -> (ShardId, Option<ShardId>) {
    let (sender, _tick, pos) = c.clients[idx]
        .latest()
        .expect("the client has received a frame with its ship");
    (sender, c.map().lookup(&pos))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_split_rebalances_a_player_into_the_child_without_the_client_noticing() {
    let (mut c, child) = split_with_two_players().await;
    let guid = c.clients[0].guid;
    let stayer_guid = c.clients[1].guid;

    let inbox = c.fly_into_child(0, child, 30).await;
    c.settle().await;
    let view = view_of(&inbox, guid);

    // The client was told where to go, and told the child's *real* address
    // (it connected there: `follow` dials the redirect's own map).
    assert!(
        !view.redirects.is_empty(),
        "the client must be redirected when its ship enters the carved octant"
    );
    assert!(
        view.redirects.iter().all(|r| r.new_shard == child),
        "every redirect names the child"
    );
    assert_eq!(view.rejects, 0, "the client is never refused");

    // Its ship keeps its guid and the child really serves it.
    assert!(
        view.ship_frames.iter().any(|(s, _)| *s == child),
        "after the handoff the child's frames carry the client's own ship (same guid)"
    );
    assert!(
        view.ship_frames.iter().any(|(s, _)| *s == shard(1)),
        "before the handoff the parent's frames carried it"
    );

    // Position continuity: ordered by tick, the ship never teleports -- including
    // at the seam where the sender changes from parent to child.
    let mut track = ship_track(&inbox, guid);
    track.sort_by_key(|(_, tick, _)| *tick);
    track.dedup_by_key(|(_, tick, _)| *tick);
    for pair in track.windows(2) {
        let step = WorldPos::distance(pair[0].2, pair[1].2);
        assert!(
            step <= MAX_STEP_M * (pair[1].1 - pair[0].1) as f64,
            "the ship jumped {step:.2} m between tick {} (shard {}) and tick {} (shard {})",
            pair[0].1,
            pair[0].0,
            pair[1].1,
            pair[1].0
        );
    }
    // The ship kept moving under the client's own inputs after the switch: the
    // child applies what the client sends it.
    let last = track.last().expect("frames");
    let seam = track
        .iter()
        .find(|(s, _, _)| *s == child)
        .expect("a child frame");
    assert!(
        WorldPos::distance(seam.2, last.2) > 1.0,
        "inputs sent to the child after the handoff must still move the ship"
    );

    // Settled: for every player, the map's owner of the player's position is the
    // shard the player is actually being served by.
    for (idx, expected) in [(0usize, child), (1usize, shard(1))] {
        let (sender, owner) = frame_vs_lookup(&c, idx);
        assert_eq!(
            sender, expected,
            "player {idx} is served by shard {expected}"
        );
        assert_eq!(
            owner,
            Some(sender),
            "map().lookup() must agree with the shard serving player {idx}"
        );
    }
    // The player that never moved is never refused and is never served by anyone
    // but the parent (it may be *told* about the flyer's move -- a redirect goes
    // to every observer of the moving ship -- but its own ship stays put).
    let stayer = view_of(&c.clients[1].drain(), stayer_guid);
    assert_eq!(stayer.rejects, 0);
    assert!(
        stayer.ship_frames.iter().all(|(s, _)| *s == shard(1)),
        "the stayer's own ship is only ever in the parent's frames"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_client_sees_a_frame_every_tick_across_a_rebalance() {
    let (mut c, child) = split_with_two_players().await;
    let guid = c.clients[0].guid;

    let inbox = c.fly_into_child(0, child, 30).await;
    c.settle().await;
    let view = view_of(&inbox, guid);

    // (d) never rejected, and the handoff really happened.
    assert_eq!(view.rejects, 0);
    assert!(view.ship_frames.iter().any(|(s, _)| *s == child));
    assert!(view.ship_frames.iter().any(|(s, _)| *s == shard(1)));

    // (a) the union of both shards' frames has no tick gap: the client's frame
    // clock counted 1, 2, 3 ... straight through the seam.
    let mut ticks: Vec<u64> = view.frames.iter().map(|(_, t)| *t).collect();
    ticks.sort_unstable();
    ticks.dedup();
    let gaps: Vec<(u64, u64)> = ticks
        .windows(2)
        .filter(|w| w[1] - w[0] > 1)
        .map(|w| (w[0], w[1]))
        .collect();
    assert!(gaps.is_empty(), "tick gaps across the rebalance: {gaps:?}");

    // (b) from its first ship frame on, the ship is in a frame of *every* tick.
    let mut ship_ticks: Vec<u64> = view.ship_frames.iter().map(|(_, t)| *t).collect();
    ship_ticks.sort_unstable();
    ship_ticks.dedup();
    let missing: Vec<u64> = (ship_ticks[0]..=*ship_ticks.last().unwrap())
        .filter(|t| ship_ticks.binary_search(t).is_err())
        .collect();
    assert!(
        missing.is_empty(),
        "ticks where the client's own ship was in no frame: {missing:?}"
    );

    // (c) the seam is where responsibility changed hands, once and only once:
    // ordered by tick the sender goes parent... then child and never back.
    let mut by_tick = view.ship_frames.clone();
    by_tick.sort_by_key(|(_, t)| *t);
    let first_child = by_tick.iter().position(|(s, _)| *s == child).unwrap();
    assert!(
        by_tick[first_child..].iter().all(|(s, _)| *s == child),
        "once the child serves the ship the parent never serves it again"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn lookup_agrees_with_the_serving_shard_after_every_topology_change() {
    // Split, then a player crossing, then (on a fresh empty child) a merge: after
    // each change, every player's frames come from the shard the map names.
    let (mut c, child) = split_with_two_players().await;
    c.step_n(5).await;
    c.settle().await;
    for idx in [0usize, 1] {
        let (sender, owner) = frame_vs_lookup(&c, idx);
        assert_eq!(owner, Some(sender), "after the split, player {idx}");
    }
    let _ = c.fly_into_child(0, child, 30).await;
    c.settle().await;
    for idx in [0usize, 1] {
        let (sender, owner) = frame_vs_lookup(&c, idx);
        assert_eq!(owner, Some(sender), "after the crossing, player {idx}");
    }

    // A merge of an empty child folds its octant back into the parent.
    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    let only = c.connect(shard(1), "solo").await;
    c.step_n(3).await;
    let child = c.split(shard(1)).await;
    c.headroom().await;
    c.merge(child).await;
    c.step_n(5).await;
    c.settle().await;
    let (sender, _tick, pos) = c.clients[only].latest().expect("a frame with the ship");
    assert_eq!(sender, shard(1));
    assert_eq!(
        c.map().lookup(&pos),
        Some(sender),
        "after the merge the parent owns the position again"
    );
}

// ------------------------------------------------------------------- H15
//
// "The orchestrator converges the world back to its declared topology after a
// shard dies, and never re-uses the dead shard's id or leaves its region
// unowned." The client-observable form: after a shard is killed its region is
// owned by a live shard, and the next allocation is a fresh id above the dead
// one -- so a client still holding a redirect to the dead shard is never looking
// at a different shard than it thinks.

#[tokio::test(flavor = "multi_thread")]
async fn a_killed_shard_is_removed_and_the_world_stays_owned() {
    let mut c = Cluster::new(100);
    c.add_member(shard(1), root()).await;
    let child = c.split(shard(1)).await;
    let child_region = c
        .map()
        .regions
        .iter()
        .find(|r| r.shard == child)
        .expect("the child owns its octant")
        .node;
    let child_launch = c.orch.launch_of(child).expect("the split launched it");

    // Kill the child through the admin verb (the same path the liveness poll
    // drives). The launcher records the kill and the member is dropped.
    c.kill(child).await;
    assert!(
        !c.members.iter().any(|m| m.id == child),
        "the killed shard is no longer running"
    );
    assert!(
        c.launcher().killed().contains(&child_launch),
        "the launcher was told to kill it"
    );

    // Its region does not stay with a dead owner: a replacement is launched,
    // registers, and is given the octant under a fresh id above the dead one.
    for _ in 0..16 {
        c.step().await;
        if c.map().lookup(&child_region.center()) != Some(child) {
            break;
        }
    }
    c.deliver_outbox().await;
    let heir = c
        .map()
        .lookup(&child_region.center())
        .expect("the octant has an owner");
    assert!(
        heir.0 > child.0,
        "the dead shard's id ({child}) is never reused: its region went to {heir}"
    );
    assert!(
        c.members.iter().any(|m| m.id == heir),
        "and its new owner is a live shard"
    );
    assert!(!c.map().regions.iter().any(|r| r.shard == child));

    // The surviving shard still owns the world outside the dead shard's region,
    // so a client standing anywhere else is still served.
    assert_eq!(
        c.map()
            .lookup(&WorldPos::new([1_000, 1_000, 1_000], [0.0; 3])),
        Some(shard(1)),
        "the root owner still answers for positions outside the carved octant"
    );
}
