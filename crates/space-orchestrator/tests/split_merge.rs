//! Phase 3 slice-3 integration tests: the orchestrator event loop and the
//! autoscaler → topology → epoch wiring (plan P3.9/P3.10).
//!
//! The orchestrator is a synchronous state machine driven by [`Orchestrator::tick`]
//! (mirroring `Shard::tick`), so the split/merge decisions are proven
//! deterministically — dwell timers advance by ticks, never by sleeping. The
//! loopback tests then prove the same wiring over real QUIC: real shards connect
//! to the control endpoint, ack a prepared epoch, and apply the committed map.
//!
//! The full client-visible H9–H15 suite (players crossing a split boundary with
//! no tick gap) is slice 4; the Docker launcher is slice 5.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use space_orchestrator::autoscale::{SPLIT_DWELL_TICKS, SPLIT_THRESHOLD};
use space_orchestrator::config::OrchestratorConfig;
use space_orchestrator::control::ControlPlane;
use space_orchestrator::epoch::EpochPublisher;
use space_orchestrator::ids::{MemoryCounterSink, PersistentShardIdAllocator};
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::launcher::{LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};
use space_orchestrator::orchestrator::Orchestrator;
use space_protocol::control::LoadReport;
use space_protocol::shard::{ShardId, ShardMap, ShardRegion};
use space_protocol::Message;
use space_world::NodeRef;

fn addr(n: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), n)
}

fn shard(n: u32) -> ShardId {
    ShardId::new(n)
}

/// A launcher that records spawns/kills and never touches a process.
#[derive(Debug, Default)]
struct RecordingLauncher {
    spawned: std::sync::Mutex<Vec<LaunchId>>,
    killed: std::sync::Mutex<Vec<LaunchId>>,
}

impl Launcher for RecordingLauncher {
    async fn spawn(
        &self,
        spec: SpawnSpec,
    ) -> Result<ShardHandle, space_orchestrator::launcher::LaunchError> {
        self.spawned.lock().unwrap().push(spec.launch);
        Ok(ShardHandle {
            launch: spec.launch,
            shard: None,
            endpoint: launched_at(spec.launch),
            process: "rec".into(),
        })
    }
    async fn kill(
        &self,
        launch: LaunchId,
    ) -> Result<(), space_orchestrator::launcher::LaunchError> {
        self.killed.lock().unwrap().push(launch);
        Ok(())
    }
    async fn poll(
        &self,
        _launch: LaunchId,
    ) -> Result<Poll, space_orchestrator::launcher::LaunchError> {
        Ok(Poll::Alive)
    }
}

/// Where the shard of `launch` listens in these tests.
fn launched_at(launch: LaunchId) -> SocketAddr {
    addr(9_500 + u16::try_from(launch.0).expect("a small launch number"))
}

/// Play the launcher for `spec`: its shard says hello — with no id and no
/// region, like every shard — and the launcher reports its handle. Returns
/// the id the shard was welcomed as.
fn register(orch: &mut Orchestrator<RecordingLauncher>, spec: &SpawnSpec) -> ShardId {
    let endpoint = launched_at(spec.launch);
    let welcome = orch
        .handle_shard_hello(&space_protocol::ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::NONE,
            endpoint,
            region: None,
            client_endpoint: None,
            secret: String::new(),
        })
        .expect("the launched shard is welcomed");
    orch.note_launch_result(
        spec.launch,
        Ok(ShardHandle {
            launch: spec.launch,
            shard: None,
            endpoint,
            process: "rec".into(),
        }),
    );
    welcome.shard
}

/// An orchestrator over `plane` in which every shard of the map holds a
/// control link (a map change is only made when every one of them can
/// acknowledge it).
fn orchestrator(plane: ControlPlane) -> Orchestrator<RecordingLauncher> {
    let mut orch = Orchestrator::new(plane, Arc::new(RecordingLauncher::default()));
    let owners: Vec<ShardId> = orch.map().regions.iter().map(|r| r.shard).collect();
    orch.note_connected(owners);
    orch
}

/// A control plane whose root (depth 40) is owned by `root_owner`.
fn plane_with_root(root_owner: ShardId) -> ControlPlane {
    let config = OrchestratorConfig::default();
    let map = ShardMap::new(
        1,
        vec![ShardRegion::new(
            config.root_node(),
            root_owner,
            addr(7_001),
        )],
    );
    let ids = PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
    ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into())
}

/// Play the end of a drain: the merged `gone` reports that it is empty, is
/// told to stop ([`Message::ShardRetire`]) and — nobody closes its link in
/// this fixture — is stopped when the grace has passed.
fn drain_and_stop(orch: &mut Orchestrator<RecordingLauncher>, gone: ShardId) {
    orch.tick();
    orch.handle_load_report(&cold_report(gone));
    orch.tick();
    let told = orch
        .drain_outbox()
        .into_iter()
        .any(|(to, m)| to == gone && matches!(m, Message::ShardRetire(_)));
    assert!(told, "the drained shard is told to stop");
    assert!(
        orch.take_pending_kill().is_empty(),
        "and left to end by itself"
    );
    for _ in 0..=space_orchestrator::config::RETIRE_GRACE_TICKS {
        orch.tick();
    }
    assert_eq!(
        orch.take_pending_kill(),
        vec![gone],
        "stopped after the grace"
    );
    assert!(orch.draining_shards().is_empty());
}

fn hot_report(s: ShardId) -> LoadReport {
    LoadReport {
        shard: s,
        players: 90,
        tick_p95_ms: 1.0,
        load: 0.0,
        ..Default::default()
    }
}

/// A cold **and empty** shard: only a shard with no players is merged away (a
/// merge removes the shard, and nothing hands its players to anyone).
fn cold_report(s: ShardId) -> LoadReport {
    LoadReport {
        shard: s,
        players: 0,
        tick_p95_ms: 1.0,
        load: 0.0,
        ..Default::default()
    }
}

/// Drive the autoscaler to a split of the root shard — it asks for an octant
/// and a shard is launched, which registers and is given it — then ack every
/// prepare recipient so the epoch commits. Returns the orchestrator's new map
/// epoch.
fn drive_split_to_commit(orch: &mut Orchestrator<RecordingLauncher>, hot: ShardId) -> u64 {
    // The metric must actually be hot.
    assert!(space_orchestrator::autoscale::load_metric(&hot_report(hot)) >= SPLIT_THRESHOLD);
    let mut prepared = false;
    for _ in 0..(SPLIT_DWELL_TICKS + 5) {
        orch.handle_load_report(&hot_report(hot));
        orch.tick();
        for spec in orch.take_pending_spawn() {
            register(orch, &spec);
        }
        let outbox = orch.drain_outbox();
        if !outbox.is_empty() {
            prepared = true;
            // Collect the recipients of the EpochPrepare and ack them all.
            let recipients: Vec<(ShardId, u64)> = outbox
                .iter()
                .filter_map(|(to, m)| match m {
                    Message::EpochPrepare(p) => Some((*to, p.epoch)),
                    _ => None,
                })
                .collect();
            assert!(
                outbox
                    .iter()
                    .any(|(_, m)| matches!(m, Message::EpochPrepare(_))),
                "the outbox carried a prepare"
            );
            for (r, epoch) in recipients {
                orch.handle_epoch_ack(r, epoch);
            }
            break;
        }
    }
    assert!(prepared, "a split prepare was emitted");
    orch.map().epoch
}

#[test]
fn a_hot_shard_splits_the_epoch_and_a_shard_is_launched_for_the_child() {
    let mut orch = orchestrator(plane_with_root(shard(1)));
    assert_eq!(orch.map().epoch, 1);
    assert_eq!(orch.map().regions.len(), 1);

    let epoch_after = drive_split_to_commit(&mut orch, shard(1));
    assert_eq!(epoch_after, 2, "the split bumped the epoch to 2");

    // The new map has the child region appended (additive split, plan F5).
    assert_eq!(orch.map().regions.len(), 2, "the child region was appended");
    let child_owner = orch
        .map()
        .regions
        .iter()
        .max_by_key(|r| r.node.depth)
        .expect("a child region")
        .shard;
    assert_ne!(child_owner, shard(1), "the child is a new shard");

    // The child is the shard that was launched for the split: it registered
    // without an id and was given this one.
    assert!(
        orch.launch_of(child_owner).is_some(),
        "the split launched the shard that became {child_owner}"
    );
    assert!(orch.take_pending_spawn().is_empty(), "one launch, no more");
}

#[test]
fn a_split_stages_a_spawn_and_the_commit_publishes_the_child_region() {
    let mut orch = orchestrator(plane_with_root(shard(1)));
    // Run the autoscaler to the split; the staged launch's shard registers
    // and is told its id, before anything is acked.
    let mut staged_child: Option<ShardId> = None;
    for _ in 0..(SPLIT_DWELL_TICKS + 5) {
        orch.handle_load_report(&hot_report(shard(1)));
        orch.tick();
        let spawns = orch.take_pending_spawn();
        if !spawns.is_empty() {
            assert!(
                orch.drain_outbox().is_empty(),
                "nothing is proposed before the launched shard has registered"
            );
            assert_eq!(orch.map().epoch, 1);
            staged_child = Some(register(&mut orch, &spawns[0]));
            // Ack the prepare so the epoch commits.
            let recipients: Vec<(ShardId, u64)> = orch
                .drain_outbox()
                .into_iter()
                .filter_map(|(to, m)| match m {
                    Message::EpochPrepare(p) => Some((to, p.epoch)),
                    _ => None,
                })
                .collect();
            for (r, epoch) in recipients {
                orch.handle_epoch_ack(r, epoch);
            }
            break;
        }
    }
    let child = staged_child.expect("a child spawn was staged");
    assert_eq!(orch.map().epoch, 2);
    assert!(
        orch.map().regions.iter().any(|r| r.shard == child),
        "the committed map carries the child region"
    );
}

#[test]
fn a_cold_child_merges_the_epoch_and_stages_a_kill() {
    // Root (depth 40) owned by 1, and one child octant owned by the shard the
    // autoscaler's own split launched: the autoscaler takes away what the
    // orchestrator started, and nothing else (a shard put into the map by
    // hand, as this test used to have it, stays until an operator merges it).
    let mut orch = orchestrator(plane_with_root(shard(1)));
    assert_eq!(drive_split_to_commit(&mut orch, shard(1)), 2);
    let _ = orch.drain_outbox();
    let child = orch
        .map()
        .regions
        .iter()
        .map(|r| r.shard)
        .find(|owner| *owner != shard(1))
        .expect("the split's child");
    assert!(
        orch.launch_of(child).is_some(),
        "launched by the orchestrator"
    );

    // Both go cold. The cooldown after the split (900 ticks) and the merge
    // dwell (1800) are long; drive them. The shard that takes the region
    // back reports too: without its report nothing is merged into it.
    let mut staged_kill = false;
    let patience = space_orchestrator::autoscale::SCALE_COOLDOWN_TICKS
        + space_orchestrator::autoscale::MERGE_DWELL_TICKS
        + 5;
    for _ in 0..patience {
        orch.handle_load_report(&cold_report(child));
        orch.handle_load_report(&cold_report(shard(1)));
        orch.tick();
        assert!(
            orch.take_pending_kill().is_empty(),
            "nothing is killed before the merge has committed"
        );
        let recipients: Vec<(ShardId, u64)> = orch
            .drain_outbox()
            .into_iter()
            .filter_map(|(to, m)| match m {
                Message::EpochPrepare(p) => Some((to, p.epoch)),
                _ => None,
            })
            .collect();
        if recipients.is_empty() {
            continue;
        }
        // The merge was proposed: the child is still running (it has to ack).
        assert_eq!(orch.pending_merge(), Some(child));
        for (r, epoch) in recipients {
            orch.handle_epoch_ack(r, epoch);
        }
        // Committed: the cold child owns nothing now and drains; it is
        // stopped once it has said that it is empty (the premise before
        // phase 6: it was killed at the commit).
        assert!(orch.take_pending_kill().is_empty(), "a commit stops nobody");
        assert_eq!(orch.draining_shards(), vec![child]);
        drain_and_stop(&mut orch, child);
        staged_kill = true;
        break;
    }
    assert!(staged_kill, "a merge staged a kill");
    assert_eq!(orch.map().epoch, 3, "the merge bumped the epoch");
    assert_eq!(orch.map().regions.len(), 1, "the child region was removed");
    assert_eq!(
        orch.map().regions[0].shard,
        shard(1),
        "the parent owns the region"
    );
}

/// The autoscaler's merge of a **carved cell**: the cold shard owns one cell
/// (depth 64) of the depth-40 root, whose node is twenty-four levels above it
/// and the only region around it. The merge is proposed, goes to both shards,
/// commits, gives the cell back to the root's owner and stages the kill.
///
/// Mutation this catches: a merge target that has to be the parent *node* —
/// the autoscaler's decision would be refused on every dwell, for ever.
#[test]
fn a_cold_carved_cell_merges_back_into_the_roots_owner() {
    let config = OrchestratorConfig::default();
    let root = config.root_node();
    let k = root.center().cell;
    let cell = NodeRef::new(64, [k[0] - 1, k[1], k[2] - 1]);
    let inside = cell.center();
    // The cell is carved for a shard the orchestrator launches (the
    // autoscaler merges away no other), through the epoch protocol.
    let mut orch = orchestrator(plane_with_root(shard(1)));
    orch.begin_carve(cell).expect("the cell can be carved");
    let spec = orch
        .take_pending_spawn()
        .pop()
        .expect("a shard is launched");
    let carved = register(&mut orch, &spec);
    for (to, message) in orch.drain_outbox() {
        if let Message::EpochPrepare(prepare) = message {
            orch.handle_epoch_ack(to, prepare.epoch);
        }
    }
    let _ = orch.drain_outbox();
    assert_eq!(orch.map().epoch, 2, "the carve committed");
    assert_eq!(orch.map().lookup(&inside), Some(carved));
    space_orchestrator::topology::validate(orch.map())
        .expect("a cell under the root is a valid map");

    let mut merged = false;
    let patience = space_orchestrator::autoscale::SCALE_COOLDOWN_TICKS
        + space_orchestrator::autoscale::MERGE_DWELL_TICKS
        + 5;
    for _ in 0..patience {
        orch.handle_load_report(&cold_report(carved));
        orch.handle_load_report(&cold_report(shard(1)));
        orch.tick();
        assert!(
            orch.take_pending_kill().is_empty(),
            "nothing is killed before the merge has committed"
        );
        let recipients: Vec<(ShardId, u64)> = orch
            .drain_outbox()
            .into_iter()
            .filter_map(|(to, m)| match m {
                Message::EpochPrepare(p) => {
                    assert_eq!(
                        p.changes,
                        vec![(cell, shard(1))],
                        "the cell reverts to shard 1"
                    );
                    Some((to, p.epoch))
                }
                _ => None,
            })
            .collect();
        if recipients.is_empty() {
            continue;
        }
        assert_eq!(recipients, vec![(shard(1), 3), (carved, 3)]);
        assert_eq!(orch.pending_merge(), Some(carved));
        assert_eq!(orch.map().lookup(&inside), Some(carved), "not yet");
        for (r, epoch) in recipients {
            orch.handle_epoch_ack(r, epoch);
        }
        assert!(orch.take_pending_kill().is_empty(), "a commit stops nobody");
        drain_and_stop(&mut orch, carved);
        merged = true;
        break;
    }
    assert!(merged, "the cold cell was merged");
    assert_eq!(orch.map().epoch, 3);
    assert_eq!(orch.map().regions.len(), 1, "the cell's region was removed");
    assert_eq!(
        orch.map().lookup(&inside),
        Some(shard(1)),
        "the cell fell back to the root's owner"
    );
}

#[test]
fn an_unacked_prepare_aborts_and_the_map_is_unchanged() {
    let mut orch = orchestrator(plane_with_root(shard(1)));
    // Drive to the split — its shard registers — but ack nobody.
    let mut prepared_epoch = None;
    for _ in 0..(SPLIT_DWELL_TICKS + 5) {
        orch.handle_load_report(&hot_report(shard(1)));
        orch.tick();
        for spec in orch.take_pending_spawn() {
            register(&mut orch, &spec);
        }
        let outbox = orch.drain_outbox();
        if outbox
            .iter()
            .any(|(_, m)| matches!(m, Message::EpochPrepare(_)))
        {
            prepared_epoch = Some(orch.map().epoch);
            break;
        }
    }
    assert_eq!(
        prepared_epoch,
        Some(1),
        "the published map is still epoch 1 mid-prepare"
    );
    // Let the ack deadline pass: the prepare aborts, the map is untouched.
    for _ in 0..(space_orchestrator::config::EPOCH_ACK_TIMEOUT_TICKS + 5) {
        orch.handle_load_report(&hot_report(shard(1)));
        orch.tick();
        let _ = orch.drain_outbox();
    }
    assert_eq!(
        orch.map().epoch,
        1,
        "an un-acked prepare leaves the map byte-identical"
    );
}

#[tokio::test]
async fn a_shard_registers_over_loopback_and_gets_its_id_and_map() {
    // The async control front end + the shard-side OrchestratorLink, over real
    // QUIC: a shard sends a ShardHello with no id and receives a ShardWelcome
    // carrying an allocated id, a region and the current map (closes Phase-2
    // MAJOR 3: the orchestrator owns allocation).
    use space_orchestrator::control::{ControlServer, Inbound};
    use space_protocol::ShardHello;
    use space_server::net::OrchestratorLink;

    let server = Arc::new(ControlServer::bind(addr(0)).expect("bind control"));
    let bound = server.local_addr().expect("bound");
    let _accept = Arc::clone(&server).spawn_accept();

    // A minimal responder: answer each hello from a control plane.
    let responder_plane = std::sync::Arc::new(std::sync::Mutex::new({
        // The binary reserves the root owner at startup; the first anonymous
        // shard adopts it rather than being allocated a fresh id.
        let mut plane = plane_with_root(shard(1));
        plane.set_unclaimed_root_owner(shard(1));
        plane
    }));
    {
        let server = Arc::clone(&server);
        let plane = Arc::clone(&responder_plane);
        tokio::spawn(async move {
            while let Some(inbound) = server.next_inbound().await {
                if let Inbound::Hello {
                    hello,
                    reply,
                    verdict,
                    ..
                } = inbound
                {
                    let welcome = plane
                        .lock()
                        .unwrap()
                        .handle_shard_hello(&hello)
                        .expect("welcome");
                    let admitted = welcome.shard;
                    let _ = reply.send(Message::ShardWelcome(welcome)).await;
                    // The verdict is what opens the connection: only then is
                    // what was queued for it — the welcome — written.
                    let _ = verdict.send(Ok(admitted));
                }
            }
        });
    }

    let hello = ShardHello {
        cluster: "space-local".into(),
        shard: ShardId::NONE,
        endpoint: addr(9_100),
        region: None,
        client_endpoint: None,
        secret: String::new(),
    };
    let (welcome, _conn) = OrchestratorLink::handshake(bound, hello)
        .await
        .expect("handshake over loopback");
    assert_eq!(
        welcome.shard,
        shard(1),
        "the first anonymous shard adopts the reserved root owner"
    );
    assert_eq!(welcome.region.shard, welcome.shard);
    assert_eq!(
        welcome.shard_map.epoch, 1,
        "the welcome carries the current map"
    );
    assert_eq!(welcome.peer_token, "secret");
    server.close();
}

/// Run the orchestrator's real loop until `done` resolves (20 s at most), then
/// a few beats more, so the loop is not stopped in the middle of delivering
/// what the last beat produced.
async fn run_the_loop_until<T>(
    orchestrator: &mut Orchestrator<LocalLauncher>,
    server: &Arc<space_orchestrator::control::ControlServer>,
    what: &str,
    done: impl std::future::Future<Output = T>,
) -> T {
    use std::time::Duration;
    let settled = async {
        let value = done.await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        value
    };
    tokio::select! {
        ended = orchestrator.run_with(Arc::clone(server)) => panic!("the loop ended: {ended:?}"),
        value = tokio::time::timeout(Duration::from_secs(20), settled) => {
            value.unwrap_or_else(|_| panic!("timed out waiting until {what}"))
        }
    }
}

/// `docs/plan-sharding.md` §1, end to end: one **cell** (depth 64) is carved
/// out of the depth-40 root for a shard the local launcher starts, the epoch
/// commits, and the merge that undoes it gives the cell back to the root's
/// owner and stops the shard.
///
/// Everything in between is production code: `Orchestrator::run_with` over the
/// QUIC control endpoint, a root shard booted by `start_orchestrated` (what
/// `space-server --orchestrator` runs) and the carved cell's shard booted by
/// [`LocalLauncher`], which registers with neither id nor region and is given
/// both. The test only asks for the carve and the merge, and — because no
/// shard sends load reports yet — says on the shard's behalf that it is empty;
/// the loop is stopped for those calls and resumed.
///
/// Mutations this catches, each on its own: a `validate` that refuses a region
/// below depth 60 (the carve is never proposed); a merge that needs the parent
/// *node* to be a region (the merge is refused). Who is sent the prepare is
/// pinned where it can be seen exactly: `epoch::tests` and
/// `orchestrator::tests` (here a shard that missed the prepare would still
/// catch up from the map push behind the commit).
#[tokio::test(flavor = "multi_thread")]
async fn a_cell_is_carved_with_the_local_launcher_committed_and_merged_back() {
    use space_orchestrator::ops::OperatorCommand;
    use space_server::orchestrated::{start_orchestrated, OrchestratedOptions};
    use std::time::Duration;

    // Port 0 everywhere: the loop resolves the port the launched shard dials.
    let config = OrchestratorConfig {
        bind: addr(0),
        ..OrchestratorConfig::default()
    };
    let cluster = config.cluster.clone();
    let root_node = config.root_node();
    assert_eq!(root_node.depth, 40);
    let map = ShardMap::new(1, vec![ShardRegion::new(root_node, shard(1), addr(0))]);
    let ids = PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
    let mut control = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
    // As `boot` does: the root's id is reserved, the first shard adopts it.
    control.set_unclaimed_root_owner(shard(1));
    let launcher = Arc::new(LocalLauncher::new());
    let mut orchestrator = Orchestrator::new(control, Arc::clone(&launcher));
    let server = orchestrator.bind().expect("bind control");
    let control_addr = server.local_addr().expect("control port");

    // --- the root shard registers.
    let root = run_the_loop_until(
        &mut orchestrator,
        &server,
        "the root shard has registered",
        start_orchestrated(
            OrchestratedOptions {
                shard: ShardId::NONE,
                bind: addr(0),
                advertise: None,
                claim_region: None,
                cluster,
                client_endpoint: None,
                secret: String::new(),
            },
            control_addr,
        ),
    )
    .await
    .expect("the root shard registers");
    assert_eq!((root.shard, root.region), (shard(1), root_node));

    // --- carve the cell next to the one the scene's spawn point lies in: of
    // the four cells that meet at the root's centre, the one towards -z.
    let k = root_node.center().cell;
    let cell = NodeRef::new(64, [k[0], k[1], k[2] - 1]);
    let inside = cell.center();
    assert_eq!(orchestrator.map().lookup(&inside), Some(shard(1)));
    let wanted = orchestrator
        .begin_carve(cell)
        .expect("the carve is asked for");
    assert_eq!((wanted.node, wanted.from), (cell, shard(1)));
    assert_eq!(orchestrator.wanted_node(), Some(cell));
    assert_eq!(orchestrator.map().epoch, 1, "asked for is not published");
    // The id of the shard that will own the cell: the next one (H9). It is
    // allocated when that shard registers, not before.
    let child = shard(100);

    // --- the loop carries it out: launch, the shard's hello (no id, no
    // region), the id, the prepare to the root shard and the new one — which
    // is connected by then — both acks, commit, map push. Done when the root
    // shard runs the new map with both regions at real addresses.
    let root_map_when = |epoch: u64, regions: usize| {
        let state = Arc::clone(&root.state);
        async move {
            loop {
                {
                    let shard = state.lock().await;
                    let map = shard.shard_map();
                    if map.epoch == epoch
                        && map.regions.len() == regions
                        && map.regions.iter().all(|r| r.endpoint.port() != 0)
                    {
                        return map.clone();
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    };
    let root_map = run_the_loop_until(
        &mut orchestrator,
        &server,
        "the carve has committed (it aborts after 3 s if the new shard never acks)",
        root_map_when(2, 2),
    )
    .await;
    assert_eq!(orchestrator.map().epoch, 2, "the carve committed");
    assert_eq!(
        orchestrator.map(),
        &root_map,
        "the root shard runs the published map"
    );
    space_orchestrator::topology::validate(orchestrator.map()).expect("valid (H11)");
    let carved = orchestrator
        .map()
        .regions
        .iter()
        .find(|r| r.shard == child)
        .expect("the cell's region is published")
        .clone();
    assert_eq!(
        carved.node, cell,
        "one cell, twenty-four levels below the root"
    );
    assert_eq!(root_map.lookup(&inside), Some(child));
    assert_eq!(
        root_map.lookup(&root_node.center()),
        Some(shard(1)),
        "the cell the spawn point lies in is still the root shard's"
    );
    assert!(orchestrator.take_pending_kill().is_empty());

    // The cell's shard is alive, serves at the published address and follows
    // the committed map.
    let child_launch = launcher
        .launch_of(child)
        .await
        .expect("the launcher started the cell's shard");
    assert_eq!(orchestrator.launch_of(child), Some(child_launch));
    assert_eq!(
        launcher.poll(child_launch).await.expect("poll"),
        Poll::Alive
    );
    // (Asked as a peer shard asks, with the cluster's peer token: a shard
    // answers a peer with its own id and map. A *player* who dials the cell's
    // shard is sent to the one that owns the spawn point — checked below.)
    let client = space_net::client_endpoint().expect("client endpoint");
    let mut child_map = None;
    for _ in 0..100 {
        let (conn, welcome) = space_net::connect(
            &client,
            carved.endpoint,
            space_server::net::SERVER_NAME,
            &space_server::net::peer_hello_with(shard(1), "secret"),
        )
        .await
        .expect("the cell's shard answers at the address the map names");
        assert_eq!(welcome.shard, child);
        conn.close(0, "probe");
        if welcome.shard_map.epoch == 2 {
            child_map = Some(welcome.shard_map);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let child_map = child_map.expect("the cell's shard installs the committed map");
    assert_eq!(child_map.lookup(&inside), Some(child));
    assert_eq!(child_map.lookup(&root_node.center()), Some(shard(1)));
    // A new player starts on the spawn point, which is still the root
    // shard's: the cell's shard sends a player's hello there.
    let joined = space_net::join(
        &client,
        carved.endpoint,
        space_server::net::SERVER_NAME,
        &space_net::hello("after-the-carve"),
    )
    .await
    .expect("the cell's shard answers a player");
    let space_net::Joined::Elsewhere(redirect) = joined else {
        panic!("a player was let in by a shard that does not own the spawn point: {joined:?}");
    };
    assert_eq!(redirect.shard, shard(1));
    assert_eq!(redirect.shard_map, child_map);

    // --- merge it back. Not before the minimum interval between two commits
    // has passed, and not before the shard is known to be empty: no shard
    // sends load reports yet, so the test says so on its behalf.
    let empty = LoadReport {
        shard: child,
        players: 0,
        tick_p95_ms: 1.0,
        load: 0.0,
        ..Default::default()
    };
    let mut refusal = Some(String::new());
    for _ in 0..40 {
        orchestrator.handle_load_report(&empty);
        refusal = orchestrator.merge_refusal(child);
        if refusal.is_none() {
            break;
        }
        run_the_loop_until(
            &mut orchestrator,
            &server,
            "some time has passed",
            tokio::time::sleep(Duration::from_millis(300)),
        )
        .await;
    }
    assert_eq!(refusal, None, "an empty carved cell can be merged");
    let reply = orchestrator.handle_operator(&OperatorCommand::Merge {
        shard: child,
        expected_epoch: Some(2),
    });
    assert!(reply.accepted, "{reply:?}");
    assert_eq!(orchestrator.pending_merge(), Some(child));
    assert_eq!(
        launcher.poll(child_launch).await.expect("poll"),
        Poll::Alive
    );

    // The loop carries it out: prepare to both shards, acks, commit, push, and
    // only then the kill. Done when the root shard runs the map without the
    // cell and the launcher has stopped the cell's shard.
    let stopped = {
        let launcher = Arc::clone(&launcher);
        let merged = root_map_when(3, 1);
        async move {
            let map = merged.await;
            while launcher.poll(child_launch).await.expect("poll") != Poll::Dead {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            map
        }
    };
    let root_map = run_the_loop_until(
        &mut orchestrator,
        &server,
        "the merge has committed and the cell's shard is stopped",
        stopped,
    )
    .await;
    assert_eq!(orchestrator.map().epoch, 3, "the merge committed");
    assert_eq!(orchestrator.map(), &root_map);
    assert_eq!(
        orchestrator.map().regions.len(),
        1,
        "the cell's region is gone"
    );
    assert_eq!(
        root_map.lookup(&inside),
        Some(shard(1)),
        "the cell fell back to the root's owner"
    );
    assert_eq!(orchestrator.pending_merge(), None);
    assert_eq!(launcher.poll(child_launch).await.expect("poll"), Poll::Dead);
    assert_eq!(orchestrator.launch_of(child), None);

    root.abort();
    server.close();
}

#[tokio::test]
async fn the_local_launcher_spawns_a_reachable_child_over_loopback() {
    // The launcher half of a split: a spawned shard answers a real client.
    // Detached — there is no orchestrator here for it to register with; the
    // registering launch is proven end to end in `boot_loopback.rs`.
    let launcher = LocalLauncher::detached();
    let spec = SpawnSpec {
        launch: LaunchId(2),
        cluster: "space-local".into(),
        control_addr: addr(7_900),
        host_port: 0,
        secret: String::new(),
    };
    let handle = launcher.spawn(spec).await.expect("spawn");
    let endpoint = space_net::client_endpoint().expect("client endpoint");
    let (conn, welcome) = space_net::connect(
        &endpoint,
        handle.endpoint,
        space_server::net::SERVER_NAME,
        &space_net::hello("player"),
    )
    .await
    .expect("the spawned child answers a client");
    assert_eq!(
        welcome.shard,
        shard(2),
        "a detached shard runs as its launch's number"
    );
    assert!(!conn.is_closed());
    assert_eq!(launcher.poll(LaunchId(2)).await.unwrap(), Poll::Alive);
    launcher.kill(LaunchId(2)).await.expect("kill");
}
