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
use space_orchestrator::launcher::{Launcher, Poll, SpawnSpec};
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
    spawned: std::sync::Mutex<Vec<ShardId>>,
    killed: std::sync::Mutex<Vec<ShardId>>,
}

impl Launcher for RecordingLauncher {
    async fn spawn(
        &self,
        spec: SpawnSpec,
    ) -> Result<space_orchestrator::launcher::ShardHandle, space_orchestrator::launcher::LaunchError>
    {
        self.spawned.lock().unwrap().push(spec.shard);
        Ok(space_orchestrator::launcher::ShardHandle {
            shard: spec.shard,
            endpoint: spec.region.endpoint,
            process: "rec".into(),
        })
    }
    async fn kill(&self, shard: ShardId) -> Result<(), space_orchestrator::launcher::LaunchError> {
        self.killed.lock().unwrap().push(shard);
        Ok(())
    }
    async fn poll(
        &self,
        _shard: ShardId,
    ) -> Result<Poll, space_orchestrator::launcher::LaunchError> {
        Ok(Poll::Alive)
    }
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

fn hot_report(s: ShardId) -> LoadReport {
    LoadReport {
        shard: s,
        players: 90,
        tick_p95_ms: 1.0,
        load: 0.0,
    }
}

fn cold_report(s: ShardId) -> LoadReport {
    LoadReport {
        shard: s,
        players: 5,
        tick_p95_ms: 1.0,
        load: 0.0,
    }
}

/// Drive the autoscaler to a split of the root shard, then ack every prepare
/// recipient so the epoch commits. Returns the orchestrator's new map epoch.
fn drive_split_to_commit(orch: &mut Orchestrator<RecordingLauncher>, hot: ShardId) -> u64 {
    // The metric must actually be hot.
    assert!(space_orchestrator::autoscale::load_metric(&hot_report(hot)) >= SPLIT_THRESHOLD);
    let mut prepared = false;
    for _ in 0..(SPLIT_DWELL_TICKS + 5) {
        orch.handle_load_report(&hot_report(hot));
        orch.tick();
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
fn a_hot_shard_splits_the_epoch_and_stages_a_child_spawn() {
    let mut orch = Orchestrator::new(
        plane_with_root(shard(1)),
        Arc::new(RecordingLauncher::default()),
    );
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

    // The split staged a spawn for that child id.
    let spawns = orch.take_pending_spawn();
    assert!(
        spawns.iter().any(|s| s.shard == child_owner),
        "the split staged a spawn for the child {child_owner}"
    );
}

#[test]
fn a_split_stages_a_spawn_and_the_commit_publishes_the_child_region() {
    let mut orch = Orchestrator::new(
        plane_with_root(shard(1)),
        Arc::new(RecordingLauncher::default()),
    );
    // Run the autoscaler to the split; capture the staged spawn before acking.
    let mut staged_child: Option<ShardId> = None;
    for _ in 0..(SPLIT_DWELL_TICKS + 5) {
        orch.handle_load_report(&hot_report(shard(1)));
        orch.tick();
        let spawns = orch.take_pending_spawn();
        if !spawns.is_empty() {
            staged_child = Some(spawns[0].shard);
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
    // Root (depth 40) owned by 1, one child octant owned by 2.
    let config = OrchestratorConfig::default();
    let root = config.root_node();
    let child = space_world::octree::children(root).unwrap()[0];
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(root, shard(1), addr(7_001)),
            ShardRegion::new(child, shard(2), addr(7_002)),
        ],
    );
    let ids = PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
    let control = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
    let mut orch = Orchestrator::new(control, Arc::new(RecordingLauncher::default()));

    // The merge dwell is long (1800 ticks); drive it.
    let mut staged_kill = false;
    for _ in 0..(space_orchestrator::autoscale::MERGE_DWELL_TICKS + 5) {
        orch.handle_load_report(&cold_report(shard(2)));
        orch.tick();
        let kills = orch.take_pending_kill();
        if !kills.is_empty() {
            staged_kill = true;
            assert!(
                kills.contains(&shard(2)),
                "the cold child is staged for kill"
            );
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
    assert!(staged_kill, "a merge staged a kill");
    assert_eq!(orch.map().epoch, 2, "the merge bumped the epoch");
    assert_eq!(orch.map().regions.len(), 1, "the child region was removed");
    assert_eq!(
        orch.map().regions[0].shard,
        shard(1),
        "the parent owns the region"
    );
}

#[test]
fn an_unacked_prepare_aborts_and_the_map_is_unchanged() {
    let mut orch = Orchestrator::new(
        plane_with_root(shard(1)),
        Arc::new(RecordingLauncher::default()),
    );
    // Drive to the split, but ack nobody.
    let mut prepared_epoch = None;
    for _ in 0..(SPLIT_DWELL_TICKS + 5) {
        orch.handle_load_report(&hot_report(shard(1)));
        orch.tick();
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
                if let Inbound::Hello { hello, reply } = inbound {
                    let welcome = plane
                        .lock()
                        .unwrap()
                        .handle_shard_hello(&hello)
                        .expect("welcome");
                    let _ = reply.send(Message::ShardWelcome(welcome)).await;
                }
            }
        });
    }

    let hello = ShardHello {
        cluster: "space-local".into(),
        shard: ShardId::NONE,
        endpoint: addr(9_100),
        region: None,
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

#[tokio::test]
async fn the_local_launcher_spawns_a_reachable_child_over_loopback() {
    // The launcher half of a split: a spawned shard answers a real client.
    let launcher = LocalLauncher::new();
    let spec = SpawnSpec {
        shard: shard(2),
        region: ShardRegion::new(NodeRef::new(40, [0, 0, 0]), shard(2), addr(0)),
        cluster: "space-local".into(),
        control_addr: addr(7_900),
        host_port: 0,
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
    assert_eq!(welcome.shard, shard(2));
    assert!(!conn.is_closed());
    assert_eq!(launcher.poll(shard(2)).await.unwrap(), Poll::Alive);
    launcher.kill(shard(2)).await.expect("kill");
}
