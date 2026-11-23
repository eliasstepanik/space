//! The booted cluster serves real clients (Phase 3 remediation, FIX 1).
//!
//! Boots the orchestrator through its real wiring (`boot`, then
//! `Orchestrator::bind` and `Orchestrator::run_with`) and ONE orchestrated shard
//! through [`space_server::orchestrated::start_orchestrated`] — the same function
//! the `space-server --orchestrator` binary runs — in-process on loopback. A real
//! QUIC client then connects to that shard and must receive frames for its own
//! ship and never be redirected to the control port.
//!
//! Before the fix the orchestrator pre-allocated root owner shard 1 with the
//! control port as its endpoint, and nothing launched shard 1: the booted shard
//! was handed id 2, and every ship in the root region was offered to shard 1, so
//! each client got a `Redirect` to the control port and never saw a frame again.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_orchestrator::boot::boot;
use space_orchestrator::config::OrchestratorConfig;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::orchestrator::Orchestrator;
use space_protocol::{Hello, Message, ShardId};
use space_server::orchestrated::{start_orchestrated, OrchestratedOptions};

fn loopback0() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 0))
}

/// Boot an orchestrator through the binary's own path; returns the cluster
/// name, the control address, its task and the temp state file.
async fn boot_orchestrator(
    tag: &str,
) -> (
    String,
    SocketAddr,
    tokio::task::JoinHandle<Result<(), std::io::Error>>,
    std::path::PathBuf,
) {
    let state_path = std::env::temp_dir().join(format!(
        "space-orch-boot-{tag}-{}-{}.id",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_file(&state_path);
    let config = OrchestratorConfig {
        bind: loopback0(),
        state_path: state_path.clone(),
        ..OrchestratorConfig::default()
    };
    let cluster = config.cluster.clone();
    let control = boot(config, "secret".into()).expect("boot");
    let mut orchestrator = Orchestrator::new(control, Arc::new(LocalLauncher::new()));
    let server = orchestrator.bind().expect("bind control");
    let control_addr = server.local_addr().expect("control port");
    let task = tokio::spawn(async move { orchestrator.run_with(server).await });
    (cluster, control_addr, task, state_path)
}

fn options(cluster: &str, claim_region: Option<space_world::NodeRef>) -> OrchestratedOptions {
    OrchestratedOptions {
        shard: ShardId::NONE,
        bind: loopback0(),
        advertise: None,
        claim_region,
        cluster: cluster.to_owned(),
        secret: String::new(),
    }
}

/// A real client joins `endpoint` and must be welcomed.
async fn a_client_is_welcomed_by(endpoint: SocketAddr) {
    let client = space_net::client_endpoint().expect("client endpoint");
    let hello = Hello {
        protocol: space_protocol::PROTOCOL_VERSION,
        player_token: "player".into(),
    };
    space_net::connect(&client, endpoint, space_server::net::SERVER_NAME, &hello)
        .await
        .expect("the shard welcomes a real client");
}

#[tokio::test]
async fn an_anonymous_claim_for_a_node_outside_the_map_is_rejected_and_burns_no_id() {
    // `space-server --orchestrator … --claim-region <a node nobody carved>`.
    //
    // This used to be welcomed ("two orchestrated shards on distinct nodes are
    // both welcomed and serve"): the orchestrator allocated an id and answered
    // with the claimed region — but never published it. The shard's own map
    // (the welcome's) still gave the whole root to shard a, so it handed every
    // ship it was given straight to a; a second shard could claim the same
    // node; and each restart burned another id. A region is given to a shard
    // that registers *without* one (the orchestrator picks it and carves it
    // through an epoch change), never claimed — so the anonymous claim is
    // refused, fast, for good and with the reason.
    let (cluster, control_addr, orch_task, state_path) = boot_orchestrator("claim").await;
    let root = OrchestratorConfig::default().root_node();
    let child = space_world::octree::children(root).expect("root splits")[0];

    let a = start_orchestrated(options(&cluster, None), control_addr)
        .await
        .expect("shard a adopts the root");
    assert_eq!(a.shard, ShardId::new(1), "a adopts the reserved root owner");

    for attempt in 0..2 {
        let claim = tokio::time::timeout(
            Duration::from_secs(5),
            start_orchestrated(options(&cluster, Some(child)), control_addr),
        )
        .await
        .expect("the rejection arrives promptly");
        let err = claim.expect_err("a node outside the published map cannot be claimed");
        assert!(
            err.contains("rejected") && err.contains("not in the published map"),
            "attempt {attempt}: the error names the rejection and its reason, got: {err}"
        );
    }
    // The refusals allocated nothing: the counter file still holds the id after
    // the root owner's (boot allocated 1, so the next id is 2).
    let persisted = std::fs::read_to_string(&state_path).expect("the counter file exists");
    assert_eq!(persisted.trim(), "2", "a refused claim must not burn an id");

    // Shard a is unaffected: it still serves, and its map is still the root's.
    a_client_is_welcomed_by(a.endpoint).await;
    assert_eq!(a.state.lock().await.shard_map().regions.len(), 1);

    a.abort();
    orch_task.abort();
    let _ = std::fs::remove_file(&state_path);
}

/// Wait until `shard` runs a map of `epoch` in which it owns its region.
async fn owns_its_region_at(shard: &space_server::orchestrated::RunningShard, epoch: u64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        {
            let state = shard.state.lock().await;
            let map = state.shard_map();
            if map.epoch >= epoch && map.lookup(&shard.region.center()) == Some(shard.shard) {
                return;
            }
        }
        assert!(
            Instant::now() < deadline,
            "{} never owned its region at epoch {epoch}",
            shard.shard
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_second_region_less_shard_is_given_the_next_layout_cell() {
    // Two shards started by hand, each with nothing but the orchestrator's
    // address — what `space-server --orchestrator <addr>` does. The first gets
    // the root. The second used to be refused ("the root is already owned"),
    // and exited; it is now given the first layout cell beside the scene's
    // spawn point, through an epoch change proposed with it already connected.
    let (cluster, control_addr, orch_task, state_path) = boot_orchestrator("second").await;
    let root = OrchestratorConfig::default().root_node();
    let cell = space_orchestrator::topology::layout_cells(root)[0];
    let a = start_orchestrated(options(&cluster, None), control_addr)
        .await
        .expect("shard a adopts the root");
    assert_eq!((a.shard, a.region), (ShardId::new(1), root));

    let started = Instant::now();
    let b = tokio::time::timeout(
        Duration::from_secs(10),
        start_orchestrated(options(&cluster, None), control_addr),
    )
    .await
    .expect("the second shard is answered promptly")
    .expect("and welcomed, not refused");
    assert_eq!(b.shard, ShardId::new(2), "a freshly allocated id (H9)");
    assert_eq!(b.region, cell, "the first layout cell");
    assert!(started.elapsed() < Duration::from_secs(5));

    // The welcome did not make it the owner: the epoch does. Both shards end
    // up running the committed map, in which each owns its own ground.
    owns_its_region_at(&b, 2).await;
    owns_its_region_at(&a, 2).await;
    let map = a.state.lock().await.shard_map().clone();
    assert_eq!(map.epoch, 2);
    assert_eq!(map.regions.len(), 2);
    assert_eq!(map.lookup(&cell.center()), Some(b.shard));
    let spawn = space_world::WorldPos::from_f64_metres_around(root.center(), [6.0, 512.0, 4.0]);
    assert_eq!(
        map.lookup(&spawn),
        Some(a.shard),
        "the spawn point stays with the root"
    );
    // Each knows where the other listens (the map push behind the commit).
    assert!(map.regions.iter().all(|r| r.endpoint.port() != 0));

    a_client_is_welcomed_by(a.endpoint).await;
    a_client_is_welcomed_by(b.endpoint).await;

    a.abort();
    b.abort();
    orch_task.abort();
    let _ = std::fs::remove_file(&state_path);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_restarting_shard_at_the_same_endpoint_replaces_itself_under_a_new_id() {
    // Shard a crashes and restarts with no id, on the same advertised address
    // (as a compose container does). It used to get id 1 back — and with it a
    // guid counter that starts over, under an id that had minted guids before.
    // It is welcomed as a NEW shard and given the old one's region through an
    // epoch change; the scene is re-issued one revision up so its bodies are
    // built once, afresh.
    let (cluster, control_addr, orch_task, state_path) = boot_orchestrator("restart").await;
    let first = start_orchestrated(options(&cluster, None), control_addr)
        .await
        .expect("adopts the root");
    let (id, endpoint, root) = (first.shard, first.endpoint, first.region);
    // It learns the scene (revision 1) like any shard.
    let deadline = Instant::now() + Duration::from_secs(5);
    while first.state.lock().await.scene().revision != 1 {
        assert!(
            Instant::now() < deadline,
            "the first shard never got the scene"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    first.abort();

    // Same advertised endpoint, new socket (the old one is aborted; use the
    // advertise knob so the hello carries the original address).
    let mut again = options(&cluster, None);
    again.advertise = Some(endpoint);
    let second = start_orchestrated(again, control_addr)
        .await
        .expect("the restarted shard is welcomed, not locked out");
    assert_ne!(second.shard, id, "never the old id");
    assert_eq!(second.shard, ShardId::new(2));
    assert_eq!(second.region, root, "the old shard's region");
    owns_its_region_at(&second, 2).await;
    {
        let state = second.state.lock().await;
        assert_eq!(state.shard_map().regions.len(), 1);
        assert_eq!(state.shard_map().regions[0].shard, second.shard);
    }
    // The scene was re-issued: revision 2, the same scene.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let scene = second.state.lock().await.scene();
        if scene.revision == 2 {
            assert_eq!(scene.scene, space_protocol::Scene::DEFAULT.id());
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the replacement never heard the re-issued scene (it runs {scene})"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    second.abort();
    orch_task.abort();
    let _ = std::fs::remove_file(&state_path);
}

#[tokio::test]
async fn a_booted_cluster_serves_frames_and_never_redirects_to_the_control_port() {
    // --- the orchestrator, through the binary's own boot path.
    let state_path = std::env::temp_dir().join(format!(
        "space-orch-boot-{}-{}.id",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_file(&state_path);
    let config = OrchestratorConfig {
        bind: loopback0(),
        state_path: state_path.clone(),
        ..OrchestratorConfig::default()
    };
    let cluster = config.cluster.clone();
    let control = boot(config, "secret".into()).expect("boot");
    assert_eq!(
        control.unclaimed_root_owner(),
        Some(ShardId::new(1)),
        "boot reserves the root owner"
    );
    let mut orchestrator = Orchestrator::new(control, Arc::new(LocalLauncher::new()));
    let server = orchestrator.bind().expect("bind control");
    let control_addr = server.local_addr().expect("control port");
    let orch_task = tokio::spawn(async move { orchestrator.run_with(server).await });

    // --- ONE orchestrated shard, no --shard: it asks the orchestrator for its id.
    let shard = start_orchestrated(
        OrchestratedOptions {
            shard: ShardId::NONE,
            bind: loopback0(),
            advertise: None,
            claim_region: None,
            cluster,
            secret: String::new(),
        },
        control_addr,
    )
    .await
    .expect("the shard registers with the orchestrator");
    assert_eq!(
        shard.shard,
        ShardId::new(1),
        "the first shard adopts the reserved root owner"
    );
    assert_ne!(
        shard.endpoint, control_addr,
        "a shard is not the control port"
    );

    // --- a real client over QUIC.
    let endpoint = space_net::client_endpoint().expect("client endpoint");
    let hello = Hello {
        protocol: space_protocol::PROTOCOL_VERSION,
        player_token: "player-1".into(),
    };
    let (conn, welcome) = space_net::connect(
        &endpoint,
        shard.endpoint,
        space_server::net::SERVER_NAME,
        &hello,
    )
    .await
    .expect("the shard accepts a real client");
    let root_owner = welcome
        .shard_map
        .regions
        .iter()
        .find(|r| r.node == shard.region)
        .expect("the root is in the client's map");
    assert_eq!(root_owner.shard, shard.shard, "the shard owns its region");
    assert_eq!(
        root_owner.endpoint, shard.endpoint,
        "the map names the live shard, not the control port"
    );
    let own_ship = shard
        .state
        .lock()
        .await
        .ship_of(welcome.session)
        .expect("a join spawns a ship");

    // Watch for ~1.5 s (45 ticks): the first frame carries the ship, and no
    // Redirect may ever follow (an idle ship is not re-sent in deltas).
    let deadline = Instant::now() + Duration::from_millis(1_500);
    let mut own_frames = 0u32;
    let mut redirects = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let next = tokio::time::timeout(remaining, async {
            tokio::select! {
                m = conn.recv_stream() => m,
                m = conn.recv_datagram() => m,
                m = conn.recv_control() => m,
            }
        })
        .await;
        let Ok(Ok(Some(message))) = next else { break };
        match message {
            Message::Snapshot(f) if f.entities.iter().any(|e| e.guid == own_ship) => {
                own_frames += 1;
            }
            Message::Delta(f) if f.entities.iter().any(|e| e.guid == own_ship) => {
                own_frames += 1;
            }
            Message::Redirect(r) => redirects.push(r),
            _ => {}
        }
    }
    assert!(
        redirects.is_empty(),
        "no Redirect expected; got {:?}",
        redirects
            .iter()
            .map(|r| (r.new_shard, r.shard_map.regions.len()))
            .collect::<Vec<_>>()
    );
    assert!(
        own_frames >= 1,
        "the client must receive a frame carrying its own ship, got {own_frames}"
    );

    shard.abort();
    orch_task.abort();
    let _ = std::fs::remove_file(&state_path);
}

/// A split commits **through the real run loop**, with the real local launcher.
///
/// Everything in between is production code: `Orchestrator::run_with` over the
/// QUIC control endpoint, a root shard booted by `start_orchestrated` (what
/// `space-server --orchestrator` runs), and the child booted by
/// [`LocalLauncher`]. The only thing the test does by hand is ask for the split
/// (the admin verb has no wire front end yet); the loop is stopped for that one
/// call and resumed.
///
/// Before the fixes this could not happen at all, for three independent
/// reasons, each of which this test catches on its own:
///
/// * the loop awaited the launch, and the launch (a registering shard) waits
///   for a `ShardHello` answer only that loop can give;
/// * the launcher started a bare shard that never dialled the orchestrator, so
///   the child never registered and never acked;
/// * the `EpochPrepare` was published before the child existed and never sent
///   again, so even a registered child had nothing to ack. (The prepare is now
///   only proposed once the child has registered, so there is no "again".)
///
/// Each one leaves the split uncommitted and the map at epoch 1.
#[tokio::test(flavor = "multi_thread")]
async fn a_split_commits_through_the_real_run_loop_and_the_local_launcher() {
    use space_orchestrator::launcher::Launcher as _;
    use space_protocol::{AdminCommand, AdminReply};

    let state_path = std::env::temp_dir().join(format!(
        "space-orch-boot-split-{}-{}.id",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_file(&state_path);
    let config = OrchestratorConfig {
        // Port 0: the loop resolves the port the child has to dial.
        bind: loopback0(),
        state_path: state_path.clone(),
        ..OrchestratorConfig::default()
    };
    let cluster = config.cluster.clone();
    let root_node = config.root_node();
    let control = boot(config, "secret".into()).expect("boot");
    let launcher = Arc::new(LocalLauncher::new());
    let mut orchestrator = Orchestrator::new(control, Arc::clone(&launcher));
    let server = orchestrator.bind().expect("bind control");
    let control_addr = server.local_addr().expect("control port");

    // --- the loop runs until the root shard has registered.
    let root = tokio::select! {
        ended = orchestrator.run_with(Arc::clone(&server)) => panic!("the loop ended: {ended:?}"),
        shard = start_orchestrated(options(&cluster, None), control_addr) => {
            shard.expect("the root shard registers")
        }
    };
    assert_eq!(root.shard, ShardId::new(1));
    assert_eq!(orchestrator.map().epoch, 1);

    // --- ask for the split (the loop is not running for this one call).
    let reply = orchestrator.handle_admin(&AdminCommand::ForceSplit(root.shard));
    assert!(
        matches!(reply, AdminReply::Accepted { .. }),
        "the split is accepted: {reply:?}"
    );

    // --- the loop carries it out: launch, the new shard's hello (no id, no
    // region), the id, the prepare, acks, commit, map push. Done when the
    // *root shard* runs the new map with both regions at real addresses —
    // i.e. it got the commit and the push.
    let root_state = Arc::clone(&root.state);
    let committed = async {
        loop {
            {
                let shard = root_state.lock().await;
                let map = shard.shard_map();
                if map.epoch == 2
                    && map.regions.len() == 2
                    && map.regions.iter().all(|r| r.endpoint.port() != 0)
                {
                    return map.clone();
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    let root_map = tokio::select! {
        ended = orchestrator.run_with(Arc::clone(&server)) => panic!("the loop ended: {ended:?}"),
        map = tokio::time::timeout(Duration::from_secs(10), committed) => {
            map.expect("the split must commit through the run loop (it aborts after 3 s if the child never acks)")
        }
    };

    // The orchestrator published exactly what the root shard runs.
    assert_eq!(orchestrator.map().epoch, 2, "the split committed");
    assert_eq!(orchestrator.map(), &root_map);
    let child_region = orchestrator
        .map()
        .regions
        .iter()
        .find(|r| r.shard != root.shard)
        .expect("the child's region is published")
        .clone();
    let child = child_region.shard;
    assert!(child > root.shard, "a fresh, greater id (H9)");
    assert_eq!(
        space_world::octree::parent(child_region.node),
        Some(root_node),
        "the child owns one octant of the root"
    );
    assert!(
        orchestrator.take_pending_kill().is_empty(),
        "a committed split kills nobody"
    );

    // The child is a live shard the launcher supervises, at the published
    // address, and it follows the committed map too.
    let child_launch = launcher
        .launch_of(child)
        .await
        .expect("the launcher started the child");
    assert_eq!(orchestrator.launch_of(child), Some(child_launch));
    assert_eq!(
        launcher.poll(child_launch).await.expect("poll"),
        space_orchestrator::launcher::Poll::Alive
    );
    let client = space_net::client_endpoint().expect("client endpoint");
    let mut seen = None;
    for _ in 0..100 {
        let hello = Hello {
            protocol: space_protocol::PROTOCOL_VERSION,
            player_token: "after-the-split".into(),
        };
        let (conn, welcome) = space_net::connect(
            &client,
            child_region.endpoint,
            space_server::net::SERVER_NAME,
            &hello,
        )
        .await
        .expect("the child answers at the address the map names");
        assert_eq!(welcome.shard, child, "it runs as the id it was allocated");
        conn.close(0, "probe");
        if welcome.shard_map.epoch == 2 {
            seen = Some(welcome.shard_map);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let child_map = seen.expect("the child installs the committed map");
    assert_eq!(
        child_map.lookup(&child_region.node.center()),
        Some(child),
        "the child knows it owns its octant"
    );
    assert_eq!(
        child_map.lookup(&root_node.center()),
        root_map.lookup(&root_node.center()),
        "and agrees with the root shard about everything else"
    );

    launcher.kill(child_launch).await.expect("kill");
    root.abort();
    server.close();
    let _ = std::fs::remove_file(&state_path);
}
