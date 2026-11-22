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
async fn two_orchestrated_shards_on_distinct_nodes_are_both_welcomed_and_serve() {
    // Defect A, the bring-up half: the second shard joins on its own region
    // (as a split's child or a `--claim-region` shard would), is allocated a
    // distinct id, and serves.
    let (cluster, control_addr, orch_task, state_path) = boot_orchestrator("two").await;
    let root = OrchestratorConfig::default().root_node();
    let child = space_world::octree::children(root).expect("root splits")[0];

    let a = start_orchestrated(options(&cluster, None), control_addr)
        .await
        .expect("shard a adopts the root");
    let b = start_orchestrated(options(&cluster, Some(child)), control_addr)
        .await
        .expect("shard b is welcomed on a free node");
    assert_eq!(a.shard, ShardId::new(1), "a adopts the reserved root owner");
    assert_ne!(a.shard, b.shard, "distinct ids");
    assert_ne!(a.region, b.region, "distinct nodes");
    a_client_is_welcomed_by(a.endpoint).await;
    a_client_is_welcomed_by(b.endpoint).await;

    a.abort();
    b.abort();
    orch_task.abort();
    let _ = std::fs::remove_file(&state_path);
}

#[tokio::test]
async fn a_second_region_less_shard_is_rejected_fast_with_a_clear_reason() {
    // Defect A, the failure half: the root is owned, so a second region-less
    // hello is refused. The orchestrator must SAY so (a `Reject` on the hello
    // connection) instead of dropping it and leaving the shard to time out.
    let (cluster, control_addr, orch_task, state_path) = boot_orchestrator("reject").await;
    let a = start_orchestrated(options(&cluster, None), control_addr)
        .await
        .expect("shard a adopts the root");

    let started = Instant::now();
    let second = tokio::time::timeout(
        Duration::from_secs(5),
        start_orchestrated(options(&cluster, None), control_addr),
    )
    .await
    .expect("the rejection arrives well before the 10 s idle timeout");
    let err = second.expect_err("the root is already owned");
    assert!(
        err.contains("rejected") && err.contains("already owned"),
        "the error names the rejection and its reason, got: {err}"
    );
    assert!(started.elapsed() < Duration::from_secs(5));

    a.abort();
    orch_task.abort();
    let _ = std::fs::remove_file(&state_path);
}

#[tokio::test]
async fn a_restarting_shard_with_the_same_endpoint_rejoins_with_the_same_id() {
    // Defect B end to end: shard a crashes and restarts with no id, on the same
    // advertised address (as a compose container does). It must get id 1 back.
    let (cluster, control_addr, orch_task, state_path) = boot_orchestrator("restart").await;
    let first = start_orchestrated(options(&cluster, None), control_addr)
        .await
        .expect("adopts the root");
    let (id, endpoint) = (first.shard, first.endpoint);
    first.abort();

    // Same advertised endpoint, new socket (the old one is aborted; use the
    // advertise knob so the hello carries the original address).
    let mut again = options(&cluster, None);
    again.advertise = Some(endpoint);
    let second = start_orchestrated(again, control_addr)
        .await
        .expect("the restarted shard is welcomed, not locked out");
    assert_eq!(second.shard, id, "same id back");

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
