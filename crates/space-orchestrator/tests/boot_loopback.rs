//! The booted cluster serves real clients (Phase 3 remediation, FIX 1).
//!
//! Boots the orchestrator through its real wiring ([`boot`] + [`Orchestrator::bind`]
//! + [`Orchestrator::run_with`]) and ONE orchestrated shard through
//! [`space_server::orchestrated::start_orchestrated`] — the same function the
//! `space-server --orchestrator` binary runs — in-process on loopback. A real
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
    assert_ne!(shard.endpoint, control_addr, "a shard is not the control port");

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

    // Watch for ~1.5 s (45 ticks): frames must keep arriving, no Redirect.
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
        own_frames >= 2,
        "the client must keep receiving frames for its own ship, got {own_frames}"
    );

    shard.abort();
    orch_task.abort();
    let _ = std::fs::remove_file(&state_path);
}
