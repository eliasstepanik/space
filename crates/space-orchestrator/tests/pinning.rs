//! Cluster certificates over the real run loop: with a cluster secret the
//! orchestrator's control port, its player port and every launched shard's
//! client endpoint present certificates issued from the CA that secret makes,
//! so a client that holds the pin of the secret gets an answer and one with the
//! pin of another secret is refused with `NetError::Untrusted`. Without a
//! secret, on a loopback control port, nothing is verified and the development
//! workflow (shards started by hand next to launched ones) still works.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_net::{ClusterPin, ClusterTrust, NetError};
use space_orchestrator::boot::boot;
use space_orchestrator::config::{OrchestratorConfig, Secret};
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::orchestrator::Orchestrator;
use space_orchestrator::snapshot::Snapshot;
use space_protocol::ShardId;
use space_server::orchestrated::{start_orchestrated, OrchestratedOptions};
use space_test_support::loopback0;

const SECRET: &str = "the-secret-of-the-pinning-cluster";

fn temp_path(tag: &str) -> PathBuf {
    space_test_support::temp_path("space-orch-pinning", tag).with_extension("id")
}

fn pin_of(secret: &str) -> ClusterPin {
    ClusterTrust::from_secret(secret).expect("a secret").pin()
}

struct Running {
    cluster: String,
    control_addr: SocketAddr,
    player_addr: SocketAddr,
    ops: space_orchestrator::ops::OpsHandle,
    task: tokio::task::JoinHandle<Result<(), std::io::Error>>,
    state_path: PathBuf,
}

impl Running {
    fn start(tag: &str, min_shards: usize, secret: Option<&str>) -> Self {
        let state_path = temp_path(tag);
        let config = OrchestratorConfig {
            bind: loopback0(),
            player_bind: Some(loopback0()),
            state_path: state_path.clone(),
            min_shards,
            cluster_secret: secret.map(Secret::new),
            ..OrchestratorConfig::default()
        };
        let cluster = config.cluster.clone();
        let control = boot(config, "peer-token-of-the-pinning-cluster".into()).expect("boot");
        let mut orchestrator = Orchestrator::new(control, Arc::new(LocalLauncher::new()));
        let ops = orchestrator.attach_ops();
        let server = orchestrator.bind().expect("bind control");
        let player = orchestrator
            .bind_player()
            .expect("bind player")
            .expect("a player port");
        let control_addr = server.local_addr().expect("control port");
        let player_addr = player.local_addr().expect("player port");
        let task =
            tokio::spawn(async move { orchestrator.run_with_player(server, Some(player)).await });
        Self {
            cluster,
            control_addr,
            player_addr,
            ops,
            task,
            state_path,
        }
    }

    async fn wait_for(&self, what: &str, done: impl Fn(&Snapshot) -> bool) -> Arc<Snapshot> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let snapshot = self.ops.snapshot();
            if done(&snapshot) {
                return snapshot;
            }
            assert!(
                Instant::now() < deadline,
                "{what} did not happen in time. Shards: {:?}; events: {:?}",
                snapshot
                    .shards
                    .iter()
                    .map(|s| (s.shard.0, s.connected))
                    .collect::<Vec<_>>(),
                snapshot
                    .events
                    .iter()
                    .take(8)
                    .map(|e| &e.text)
                    .collect::<Vec<_>>(),
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    async fn stop(self) {
        self.task.abort();
        let _ = self.task.await;
        let _ = std::fs::remove_file(&self.state_path);
        let _ = std::fs::remove_file(space_orchestrator::scene::scene_path(&self.state_path));
    }
}

/// Every shard of `snapshot` is registered (connected) and owns a region.
fn all_registered(snapshot: &Snapshot, count: usize) -> bool {
    snapshot.shards.len() == count
        && snapshot.shards.iter().all(|s| s.connected)
        && snapshot.regions.len() == count
        && snapshot.regions.iter().all(|r| r.endpoint.is_some())
}

/// What a client holding `pin` gets from a player's hello at `addr`.
async fn join_with(addr: SocketAddr, pin: &ClusterPin) -> Result<(), NetError> {
    let client = space_net::client_endpoint_pinned(addr, Some(pin)).expect("client endpoint");
    let joined = space_net::join(
        &client,
        addr,
        space_server::net::SERVER_NAME,
        &space_net::hello("a-player"),
    )
    .await?;
    if let space_net::Joined::Shard(conn, _) = joined {
        conn.close(0, "seen enough");
    }
    Ok(())
}

/// A cluster with a secret comes up, and everything a client can dial
/// presents a certificate of that secret's CA.
///
/// Mutation this catches: an endpoint (player port, a shard's client
/// endpoint) that still presents a self-signed certificate — a pinned client
/// would be refused — or one that presents the cluster certificate to
/// everybody's pin.
#[tokio::test(flavor = "multi_thread")]
async fn a_cluster_with_a_secret_is_dialled_only_by_clients_that_hold_its_pin() {
    let orch = Running::start("secret", 2, Some(SECRET));
    let snapshot = orch
        .wait_for("two registered shards", |s| all_registered(s, 2))
        .await;
    assert!(snapshot.hello_auth_required);

    let mut addrs = vec![("the player port", orch.player_addr)];
    for region in &snapshot.regions {
        addrs.push(("a shard", region.endpoint.expect("registered")));
    }
    assert_eq!(addrs.len(), 3);
    for (what, addr) in &addrs {
        join_with(*addr, &pin_of(SECRET))
            .await
            .unwrap_or_else(|e| panic!("{what} at {addr} must answer the right pin: {e}"));
        let err = join_with(*addr, &pin_of("another-secret-entirely"))
            .await
            .expect_err("another secret's pin");
        assert!(
            matches!(err, NetError::Untrusted(_)),
            "{what} at {addr}: {err:?}"
        );
    }
    orch.stop().await;
}

/// Nothing configured on a loopback control port: no verification anywhere,
/// and a shard started by hand with an empty secret registers next to the
/// launched ones — today's development workflow.
#[tokio::test(flavor = "multi_thread")]
async fn without_a_secret_launched_and_hand_started_shards_all_register() {
    let orch = Running::start("open", 2, None);
    orch.wait_for("two launched shards", |s| all_registered(s, 2))
        .await;
    let by_hand = OrchestratedOptions {
        shard: ShardId::NONE,
        bind: loopback0(),
        advertise: None,
        claim_region: None,
        cluster: orch.cluster.clone(),
        client_endpoint: None,
        secret: String::new(),
    };
    let shard = tokio::time::timeout(
        Duration::from_secs(20),
        start_orchestrated(by_hand, orch.control_addr),
    )
    .await
    .expect("registered in time")
    .expect("a shard with no secret registers on an open loopback cluster");
    orch.wait_for("three registered shards", |s| all_registered(s, 3))
        .await;
    shard.abort();
    orch.stop().await;
}
