//! The shard lifecycle through the **real run loop** (`docs/plan-sharding.md`
//! §2): the orchestrator starts the shards it is asked to keep running, gives
//! each its region when it registers, replaces one that dies, lets nobody in
//! without the cluster secret when one is required — and hears nothing from a
//! connection it did not let in, and from one it did only what that shard
//! says about itself.
//!
//! Everything here is production code over loopback QUIC:
//! `Orchestrator::run_with`, real shards — in-process ones started by
//! [`LocalLauncher`] through `space_server::orchestrated::start_orchestrated`,
//! and, in the last test, real `space-server` **processes** started by
//! [`ProcessLauncher`]. The tests only watch (the snapshots the loop publishes
//! for its web UI) and, to make a shard die, stop it behind the orchestrator's
//! back.
//!
//! The rules themselves — the order regions are handed out in, who is told to
//! wait, what is refused for good, who has to acknowledge a replacement — are
//! pinned one by one in `orchestrator::tests`, without a socket.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_orchestrator::boot::boot;
use space_orchestrator::config::{LauncherKind, OrchestratorConfig, Secret};
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::launcher::process::{ProcessLauncher, SHARD_EXE};
use space_orchestrator::launcher::{LaunchId, Launcher};
use space_orchestrator::ops::OpsHandle;
use space_orchestrator::orchestrator::Orchestrator;
use space_orchestrator::snapshot::{ShardRole, Snapshot};
use space_orchestrator::topology;
use space_protocol::control::LoadReport;
use space_protocol::shard::EpochAck;
use space_protocol::{Message, Reject, RejectReason, SceneAck, SceneState, ShardHello, ShardId};
use space_server::orchestrated::{start_orchestrated, OrchestratedOptions};
use space_test_support::loopback0;

fn temp_path(tag: &str) -> PathBuf {
    space_test_support::temp_path("space-orch-lifecycle", tag)
}

/// A running orchestrator: its loop in a task, and the handle its web UI
/// would read snapshots through.
struct Running {
    cluster: String,
    control_addr: SocketAddr,
    ops: OpsHandle,
    task: tokio::task::JoinHandle<Result<(), std::io::Error>>,
    state_path: PathBuf,
}

impl Running {
    /// Boot through the binary's own path (`boot`, `bind`, `run_with`) with
    /// `launcher`, on a loopback control port chosen by the system.
    fn start<L: Launcher + 'static>(
        tag: &str,
        config: OrchestratorConfig,
        launcher: Arc<L>,
    ) -> Self {
        let state_path = temp_path(tag).with_extension("id");
        let _ = std::fs::remove_file(&state_path);
        let config = OrchestratorConfig {
            bind: loopback0(),
            state_path: state_path.clone(),
            ..config
        };
        let cluster = config.cluster.clone();
        let control = boot(config, "peer-token-of-the-test-cluster".into()).expect("boot");
        let mut orchestrator = Orchestrator::new(control, launcher);
        let ops = orchestrator.attach_ops();
        let server = orchestrator.bind().expect("bind control");
        let control_addr = server.local_addr().expect("control port");
        let task = tokio::spawn(async move { orchestrator.run_with(server).await });
        Self {
            cluster,
            control_addr,
            ops,
            task,
            state_path,
        }
    }

    /// The latest snapshot `done` is satisfied with, within `limit`.
    async fn wait_for(
        &self,
        what: &str,
        limit: Duration,
        done: impl Fn(&Snapshot) -> bool,
    ) -> Arc<Snapshot> {
        let deadline = Instant::now() + limit;
        loop {
            let snapshot = self.ops.snapshot();
            if done(&snapshot) {
                return snapshot;
            }
            assert!(
                Instant::now() < deadline,
                "{what} did not happen within {limit:?}. Shards: {:?}; regions: {:?}; in flight: \
                 {:?}; newest events: {:?}",
                snapshot
                    .shards
                    .iter()
                    .map(|s| (s.shard.0, s.role.label(), s.connected))
                    .collect::<Vec<_>>(),
                snapshot
                    .regions
                    .iter()
                    .map(|r| (r.node.depth, r.shard.0))
                    .collect::<Vec<_>>(),
                snapshot.in_flight.as_ref().map(|f| &f.what),
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

    /// Stop the loop and wait until it has let go of everything it held.
    async fn stop(self) {
        self.task.abort();
        let _ = self.task.await;
        let _ = std::fs::remove_file(&self.state_path);
        let _ = std::fs::remove_file(space_orchestrator::scene::scene_path(&self.state_path));
    }
}

/// `count` regions, each owned by a connected shard that owns nothing else,
/// nothing in flight, nothing starting, nobody dead — and every shard has
/// confirmed the current scene.
fn settled_with(snapshot: &Snapshot, count: usize) -> bool {
    let current = snapshot.scene.current();
    snapshot.regions.len() == count
        && snapshot.shards.len() == count
        && snapshot.in_flight.is_none()
        && snapshot.starting == 0
        && snapshot.shards.iter().all(|shard| {
            shard.role == ShardRole::Owner
                && shard.connected
                && shard.regions.len() == 1
                && shard.scene == Some(current)
        })
        && snapshot
            .regions
            .iter()
            .all(|region| region.endpoint.is_some())
}

/// The owner of the point `(x, z)` metres from the scene's spawn point in
/// `snapshot`'s map (the deepest region that holds it).
fn owner_at(snapshot: &Snapshot, x: f64, z: f64) -> ShardId {
    // The anchor rule (`space_scene::ANCHOR_OFFSET`): the root's centre corner
    // plus (6, 512, 4) metres.
    let pos = space_world::WorldPos::from_f64_metres_around(
        snapshot.root.center(),
        [6.0 + x, 512.0, 4.0 + z],
    );
    snapshot
        .regions
        .iter()
        .filter(|region| space_world::octree::contains(region.node, pos))
        .max_by_key(|region| region.node.depth)
        .map(|region| region.shard)
        .expect("the root holds every point")
}

/// Join `endpoint` as a player and return the map of the welcome and the
/// guids of the first snapshot.
async fn join_and_look(
    endpoint: SocketAddr,
) -> (space_protocol::ShardMap, Vec<space_protocol::EntityGuid>) {
    let client = space_net::client_endpoint_for(endpoint).expect("client endpoint");
    let (conn, welcome) = space_net::connect(
        &client,
        endpoint,
        space_server::net::SERVER_NAME,
        &space_net::hello("a-player"),
    )
    .await
    .expect("the shard welcomes a player");
    let seen = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let next = tokio::select! {
                m = conn.recv_stream() => m,
                m = conn.recv_datagram() => m,
                m = conn.recv_control() => m,
            };
            if let Ok(Some(Message::Snapshot(snapshot))) = next {
                return snapshot.entities.iter().map(|e| e.guid).collect::<Vec<_>>();
            }
        }
    })
    .await
    .expect("the first snapshot arrives");
    conn.close(0, "seen enough");
    (welcome.shard_map, seen)
}

/// Say hello at `endpoint` as a player and return the map the shard there
/// runs, and the shard it sent the player on to (`None`: it welcomed the
/// player itself).
async fn hello_and_map(endpoint: SocketAddr) -> (space_protocol::ShardMap, Option<ShardId>) {
    let client = space_net::client_endpoint_for(endpoint).expect("client endpoint");
    let joined = space_net::join(
        &client,
        endpoint,
        space_server::net::SERVER_NAME,
        &space_net::hello("a-player"),
    )
    .await
    .expect("the shard answers a player");
    match joined {
        space_net::Joined::Shard(conn, welcome) => {
            conn.close(0, "seen enough");
            (welcome.shard_map, None)
        }
        space_net::Joined::Elsewhere(redirect) => (redirect.shard_map, Some(redirect.shard)),
    }
}

/// `--min-shards 4` with the in-process launcher, through the real loop: with
/// nothing started by hand the cluster comes up as four shards, each given its
/// region at its hello — the root, then the three layout cells beside the
/// spawn point — and settles as the four quadrants of the scene. The shards
/// themselves run the committed map and reach each other.
///
/// Mutation this catches (beyond the unit tests): anything in the real path
/// that the synchronous model does not have — a welcome that reaches the shard
/// after its prepare, a launched shard that gives up when told to wait its
/// turn, a reconciler that launches while its own launches are still starting.
#[tokio::test(flavor = "multi_thread")]
async fn min_shards_4_through_the_real_loop_ends_with_four_live_shards_in_four_quadrants() {
    let launcher = Arc::new(LocalLauncher::new());
    let started = Instant::now();
    let orch = Running::start(
        "four",
        OrchestratorConfig {
            min_shards: 4,
            ..OrchestratorConfig::default()
        },
        Arc::clone(&launcher),
    );
    let snapshot = orch
        .wait_for("four settled shards", Duration::from_secs(30), |s| {
            settled_with(s, 4)
        })
        .await;
    println!(
        "four regions owned by four live shards {:.2?} after the orchestrator started (epoch {})",
        started.elapsed(),
        snapshot.epoch
    );
    assert_eq!(
        snapshot.epoch, 4,
        "the root, then one epoch per layout cell"
    );
    assert_eq!((snapshot.min_shards, snapshot.max_shards), (4, 8));

    let root = snapshot.root;
    let cells = topology::layout_cells(root);
    let owner_of = |node| {
        snapshot
            .regions
            .iter()
            .find(|region| region.node == node)
            .map(|region| region.shard)
    };
    assert_eq!(
        owner_of(root),
        Some(ShardId::new(1)),
        "the reserved root id"
    );
    let owners: Vec<ShardId> = [root, cells[0], cells[1], cells[2]]
        .into_iter()
        .map(|node| owner_of(node).expect("the node is a region"))
        .collect();
    let mut distinct = owners.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), 4, "four regions, four owners: {owners:?}");
    assert!(
        snapshot.shards.iter().all(|s| s.launched),
        "every one of them was started by the orchestrator"
    );
    // The scene's four quadrants, meeting at x = -6 / z = -4.
    assert_eq!(owner_at(&snapshot, 0.0, 0.0), owners[0]);
    assert_eq!(owner_at(&snapshot, -5.0, -3.0), owners[0]);
    assert_eq!(owner_at(&snapshot, -5.0, -5.0), owners[1]);
    assert_eq!(owner_at(&snapshot, -7.0, -3.0), owners[2]);
    assert_eq!(owner_at(&snapshot, -7.0, -5.0), owners[3]);

    // Every shard itself runs the committed map, with every address in it: a
    // player who says hello to any of them is handed the same four regions —
    // by the root's shard with its welcome (it owns the spawn point), by the
    // other three with the redirect that sends the player there.
    assert_eq!(snapshot.spawn_owner, Some(owners[0]));
    for region in &snapshot.regions {
        let endpoint = region.endpoint.expect("a registered owner");
        let (map, sent_to) = hello_and_map(endpoint).await;
        let expected = (region.shard != owners[0]).then_some(owners[0]);
        assert_eq!(
            sent_to, expected,
            "{}: a new player starts on the spawn point's shard",
            region.shard
        );
        assert_eq!(map.epoch, 4, "{} runs the committed map", region.shard);
        assert_eq!(map.regions.len(), 4);
        for (node, owner) in [root, cells[0], cells[1], cells[2]]
            .into_iter()
            .zip(&owners)
        {
            assert_eq!(map.lookup(&node.center()), Some(*owner));
        }
        assert!(map.regions.iter().all(|r| r.endpoint.port() != 0));
    }

    // It stays that way: nothing more is launched, nothing flaps.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let later = orch.ops.snapshot();
    assert!(settled_with(&later, 4));
    assert_eq!(later.epoch, 4);
    assert_eq!(later.next_shard_id, 5);

    for launch in 1..=4 {
        launcher.kill(LaunchId(launch)).await.expect("stop");
    }
    orch.stop().await;
}

/// Supervision through the real loop: the shard that owns the root — the one
/// that spawned the scene's bodies — is stopped behind the orchestrator's
/// back. The orchestrator notices (the launcher's poll, and the closed link),
/// starts a replacement, gives it the root under a **new id** through an
/// epoch change the dead shard is not asked to acknowledge, and then raises
/// the scene's revision, once. The replacement builds the scene's bodies
/// afresh: a player who joins it sees bodies, all of the new revision and
/// minted by the new shard.
#[tokio::test(flavor = "multi_thread")]
async fn a_shard_that_dies_is_replaced_under_a_new_id_and_the_scene_is_built_afresh() {
    let launcher = Arc::new(LocalLauncher::new());
    let orch = Running::start(
        "replace",
        OrchestratorConfig {
            min_shards: 2,
            ..OrchestratorConfig::default()
        },
        Arc::clone(&launcher),
    );
    let before = orch
        .wait_for("two settled shards", Duration::from_secs(30), |s| {
            settled_with(s, 2)
        })
        .await;
    let root = before.root;
    let dead = ShardId::new(1);
    let other = before
        .shards
        .iter()
        .map(|s| s.shard)
        .find(|s| *s != dead)
        .expect("a second shard");
    let scene_before = before.scene.current();
    assert_eq!(scene_before.revision, 1);
    // The root's shard spawned the scene: a player there sees its bodies, all
    // minted by shard 1 for revision 1.
    let root_endpoint = before.regions[0].endpoint.expect("registered");
    assert_eq!(before.regions[0].shard, dead);
    let (_, guids) = join_and_look(root_endpoint).await;
    let bodies_before: Vec<_> = guids
        .iter()
        .filter(|g| space_server::border::scene_revision_of(**g).is_some())
        .collect();
    assert!(
        !bodies_before.is_empty(),
        "the demo area has bodies: {guids:?}"
    );

    // The root's shard dies.
    let launch = launcher.launch_of(dead).await.expect("launched");
    let died = Instant::now();
    launcher
        .kill(launch)
        .await
        .expect("stopped behind the orchestrator's back");

    let after = orch
        .wait_for(
            "the dead shard to be replaced and the scene confirmed again",
            Duration::from_secs(30),
            |s| settled_with(s, 2) && !s.shards.iter().any(|shard| shard.shard == dead),
        )
        .await;
    println!(
        "the dead shard was replaced and the scene confirmed again {:.2?} after it died",
        died.elapsed()
    );
    let new = after.regions[0].shard;
    assert_eq!(after.regions[0].node, root);
    assert!(
        new > other && new > dead,
        "a new id, above every id issued: {new}"
    );
    assert_eq!(after.epoch, before.epoch + 1, "one epoch change");
    assert_eq!(after.regions.len(), 2);
    assert!(after.shards.iter().any(|s| s.shard == other && s.connected));
    assert_eq!(
        after.scene.current(),
        SceneState::new(scene_before.scene, scene_before.revision + 1),
        "the same scene, one revision up"
    );
    let texts: Vec<&str> = after.events.iter().map(|e| e.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.starts_with("shard 1 lost:")),
        "{texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t.contains("committed: replacement of shard 1")),
        "{texts:?}"
    );
    assert_eq!(
        texts
            .iter()
            .filter(|t| t.contains("replaced shard 1: the scene"))
            .count(),
        1,
        "the revision is raised once: {texts:?}"
    );

    // The replacement spawned the scene afresh: bodies again, every one of
    // revision 2 and minted by the new shard — none of the dead shard's.
    let new_endpoint = after.regions[0].endpoint.expect("registered");
    let deadline = Instant::now() + Duration::from_secs(10);
    let bodies = loop {
        let (map, guids) = join_and_look(new_endpoint).await;
        assert_eq!(map.lookup(&root.center()), Some(new));
        let bodies: Vec<_> = guids
            .into_iter()
            .filter(|g| space_server::border::scene_revision_of(*g).is_some())
            .collect();
        if !bodies.is_empty() {
            break bodies;
        }
        assert!(
            Instant::now() < deadline,
            "the replacement never built the scene"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert_eq!(
        bodies.len(),
        bodies_before.len(),
        "as many bodies as before, not twice as many"
    );
    for guid in &bodies {
        assert_eq!(
            space_server::border::scene_revision_of(*guid),
            Some(2),
            "{guid:?}"
        );
        assert_eq!(guid.shard, new, "minted by the new spawner: {guid:?}");
    }

    // It does not happen again by itself.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let later = orch.ops.snapshot();
    assert_eq!(later.scene.revision, 2);
    assert_eq!(later.epoch, after.epoch);
    assert!(settled_with(&later, 2));

    for launch in 1..=3 {
        launcher.kill(LaunchId(launch)).await.expect("stop");
    }
    orch.stop().await;
}

/// The cluster secret, over the real transport. With one configured, a shard
/// started by hand is refused **for good** without it — and with a wrong one —
/// and says so; with it, it is welcomed. The shard the orchestrator starts
/// itself is handed the secret and needs nothing.
#[tokio::test(flavor = "multi_thread")]
async fn a_shard_without_the_cluster_secret_is_refused_for_good() {
    const SECRET: &str = "the-secret-of-this-test-cluster";
    let launcher = Arc::new(LocalLauncher::new());
    let orch = Running::start(
        "secret",
        OrchestratorConfig {
            min_shards: 1,
            cluster_secret: Some(Secret::new(SECRET)),
            ..OrchestratorConfig::default()
        },
        Arc::clone(&launcher),
    );
    // The launched shard presents the secret it was handed: the root is taken.
    let first = orch
        .wait_for("the launched shard", Duration::from_secs(30), |s| {
            settled_with(s, 1)
        })
        .await;
    assert!(first.hello_auth_required);

    let by_hand = |secret: &str| OrchestratedOptions {
        shard: ShardId::NONE,
        bind: loopback0(),
        advertise: None,
        claim_region: None,
        cluster: orch.cluster.clone(),
        client_endpoint: None,
        secret: secret.to_owned(),
    };
    for wrong in ["", "not-the-secret-of-this-cluster"] {
        let started = Instant::now();
        let err = tokio::time::timeout(
            Duration::from_secs(10),
            start_orchestrated(by_hand(wrong), orch.control_addr),
        )
        .await
        .expect("a refusal is not waited out")
        .expect_err("no secret, no region");
        if wrong.is_empty() {
            // No secret: nothing to verify with; the orchestrator refuses the hello.
            assert!(
                err.contains("rejected") && err.contains("cluster secret"),
                "{wrong:?}: {err}"
            );
        } else {
            // A wrong secret: the shard sees the orchestrator's certificate is
            // not of its secret's CA and never sends the hello.
            assert!(
                err.contains("did not prove it holds") && err.contains("cluster secret"),
                "{wrong:?}: {err}"
            );
        }
        assert!(!err.contains(SECRET), "the secret is never said: {err}");
        assert!(started.elapsed() < Duration::from_secs(5), "final, at once");
    }
    let snapshot = orch
        .wait_for("the refusal to be logged", Duration::from_secs(5), |s| {
            s.events.iter().any(|e| {
                e.text
                    .contains("refused: the cluster secret is missing or wrong")
            })
        })
        .await;
    assert_eq!(snapshot.next_shard_id, 2, "a refused hello burned no id");
    assert_eq!(snapshot.epoch, 1);
    assert!(
        !format!("{snapshot:?}").contains(SECRET),
        "and the secret is in no snapshot"
    );

    let welcomed = start_orchestrated(by_hand(SECRET), orch.control_addr)
        .await
        .expect("with the secret it is welcomed");
    assert_eq!(welcomed.shard, ShardId::new(2));
    orch.wait_for("its cell", Duration::from_secs(10), |s| settled_with(s, 2))
        .await;

    welcomed.abort();
    launcher.kill(LaunchId(1)).await.expect("stop");
    orch.stop().await;
}

// ------------------------------------------- who may say what on the control port

/// Dial the control port as a shard does and say `hello` — but by hand, so
/// that the test keeps the connection whatever the answer is. Returns the
/// connection and the orchestrator's answer to the hello.
async fn hello_by_hand(control: SocketAddr, hello: ShardHello) -> (space_net::Conn, Message) {
    let endpoint = space_net::client_endpoint_for(control).expect("client endpoint");
    let (conn, _placeholder) = space_net::connect(
        &endpoint,
        control,
        space_orchestrator::control::SERVER_NAME,
        &space_net::hello("shard:0"),
    )
    .await
    .expect("the transport handshake completes");
    conn.send_control(&Message::ShardHello(hello))
        .await
        .expect("the hello is sent");
    let answer = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::select! {
            m = conn.recv_control() => m,
            m = conn.recv_stream() => m,
        }
    })
    .await
    .expect("the orchestrator answers a hello")
    .expect("the answer can be read")
    .expect("the answer is a message");
    (conn, answer)
}

/// A hello with neither id nor region, as every shard says it.
fn plain_hello(cluster: &str, port: u16, secret: &str) -> ShardHello {
    ShardHello {
        cluster: cluster.to_owned(),
        shard: ShardId::NONE,
        endpoint: SocketAddr::from(([127, 0, 0, 1], port)),
        region: None,
        client_endpoint: None,
        secret: secret.to_owned(),
    }
}

/// Everything a shard may say after its hello, forged in `shard`'s name.
fn forged_as(shard: ShardId, epoch: u64, scene: SceneState) -> Vec<Message> {
    vec![
        Message::LoadReport(LoadReport {
            shard,
            players: 77,
            tick_p95_ms: 1.0,
            load: 0.0,
            ..Default::default()
        }),
        Message::EpochAck(EpochAck { shard, epoch }),
        Message::SceneAck(SceneAck {
            shard,
            applied: scene,
        }),
    ]
}

/// Send `messages` on `conn` the way a shard does (best effort: a connection
/// the orchestrator is closing may refuse them, which is as good).
async fn say(conn: &space_net::Conn, messages: &[Message]) {
    for message in messages {
        let _ = conn.send_reliable(message).await;
    }
}

/// Wait until the orchestrator has closed `conn`, and say how.
async fn closed_by_the_orchestrator(conn: &space_net::Conn, what: &str) -> quinn::ConnectionError {
    tokio::time::timeout(Duration::from_secs(5), conn.connection().closed())
        .await
        .unwrap_or_else(|_| panic!("{what}: the orchestrator must close the connection"))
}

/// The cluster secret gates the hello, and the hello gates everything else:
/// a connection whose hello is refused is told why and **closed**, and
/// nothing else it says — before the close, in the name of a real shard —
/// ever reaches the orchestrator.
///
/// Mutation this catches: reading on after a refused hello. Reports and acks
/// name their own sender, so anybody who could reach the control port could
/// report load for a shard (and have it split or merged), acknowledge a map
/// change in its name, or "confirm" a scene revision that makes the
/// orchestrator re-issue the scene — all without the secret.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_hello_is_closed_and_nothing_else_it_says_is_heard() {
    const SECRET: &str = "the-secret-of-the-gated-cluster";
    let launcher = Arc::new(LocalLauncher::new());
    let orch = Running::start(
        "gate",
        OrchestratorConfig {
            min_shards: 1,
            cluster_secret: Some(Secret::new(SECRET)),
            ..OrchestratorConfig::default()
        },
        Arc::clone(&launcher),
    );
    let before = orch
        .wait_for("the launched shard", Duration::from_secs(30), |s| {
            settled_with(s, 1)
        })
        .await;
    let real = before.shards[0].shard;
    let scene = before.scene.current();
    // A revision a little ahead of the cluster's: believed, it would make the
    // orchestrator re-issue the scene one above it within a second.
    let ahead = SceneState::new(scene.scene, scene.revision + 3);

    for (what, hello, expected) in [
        (
            "no secret",
            plain_hello(&orch.cluster, 19_001, ""),
            RejectReason::Unauthorized,
        ),
        (
            "a wrong secret",
            plain_hello(&orch.cluster, 19_002, "not-the-secret-of-this-cluster"),
            RejectReason::Unauthorized,
        ),
        (
            // Without the secret nothing is said about the cluster, not even
            // that the hello named the wrong one.
            "another cluster and no secret",
            plain_hello("some-other-cluster", 19_003, ""),
            RejectReason::Unauthorized,
        ),
    ] {
        let (conn, answer) = hello_by_hand(orch.control_addr, hello).await;
        assert_eq!(
            answer,
            Message::Reject(Reject {
                reason: expected.clone()
            }),
            "{what}"
        );
        // Still connected for a moment (the orchestrator gives the refusal
        // time to be read): everything a shard could say, in the real
        // shard's name — and a second hello, this time with the secret.
        say(&conn, &forged_as(real, before.epoch + 1, ahead)).await;
        say(
            &conn,
            &[Message::ShardHello(plain_hello(
                &orch.cluster,
                19_009,
                SECRET,
            ))],
        )
        .await;
        let closed = closed_by_the_orchestrator(&conn, what).await;
        assert!(
            matches!(closed, quinn::ConnectionError::ApplicationClosed(_)),
            "{what}: closed by the orchestrator, not timed out: {closed:?}"
        );
    }

    // Long enough for a forged scene ack to have raised the revision (once a
    // second) and for a forged report to show.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let after = orch.ops.snapshot();
    assert_eq!(after.scene.current(), scene, "no scene revision was forged");
    assert_eq!(after.epoch, before.epoch);
    assert_eq!(
        after.shards.len(),
        1,
        "nobody was let in on a refused connection"
    );
    assert_eq!(after.next_shard_id, before.next_shard_id, "and no id spent");
    let shard = &after.shards[0];
    // (Its own report, yes — the shard reports once a second — but not the
    // forged one, which claims 77 players.)
    assert_eq!(
        shard.report.map(|report| report.players),
        Some(0),
        "no forged load report: {shard:?}"
    );
    assert_eq!(
        shard.scene,
        Some(scene),
        "what it confirmed is its own word"
    );
    assert!(settled_with(&after, 1));
    assert!(
        !format!("{after:?}").contains(SECRET),
        "and the secret is in no snapshot"
    );

    launcher.kill(LaunchId(1)).await.expect("stop");
    orch.stop().await;
}

/// A connection that was let in is let in **as one shard**: it may report
/// and acknowledge for the id it was welcomed as, and for no other. A hello
/// that is told to wait is refused like any other — told why, and closed; the
/// shard asks again on a new connection.
///
/// Mutation this catches: believing the id inside a report or an ack. A
/// registered shard could then acknowledge a map change for a neighbour that
/// had not seen it (the change would commit without it), report load in its
/// name, or confirm its scene.
#[tokio::test(flavor = "multi_thread")]
async fn a_connection_speaks_only_for_the_shard_it_was_admitted_as() {
    let launcher = Arc::new(LocalLauncher::new());
    let orch = Running::start("speak", OrchestratorConfig::default(), launcher);
    let cluster = orch.cluster.clone();
    let scene = orch.ops.snapshot().scene.current();

    // Two "shards", played by hand: the first is given the root, the second
    // a cell — through a map change both have to acknowledge.
    let (first, welcome) =
        hello_by_hand(orch.control_addr, plain_hello(&cluster, 19_101, "")).await;
    let Message::ShardWelcome(welcome) = welcome else {
        panic!("the first hello is welcomed, got {welcome:?}");
    };
    let one = welcome.shard;
    let (second, welcome) =
        hello_by_hand(orch.control_addr, plain_hello(&cluster, 19_102, "")).await;
    let Message::ShardWelcome(welcome) = welcome else {
        panic!("the second hello is welcomed, got {welcome:?}");
    };
    let two = welcome.shard;
    assert_ne!(one, two);
    let flight = orch
        .wait_for(
            "the second shard's map change",
            Duration::from_secs(10),
            |s| s.in_flight.is_some(),
        )
        .await;
    let epoch = flight.in_flight.as_ref().expect("in flight").epoch;
    assert_eq!(epoch, flight.epoch + 1);

    // A third hello while that change is in flight: "not yet" — and the
    // connection is closed like any refused one. What it says in the
    // meantime, in the first shard's name, is not heard.
    let (third, answer) = hello_by_hand(orch.control_addr, plain_hello(&cluster, 19_103, "")).await;
    assert_eq!(
        answer,
        Message::Reject(Reject {
            reason: RejectReason::Draining
        }),
        "one change at a time: ask again"
    );
    say(&third, &forged_as(one, epoch, scene)).await;
    // (That it is closed is looked at below: the orchestrator gives a refusal
    // a second to be read, and the change in flight has a deadline.)

    // The second shard's connection speaking for the first: dropped, every
    // message of it. Its own ack counts.
    say(&second, &forged_as(one, epoch, scene)).await;
    say(
        &second,
        &[Message::EpochAck(EpochAck { shard: two, epoch })],
    )
    .await;
    // (And a second hello on a connection that is already somebody changes
    // nothing: a connection is admitted once.)
    say(
        &second,
        &[Message::ShardHello(plain_hello(&cluster, 19_104, ""))],
    )
    .await;
    let acked = orch
        .wait_for("the second shard's own ack", Duration::from_secs(10), |s| {
            s.in_flight.as_ref().is_some_and(|flight| {
                flight
                    .recipients
                    .iter()
                    .any(|(shard, acked)| *shard == two && *acked)
            })
        })
        .await;
    let recipients = &acked.in_flight.as_ref().expect("in flight").recipients;
    assert_eq!(
        recipients,
        &vec![(one, false), (two, true)],
        "the first shard has not acknowledged anything"
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    let still = orch.ops.snapshot();
    assert_eq!(
        still.epoch,
        epoch - 1,
        "nothing committed on the second shard's word"
    );
    let view = |snapshot: &Snapshot, shard: ShardId| {
        snapshot
            .shards
            .iter()
            .find(|s| s.shard == shard)
            .cloned()
            .expect("the shard is known")
    };
    assert!(view(&still, one).report.is_none(), "no report in its name");
    assert_eq!(
        view(&still, one).scene,
        None,
        "no scene confirmed in its name"
    );
    assert_eq!(still.shards.len(), 2, "and nobody new was let in");
    assert_eq!(still.next_shard_id, u64::from(two.0) + 1);

    // The first shard speaks for itself: the change commits, its report and
    // its scene are taken.
    say(&first, &[Message::EpochAck(EpochAck { shard: one, epoch })]).await;
    orch.wait_for("the commit", Duration::from_secs(10), |s| s.epoch == epoch)
        .await;
    say(
        &first,
        &[
            Message::LoadReport(LoadReport {
                shard: one,
                players: 3,
                tick_p95_ms: 1.0,
                load: 0.0,
                ..Default::default()
            }),
            Message::SceneAck(SceneAck {
                shard: one,
                applied: scene,
            }),
        ],
    )
    .await;
    let heard = orch
        .wait_for(
            "the first shard's own report and ack",
            Duration::from_secs(10),
            |s| {
                s.shards.iter().any(|shard| {
                    shard.shard == one && shard.report.is_some() && shard.scene.is_some()
                })
            },
        )
        .await;
    assert_eq!(view(&heard, one).report.map(|r| r.players), Some(3));
    assert_eq!(view(&heard, one).scene, Some(scene));
    assert_eq!(heard.regions.len(), 2);

    // The hello that was told to wait: its connection was closed by the
    // orchestrator, and nothing it said was heard (see above: no report, no
    // scene, no ack in the first shard's name).
    let closed = closed_by_the_orchestrator(&third, "a hello that is told to wait").await;
    assert!(
        matches!(closed, quinn::ConnectionError::ApplicationClosed(_)),
        "{closed:?}"
    );

    first.close(0, "done");
    second.close(0, "done");
    orch.stop().await;
}

/// A `ShardHello` the run loop gets to **after its shard has given up on
/// it** is not served: no shard is registered, no id is spent, no change of
/// the map is proposed — and the shard, which asks again on a new
/// connection, is served as if it were asking for the first time.
///
/// The test is the run loop here: the front end accepts connections and
/// queues their hellos, and the test decides when each is answered and how
/// old it is by then. Three hellos:
///
/// 1. answered after the shard has **closed the connection** (what a shard
///    does when it has waited `HANDSHAKE_TIMEOUT` for the answer);
/// 2. answered `HELLO_STALE_AFTER` after it arrived, on a connection that is
///    still open: told "not yet" — the shard asks again at once;
/// 3. answered at once: welcomed, as the first shard of the cluster.
///
/// Mutation this catches (run): serving a hello whatever its age and
/// whatever became of its connection (`hello_is_stale` always `false`). The
/// first hello was then given the root and registered on a connection
/// nobody was on; the orchestrator found that link closed a beat later, and
/// the shard's next hello was a *replacement* under another id, through an
/// epoch — with a launched shard, whose process was by then staged for
/// stopping, that took the epoch's deadline and a launch pause: 14 s instead
/// of half a second in CI.
#[tokio::test(flavor = "multi_thread")]
async fn a_hello_whose_shard_has_given_up_on_it_is_not_served() {
    use space_orchestrator::control::{Inbound, HELLO_STALE_AFTER};

    let state_path = temp_path("stale").with_extension("id");
    let _ = std::fs::remove_file(&state_path);
    let config = OrchestratorConfig {
        bind: loopback0(),
        state_path: state_path.clone(),
        ..OrchestratorConfig::default()
    };
    let cluster = config.cluster.clone();
    let control = boot(config, "peer-token-of-the-test-cluster".into()).expect("boot");
    let mut orchestrator = Orchestrator::new(control, Arc::new(LocalLauncher::new()));
    let server = orchestrator.bind().expect("bind control");
    let control_addr = server.local_addr().expect("control port");
    let accept = Arc::clone(&server).spawn_accept();
    let ids = orchestrator.snapshot().next_shard_id;

    // Say hello as a shard does, and hand back the connection and the hello
    // as it reaches the run loop's queue.
    let say_hello = |port: u16| {
        let (server, cluster) = (Arc::clone(&server), cluster.clone());
        async move {
            let endpoint = space_net::client_endpoint_for(control_addr).expect("client endpoint");
            let (conn, _placeholder) = space_net::connect(
                &endpoint,
                control_addr,
                space_orchestrator::control::SERVER_NAME,
                &space_net::hello("shard:0"),
            )
            .await
            .expect("the transport handshake completes");
            conn.send_control(&Message::ShardHello(plain_hello(&cluster, port, "")))
                .await
                .expect("the hello is sent");
            let queued = tokio::time::timeout(Duration::from_secs(10), server.next_inbound())
                .await
                .expect("the hello reaches the run loop's queue")
                .expect("the front end is there");
            (conn, queued)
        }
    };
    let answer = |conn: space_net::Conn| async move {
        tokio::time::timeout(Duration::from_secs(10), async {
            tokio::select! {
                m = conn.recv_control() => m,
                m = conn.recv_stream() => m,
            }
        })
        .await
        .expect("the orchestrator answers")
        .expect("the answer can be read")
        .expect("the answer is a message")
    };
    let nothing_decided = |orchestrator: &Orchestrator<LocalLauncher>, what: &str| {
        let snapshot = orchestrator.snapshot();
        // (The root's reserved owner is listed from the start: expected, and
        // not there.)
        assert!(
            snapshot
                .shards
                .iter()
                .all(|shard| !shard.connected && shard.advertised.is_none()),
            "{what}: nobody is registered: {:?}",
            snapshot.shards
        );
        assert!(orchestrator.live_shards().is_empty(), "{what}");
        assert!(orchestrator.dead_shards().is_empty(), "{what}");
        assert_eq!(snapshot.next_shard_id, ids, "{what}: no id was spent");
        assert!(snapshot.in_flight.is_none(), "{what}: nothing is proposed");
        assert_eq!(snapshot.epoch, 1, "{what}");
    };

    // 1. The shard gives up before the loop gets to its hello.
    let (conn, queued) = say_hello(19_101).await;
    let Inbound::Hello {
        hello,
        reply,
        conn: its_conn,
        verdict,
        ..
    } = queued
    else {
        panic!("a hello, got {queued:?}");
    };
    conn.close(0, "gave up waiting");
    tokio::time::timeout(Duration::from_secs(10), its_conn.connection().closed())
        .await
        .expect("the orchestrator's end sees the close");
    orchestrator
        .answer_hello(&server, &hello, reply, its_conn, verdict, Duration::ZERO)
        .await;
    assert_eq!(orchestrator.abandoned_hellos(), 1);
    nothing_decided(&orchestrator, "a hello on a closed connection");

    // 2. The loop gets to a hello later than a shard waits for its answer.
    let (conn, queued) = say_hello(19_101).await;
    let Inbound::Hello {
        hello,
        reply,
        conn: its_conn,
        verdict,
        ..
    } = queued
    else {
        panic!("a hello, got {queued:?}");
    };
    orchestrator
        .answer_hello(&server, &hello, reply, its_conn, verdict, HELLO_STALE_AFTER)
        .await;
    assert_eq!(orchestrator.abandoned_hellos(), 2);
    nothing_decided(&orchestrator, "a hello answered too late");
    assert_eq!(
        answer(conn).await,
        Message::Reject(Reject {
            reason: RejectReason::Draining
        }),
        "told to ask again"
    );
    let said: Vec<String> = orchestrator
        .events()
        .newest_first()
        .filter(|e| e.text.contains("was not served"))
        .map(|e| e.text.clone())
        .collect();
    assert_eq!(said.len(), 2, "each is in the event log: {said:?}");
    assert!(said[0].contains("127.0.0.1:19101"), "{said:?}");
    assert!(said[0].contains("4.0 s after it arrived"), "{said:?}");
    assert!(said[1].contains("its connection was closed"), "{said:?}");

    // 3. The shard asks again and is answered at once: the first shard of
    //    the cluster, under the reserved id — not the replacement of a
    //    shard that never was.
    let (conn, queued) = say_hello(19_101).await;
    let Inbound::Hello {
        hello,
        reply,
        conn: its_conn,
        verdict,
        received,
    } = queued
    else {
        panic!("a hello, got {queued:?}");
    };
    orchestrator
        .answer_hello(
            &server,
            &hello,
            reply,
            its_conn,
            verdict,
            received.elapsed(),
        )
        .await;
    let Message::ShardWelcome(welcome) = answer(conn.clone()).await else {
        panic!("a welcome");
    };
    assert_eq!(welcome.shard, ShardId::new(1), "the reserved root id");
    assert_eq!(welcome.shard_map.epoch, 1, "no epoch was needed");
    assert_eq!(orchestrator.abandoned_hellos(), 2);
    let snapshot = orchestrator.snapshot();
    assert_eq!(snapshot.shards.len(), 1);
    assert_eq!(snapshot.next_shard_id, ids, "the root's id was reserved");
    assert!(orchestrator.dead_shards().is_empty());

    conn.close(0, "done");
    accept.abort();
    server.close();
    let _ = std::fs::remove_file(&state_path);
    let _ = std::fs::remove_file(space_orchestrator::scene::scene_path(&state_path));
}

// ------------------------------------------------------- the process launcher

/// The `space-server` executable the process launcher would start for the
/// orchestrator binary of this build: the one lying next to it.
///
/// Cargo builds the orchestrator's own binary for this test
/// (`CARGO_BIN_EXE_space-orchestrator`); `space-server` belongs to another
/// package, so it is only there if something built it — and then possibly from
/// an older commit. This test therefore builds it itself first, with the same
/// cargo, profile and target directory (a no-op when it is fresh).
fn space_server_exe() -> PathBuf {
    let orchestrator_exe = PathBuf::from(env!("CARGO_BIN_EXE_space-orchestrator"));
    let profile_dir = orchestrator_exe
        .parent()
        .expect("the orchestrator binary lies in a profile directory")
        .to_path_buf();
    let exe = profile_dir.join(SHARD_EXE);
    let profile = profile_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_owned();
    let target_dir = profile_dir
        .parent()
        .expect("a profile directory lies in a target directory");
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let mut build = std::process::Command::new(cargo);
    build
        .current_dir(&workspace)
        .args([
            "build",
            "--locked",
            "-p",
            "space-server",
            "--bin",
            "space-server",
        ])
        .arg("--target-dir")
        .arg(target_dir);
    match profile.as_str() {
        "debug" => {}
        "release" => {
            build.arg("--release");
        }
        other => {
            build.args(["--profile", other]);
        }
    }
    let status = build
        .status()
        .expect("cargo can be run to build the space-server binary");
    assert!(
        status.success(),
        "building space-server (for {}) failed",
        exe.display()
    );
    assert!(exe.is_file(), "{} was not built", exe.display());
    exe
}

/// `count` consecutive UDP ports on loopback that are free right now,
/// **below the range the operating system hands out by itself**.
///
/// Not "ask for port 0 and take what comes": the ports that gives are the
/// very next ones the system gives to somebody else — the orchestrator's
/// control port, the socket a shard dials the orchestrator from — and a shard
/// that then binds `127.0.0.1:<that port>` takes over the datagrams of a
/// socket bound to the wildcard address on the same port. (That is how this
/// test first failed: the root shard stopped hearing the orchestrator the
/// moment its neighbour was started. A real deployment's host port range,
/// 7870-7899 by default, is not in the dynamic range either.)
fn free_port_range(count: u16) -> (u16, u16) {
    let spread = u16::try_from(std::process::id() % 4_000).unwrap_or(0);
    for attempt in 0..200u16 {
        let first = 20_000 + (spread + attempt * 16) % 9_000;
        let held: Vec<_> = (0..count)
            .map(|n| std::net::UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], first + n))))
            .collect();
        if held.iter().all(Result::is_ok) {
            return (first, first + count - 1);
        }
    }
    panic!("no run of {count} free UDP ports was found");
}

/// Stop the process `pid` the hard way, as a crash would.
///
/// With what every system this runs on has: `taskkill` is part of Windows,
/// and elsewhere `kill` is asked of `/bin/sh`, where it is a **builtin** —
/// not the `kill` program, which a minimal image does not have (the CI's
/// `rust:1-slim-bookworm` build image has no procps: this test failed there
/// with "No such file or directory" before it had killed anything).
fn kill_process(pid: u32) {
    let pid = pid.to_string();
    let status = if cfg!(windows) {
        std::process::Command::new("taskkill")
            .args(["/PID", &pid, "/F"])
            .stdout(std::process::Stdio::null())
            .status()
    } else {
        std::process::Command::new("/bin/sh")
            .args(["-c", &format!("kill -9 {pid}")])
            .status()
    };
    assert!(
        status.expect("the kill command runs").success(),
        "process {pid} could not be stopped"
    );
}

/// The `process` launcher against the real `space-server` binary: two shards
/// are started as processes, each on a port of the range, each logging to a
/// file of its own and noted in the pid file; they register (no id, no
/// region) and are given the root and a cell. One process is then killed from
/// outside: the orchestrator notices, starts another and gives it the dead
/// one's region under a new id — and then that one is killed too. When the
/// launcher goes, its processes go, and the pid file with them.
///
/// What is checked here that nothing else checks: the command line the
/// launcher builds is one the real binary accepts, the secret reaches the
/// child through its environment (the orchestrator requires it here), the
/// hello of a process is matched to its launch by the address it advertises,
/// a process that exits is found by `poll`, and the pid file lists exactly
/// the processes that are running.
///
/// The host port range is **exactly as large as the fleet**, so a replacement
/// is handed the very port of the shard it replaces — twice. That is the
/// sequence in which a launch used to be put down to the dead shard that
/// still advertised that address: the replacement was then supervised by
/// nobody, its death was only noticed when its control link timed out (ten
/// seconds), and its port was never free again.
#[tokio::test(flavor = "multi_thread")]
async fn the_process_launcher_starts_real_shards_and_replaces_one_that_is_killed() {
    let exe = space_server_exe();
    let log_dir = temp_path("process");
    let ports = free_port_range(2);
    let bind = loopback0();
    let launcher = Arc::new(ProcessLauncher::new(&exe, &log_dir, bind));
    launcher
        .ensure_available()
        .expect("the binary was just built");

    let started = Instant::now();
    let orch = Running::start(
        "process",
        OrchestratorConfig {
            min_shards: 2,
            launcher: LauncherKind::Process,
            host_port_range: ports,
            // Required of every hello: the children can only know it from
            // the environment the launcher gives them.
            cluster_secret: Some(Secret::new("a-secret-only-the-launcher-hands-out")),
            ..OrchestratorConfig::default()
        },
        Arc::clone(&launcher),
    );
    let before = orch
        .wait_for("two shard processes", Duration::from_secs(60), |s| {
            settled_with(s, 2)
        })
        .await;
    println!(
        "two regions owned by two shard processes {:.2?} after the orchestrator started",
        started.elapsed()
    );
    assert!(before.shards.iter().all(|s| s.launched && s.connected));
    // Each listens on a port of the range, on loopback, as a process the
    // launcher supervises.
    for shard in &before.shards {
        let at = shard.advertised.expect("registered");
        assert!(
            at.ip().is_loopback() && (ports.0..=ports.1).contains(&at.port()),
            "{at}"
        );
        assert!(
            launcher.log_path(at.port()).is_file(),
            "a log file for the shard on {at}"
        );
        assert!(
            launcher.pid_at(at).is_some(),
            "a process for the shard on {at}"
        );
    }
    // The pid file lists exactly the processes that are running: the
    // launcher's own table, and one per shard — whatever it took to get
    // there (a launch that had to be tried again leaves nothing behind).
    let noted = |launcher: &ProcessLauncher| -> Vec<u32> {
        match std::fs::read_to_string(launcher.pid_path()) {
            Ok(text) => text
                .lines()
                .map(|line| line.trim().parse().expect("a pid a line"))
                .collect(),
            Err(_) => Vec::new(),
        }
    };
    let pid_of = |snapshot: &Snapshot, shard: ShardId| -> u32 {
        let at = snapshot
            .shards
            .iter()
            .find(|s| s.shard == shard)
            .and_then(|s| s.advertised)
            .expect("registered");
        launcher.pid_at(at).expect("a process the launcher runs")
    };
    let pids = noted(&launcher);
    assert_eq!(pids, launcher.live_pids(), "the file is the table");
    assert_eq!(pids.len(), before.shards.len(), "one process per shard");
    for shard in &before.shards {
        assert!(pids.contains(&pid_of(&before, shard.shard)), "{pids:?}");
    }
    // A player is served by a real process.
    let root_endpoint = before.regions[0].endpoint.expect("registered");
    let (map, _) = join_and_look(root_endpoint).await;
    assert_eq!(map.regions.len(), 2);

    // The second shard's process is killed from outside — and then, when it
    // has been replaced, its replacement: the same port, twice.
    let root = before.regions[0].shard;
    let mut victim = before.regions[1].shard;
    let victim_port = before.regions[1].endpoint.expect("registered").port();
    assert_ne!(victim, root);
    let mut current = before;
    for round in 1..=2u32 {
        let victim_pid = pid_of(&current, victim);
        let died = Instant::now();
        kill_process(victim_pid);
        let after = orch
            .wait_for(
                "the killed process to be replaced",
                Duration::from_secs(60),
                |s| settled_with(s, 2) && !s.shards.iter().any(|shard| shard.shard == victim),
            )
            .await;
        let took = died.elapsed();
        println!(
            "round {round}: the killed shard process was replaced and the scene confirmed again \
             {took:.2?} after the kill"
        );
        let heir = after.regions[1].shard;
        assert_eq!(
            after.regions[1].node, current.regions[1].node,
            "round {round}: the same region"
        );
        assert!(heir > victim, "round {round}: under a new id: {heir}");
        assert_eq!(
            after.regions[0].shard, root,
            "round {round}: the root's shard is untouched"
        );
        assert_eq!(after.scene.revision, current.scene.revision + 1);
        assert_eq!(
            after.regions[1].endpoint.map(|at| at.port()),
            Some(victim_port),
            "round {round}: the only free port is the dead shard's"
        );
        // Noticed through the launcher (asked twice a second), not when the
        // dead process's control link timed out ten seconds later.
        let texts: Vec<&str> = after.events.iter().map(|e| e.text.as_str()).collect();
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with(&format!("{victim} lost: its process exited"))),
            "round {round}: {texts:?}"
        );
        // And replaced in one go: no change of the map ran into its
        // deadline on the way (three seconds for an ack that cannot come —
        // what a replacement cost when a hello of the process that was
        // being stopped was welcomed), no launch failed, and no launch was
        // followed by a pause.
        //
        // (Not "within so many seconds": on a loaded machine that also
        // measures how long the machine kept the orchestrator and the new
        // process from running — an 8 s bound here failed in CI for a run
        // loop that was held up for more than five. What it stood for is
        // said above and below. Locally: 0.4 to 0.5 s.)
        for defect in [
            "aborted at its deadline",
            "launching a shard failed",
            "exited before it registered",
            "did not register within",
            "no shard can be started",
        ] {
            assert!(
                !texts.iter().any(|t| t.contains(defect)),
                "round {round} ({took:.2?}): {texts:?}"
            );
        }
        let heir_shard = after
            .shards
            .iter()
            .find(|s| s.shard == heir)
            .expect("the heir");
        assert!(
            heir_shard.launched,
            "round {round}: the replacement is a launch the orchestrator supervises"
        );
        // The pid file: the two that run, and not the one that was killed.
        let pids_after = noted(&launcher);
        assert_eq!(pids_after, launcher.live_pids(), "round {round}");
        assert_eq!(pids_after.len(), 2, "round {round}: {pids_after:?}");
        assert!(!pids_after.contains(&victim_pid), "round {round}");
        assert!(pids_after.contains(&pid_of(&after, heir)), "round {round}");
        assert!(pids_after.contains(&pid_of(&after, root)), "round {round}");
        victim = heir;
        current = after;
    }
    // What the shards said is in their logs, and never the secret.
    let log =
        std::fs::read_to_string(launcher.log_path(root_endpoint.port())).expect("the root's log");
    assert!(log.contains("orchestrated by"), "{log}");
    for entry in std::fs::read_dir(&log_dir)
        .expect("the log directory")
        .flatten()
    {
        let text = std::fs::read_to_string(entry.path()).unwrap_or_default();
        assert!(
            !text.contains("a-secret-only-the-launcher-hands-out"),
            "{} contains the secret",
            entry.path().display()
        );
    }

    // The orchestrator ends: the launcher goes, and its processes with it.
    let shard_ports: Vec<u16> = current
        .shards
        .iter()
        .filter_map(|s| s.advertised.map(|at| at.port()))
        .collect();
    let pid_path = launcher.pid_path();
    orch.stop().await;
    // The loop asks its launcher off the loop: a question that was on its
    // way when the loop was stopped still holds the launcher for a moment.
    // It goes — and stops its processes — when the last holder lets go.
    let deadline = Instant::now() + Duration::from_secs(10);
    while Arc::strong_count(&launcher) > 1 {
        assert!(
            Instant::now() < deadline,
            "a task of the stopped orchestrator still holds its launcher"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    drop(launcher);
    assert!(
        !pid_path.exists(),
        "nothing is running, so there is no pid file"
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let free = shard_ports.iter().all(|port| {
            std::net::UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], *port))).is_ok()
        });
        if free {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "a shard process still holds one of the ports {shard_ports:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let _ = std::fs::remove_dir_all(&log_dir);
}

/// A launcher that starts nothing and, once asked whether what it started is
/// alive — or told to stop it — never answers: a container daemon that hangs.
#[derive(Debug, Default)]
struct SilentLauncher {
    asked: std::sync::atomic::AtomicU32,
}

impl Launcher for SilentLauncher {
    async fn spawn(
        &self,
        spec: space_orchestrator::launcher::SpawnSpec,
    ) -> Result<space_orchestrator::launcher::ShardHandle, space_orchestrator::launcher::LaunchError>
    {
        Ok(space_orchestrator::launcher::ShardHandle {
            launch: spec.launch,
            shard: None,
            endpoint: SocketAddr::from(([127, 0, 0, 1], spec.host_port)),
            process: "nothing".into(),
        })
    }

    async fn kill(
        &self,
        _launch: LaunchId,
    ) -> Result<(), space_orchestrator::launcher::LaunchError> {
        std::future::pending().await
    }

    async fn poll(
        &self,
        _launch: LaunchId,
    ) -> Result<space_orchestrator::launcher::Poll, space_orchestrator::launcher::LaunchError> {
        self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        std::future::pending().await
    }
}

/// The run loop does not wait for its launcher: with a launcher that never
/// answers a liveness question, the orchestrator's clock runs on, snapshots
/// are published and an operator is answered.
///
/// Mutation this catches (run): awaiting `launcher.poll` — or `kill` — in the
/// beat (the state before: the loop stood still for as long as the launcher
/// took; with the Docker launcher's bounds, 10 to 15 s per shard with a
/// daemon that hangs. Here: for ever, and the tick never moves again).
#[tokio::test(flavor = "multi_thread")]
async fn a_launcher_that_never_answers_does_not_stop_the_clock() {
    let launcher = Arc::new(SilentLauncher::default());
    let config = OrchestratorConfig {
        cluster: "silent".into(),
        min_shards: 1,
        launcher: LauncherKind::Process,
        host_port_range: free_port_range(2),
        ..OrchestratorConfig::default()
    };
    let orch = Running::start("silent", config, Arc::clone(&launcher));
    // The launch "started" (nothing registers), and the loop asks about it.
    let deadline = Instant::now() + Duration::from_secs(10);
    while launcher.asked.load(std::sync::atomic::Ordering::SeqCst) == 0 {
        assert!(Instant::now() < deadline, "the launcher was never asked");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // That question is never answered. Two seconds of ticks later the loop
    // is still beating —
    let asked_at = orch.ops.snapshot().tick;
    let later = orch
        .wait_for("the clock to run on", Duration::from_secs(20), |s| {
            s.tick >= asked_at + 60
        })
        .await;
    assert!(later.tick >= asked_at + 60);
    // — still answers an operator —
    let reply = tokio::time::timeout(
        Duration::from_secs(5),
        orch.ops
            .request(space_orchestrator::ops::OperatorCommand::PauseAutoscaler),
    )
    .await
    .expect("an operator is answered while the launcher hangs")
    .expect("the loop is there");
    assert!(reply.accepted, "{reply:?}");
    // — and has asked once, not once per poll interval.
    assert_eq!(
        launcher.asked.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "a question that is open is not asked again"
    );
    orch.stop().await;
}
