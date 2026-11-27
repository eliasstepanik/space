//! The orchestrator's **player port** through the real run loop
//! (`docs/plan-sharding.md` §4): the one address players type.
//!
//! Everything here is production code over loopback QUIC:
//! `Orchestrator::run_with_player`, the real [`PlayerServer`], real shards
//! started by `space_server::orchestrated::start_orchestrated`, and a client's
//! half of the handshake through `space_net::join` — the function the game
//! client joins through.
//!
//! | test | rule |
//! |---|---|
//! | [`the_player_port_sends_a_player_to_the_spawn_points_owner_and_says_starting_until_there_is_one`] | "starting" while nobody owns the spawn point; then a redirect naming the owner with the published map; the address in it is a shard that welcomes the player |
//! | [`after_the_roots_shard_is_replaced_the_player_port_names_the_new_owner`] | a dead owner is not named; its replacement is, under its new id |
//! | [`nothing_said_on_the_player_port_reaches_the_orchestrator`] | no shard registers there, forged acks and reports go nowhere, the state is untouched |
//! | [`a_player_who_dials_the_control_port_is_told_what_it_is`] | the control port answers a player with a readable refusal, not a placeholder welcome |
//! | [`a_shard_that_registers_without_an_ip_is_listed_with_the_one_it_is_seen_from`] | an unspecified internal IP is filled in from the control connection; the client endpoint asked for is in the map |

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_net::{Joined, NetError};
use space_orchestrator::boot::boot;
use space_orchestrator::config::OrchestratorConfig;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::ops::OpsHandle;
use space_orchestrator::orchestrator::Orchestrator;
use space_orchestrator::player::PlayerServer;
use space_orchestrator::snapshot::{ShardRole, Snapshot};
use space_protocol::control::LoadReport;
use space_protocol::shard::EpochAck;
use space_protocol::{JoinRedirect, Message, RejectReason, SceneAck, ShardHello, ShardId};
use space_server::net::SERVER_NAME;
use space_server::orchestrated::{start_orchestrated, OrchestratedOptions, RunningShard};
use space_test_support::loopback0;

fn temp_path(tag: &str) -> PathBuf {
    space_test_support::temp_path("space-orch-player", tag)
}

/// A running orchestrator with a player port: its loop in a task, the two
/// addresses, the player port itself (for its counters) and the handle its
/// web UI would read snapshots through.
struct Running {
    cluster: String,
    control_addr: SocketAddr,
    player_addr: SocketAddr,
    player: Arc<PlayerServer>,
    ops: OpsHandle,
    task: tokio::task::JoinHandle<Result<(), std::io::Error>>,
    state_path: PathBuf,
}

impl Running {
    /// Boot through the binary's own path (`boot`, `bind`, `bind_player`,
    /// `run_with_player`) on loopback ports chosen by the system. No shard is
    /// started by the orchestrator: the tests start theirs by hand.
    fn start(tag: &str) -> Self {
        let state_path = temp_path(tag).with_extension("id");
        let _ = std::fs::remove_file(&state_path);
        let config = OrchestratorConfig {
            bind: loopback0(),
            player_bind: Some(loopback0()),
            state_path: state_path.clone(),
            ..OrchestratorConfig::default()
        };
        let cluster = config.cluster.clone();
        let control = boot(config, "peer-token-of-the-test-cluster".into()).expect("boot");
        let mut orchestrator = Orchestrator::new(control, Arc::new(LocalLauncher::new()));
        let ops = orchestrator.attach_ops();
        let server = orchestrator.bind().expect("bind control");
        let control_addr = server.local_addr().expect("control port");
        let player = orchestrator
            .bind_player()
            .expect("bind the player port")
            .expect("a player port is configured");
        let player_addr = player.local_addr().expect("player port");
        let served = Arc::clone(&player);
        let task =
            tokio::spawn(async move { orchestrator.run_with_player(server, Some(served)).await });
        Self {
            cluster,
            control_addr,
            player_addr,
            player,
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
                "{what} did not happen within {limit:?}. Shards: {:?}; player port: {:?}; \
                 newest events: {:?}",
                snapshot
                    .shards
                    .iter()
                    .map(|s| (s.shard.0, s.role.label(), s.connected))
                    .collect::<Vec<_>>(),
                snapshot.player_port,
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

    /// A shard started by hand, as `space-server --orchestrator <control>`
    /// starts one.
    async fn shard(&self) -> RunningShard {
        start_orchestrated(self.options(), self.control_addr)
            .await
            .expect("the shard registers")
    }

    fn options(&self) -> OrchestratedOptions {
        OrchestratedOptions {
            shard: ShardId::NONE,
            bind: loopback0(),
            advertise: None,
            client_endpoint: None,
            claim_region: None,
            cluster: self.cluster.clone(),
            secret: String::new(),
        }
    }

    async fn stop(self) {
        self.task.abort();
        let _ = self.task.await;
        let _ = std::fs::remove_file(&self.state_path);
        let _ = std::fs::remove_file(space_orchestrator::scene::scene_path(&self.state_path));
    }
}

/// Say hello at `addr` as a player does (`space_net::join`).
async fn hello_at(addr: SocketAddr, token: &str) -> Result<Joined, NetError> {
    let endpoint = space_net::client_endpoint_for(addr).expect("client endpoint");
    tokio::time::timeout(
        Duration::from_secs(10),
        space_net::join(&endpoint, addr, SERVER_NAME, &space_net::hello(token)),
    )
    .await
    .expect("an answer within the handshake bound")
}

/// The redirect the player port answers with, once it has one.
async fn redirect_from(player_addr: SocketAddr, what: &str) -> JoinRedirect {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match hello_at(player_addr, "space-client-test").await {
            Ok(Joined::Elsewhere(redirect)) => return redirect,
            Err(NetError::Rejected(RejectReason::Starting)) => {}
            other => panic!("{what}: neither a redirect nor \"starting\": {other:?}"),
        }
        assert!(Instant::now() < deadline, "{what}: still \"starting\"");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// The player port, end to end. With no shard it says "starting" — a refusal
/// with a reason of its own, never a redirect to the reserved root id, whose
/// address in the map is a placeholder. Once a shard owns the spawn point a
/// hello is answered with a `JoinRedirect` naming it and carrying the
/// published map, and the address the map names for its players is a shard
/// that welcomes the player: the two steps a client takes.
///
/// Mutation this catches: redirecting to whatever the map says before a
/// shard is there (a client dials the control port), or a redirect without
/// the map (a client has no address to dial).
#[tokio::test(flavor = "multi_thread")]
async fn the_player_port_sends_a_player_to_the_spawn_points_owner_and_says_starting_until_there_is_one(
) {
    let orch = Running::start("entry");

    // Nobody owns the spawn point: "starting", as often as it is asked.
    for _ in 0..3 {
        let refused = hello_at(orch.player_addr, "space-client-early")
            .await
            .expect_err("no shard to send a player to");
        assert_eq!(refused, NetError::Rejected(RejectReason::Starting));
    }
    assert_eq!(orch.player.turned_away(), 3);
    assert_eq!(orch.player.redirected(), 0);
    // The operator's view says so: where the port listens, that nobody is
    // sent anywhere, and how often that was said.
    let empty = orch
        .wait_for(
            "the snapshot to count the hellos",
            Duration::from_secs(5),
            |s| s.player_port.is_some_and(|p| p.turned_away == 3),
        )
        .await;
    let port = empty
        .player_port
        .expect("the snapshot shows the player port");
    assert_eq!(port.addr, orch.player_addr);
    assert_eq!((port.sends_to, port.redirected), (None, 0));

    // A shard registers and is given the root, and with it the spawn point.
    let root = orch.shard().await;
    assert_eq!(root.shard, ShardId::new(1));
    let redirect = redirect_from(orch.player_addr, "after the root's shard registered").await;
    assert_eq!(redirect.shard, root.shard);
    let snapshot = orch
        .wait_for(
            "the snapshot to name the owner",
            Duration::from_secs(5),
            |s| {
                s.player_port
                    .is_some_and(|p| p.sends_to == Some(root.shard))
            },
        )
        .await;
    assert_eq!(snapshot.spawn_owner, Some(root.shard));
    assert_eq!(redirect.shard_map.epoch, snapshot.epoch);
    assert_eq!(redirect.shard_map.regions.len(), snapshot.regions.len());
    assert_eq!(
        redirect.shard_map.client_endpoint_of(root.shard),
        Some(root.client_endpoint),
        "the map says where players reach that shard"
    );
    assert_eq!(
        snapshot.regions[0].client_endpoint,
        Some(root.client_endpoint)
    );
    assert_eq!(
        snapshot.shards[0].client_endpoint,
        Some(root.client_endpoint)
    );

    // The second step: that address is a shard, and it welcomes the player.
    let at = redirect
        .shard_map
        .client_endpoint_of(redirect.shard)
        .expect("an address");
    let joined = hello_at(at, "space-client-test").await.expect("an answer");
    let Joined::Shard(conn, welcome) = joined else {
        panic!("the spawn point's owner welcomes a player, got {joined:?}");
    };
    assert_eq!(welcome.shard, root.shard);
    conn.close(0, "seen enough");

    // A second shard takes a layout cell. Players are still sent to the
    // first one — and that second shard, dialed directly, sends them there
    // too: every address leads to the spawn point.
    let second = orch.shard().await;
    orch.wait_for("the second shard's cell", Duration::from_secs(10), |s| {
        s.regions.len() == 2
            && s.in_flight.is_none()
            && s.shards.iter().all(|shard| shard.role == ShardRole::Owner)
    })
    .await;
    let redirect = redirect_from(orch.player_addr, "with two shards").await;
    assert_eq!(redirect.shard, root.shard);
    assert_eq!(redirect.shard_map.regions.len(), 2);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match hello_at(second.endpoint, "space-client-test").await {
            Ok(Joined::Elsewhere(by_shard)) => {
                assert_eq!(by_shard.shard, root.shard);
                assert_eq!(
                    by_shard.shard_map.client_endpoint_of(root.shard),
                    Some(root.client_endpoint)
                );
                break;
            }
            // Its own map does not name the owner's address yet (the push
            // behind its commit is on its way).
            Err(NetError::Rejected(RejectReason::Starting)) => {}
            other => panic!("a shard off the spawn point welcomed a player: {other:?}"),
        }
        assert!(Instant::now() < deadline, "the second shard never knew");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(orch.player.redirected() >= 2);

    root.abort();
    second.abort();
    orch.stop().await;
}

/// The shard that owns the spawn point dies. From then on the player port
/// does not name it — it says "starting" — and when its replacement has
/// taken over the root, under a new id and at another address, the player
/// port names that one. A client that asks again ends up on a living shard.
///
/// Mutation this catches: redirecting by the map alone (the dead shard stays
/// in it until the replacement's epoch commits — a client is sent to an
/// address nobody listens on, every time it asks).
#[tokio::test(flavor = "multi_thread")]
async fn after_the_roots_shard_is_replaced_the_player_port_names_the_new_owner() {
    let orch = Running::start("replaced");
    let first = orch.shard().await;
    let dead = first.shard;
    let dead_at = first.client_endpoint;
    let before = redirect_from(orch.player_addr, "the first owner").await;
    assert_eq!(before.shard, dead);

    // It dies (its process is gone: the control link closes with it).
    first.abort();
    orch.wait_for("the death to be noticed", Duration::from_secs(15), |s| {
        s.shards
            .iter()
            .any(|shard| shard.shard == dead && shard.role == ShardRole::Dead)
    })
    .await;
    let refused = hello_at(orch.player_addr, "space-client-test")
        .await
        .expect_err("the owner is dead and not replaced yet");
    assert_eq!(refused, NetError::Rejected(RejectReason::Starting));
    assert_eq!(
        orch.ops.snapshot().player_port.expect("shown").sends_to,
        None
    );

    // A replacement registers and is given the dead shard's root.
    let replacement = orch.shard().await;
    assert_ne!(replacement.shard, dead, "a new id");
    assert_ne!(replacement.client_endpoint, dead_at, "another address");
    let after = redirect_from(orch.player_addr, "the replacement").await;
    assert_eq!(after.shard, replacement.shard);
    assert_eq!(
        after.shard_map.client_endpoint_of(after.shard),
        Some(replacement.client_endpoint)
    );
    assert!(
        after.shard_map.regions.iter().all(|r| r.shard != dead),
        "the dead shard is out of the map"
    );
    let joined = hello_at(replacement.client_endpoint, "space-client-test")
        .await
        .expect("an answer");
    assert!(
        matches!(&joined, Joined::Shard(_, welcome) if welcome.shard == replacement.shard),
        "the replacement welcomes the player: {joined:?}"
    );

    replacement.abort();
    orch.stop().await;
}

/// What the cluster's state looks like to a test: the shards the
/// orchestrator knows, the id it would hand out next, the map's epoch.
fn state_of(snapshot: &Snapshot) -> (Vec<(u32, &'static str, bool)>, u64, u64, usize) {
    (
        snapshot
            .shards
            .iter()
            .map(|s| (s.shard.0, s.role.label(), s.connected))
            .collect(),
        snapshot.next_shard_id,
        snapshot.epoch,
        snapshot.regions.len(),
    )
}

/// The player port is not a way into the cluster. A shard that is pointed at
/// it cannot register: it is refused, for good and with the reason. A peer
/// that says hello as a player and then sends what a shard would — a
/// `ShardHello`, an epoch ack, a load report, a scene ack — is given the
/// same answer as any player and nothing it sent is read. Afterwards the
/// orchestrator knows exactly what it knew before: no shard, no id spent, no
/// epoch, nothing in its event log.
///
/// Mutation this catches: serving the player port with the control port's
/// connection handler (anybody who can reach the public port registers a
/// shard without the cluster secret), or reading on after the hello.
#[tokio::test(flavor = "multi_thread")]
async fn nothing_said_on_the_player_port_reaches_the_orchestrator() {
    let orch = Running::start("closed");
    let root = orch.shard().await;
    // (Settled: the shard has confirmed the scene, so nothing of its own
    // registration is still on its way into the event log.)
    let before = orch
        .wait_for("the root's shard", Duration::from_secs(10), |s| {
            s.player_port
                .is_some_and(|p| p.sends_to == Some(root.shard))
                && s.shards
                    .first()
                    .is_some_and(|shard| shard.scene == Some(s.scene.current()))
        })
        .await;
    let known = state_of(&before);
    let events_before = before.events.len();

    // A real shard, pointed at the player port: refused for good, at once.
    let started = Instant::now();
    let err = tokio::time::timeout(
        Duration::from_secs(10),
        start_orchestrated(orch.options(), orch.player_addr),
    )
    .await
    .expect("a refusal is not waited out")
    .expect_err("no shard registers at the player port");
    assert!(
        err.contains("player port") && err.contains("control port"),
        "the shard is told what it dialed: {err}"
    );
    assert!(started.elapsed() < Duration::from_secs(5), "final, at once");

    // A peer that says hello as a player and then talks like a shard, on
    // every path a shard uses. It is answered like any player...
    let endpoint = space_net::client_endpoint_for(orch.player_addr).expect("client endpoint");
    let connection = endpoint
        .connect(orch.player_addr, SERVER_NAME)
        .expect("dial")
        .await
        .expect("the transport handshake");
    let (mut send, mut recv) = connection.open_bi().await.expect("the control stream");
    let forged: Vec<Message> = vec![
        Message::Hello(space_net::hello("space-client-forger")),
        Message::ShardHello(ShardHello {
            cluster: orch.cluster.clone(),
            shard: ShardId::NONE,
            endpoint: "127.0.0.1:7999".parse().expect("an address"),
            region: None,
            client_endpoint: None,
            secret: String::new(),
        }),
        Message::EpochAck(EpochAck {
            epoch: before.epoch + 1,
            shard: root.shard,
        }),
        Message::LoadReport(LoadReport {
            shard: root.shard,
            players: 99,
            tick_p95_ms: 30.0,
            load: 1.0,
            ..Default::default()
        }),
        Message::SceneAck(SceneAck {
            shard: root.shard,
            applied: space_protocol::SceneState::new(space_protocol::Scene::Materials.id(), 77),
        }),
    ];
    for message in &forged {
        let frame = space_protocol::encode_frame(message);
        // (Best effort behind the hello: the port may already have answered
        // and closed, which is the point.)
        let _ = send.write_all(&frame).await;
    }
    for message in &forged[1..] {
        if let Ok(mut uni) = connection.open_uni().await {
            let _ = uni.write_all(&space_protocol::encode_frame(message)).await;
            let _ = uni.finish();
        }
    }
    let answer = tokio::time::timeout(Duration::from_secs(5), space_net::read_frame(&mut recv))
        .await
        .expect("an answer")
        .expect("readable")
        .expect("a message");
    assert!(
        matches!(&answer, Message::JoinRedirect(r) if r.shard == root.shard),
        "the same answer as any player: {answer:?}"
    );
    // ...and the connection is finished from the port's side, within its
    // drain, whether the peer closes or not.
    let closed = tokio::time::timeout(Duration::from_secs(3), connection.closed()).await;
    assert!(closed.is_ok(), "the player port holds no connection open");

    // The orchestrator knows what it knew.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let after = orch.ops.snapshot();
    assert_eq!(state_of(&after), known);
    assert_eq!(after.events.len(), events_before, "nothing was logged");
    // (The shard's own report is there — it reports once a second — and
    // says what the shard said, not the 99 players of the forgery.)
    assert_eq!(
        after.shards[0].report.map(|report| report.players),
        Some(0),
        "no load was forged for the shard"
    );
    assert_eq!(after.scene, before.scene);

    root.abort();
    orch.stop().await;
}

/// A client pointed at the **control** port by mistake used to be answered
/// with the placeholder welcome a shard's transport handshake gets, took the
/// control port for a shard, and sat on an empty world until the port's
/// hello timeout closed the connection. It is told what it dialed instead —
/// at once, in words a connect screen can show — and nothing about the
/// cluster. A shard's own handshake (`shard:<id>`) is untouched.
#[tokio::test(flavor = "multi_thread")]
async fn a_player_who_dials_the_control_port_is_told_what_it_is() {
    let orch = Running::start("wrong-port");
    let started = Instant::now();
    let refused = hello_at(orch.control_addr, "space-client-0123456789abcdef")
        .await
        .expect_err("the control port is not a shard");
    let NetError::Rejected(RejectReason::Other(why)) = &refused else {
        panic!("a readable refusal, got {refused:?}");
    };
    assert!(
        why.contains("control port") && why.contains("player port"),
        "{why}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "said at once, not after the hello timeout"
    );
    // Shards still register there.
    let shard = orch.shard().await;
    assert_eq!(shard.shard, ShardId::new(1));
    shard.abort();
    orch.stop().await;
}

/// A shard that advertises an address without an IP — `0.0.0.0:<port>`, what
/// a shard in a container knows of itself — and asks for a client endpoint
/// of its own is listed with the IP its control connection comes from, and
/// the map sends its players where it asked. The shard itself runs on those
/// two addresses.
///
/// Mutation this catches: recording the advertised address as it is (the map
/// tells every neighbour to dial `0.0.0.0`), or filling the observed IP into
/// the client endpoint as well (a player on another machine is sent to the
/// orchestrator's view of the shard instead of the host it dialed).
#[tokio::test(flavor = "multi_thread")]
async fn a_shard_that_registers_without_an_ip_is_listed_with_the_one_it_is_seen_from() {
    let orch = Running::start("observed");
    // The port it advertises is the one it binds. It has to be named
    // before it is bound, so it is looked for first — and whoever else is
    // handed ports by the system (the other tests of this file, in this
    // process) may take it in between: then another one is looked for.
    // ("bind: … only one usage of each socket address", seen here once, on Windows:
    // the test then failed before it had tested anything.)
    let mut attempt = 0;
    let (shard, any, observed) = loop {
        attempt += 1;
        let port = {
            let probe = std::net::UdpSocket::bind(loopback0()).expect("a free port");
            probe.local_addr().expect("bound").port()
        };
        let any = SocketAddr::from(([0, 0, 0, 0], port));
        let observed = SocketAddr::from(([127, 0, 0, 1], port));
        let mut options = orch.options();
        options.bind = observed;
        options.advertise = Some(any);
        options.client_endpoint = Some(any);
        match start_orchestrated(options, orch.control_addr).await {
            Ok(shard) => break (shard, any, observed),
            Err(why) if why.starts_with("bind:") && attempt < 20 => {}
            Err(why) => panic!("the shard registers: {why}"),
        }
    };
    assert_eq!(shard.endpoint, observed, "the address it is seen from");
    assert_eq!(shard.client_endpoint, any, "the host the player dialed");

    let snapshot = orch
        .wait_for("the shard to be listed", Duration::from_secs(10), |s| {
            s.shards
                .iter()
                .any(|v| v.shard == shard.shard && v.connected)
        })
        .await;
    let listed = snapshot
        .shards
        .iter()
        .find(|v| v.shard == shard.shard)
        .expect("listed");
    assert_eq!(listed.advertised, Some(observed));
    assert_eq!(listed.client_endpoint, Some(any));
    assert_eq!(snapshot.regions[0].endpoint, Some(observed));
    assert_eq!(snapshot.regions[0].client_endpoint, Some(any));

    // A player is redirected to `0.0.0.0:<port>`: the host it dialed the
    // player port on, that port — which is the shard.
    let redirect = redirect_from(orch.player_addr, "the shard without an IP").await;
    let players_at = redirect
        .shard_map
        .client_endpoint_of(redirect.shard)
        .expect("an address");
    assert_eq!(players_at, any);
    let dial = space_protocol::resolve_client_endpoint(players_at, orch.player_addr.ip())
        .expect("a port to dial");
    assert_eq!(dial, observed);
    let joined = hello_at(dial, "space-client-test")
        .await
        .expect("an answer");
    assert!(
        matches!(&joined, Joined::Shard(_, welcome) if welcome.shard == shard.shard),
        "{joined:?}"
    );

    shard.abort();
    orch.stop().await;
}
