//! One address (`docs/plan-sharding.md` §4), seen from the client: whatever
//! address a player dials, the client ends up on the shard that owns the
//! scene's spawn point, at the spawn point — and comes back through that
//! address when its shard is gone.
//!
//! Everything here is production code over loopback QUIC: the real
//! orchestrator loop with its player port, real shards (started in process by
//! the orchestrator's `local` launcher, or standalone), and the client's own
//! core ([`App`]) or link ([`ServerLink`]). No GPU is needed: frames are built
//! and not rendered.
//!
//! | test | rule |
//! |---|---|
//! | [`a_client_that_dials_the_player_port_starts_on_the_spawn_point_and_one_that_dials_another_shard_does_too`] | the player port and a shard off the spawn point both lead to the spawn point's owner; the state says the entry address and the shard's |
//! | [`a_client_whose_shard_dies_comes_back_through_the_entry_address_onto_the_replacement`] | a `--server` client retries the **entry** address, not the dead shard's |
//! | [`while_the_cluster_is_starting_the_client_says_so_and_gets_in_when_a_shard_is_there`] | "starting" is shown as such and retried |
//! | [`a_client_pointed_at_the_control_port_is_told_what_it_dialed`] | the control port's refusal reaches the connect screen in words |
//! | [`two_redirects_are_followed_and_a_third_ends_the_join_with_a_reason`] | the hop limit; a redirect to nowhere |
//! | [`a_client_endpoint_without_an_ip_is_dialed_on_the_host_of_the_entry_address`] | `0.0.0.0:port` in a join redirect |
//!
//! The same rule for a **handoff**'s redirect is exercised by
//! `tests/handoff.rs`, whose map sends players to `0.0.0.0:<port>`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_client::address::explain_join;
use space_client::app::{App, AppConfig};
use space_client::link::{JoinFailure, ServerLink, MAX_JOIN_REDIRECTS};
use space_harness::{Control, Hub, Screen};
use space_net::NetError;
use space_orchestrator::boot::boot;
use space_orchestrator::config::OrchestratorConfig;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::launcher::Launcher;
use space_orchestrator::ops::OpsHandle;
use space_orchestrator::orchestrator::Orchestrator;
use space_orchestrator::snapshot::{ShardRole, Snapshot};
use space_protocol::{JoinRedirect, Message, ShardId, ShardMap, ShardRegion};
use space_render::QualityPreset;
use space_server::net::ShardNet;
use space_server::{Shard, ShardConfig};
use space_world::NodeRef;
use tokio::sync::Mutex as AsyncMutex;

const LONG: Duration = Duration::from_secs(40);

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("runtime")
}

fn loopback0() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 0))
}

// ---------------------------------------------------------------- a cluster

/// A running orchestrator with a player port, keeping `min_shards` shards
/// running in this process.
struct Cluster {
    control_addr: SocketAddr,
    player_addr: SocketAddr,
    cluster: String,
    ops: OpsHandle,
    launcher: Arc<LocalLauncher>,
    task: tokio::task::JoinHandle<Result<(), std::io::Error>>,
    state_path: std::path::PathBuf,
}

impl Cluster {
    fn start(rt: &tokio::runtime::Runtime, tag: &str, min_shards: usize) -> Self {
        let _inside = rt.enter();
        let state_path = std::env::temp_dir().join(format!(
            "space-client-entry-{tag}-{}-{}.id",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_file(&state_path);
        let config = OrchestratorConfig {
            bind: loopback0(),
            player_bind: Some(loopback0()),
            state_path: state_path.clone(),
            min_shards,
            ..OrchestratorConfig::default()
        };
        let cluster = config.cluster.clone();
        let control = boot(config, "peer-token-of-the-test-cluster".into()).expect("boot");
        let launcher = Arc::new(LocalLauncher::new());
        let mut orchestrator = Orchestrator::new(control, Arc::clone(&launcher));
        let ops = orchestrator.attach_ops();
        let server = orchestrator.bind().expect("bind control");
        let control_addr = server.local_addr().expect("control port");
        let player = orchestrator
            .bind_player()
            .expect("bind the player port")
            .expect("a player port is configured");
        let player_addr = player.local_addr().expect("player port");
        let task =
            tokio::spawn(async move { orchestrator.run_with_player(server, Some(player)).await });
        Self {
            control_addr,
            player_addr,
            cluster,
            ops,
            launcher,
            task,
            state_path,
        }
    }

    /// The latest snapshot `done` is satisfied with.
    fn wait_for(&self, what: &str, done: impl Fn(&Snapshot) -> bool) -> Arc<Snapshot> {
        let deadline = Instant::now() + LONG;
        loop {
            let snapshot = self.ops.snapshot();
            if done(&snapshot) {
                return snapshot;
            }
            assert!(
                Instant::now() < deadline,
                "{what} did not happen. Shards: {:?}; player port: {:?}",
                snapshot
                    .shards
                    .iter()
                    .map(|s| (s.shard.0, s.role.label(), s.connected))
                    .collect::<Vec<_>>(),
                snapshot.player_port
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    /// `count` regions, each owned by a connected shard, nothing in flight,
    /// and the player port sending players to the spawn point's owner.
    fn settled(&self, count: usize) -> Arc<Snapshot> {
        self.wait_for("the cluster to settle", |s| {
            s.regions.len() == count
                && s.shards.len() == count
                && s.in_flight.is_none()
                && s.starting == 0
                && s.shards
                    .iter()
                    .all(|shard| shard.role == ShardRole::Owner && shard.connected)
                && s.player_port.is_some_and(|p| p.sends_to.is_some())
        })
    }

    fn stop(self, rt: &tokio::runtime::Runtime) {
        self.task.abort();
        let _ = rt.block_on(self.task);
        let _ = std::fs::remove_file(&self.state_path);
        let _ = std::fs::remove_file(space_orchestrator::scene::scene_path(&self.state_path));
    }
}

/// Where the map of `snapshot` sends the players of `shard`.
fn players_dial(snapshot: &Snapshot, shard: ShardId) -> SocketAddr {
    snapshot
        .shards
        .iter()
        .find(|view| view.shard == shard)
        .and_then(|view| view.client_endpoint)
        .expect("a registered shard")
}

// ------------------------------------------------------------------ a client

/// An app run without a GPU: frames are built but not rendered.
struct Dry {
    app: App,
}

impl Dry {
    /// A client that was given `server` on the command line (`--server`): it
    /// connects at once and keeps the connection up by itself.
    fn unattended(rt: &tokio::runtime::Runtime, server: SocketAddr) -> Self {
        Self {
            app: App::new(
                AppConfig {
                    server: Some(server.to_string()),
                    quality: QualityPreset::High,
                    model: None,
                    escape_quits: false,
                },
                rt.handle().clone(),
                Hub::new(),
            ),
        }
    }

    /// A window client on its connect screen (nothing is retried by itself).
    fn windowed(rt: &tokio::runtime::Runtime) -> Self {
        Self {
            app: App::new(
                AppConfig {
                    server: None,
                    quality: QualityPreset::High,
                    model: None,
                    escape_quits: true,
                },
                rt.handle().clone(),
                Hub::new(),
            ),
        }
    }

    fn frame(&mut self) {
        self.app.update();
        let _ = self.app.scene(640, 360);
        self.app.finish_frame(640, 360, None);
    }

    fn until(&mut self, what: &str, done: impl Fn(&App) -> bool) {
        let deadline = Instant::now() + LONG;
        loop {
            self.frame();
            if done(&self.app) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "never {what}: {:?} {:?}",
                self.app,
                self.app.client_state(640, 360)
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn online(&mut self) {
        self.until("came online", |app| {
            app.screen() == Screen::Online && app.replica().is_some_and(|r| r.state.is_some())
        });
    }
}

/// How far `app`'s ship is from the scene's spawn point, on the floor and
/// above it.
fn from_spawn(app: &App) -> (f64, f64) {
    let at = app.position_in_scene().expect("a ship");
    ((at[0] * at[0] + at[2] * at[2]).sqrt(), at[1].abs())
}

/// The player port leads to the shard that owns the spawn point, and so does
/// every other shard's address. A client that dials the orchestrator's player
/// port comes online on the root's shard, on the spawn point, and its state
/// names both addresses: the one that was dialed and the shard's. A second
/// client that dials **another** shard directly is sent to the same shard and
/// arrives beside the first — not in the middle of the cell of the shard it
/// dialed. Neither took a handoff to get there.
#[test]
fn a_client_that_dials_the_player_port_starts_on_the_spawn_point_and_one_that_dials_another_shard_does_too(
) {
    let rt = runtime();
    let cluster = Cluster::start(&rt, "entry", 2);
    let settled = cluster.settled(2);
    let owner = settled.spawn_owner.expect("a spawn owner");
    assert_eq!(settled.regions[0].shard, owner, "the root's shard");
    let owner_at = players_dial(&settled, owner);
    let other = settled
        .shards
        .iter()
        .find(|view| view.shard != owner)
        .expect("a second shard");
    let other_at = other.client_endpoint.expect("registered");
    assert_ne!(other_at, owner_at);

    // Through the player port.
    let mut first = Dry::unattended(&rt, cluster.player_addr);
    first.online();
    let state = first.app.client_state(640, 360);
    assert_eq!(state.address, cluster.player_addr.to_string(), "the entry");
    assert_eq!(state.shard_address, Some(owner_at.to_string()));
    assert_eq!(state.shard, Some(owner.0));
    assert_eq!(state.handoffs, 0);
    assert_eq!(from_spawn(&first.app), (0.0, 0.0), "on the spawn point");

    // Through a shard that does not own the spawn point.
    let mut second = Dry::unattended(&rt, other_at);
    second.online();
    let state = second.app.client_state(640, 360);
    assert_eq!(state.address, other_at.to_string(), "what was dialed");
    assert_eq!(state.shard_address, Some(owner_at.to_string()));
    assert_eq!(state.shard, Some(owner.0), "sent on to the owner");
    assert_eq!(state.handoffs, 0, "sent on, not handed over");
    let (beside, above) = from_spawn(&second.app);
    assert!(
        above < 0.05 && (1.0..16.0).contains(&beside),
        "beside the spawn point, not half a cell away: {beside:.2} m, {above:.2} m up"
    );
    for client in [&mut first, &mut second] {
        client.until("saw the other player", |app| {
            app.replica().is_some_and(|r| r.others().count() == 1)
        });
    }

    drop(first);
    drop(second);
    cluster.stop(&rt);
}

/// A client that was given the player port keeps its connection up through
/// it. Its shard — the only one, the owner of the spawn point — dies. The
/// client loses the connection, says so, and tries again **at the entry
/// address**; the orchestrator, which has replaced the shard under a new id
/// and at another address, sends it there. The address the client shows never
/// changed.
///
/// Mutation this catches: retrying the address of the shard that was being
/// played on (the state before the player port: the client knocked on a dead
/// port for ever, because the replacement listens somewhere else).
#[test]
fn a_client_whose_shard_dies_comes_back_through_the_entry_address_onto_the_replacement() {
    let rt = runtime();
    let cluster = Cluster::start(&rt, "comes-back", 1);
    let before = cluster.settled(1);
    let dead = before.spawn_owner.expect("a spawn owner");
    let dead_at = players_dial(&before, dead);

    let mut client = Dry::unattended(&rt, cluster.player_addr);
    client.online();
    assert_eq!(client.app.client_state(1, 1).shard, Some(dead.0));

    // The shard dies behind everybody's back.
    let launch = rt
        .block_on(cluster.launcher.launch_of(dead))
        .expect("the orchestrator launched it");
    let died = Instant::now();
    rt.block_on(cluster.launcher.kill(launch)).expect("stopped");

    // The client notices, and says what it is doing about it.
    client.until("lost its shard", |app| app.screen() != Screen::Online);
    let lost = client.app.client_state(640, 360);
    assert_eq!(lost.address, cluster.player_addr.to_string());
    assert_eq!(lost.shard_address, None, "on no shard right now");

    // It comes back by itself, on the replacement.
    client.until("came back", |app| {
        app.screen() == Screen::Online
            && app
                .replica()
                .is_some_and(|r| r.state.is_some() && r.shard != Some(dead))
    });
    println!(
        "back on a shard {:.2?} after the first one died",
        died.elapsed()
    );
    let back = client.app.client_state(640, 360);
    let after = cluster.settled(1);
    let replacement = after.spawn_owner.expect("a spawn owner");
    assert_ne!(replacement, dead, "a new id");
    assert_eq!(back.shard, Some(replacement.0));
    assert_eq!(back.address, cluster.player_addr.to_string(), "the entry");
    let replacement_at = players_dial(&after, replacement);
    assert_ne!(replacement_at, dead_at, "another address");
    assert_eq!(back.shard_address, Some(replacement_at.to_string()));
    assert_eq!(from_spawn(&client.app), (0.0, 0.0), "on the spawn point");
    assert_eq!((back.error, back.retry_in_s), (None, None));

    drop(client);
    cluster.stop(&rt);
}

/// A cluster that has no shard yet answers a player with "starting". The
/// client shows that — in those words, not as a refusal or a missing server —
/// and, having been given its server on the command line, tries again until a
/// shard is there; then it is in.
#[test]
fn while_the_cluster_is_starting_the_client_says_so_and_gets_in_when_a_shard_is_there() {
    let rt = runtime();
    // No shard is started by the orchestrator: one is started by hand, later.
    let cluster = Cluster::start(&rt, "starting", 0);
    let mut client = Dry::unattended(&rt, cluster.player_addr);
    client.until("was told to wait", |app| app.screen() == Screen::Error);
    let waiting = client.app.client_state(640, 360);
    let said = waiting.error.expect("a reason");
    assert!(
        said.contains("the cluster is starting") && said.contains(&cluster.player_addr.to_string()),
        "{said}"
    );
    assert!(waiting.retry_in_s.is_some(), "it tries again by itself");
    assert_eq!(waiting.address, cluster.player_addr.to_string());

    // A shard registers (as `space-server --orchestrator <control>` does).
    let options = space_server::orchestrated::OrchestratedOptions {
        shard: ShardId::NONE,
        bind: loopback0(),
        advertise: None,
        client_endpoint: None,
        claim_region: None,
        cluster: cluster.cluster.clone(),
        secret: String::new(),
    };
    let shard = rt
        .block_on(space_server::orchestrated::start_orchestrated(
            options,
            cluster.control_addr,
        ))
        .expect("the shard registers");
    client.online();
    let state = client.app.client_state(640, 360);
    assert_eq!(state.shard, Some(shard.shard.0));
    assert_eq!(state.shard_address, Some(shard.client_endpoint.to_string()));
    assert_eq!(from_spawn(&client.app), (0.0, 0.0));

    drop(client);
    shard.abort();
    cluster.stop(&rt);
}

/// A player who types the orchestrator's **control** port is not left on a
/// connect screen that says nothing (the control port used to answer with the
/// placeholder welcome it gives a shard's transport handshake, and the client
/// took it for a shard): the control port says what it is, and the connect
/// screen shows it.
#[test]
fn a_client_pointed_at_the_control_port_is_told_what_it_dialed() {
    let rt = runtime();
    let cluster = Cluster::start(&rt, "control", 1);
    cluster.settled(1);
    let mut client = Dry::windowed(&rt);
    client
        .app
        .hub()
        .connect(&cluster.control_addr.to_string())
        .expect("the client takes the request");
    client.until("was refused", |app| app.screen() == Screen::Error);
    let state = client.app.client_state(640, 360);
    let said = state.error.expect("a reason");
    assert!(
        said.contains("control port") && said.contains("player port"),
        "{said}"
    );
    assert!(
        !said.contains("Other("),
        "in words, not a debug print: {said}"
    );
    assert_eq!(
        state.retry_in_s, None,
        "a person is there: nothing is retried"
    );

    drop(client);
    cluster.stop(&rt);
}

// -------------------------------------------------- the link, and fake entries

/// A standalone shard alone in the world, serving and ticking.
struct Standalone {
    net: Arc<ShardNet>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl Standalone {
    fn start() -> Self {
        let net = Arc::new(ShardNet::bind(loopback0()).expect("bind"));
        let shard = Arc::new(AsyncMutex::new(Shard::new(ShardConfig::new(
            ShardId::new(1),
            net.local_addr().expect("addr"),
            NodeRef::new(64, [0, 0, 0]),
        ))));
        let serve = tokio::spawn({
            let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
            async move {
                let _ = net.serve(shard).await;
            }
        });
        let ticker = tokio::spawn({
            let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
            async move {
                let mut clock =
                    tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
                clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    clock.tick().await;
                    shard.lock().await.tick();
                    net.flush(&shard).await;
                }
            }
        });
        Self {
            net,
            tasks: vec![serve, ticker],
        }
    }

    fn addr(&self) -> SocketAddr {
        self.net.local_addr().expect("bound")
    }

    fn stop(self) {
        for task in &self.tasks {
            task.abort();
        }
        self.net.close();
    }
}

/// An entry that answers every hello with a `JoinRedirect` naming shard
/// `shard`, whose players its map sends to `client_endpoint`. Returns its
/// address; it serves until the runtime ends.
fn redirector(shard: u32, client_endpoint: SocketAddr) -> SocketAddr {
    let (config, _cert) = space_net::crypto::make_server_config();
    let endpoint = space_net::server_endpoint(loopback0(), config).expect("bind a redirector");
    let addr = endpoint.local_addr().expect("bound");
    let redirect = JoinRedirect {
        shard: ShardId::new(shard),
        shard_map: ShardMap::new(
            1,
            vec![ShardRegion::new(
                NodeRef::new(64, [0, 0, 0]),
                ShardId::new(shard),
                // (Where *shards* would dial it: never where a player does.)
                "10.255.255.1:9".parse().expect("an address"),
            )
            .with_client_endpoint(client_endpoint)],
        ),
    };
    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            let redirect = redirect.clone();
            tokio::spawn(async move {
                if let Ok((conn, _hello)) = space_net::accept_incoming(incoming).await {
                    conn.turn_away(&Message::JoinRedirect(redirect)).await;
                }
            });
        }
    });
    addr
}

async fn eventually(seconds: u64, mut condition: impl FnMut() -> bool) -> bool {
    for _ in 0..seconds * 20 {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    condition()
}

/// A join follows a redirect, and the redirect of the address that led to,
/// and no further: the third `JoinRedirect` in a row ends the attempt with an
/// error that names the address and says what happened. A redirect to a shard
/// the map gives no address for ends it at once.
///
/// Mutation this catches: following redirects without a bound (two entries
/// that name each other keep a client dialing for ever), or reporting the
/// loop as "not a space shard".
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_redirects_are_followed_and_a_third_ends_the_join_with_a_reason() {
    assert_eq!(MAX_JOIN_REDIRECTS, 2);
    let shard = Standalone::start();
    let first = redirector(1, shard.addr());
    let second = redirector(7, first);
    let third = redirector(8, second);

    // One redirect, and two: followed to the shard.
    for entry in [first, second] {
        let link = ServerLink::join(entry).await.expect("within the limit");
        assert!(
            eventually(5, || link.replica().state.is_some()).await,
            "never got a ship through {entry}"
        );
        assert_eq!(link.replica().shard, Some(ShardId::new(1)));
        assert_eq!(link.entry(), entry);
        assert_eq!(link.shard_address(), shard.addr());
        link.close();
    }

    // Three: the attempt ends, with the address that sent it on a third time
    // and the shards it was sent to.
    let err = ServerLink::join(third).await.expect_err("one too many");
    assert_eq!(err.at, first, "the third redirect came from there");
    assert_eq!(
        err.failure,
        JoinFailure::TooManyRedirects(vec![ShardId::new(8), ShardId::new(7), ShardId::new(1)])
    );
    let said = explain_join(third, &err);
    assert!(
        said.contains("kept sending this client on")
            && said.contains(&first.to_string())
            && said.contains(&format!("sent there by {third}")),
        "{said}"
    );
    // The plain form says it too, as a protocol error.
    let plain = ServerLink::connect(third).await.expect_err("one too many");
    assert!(
        matches!(&plain, NetError::Protocol(why) if why.contains("at most 2")),
        "{plain:?}"
    );

    // A redirect to a shard whose address the map does not know (port 0).
    let nowhere = redirector(5, "0.0.0.0:0".parse().expect("an address"));
    let err = ServerLink::join(nowhere).await.expect_err("no address");
    assert_eq!(err.at, nowhere);
    assert_eq!(err.failure, JoinFailure::NoAddress(ShardId::new(5)));
    assert!(
        explain_join(nowhere, &err).contains("named no address"),
        "{}",
        explain_join(nowhere, &err)
    );

    shard.stop();
}

/// A client endpoint without an IP — `0.0.0.0:<port>`, what the orchestrator
/// puts into the map for the shards it starts — is dialed on the host of the
/// address the player dialed **first**, with that port: here the shard's
/// port on the loopback host the entry was reached on. (The rule itself, with
/// hosts that differ, is pinned in `link`'s unit tests and in
/// `space-protocol`; a handoff's redirect follows it in `tests/handoff.rs`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_endpoint_without_an_ip_is_dialed_on_the_host_of_the_entry_address() {
    let shard = Standalone::start();
    let any_host = SocketAddr::from(([0, 0, 0, 0], shard.addr().port()));
    let entry = redirector(1, any_host);
    // Through a second hop too: it is the *entry*'s host, whoever redirects.
    let outer = redirector(9, entry);
    for through in [entry, outer] {
        let link = ServerLink::join(through)
            .await
            .expect("the shard is reached");
        assert!(eventually(5, || link.replica().state.is_some()).await);
        assert_eq!(link.shard_address(), shard.addr());
        assert_eq!(link.entry(), through);
        link.close();
    }
    shard.stop();
}
