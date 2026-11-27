//! One flight, flown against a whole cluster: join, fly across the shard border
//! at x = -6, get handed to the next shard, park there.
//!
//! The flight is one routine ([`cross_one_border`]) with two callers: an
//! in-process orchestrator with four shards, and a running Docker compose
//! stack. Nothing renders (no GPU): the client's link is driven directly.
//!
//! | test | rule |
//! |---|---|
//! | [`the_pilot_flies_towards_the_target_and_brakes_at_it`] | the pure steering law |
//! | [`a_player_who_enters_through_the_player_port_is_handed_to_the_next_shard_and_parks_there`] | the flight against an in-process orchestrator with four shards |
//! | [`a_player_on_the_running_stack_is_handed_to_the_next_shard`] | the same flight against the compose stack named by `SPACE_E2E_SERVER`; `#[ignore]`d, run by `cargo run -p xtask -- e2e` |

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_client::address::{dial, ServerAddress};
use space_client::link::{Replica, ServerLink};
use space_orchestrator::boot::boot;
use space_orchestrator::config::OrchestratorConfig;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::ops::OpsHandle;
use space_orchestrator::orchestrator::Orchestrator;
use space_orchestrator::snapshot::{ShardRole, Snapshot};

const SERVER_ENV: &str = "SPACE_E2E_SERVER";
const LONG: Duration = Duration::from_secs(40);
/// Metres above the spawn point the flight is flown at: over the floating crates.
const ALTITUDE: f64 = 5.0;
/// Where the flight ends, metres from the spawn point along x: 8 m beyond the border at x = -6.
const BEYOND: f64 = -14.0;
/// Position error to wanted speed, 1/s; and the most speed wanted on an axis, m/s.
const GAIN: f64 = 0.8;
const CRUISE: f64 = 4.0;

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

/// Poll `condition` every 50 ms for up to `seconds`.
async fn eventually(seconds: u64, mut condition: impl FnMut() -> bool) -> bool {
    for _ in 0..seconds * 20 {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    condition()
}

// ---------------------------------------------------------------- a cluster

/// A running orchestrator with a player port, keeping `min_shards` shards
/// running in this process.
struct Cluster {
    player_addr: SocketAddr,
    ops: OpsHandle,
    task: tokio::task::JoinHandle<Result<(), std::io::Error>>,
    state_path: std::path::PathBuf,
}

impl Cluster {
    fn start(rt: &tokio::runtime::Runtime, tag: &str, min_shards: usize) -> Self {
        let _inside = rt.enter();
        let state_path = std::env::temp_dir().join(format!(
            "space-client-live-stack-{tag}-{}-{}.id",
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
        let control = boot(config, "peer-token-of-the-test-cluster".into()).expect("boot");
        let mut orchestrator = Orchestrator::new(control, Arc::new(LocalLauncher::new()));
        let ops = orchestrator.attach_ops();
        let server = orchestrator.bind().expect("bind control");
        let player = orchestrator
            .bind_player()
            .expect("bind the player port")
            .expect("a player port is configured");
        let player_addr = player.local_addr().expect("player port");
        let task =
            tokio::spawn(async move { orchestrator.run_with_player(server, Some(player)).await });
        Self {
            player_addr,
            ops,
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

// ---------------------------------------------------------------- the pilot

/// Thrust (each axis in -1..=1) that flies towards `target` and stops there.
fn thrust_towards(target: [f64; 3], at: [f64; 3], vel: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| {
        let want = ((target[i] - at[i]) * GAIN).clamp(-CRUISE, CRUISE);
        ((want - f64::from(vel[i])) / 2.0).clamp(-1.0, 1.0) as f32
    })
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}

fn speed(vel: [f32; 3]) -> f64 {
    vel.iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Steer towards `target` (metres from the spawn point), looking every 50 ms,
/// until `done` says so or `seconds` are over. Thrust is zero afterwards.
async fn fly(
    link: &ServerLink,
    target: [f64; 3],
    seconds: u64,
    mut done: impl FnMut(&Replica) -> bool,
) -> bool {
    for _ in 0..seconds * 20 {
        let replica = link.replica();
        if done(&replica) {
            link.set_thrust([0.0; 3]);
            return true;
        }
        if let Some(state) = replica.state {
            link.set_thrust(thrust_towards(target, replica.offset(), state.vel));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    link.set_thrust([0.0; 3]);
    done(&link.replica())
}

/// Join has happened through `entry`; fly over the border, get handed over, park.
async fn cross_one_border(link: &ServerLink, entry: SocketAddr) {
    assert!(
        eventually(30, || link.replica().state.is_some()).await,
        "the ship never appeared: {:?}",
        link.replica()
    );
    let joined = link.replica();
    let ship = joined.ship.expect("the joined replica has a ship");
    let first_shard = joined.shard.expect("the joined replica has a shard");
    let first_address = link.shard_address();
    assert_eq!(joined.handoffs, 0, "a join redirect is not a handoff");
    assert_eq!(link.entry(), entry);
    assert_eq!(
        first_address.ip(),
        entry.ip(),
        "the shard is dialed on the host the player dialed"
    );

    // Climb.
    let up = [0.0, ALTITUDE, 0.0];
    assert!(
        fly(link, up, 20, |r| distance(r.offset(), up) < 1.0).await,
        "the ship never climbed to {ALTITUDE} m: {:?}",
        link.replica()
    );

    // Cross.
    let beyond = [BEYOND, ALTITUDE, 0.0];
    assert!(
        fly(link, beyond, 60, |r| r.handoffs > 0).await,
        "never handed to another shard: {:?}",
        link.replica()
    );
    let arrived = link.replica();
    assert_eq!(arrived.ship, Some(ship), "the same ship, by guid");
    assert!(arrived.shard.is_some() && arrived.shard != Some(first_shard));
    assert!(arrived.offset()[0] < -6.0, "{arrived:?}");
    assert!(!link.is_closed());
    assert_eq!(link.entry(), entry, "the entry address never changes");
    assert_ne!(link.shard_address(), first_address);
    assert_eq!(link.shard_address().ip(), entry.ip());

    // Park: a coasting ship would overshoot, so this proves the new shard
    // applies this client's inputs.
    assert!(
        fly(link, beyond, 30, |r| {
            distance(r.offset(), beyond) < 1.0 && r.state.is_some_and(|s| speed(s.vel) < 0.5)
        })
        .await,
        "the ship never came to rest at the target on the new shard: {:?}",
        link.replica()
    );
    let parked = link.replica();
    assert_eq!(parked.shard, arrived.shard);
    assert_eq!(parked.ship, Some(ship));
    eprintln!(
        "live stack: {first_shard} at {first_address} -> {} at {}, handoffs {}, tick epoch {}",
        parked.shard.expect("a shard"),
        link.shard_address(),
        parked.handoffs,
        parked.tick_epoch
    );
}

// ---------------------------------------------------------------- the tests

#[test]
fn the_pilot_flies_towards_the_target_and_brakes_at_it() {
    let target = [-14.0, 0.0, 0.0];
    assert_eq!(thrust_towards(target, target, [0.0; 3]), [0.0, 0.0, 0.0]);
    let t = thrust_towards(target, [0.0; 3], [0.0; 3]);
    assert_eq!((t[0], t[1], t[2]), (-1.0, 0.0, 0.0));
    let t = thrust_towards(target, target, [-4.0, 0.0, 0.0]);
    assert_eq!(t[0], 1.0);
    let far = thrust_towards([500.0, -300.0, 900.0], [0.0; 3], [9.0, -9.0, 0.0]);
    assert!(far.iter().all(|c| (-1.0..=1.0).contains(c)), "{far:?}");
}

#[test]
fn a_player_who_enters_through_the_player_port_is_handed_to_the_next_shard_and_parks_there() {
    let rt = runtime();
    let cluster = Cluster::start(&rt, "live-stack", 4);
    cluster.settled(4);
    rt.block_on(async {
        let link = ServerLink::join(cluster.player_addr)
            .await
            .expect("join through the player port");
        cross_one_border(&link, cluster.player_addr).await;
        link.close();
    });
    cluster.stop(&rt);
}

#[test]
#[ignore = "needs the compose stack: run it with `cargo run -p xtask -- e2e`"]
fn a_player_on_the_running_stack_is_handed_to_the_next_shard() {
    let Ok(server) = std::env::var(SERVER_ENV) else {
        panic!("SPACE_E2E_SERVER is not set: this test is run by `cargo run -p xtask -- e2e`");
    };
    let address = ServerAddress::parse(&server).unwrap_or_else(|e| panic!("{e}"));
    runtime().block_on(async {
        let started = Instant::now();
        let link = loop {
            match dial(&address).await {
                Ok(link) => break link,
                Err(reason) => {
                    eprintln!("live stack: not in yet: {reason}");
                    if started.elapsed() >= Duration::from_secs(120) {
                        panic!("could not join {server} in 120 s: {reason}");
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        };
        let entry = link.entry();
        cross_one_border(&link, entry).await;
        link.close();
    });
}
