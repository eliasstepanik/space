//! The in-process launcher (plan §3.4, impl 2).
//!
//! [`LocalLauncher`] runs a real orchestrated shard inside the caller's tokio
//! runtime: it calls [`space_server::orchestrated::start_orchestrated`] — the
//! function the `space-server --orchestrator` binary runs — on `127.0.0.1:0`.
//! Real QUIC, a real 30 Hz tick loop, real handoffs: this is what makes the
//! whole lifecycle (regions given at hello, split, merge, replacement)
//! verifiable in-sandbox, with no process and no Docker daemon.
//!
//! # Registering with the orchestrator
//!
//! A launched shard registers like any other: it dials the control address in
//! its [`SpawnSpec`], says `ShardHello` with **no id and no region**, asks
//! again while it is told "not yet", runs the map the `ShardWelcome` hands
//! back and follows every epoch pushed over that link. [`Launcher::spawn`]
//! returns when that has happened, so the handle names the shard's id.
//! [`LocalLauncher::detached`] keeps a bare mode for tests of the launcher
//! mechanics that have no orchestrator to talk to.
//!
//! # Liveness
//!
//! [`Launcher::poll`] reports a shard dead when it is no longer serving,
//! ticking **or linked to the orchestrator** — a shard does not run without
//! its orchestrator and stops itself when the link is lost
//! (`space_server::orchestrated`), so a shard whose link was closed shows up
//! here as dead exactly like one whose task was aborted.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

use space_protocol::shard::ShardId;
use space_server::net::ShardNet;
use space_server::orchestrated::{start_orchestrated, OrchestratedOptions, RunningShard};
use space_server::shard::Shard;
use space_server::ShardConfig;
use tokio::sync::{Mutex, Notify};

use super::{dialable, LaunchError, LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};

/// A shard the launcher is running.
enum Running {
    /// Registered with the orchestrator (the normal case).
    Orchestrated(RunningShard),
    /// Bare: bound, serving and ticking, known to no orchestrator.
    Detached {
        net: Arc<ShardNet>,
        tick: tokio::task::JoinHandle<()>,
        serve: tokio::task::JoinHandle<()>,
        endpoint: SocketAddr,
    },
}

impl Running {
    fn endpoint(&self) -> SocketAddr {
        match self {
            Running::Orchestrated(shard) => shard.endpoint,
            Running::Detached { endpoint, .. } => *endpoint,
        }
    }

    fn shard(&self) -> Option<ShardId> {
        match self {
            Running::Orchestrated(shard) => Some(shard.shard),
            Running::Detached { .. } => None,
        }
    }

    fn is_running(&self) -> bool {
        match self {
            Running::Orchestrated(shard) => shard.is_running(),
            // The tick loop is the shard's heartbeat: if its task has finished
            // (aborted, or panicked) the shard is no longer simulating.
            Running::Detached { tick, serve, .. } => !(tick.is_finished() || serve.is_finished()),
        }
    }

    /// Stop the shard: its tasks, its control link and its endpoint.
    fn stop(self) {
        match self {
            Running::Orchestrated(shard) => shard.abort(),
            Running::Detached {
                net, tick, serve, ..
            } => {
                tick.abort();
                serve.abort();
                net.close();
            }
        }
    }
}

impl std::fmt::Debug for Running {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Running")
            .field("endpoint", &self.endpoint())
            .field("shard", &self.shard())
            .finish_non_exhaustive()
    }
}

/// Everything the launcher supervises, behind one lock so a `kill` and a
/// launch finishing can never miss each other.
#[derive(Debug, Default)]
struct Table {
    /// The shards that are up.
    shards: HashMap<LaunchId, Running>,
    /// Launches in flight (still shaking hands, or told to wait their turn),
    /// each with the signal that stops it.
    launching: HashMap<LaunchId, Arc<Notify>>,
    /// Launches killed while in flight: the launch tears the shard down the
    /// moment it would have registered it (the signal above usually stops it
    /// before that).
    cancelled: HashSet<LaunchId>,
}

/// The in-process launcher: one real shard per [`SpawnSpec`], in the caller's
/// runtime.
///
/// `spawn` starts an orchestrated shard on `127.0.0.1:0` that registers at the
/// spec's control address (see the module docs) and returns once it is up.
/// `kill` stops it — or, for a shard still being launched, ends that launch.
/// `poll` reports `Dead` once it has stopped, by a kill or by itself.
#[derive(Debug)]
pub struct LocalLauncher {
    running: Mutex<Table>,
    /// Whether a launched shard registers with the orchestrator.
    registers: bool,
}

impl Default for LocalLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalLauncher {
    /// A launcher with nothing running, whose shards register with the
    /// orchestrator at [`SpawnSpec::control_addr`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            running: Mutex::new(Table::default()),
            registers: true,
        }
    }

    /// A launcher whose shards do **not** register with any orchestrator: each
    /// one owns a depth-40 root as the shard whose id is the launch's number,
    /// and serves. For tests of the launcher mechanics (bind, liveness, kill)
    /// that have no control plane; a detached shard can never take part in an
    /// epoch.
    #[must_use]
    pub fn detached() -> Self {
        Self {
            running: Mutex::new(Table::default()),
            registers: false,
        }
    }

    /// The id the shard of `launch` registered as, if it is up and registered.
    pub async fn shard_of(&self, launch: LaunchId) -> Option<ShardId> {
        self.running
            .lock()
            .await
            .shards
            .get(&launch)
            .and_then(Running::shard)
    }

    /// The launch that became `shard`, if it is one of this launcher's.
    pub async fn launch_of(&self, shard: ShardId) -> Option<LaunchId> {
        self.running
            .lock()
            .await
            .shards
            .iter()
            .find(|(_, running)| running.shard() == Some(shard))
            .map(|(launch, _)| *launch)
    }

    /// Whether the shard registered as `shard` is one of this launcher's and
    /// still running (what [`Launcher::poll`] says for its launch).
    pub async fn is_alive(&self, shard: ShardId) -> bool {
        self.running
            .lock()
            .await
            .shards
            .values()
            .any(|running| running.shard() == Some(shard) && running.is_running())
    }

    /// Bring one shard up (bound, registered unless detached, serving, ticking).
    async fn launch(&self, spec: &SpawnSpec) -> Result<Running, LaunchError> {
        // Bind on loopback, port 0: the launcher never assumes a fixed port.
        let loopback = SocketAddr::from(([127, 0, 0, 1], 0));
        if self.registers {
            let control = dialable(spec.control_addr);
            // No id, no region: the orchestrator assigns both at the hello
            // (H9: never one the shard chose).
            let options = OrchestratedOptions {
                shard: ShardId::NONE,
                bind: loopback,
                advertise: None,
                claim_region: None,
                cluster: spec.cluster.clone(),
                secret: spec.secret.clone(),
            };
            return start_orchestrated(options, control)
                .await
                .map(Running::Orchestrated)
                .map_err(|err| {
                    LaunchError::Spawn(format!(
                        "the shard of {} could not register with the orchestrator at \
                         {control}: {err}",
                        spec.launch
                    ))
                });
        }
        let spawn_error = |e: std::io::Error| LaunchError::Spawn(e.to_string());
        let net = Arc::new(ShardNet::bind(loopback).map_err(spawn_error)?);
        let bound = net.local_addr().map_err(spawn_error)?;
        let id = ShardId::new(u32::try_from(spec.launch.0).unwrap_or(u32::MAX - 1));
        let config = ShardConfig::new(id, bound, space_world::NodeRef::new(40, [0, 0, 0]));
        let shard = Arc::new(Mutex::new(Shard::new(config)));
        let serve = {
            let net = Arc::clone(&net);
            let shard = Arc::clone(&shard);
            tokio::spawn(async move { net.serve(shard).await })
        };
        let tick = {
            let net = Arc::clone(&net);
            let shard = Arc::clone(&shard);
            tokio::spawn(async move {
                let mut clock = tokio::time::interval(std::time::Duration::from_secs_f64(
                    f64::from(space_sim::dt()),
                ));
                clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    clock.tick().await;
                    shard.lock().await.tick();
                    net.flush(&shard).await;
                }
            })
        };
        Ok(Running::Detached {
            net,
            tick,
            serve,
            endpoint: bound,
        })
    }
}

impl Launcher for LocalLauncher {
    async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
        let id = spec.launch;
        let stop = Arc::new(Notify::new());
        self.running
            .lock()
            .await
            .launching
            .insert(id, Arc::clone(&stop));
        // The lock is not held while the shard registers: that takes as long
        // as the orchestrator makes it wait its turn, and `kill`/`poll` must
        // not wait for it. A kill meanwhile ends the launch where it stands.
        let killed = || {
            LaunchError::Spawn(format!(
                "the shard of {id} was killed while it was starting"
            ))
        };
        let launched = tokio::select! {
            launched = self.launch(&spec) => launched,
            () = stop.notified() => Err(killed()),
        };

        let mut table = self.running.lock().await;
        table.launching.remove(&id);
        let cancelled = table.cancelled.remove(&id);
        let running = launched?;
        if cancelled {
            // Killed in the instant it came up: it must not stay up after all.
            drop(table);
            running.stop();
            return Err(killed());
        }
        let handle = ShardHandle {
            launch: id,
            shard: running.shard(),
            endpoint: running.endpoint(),
            process: format!("in-process:{}", id.0),
        };
        table.shards.insert(id, running);
        Ok(handle)
    }

    async fn kill(&self, launch: LaunchId) -> Result<(), LaunchError> {
        let running = {
            let mut table = self.running.lock().await;
            let running = table.shards.remove(&launch);
            if running.is_none() {
                if let Some(stop) = table.launching.get(&launch) {
                    // Still starting: end the launch, and remember the kill in
                    // case it finishes in this very instant.
                    stop.notify_one();
                    table.cancelled.insert(launch);
                }
            }
            running
        };
        // Not running (and not starting): killing it is a no-op, not an error —
        // the orchestrator may kill on a path that raced a death.
        if let Some(running) = running {
            running.stop();
        }
        Ok(())
    }

    async fn poll(&self, launch: LaunchId) -> Result<Poll, LaunchError> {
        let table = self.running.lock().await;
        Ok(match table.shards.get(&launch) {
            Some(running) if running.is_running() => Poll::Alive,
            // Unknown, still starting, or stopped by itself (its tick loop
            // died, or it lost the orchestrator).
            _ => Poll::Dead,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(launch: u64) -> SpawnSpec {
        SpawnSpec {
            launch: LaunchId(launch),
            cluster: "test".into(),
            control_addr: SocketAddr::from(([127, 0, 0, 1], 7_900)),
            host_port: 0,
            secret: String::new(),
        }
    }

    #[tokio::test]
    async fn spawn_returns_a_reachable_endpoint_and_poll_reports_alive() {
        let launcher = LocalLauncher::detached();
        let handle = launcher.spawn(spec(1)).await.expect("spawn");
        assert!(handle.endpoint.port() != 0, "a real port was bound");
        assert_eq!(handle.launch, LaunchId(1));
        assert_eq!(handle.shard, None, "a detached shard registered nowhere");
        assert_eq!(launcher.poll(LaunchId(1)).await.unwrap(), Poll::Alive);
        launcher.kill(LaunchId(1)).await.expect("kill");
    }

    #[tokio::test]
    async fn kill_makes_poll_report_dead() {
        let launcher = LocalLauncher::detached();
        launcher.spawn(spec(2)).await.expect("spawn");
        assert_eq!(launcher.poll(LaunchId(2)).await.unwrap(), Poll::Alive);
        launcher.kill(LaunchId(2)).await.expect("kill");
        // After the kill the shard is no longer supervised: poll says Dead.
        assert_eq!(launcher.poll(LaunchId(2)).await.unwrap(), Poll::Dead);
    }

    /// The liveness branch: a shard still *registered* whose tick loop has died
    /// (a panic, or an abort that did not go through `kill`) is reported dead.
    /// The entry stays in the map, so this is not the "unknown shard" path.
    ///
    /// Mutation this catches: dropping the `is_finished` checks in `poll`.
    #[tokio::test]
    async fn poll_reports_dead_when_a_registered_shards_tick_loop_stops() {
        for abort_tick in [true, false] {
            let launcher = LocalLauncher::detached();
            let id = LaunchId(if abort_tick { 30 } else { 31 });
            launcher.spawn(spec(id.0)).await.expect("spawn");
            assert_eq!(launcher.poll(id).await.unwrap(), Poll::Alive);

            let task = {
                let guard = launcher.running.lock().await;
                let Some(Running::Detached { tick, serve, .. }) = guard.shards.get(&id) else {
                    panic!("a detached shard is running");
                };
                let task = if abort_tick { tick } else { serve };
                task.abort();
                task.abort_handle()
            };
            // Wait for the abort to land, then poll: bounded, no fixed sleep.
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while !task.is_finished() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("the aborted task finishes");

            assert_eq!(launcher.poll(id).await.unwrap(), Poll::Dead);
            assert!(
                launcher.running.lock().await.shards.contains_key(&id),
                "still registered: the liveness branch fired, not the unknown-shard one"
            );
            launcher.kill(id).await.expect("kill");
        }
    }

    /// A registering launch whose orchestrator never answers is still
    /// "starting" — and a `kill` that arrives meanwhile ends it at once:
    /// nothing is left running, supervised or half-registered.
    ///
    /// Mutation this catches: a kill that only marks the launch (it would go
    /// on dialling an orchestrator that is not there for as long as its
    /// patience lasts, and one that is told to wait its turn for ever).
    #[tokio::test]
    async fn a_kill_ends_a_launch_that_is_still_trying_to_register() {
        // A bound UDP socket nobody reads: the handshake gets no answer.
        let silent = std::net::UdpSocket::bind("127.0.0.1:0").expect("bind a silent port");
        let mut spec = spec(40);
        spec.control_addr = silent.local_addr().expect("silent addr");

        let launcher = Arc::new(LocalLauncher::new());
        let launch = {
            let launcher = Arc::clone(&launcher);
            tokio::spawn(async move { launcher.spawn(spec).await })
        };
        // Wait until the launch is in flight, then kill it.
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !launcher
                .running
                .lock()
                .await
                .launching
                .contains_key(&LaunchId(40))
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the launch starts");
        assert_eq!(
            launcher.poll(LaunchId(40)).await.unwrap(),
            Poll::Dead,
            "a shard that is still starting is not alive yet"
        );
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            launcher.kill(LaunchId(40)),
        )
        .await
        .expect("a kill never waits for a launch in flight")
        .expect("kill");

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(3), launch)
            .await
            .expect("the launch ends at once, not when its patience runs out")
            .expect("the launch task did not panic");
        let err = outcome.expect_err("it was killed");
        assert!(
            matches!(&err, LaunchError::Spawn(why) if why.contains("killed while it was starting")),
            "{err:?}"
        );
        let table = launcher.running.lock().await;
        assert!(table.shards.is_empty(), "nothing was left running");
        assert!(
            table.launching.is_empty() && table.cancelled.is_empty(),
            "and nothing is remembered about the launch"
        );
    }

    #[tokio::test]
    async fn poll_on_an_unknown_launch_is_dead() {
        let launcher = LocalLauncher::detached();
        assert_eq!(
            launcher.poll(LaunchId(99)).await.unwrap(),
            Poll::Dead,
            "a shard the launcher never spawned is not alive"
        );
    }

    #[tokio::test]
    async fn killing_a_shard_that_is_not_running_is_a_noop() {
        let launcher = LocalLauncher::detached();
        launcher
            .kill(LaunchId(123))
            .await
            .expect("killing an unknown shard is not an error");
    }

    /// The launched shard is reachable over the real transport: a client connects
    /// through quinn and is welcomed with a session and a ship.
    #[tokio::test]
    async fn a_spawned_shard_answers_a_real_client_connection() {
        let launcher = LocalLauncher::detached();
        let handle = launcher.spawn(spec(7)).await.expect("spawn");
        let endpoint = space_net::client_endpoint().expect("client endpoint");
        let (conn, welcome) = space_net::connect(
            &endpoint,
            handle.endpoint,
            space_server::net::SERVER_NAME,
            &space_net::hello("player"),
        )
        .await
        .expect("the launched shard must accept a client");
        assert_eq!(
            welcome.shard,
            ShardId::new(7),
            "a detached shard runs as its launch's number"
        );
        assert_eq!(welcome.session, 1);
        assert!(!conn.is_closed(), "the connection is live");
        launcher.kill(LaunchId(7)).await.expect("kill");
    }
}
