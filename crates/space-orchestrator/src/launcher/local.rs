//! The in-process launcher (plan §3.4, impl 2).
//!
//! [`LocalLauncher`] runs a real [`Shard`] + [`ShardNet`] inside the caller's
//! tokio runtime — literally the `RunningShard` harness from
//! `crates/space-server/tests/acceptance.rs`, lifted behind the [`Launcher`]
//! seam so the orchestrator's tests can start real shards without duplicating
//! the wiring. Real QUIC, a real 30 Hz tick loop, real handoffs: this is what
//! makes the whole split/merge/rebalance loop verifiable in-sandbox, with no
//! Docker daemon.
//!
//! # Registering with the orchestrator
//!
//! A launched shard is an *orchestrated* shard: it dials the control address in
//! its [`SpawnSpec`], says `ShardHello` with the id the orchestrator allocated
//! for it, runs the map the `ShardWelcome` hands back and follows every epoch
//! pushed over that link — the same boot `space-server --orchestrator` does
//! (`space_server::orchestrated`), built from the same public pieces so the
//! launcher keeps the task handles its liveness probe needs. A shard that only
//! binds and ticks, as this launcher used to start, can never ack the
//! `EpochPrepare` that creates its region, so the split that launched it could
//! never commit. [`LocalLauncher::detached`] keeps that bare mode for tests of
//! the launcher mechanics that have no orchestrator to talk to.
//!
//! # Liveness
//!
//! [`Launcher::poll`] reports a shard dead when its tick loop has stopped
//! advancing. A shard that is running ticks its [`Shard`] on a wall-clock
//! interval; if the loop task has been aborted (a `kill`, or a panic) the tick
//! counter stops moving, and the next `poll` sees no progress and reports
//! [`Poll::Dead`]. This is a liveness *probe*, not a heartbeat protocol — the
//! orchestrator's own `LOAD_REPORT_INTERVAL` beat is the production liveness
//! signal (H15); this probe is the launcher-local half that lets a test prove a
//! killed shard is detected.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use space_protocol::shard::{ShardId, ShardRegion};
use space_protocol::ShardHello;
use space_server::net::{OrchestratorLink, ShardNet};
use space_server::shard::Shard;
use space_server::ShardConfig;
use tokio::sync::Mutex;

use super::{LaunchError, Launcher, Poll, ShardHandle, SpawnSpec};

/// A shard the launcher is running: its net front end, the two tasks that
/// drive it and (unless detached) its control link to the orchestrator.
struct Running {
    net: Arc<ShardNet>,
    tick: tokio::task::JoinHandle<()>,
    serve: tokio::task::JoinHandle<()>,
    /// The control connection to the orchestrator (`None` when detached).
    link: Option<OrchestratorLink>,
    /// The endpoint the shard bound to (resolved after bind).
    endpoint: SocketAddr,
}

impl Running {
    /// Stop the shard: its tasks, its control link and its endpoint.
    fn stop(self) {
        self.tick.abort();
        self.serve.abort();
        if let Some(link) = self.link {
            link.close();
        }
        self.net.close();
    }
}

impl std::fmt::Debug for Running {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Running")
            .field("endpoint", &self.endpoint)
            .field("registered", &self.link.is_some())
            .finish_non_exhaustive()
    }
}

/// Everything the launcher supervises, behind one lock so a `kill` and a
/// launch finishing can never miss each other.
#[derive(Debug, Default)]
struct Table {
    /// The shards that are up.
    shards: HashMap<ShardId, Running>,
    /// Shards whose launch is in flight (still shaking hands).
    launching: HashSet<ShardId>,
    /// Shards killed while their launch was in flight: the launch tears the
    /// shard down the moment it would have registered it.
    cancelled: HashSet<ShardId>,
}

/// The in-process launcher: one real [`Shard`] per [`SpawnSpec`], in the
/// caller's runtime.
///
/// `spawn` binds a [`ShardNet`] on `127.0.0.1:0`, registers the shard with the
/// orchestrator at the spec's control address (see the module docs), starts the
/// accept loop and the 30 Hz tick loop, and returns once the shard is up.
/// `kill` aborts the tasks and closes the endpoint — or, for a shard still being
/// launched, makes that launch end in a teardown. `poll` reports `Dead` once the
/// tick task has been aborted.
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
    /// one just owns the spec's region and serves. For tests of the launcher
    /// mechanics (bind, liveness, kill) that have no control plane; a detached
    /// shard can never take part in an epoch.
    #[must_use]
    pub fn detached() -> Self {
        Self {
            running: Mutex::new(Table::default()),
            registers: false,
        }
    }

    /// Bring one shard up (bound, registered unless detached, serving, ticking).
    async fn launch(&self, spec: &SpawnSpec) -> Result<Running, LaunchError> {
        let spawn_error = |e: std::io::Error| LaunchError::Spawn(e.to_string());
        // Bind on loopback, port 0: the launcher never assumes a fixed port.
        let loopback = SocketAddr::from(([127, 0, 0, 1], 0));
        let net = Arc::new(ShardNet::bind(loopback).map_err(spawn_error)?);
        let bound = net.local_addr().map_err(spawn_error)?;

        let (shard, link) = if self.registers {
            let control = dialable(spec.control_addr);
            // The id is the one the orchestrator allocated (H9: never one the
            // shard chose); the hello only confirms it and names the endpoint
            // the shard really bound.
            let hello = ShardHello {
                cluster: spec.cluster.clone(),
                shard: spec.shard,
                endpoint: bound,
                region: Some(ShardRegion::new(spec.region.node, spec.shard, bound)),
            };
            let (welcome, conn) = match OrchestratorLink::handshake(control, hello).await {
                Ok(answer) => answer,
                Err(err) => {
                    net.close();
                    return Err(LaunchError::Spawn(format!(
                        "{} could not register with the orchestrator at {control}: {err}",
                        spec.shard
                    )));
                }
            };
            if welcome.shard != spec.shard {
                net.close();
                return Err(LaunchError::Spawn(format!(
                    "the orchestrator welcomed {} as {}",
                    spec.shard, welcome.shard
                )));
            }
            let mut shard = Shard::new(ShardConfig::new(welcome.shard, bound, welcome.region.node));
            shard.set_shard_map(welcome.shard_map.clone());
            shard.set_endpoint(bound);
            let shard = Arc::new(Mutex::new(shard));
            for region in &welcome.shard_map.regions {
                if region.shard != welcome.shard {
                    net.set_shard_endpoint(region.shard, region.endpoint).await;
                }
            }
            // Epoch prepares/commits/aborts and map pushes come in over the
            // link; the shard's acks go out through it.
            let link = OrchestratorLink::attach(conn, Arc::clone(&shard));
            net.set_orchestrator_sender(link.tx.clone()).await;
            (shard, Some(link))
        } else {
            let config = ShardConfig::new(spec.shard, bound, spec.region.node);
            (Arc::new(Mutex::new(Shard::new(config))), None)
        };

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
        Ok(Running {
            net,
            tick,
            serve,
            link,
            endpoint: bound,
        })
    }
}

/// The address an in-process shard dials to reach `control`.
///
/// The orchestrator is this very process, so a wildcard bind address
/// (`0.0.0.0:7900`, which cannot be dialled — on Windows not at all) means
/// "loopback, that port".
fn dialable(control: SocketAddr) -> SocketAddr {
    if !control.ip().is_unspecified() {
        return control;
    }
    let loopback = match control {
        SocketAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
        SocketAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
    };
    SocketAddr::new(loopback, control.port())
}

impl Launcher for LocalLauncher {
    async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
        let id = spec.shard;
        self.running.lock().await.launching.insert(id);
        // The lock is not held while the shard shakes hands with the
        // orchestrator: that can take seconds, and `kill`/`poll` must not wait.
        let launched = self.launch(&spec).await;

        let mut table = self.running.lock().await;
        table.launching.remove(&id);
        let cancelled = table.cancelled.remove(&id);
        let running = launched?;
        if cancelled {
            // Killed while it was starting (its split aborted before it had
            // registered): it must not come up after all.
            drop(table);
            running.stop();
            return Err(LaunchError::Spawn(format!(
                "{id} was killed while it was starting"
            )));
        }
        let handle = ShardHandle {
            shard: id,
            endpoint: running.endpoint,
            process: format!("in-process:{}", id.0),
        };
        table.shards.insert(id, running);
        Ok(handle)
    }

    async fn kill(&self, shard: ShardId) -> Result<(), LaunchError> {
        let running = {
            let mut table = self.running.lock().await;
            let running = table.shards.remove(&shard);
            if running.is_none() && table.launching.contains(&shard) {
                // Still starting: remember the kill; `spawn` carries it out.
                table.cancelled.insert(shard);
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

    async fn poll(&self, shard: ShardId) -> Result<Poll, LaunchError> {
        let table = self.running.lock().await;
        let Some(running) = table.shards.get(&shard) else {
            return Ok(Poll::Dead);
        };
        // The tick loop is the shard's heartbeat: if its task has finished
        // (aborted by a kill, or panicked) the shard is no longer simulating.
        if running.tick.is_finished() || running.serve.is_finished() {
            return Ok(Poll::Dead);
        }
        Ok(Poll::Alive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::shard::ShardRegion;
    use space_world::NodeRef;

    fn spec(shard: u32) -> SpawnSpec {
        SpawnSpec {
            shard: ShardId::new(shard),
            region: ShardRegion::new(
                NodeRef::new(40, [0, 0, 0]),
                ShardId::new(shard),
                SocketAddr::from(([127, 0, 0, 1], 0)),
            ),
            cluster: "test".into(),
            control_addr: SocketAddr::from(([127, 0, 0, 1], 7_900)),
            host_port: 0,
        }
    }

    #[tokio::test]
    async fn spawn_returns_a_reachable_endpoint_and_poll_reports_alive() {
        let launcher = LocalLauncher::detached();
        let handle = launcher.spawn(spec(1)).await.expect("spawn");
        assert!(handle.endpoint.port() != 0, "a real port was bound");
        assert_eq!(handle.shard, ShardId::new(1));
        assert_eq!(launcher.poll(ShardId::new(1)).await.unwrap(), Poll::Alive);
        launcher.kill(ShardId::new(1)).await.expect("kill");
    }

    #[tokio::test]
    async fn kill_makes_poll_report_dead() {
        let launcher = LocalLauncher::detached();
        launcher.spawn(spec(2)).await.expect("spawn");
        assert_eq!(launcher.poll(ShardId::new(2)).await.unwrap(), Poll::Alive);
        launcher.kill(ShardId::new(2)).await.expect("kill");
        // After the kill the shard is no longer supervised: poll says Dead.
        assert_eq!(launcher.poll(ShardId::new(2)).await.unwrap(), Poll::Dead);
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
            let id = ShardId::new(if abort_tick { 30 } else { 31 });
            launcher.spawn(spec(id.0)).await.expect("spawn");
            assert_eq!(launcher.poll(id).await.unwrap(), Poll::Alive);

            let task = {
                let guard = launcher.running.lock().await;
                let running = guard.shards.get(&id).expect("registered");
                let task = if abort_tick {
                    &running.tick
                } else {
                    &running.serve
                };
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

    #[test]
    fn a_wildcard_control_address_is_dialled_on_loopback() {
        let wildcard: SocketAddr = "0.0.0.0:7900".parse().unwrap();
        assert_eq!(dialable(wildcard), "127.0.0.1:7900".parse().unwrap());
        let wildcard6: SocketAddr = "[::]:7900".parse().unwrap();
        assert_eq!(dialable(wildcard6), "[::1]:7900".parse().unwrap());
        let concrete: SocketAddr = "10.0.0.2:7900".parse().unwrap();
        assert_eq!(dialable(concrete), concrete, "a real address is kept");
    }

    /// A registering launch whose orchestrator never answers fails — and a
    /// `kill` that arrives while it is still trying is honoured: nothing is
    /// left running, supervised or half-registered.
    ///
    /// (Bounded by the transport's handshake timeout, a few seconds.)
    #[tokio::test]
    async fn a_launch_that_cannot_register_fails_and_a_kill_meanwhile_is_honoured() {
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
                .contains(&ShardId::new(40))
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the launch starts");
        assert_eq!(
            launcher.poll(ShardId::new(40)).await.unwrap(),
            Poll::Dead,
            "a shard that is still starting is not alive yet"
        );
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            launcher.kill(ShardId::new(40)),
        )
        .await
        .expect("a kill never waits for a launch in flight")
        .expect("kill");

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(30), launch)
            .await
            .expect("the launch gives up")
            .expect("the launch task did not panic");
        let err = outcome.expect_err("no orchestrator answered");
        assert!(matches!(err, LaunchError::Spawn(_)), "{err:?}");
        let table = launcher.running.lock().await;
        assert!(table.shards.is_empty(), "nothing was left running");
        assert!(
            table.launching.is_empty() && table.cancelled.is_empty(),
            "and nothing is remembered about the failed launch"
        );
    }

    #[tokio::test]
    async fn poll_on_an_unknown_shard_is_dead() {
        let launcher = LocalLauncher::detached();
        assert_eq!(
            launcher.poll(ShardId::new(99)).await.unwrap(),
            Poll::Dead,
            "a shard the launcher never spawned is not alive"
        );
    }

    #[tokio::test]
    async fn killing_a_shard_that_is_not_running_is_a_noop() {
        let launcher = LocalLauncher::detached();
        launcher
            .kill(ShardId::new(123))
            .await
            .expect("killing an unknown shard is not an error");
    }

    /// The launched shard is reachable over the real transport: a client connects
    /// through quinn and is welcomed with a session and a ship.
    ///
    /// This is the property the orchestrator depends on — `spawn` must not return
    /// until the shard answers a connection — proven against a real QUIC endpoint
    /// rather than a mock.
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
        assert_eq!(welcome.shard, ShardId::new(7));
        assert_eq!(welcome.session, 1);
        assert!(!conn.is_closed(), "the connection is live");
        launcher.kill(ShardId::new(7)).await.expect("kill");
    }
}
