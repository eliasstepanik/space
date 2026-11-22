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

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use space_protocol::shard::ShardId;
use space_server::net::ShardNet;
use space_server::shard::Shard;
use space_server::ShardConfig;
use tokio::sync::Mutex;

use super::{LaunchError, Launcher, Poll, ShardHandle, SpawnSpec};

/// A shard the launcher is running: its net front end and the two tasks that
/// drive it.
struct Running {
    net: Arc<ShardNet>,
    tick: tokio::task::JoinHandle<()>,
    serve: tokio::task::JoinHandle<()>,
    /// The endpoint the shard bound to (resolved after bind).
    endpoint: SocketAddr,
}

impl std::fmt::Debug for Running {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Running")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

/// The in-process launcher: one real [`Shard`] per [`SpawnSpec`], in the
/// caller's runtime.
///
/// `spawn` binds a [`ShardNet`] on `127.0.0.1:0`, starts the accept loop and the
/// 30 Hz tick loop, and returns once the endpoint is bound (so the orchestrator
/// can immediately dial it). `kill` aborts the tasks and closes the endpoint.
/// `poll` reports `Dead` once the tick task has been aborted.
#[derive(Debug, Default)]
pub struct LocalLauncher {
    running: Mutex<HashMap<ShardId, Running>>,
}

impl LocalLauncher {
    /// A launcher with nothing running.
    #[must_use]
    pub fn new() -> Self {
        Self {
            running: Mutex::new(HashMap::new()),
        }
    }
}

impl Launcher for LocalLauncher {
    async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
        // Bind on loopback, port 0: the launcher never assumes a fixed port.
        let loopback = SocketAddr::from(([127, 0, 0, 1], 0));
        let net =
            Arc::new(ShardNet::bind(loopback).map_err(|e| LaunchError::Spawn(e.to_string()))?);
        let bound = net
            .local_addr()
            .map_err(|e| LaunchError::Spawn(e.to_string()))?;
        let config = ShardConfig::new(spec.shard, bound, spec.region.node);
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

        let handle = ShardHandle {
            shard: spec.shard,
            endpoint: bound,
            process: format!("in-process:{}", spec.shard.0),
        };
        self.running.lock().await.insert(
            spec.shard,
            Running {
                net,
                tick,
                serve,
                endpoint: bound,
            },
        );
        Ok(handle)
    }

    async fn kill(&self, shard: ShardId) -> Result<(), LaunchError> {
        let Some(running) = self.running.lock().await.remove(&shard) else {
            // Already gone: killing a shard that is not running is a no-op, not
            // an error — the orchestrator may kill on a path that raced a death.
            return Ok(());
        };
        running.tick.abort();
        running.serve.abort();
        running.net.close();
        Ok(())
    }

    async fn poll(&self, shard: ShardId) -> Result<Poll, LaunchError> {
        let guard = self.running.lock().await;
        let Some(running) = guard.get(&shard) else {
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
        let launcher = LocalLauncher::new();
        let handle = launcher.spawn(spec(1)).await.expect("spawn");
        assert!(handle.endpoint.port() != 0, "a real port was bound");
        assert_eq!(handle.shard, ShardId::new(1));
        assert_eq!(launcher.poll(ShardId::new(1)).await.unwrap(), Poll::Alive);
        launcher.kill(ShardId::new(1)).await.expect("kill");
    }

    #[tokio::test]
    async fn kill_makes_poll_report_dead() {
        let launcher = LocalLauncher::new();
        launcher.spawn(spec(2)).await.expect("spawn");
        assert_eq!(launcher.poll(ShardId::new(2)).await.unwrap(), Poll::Alive);
        launcher.kill(ShardId::new(2)).await.expect("kill");
        // After the kill the shard is no longer supervised: poll says Dead.
        assert_eq!(launcher.poll(ShardId::new(2)).await.unwrap(), Poll::Dead);
    }

<<<<<<< HEAD
    /// The liveness branch: a shard still *registered* whose tick loop has died
    /// (a panic, or an abort that did not go through `kill`) is reported dead.
    /// The entry stays in the map, so this is not the "unknown shard" path.
    ///
    /// Mutation this catches: dropping the `is_finished` checks in `poll`.
    #[tokio::test]
    async fn poll_reports_dead_when_a_registered_shards_tick_loop_stops() {
        for abort_tick in [true, false] {
            let launcher = LocalLauncher::new();
            let id = ShardId::new(if abort_tick { 30 } else { 31 });
            launcher.spawn(spec(id.0)).await.expect("spawn");
            assert_eq!(launcher.poll(id).await.unwrap(), Poll::Alive);

            let task = {
                let guard = launcher.running.lock().await;
                let running = guard.get(&id).expect("registered");
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
                launcher.running.lock().await.contains_key(&id),
                "still registered: the liveness branch fired, not the unknown-shard one"
            );
            launcher.kill(id).await.expect("kill");
        }
    }

=======
>>>>>>> refs/remotes/origin/main
    #[tokio::test]
    async fn poll_on_an_unknown_shard_is_dead() {
        let launcher = LocalLauncher::new();
        assert_eq!(
            launcher.poll(ShardId::new(99)).await.unwrap(),
            Poll::Dead,
            "a shard the launcher never spawned is not alive"
        );
    }

    #[tokio::test]
    async fn killing_a_shard_that_is_not_running_is_a_noop() {
        let launcher = LocalLauncher::new();
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
        let launcher = LocalLauncher::new();
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
