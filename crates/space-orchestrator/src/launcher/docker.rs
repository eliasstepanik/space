//! The Docker launcher — **DEFERRED** (plan §3.4 impl 3, P3.8).
//!
//! # What is deferred
//!
//! The real `bollard`-backed `DockerLauncher`: create/start a shard container
//! from `deploy/Dockerfile.shard`, wait for its `ShardHello`, `docker rm -f` on
//! kill, list-by-label for `poll` and orphan reaping. None of that is here.
//!
//! # Why
//!
//! `bollard` and its whole hyper/http/futures dependency tree are **not in the
//! offline cargo cache** (`cargo metadata --offline` fails with "no matching
//! package named `bollard`"), and the build sandbox has no network access, so the
//! dependency cannot be fetched. The default build must stay offline-clean, so
//! no dependency was added. Landing it needs network access to fetch the crates,
//! then `bollard = { default-features = false, features = ["http"] }` (plan F17)
//! and a Docker daemon to run the `P3.0` end-to-end entry criterion.
//!
//! # What this stub does instead
//!
//! [`DockerLauncher`] implements [`Launcher`] so `--launcher docker` is wired end
//! to end, but every operation returns [`LaunchError::Unavailable`] — a clear
//! runtime error, never a panic. The deployment layer it will drive
//! (`deploy/Dockerfile.*`, `deploy/docker-compose.yaml`) and the container-path
//! flags (`space-server --bind/--advertise`, orchestrator `--control-addr`) are
//! already in place. See "Deferred: bollard DockerLauncher" in
//! `docs/plan-phase3.md`.

use super::{LaunchError, LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};

/// Why the Docker launcher cannot run in this build.
pub const UNAVAILABLE: &str = "docker launcher requires the bollard dependency, which is not \
     available in this offline build — see docs/plan-phase3.md deferred items";

/// The deferred Docker launcher: constructible, but every call fails with
/// [`LaunchError::Unavailable`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DockerLauncher;

impl DockerLauncher {
    /// The (stub) launcher.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Whether this launcher can run at all; lets the binary fail fast at startup
    /// instead of at the first split.
    ///
    /// # Errors
    ///
    /// Always [`LaunchError::Unavailable`] in this build.
    pub fn ensure_available(&self) -> Result<(), LaunchError> {
        Err(unavailable())
    }
}

fn unavailable() -> LaunchError {
    LaunchError::Unavailable(UNAVAILABLE.to_owned())
}

impl Launcher for DockerLauncher {
    async fn spawn(&self, _spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
        Err(unavailable())
    }

    async fn kill(&self, _launch: LaunchId) -> Result<(), LaunchError> {
        Err(unavailable())
    }

    async fn poll(&self, _launch: LaunchId) -> Result<Poll, LaunchError> {
        Err(unavailable())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn spec() -> SpawnSpec {
        SpawnSpec {
            launch: LaunchId(1),
            cluster: "test".into(),
            control_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            host_port: 0,
            secret: String::new(),
        }
    }

    #[tokio::test]
    async fn every_operation_is_a_clear_unavailable_error_not_a_panic() {
        let launcher = DockerLauncher::new();
        let want = LaunchError::Unavailable(UNAVAILABLE.to_owned());
        assert_eq!(launcher.spawn(spec()).await.unwrap_err(), want);
        assert_eq!(launcher.kill(LaunchId(1)).await.unwrap_err(), want);
        assert_eq!(launcher.poll(LaunchId(1)).await.unwrap_err(), want);
        assert_eq!(launcher.ensure_available().unwrap_err(), want);
        assert!(want.to_string().contains("bollard"));
    }
}
