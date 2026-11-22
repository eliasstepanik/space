//! The [`Launcher`] seam: how a shard process comes to exist (plan §3.4).
//!
//! The orchestrator never spawns a process or talks to a daemon directly; it
//! asks a `Launcher`. The impls:
//!
//! * [`local::LocalLauncher`] — a real in-process [`space_server::shard::Shard`]
//!   with a [`space_server::net::ShardNet`] per shard, in the caller's tokio
//!   runtime. The sandbox-testable impl: no Docker daemon is required, and it
//!   is what makes the split/merge loop verifiable end to end.
//! * [`docker::DockerLauncher`] — **deferred stub.** The real `bollard`-backed
//!   impl needs `bollard` and the hyper stack, which are not in the offline cargo
//!   cache; every call returns [`LaunchError::Unavailable`] (see `docker.rs`).
//!
//! The trait is deliberately small and *asynchronous*: `spawn` must not return
//! until the shard is reachable, because the orchestrator's next step is to
//! open a control connection to it.

use space_protocol::shard::{ShardId, ShardRegion};
use std::net::SocketAddr;

pub mod docker;
pub mod local;

/// Everything the launcher needs to bring one shard up.
#[derive(Debug, Clone)]
pub struct SpawnSpec {
    /// The id the orchestrator allocated for this shard (H9: never argv).
    pub shard: ShardId,
    /// The region the shard owns, including the endpoint it must bind.
    pub region: ShardRegion,
    /// The cluster id, forwarded so the shard's `ShardHello` matches.
    pub cluster: String,
    /// The orchestrator's control address the shard dials back to.
    pub control_addr: SocketAddr,
    /// Host UDP port to publish for this shard (from the config's port range).
    pub host_port: u16,
}

/// A shard process the launcher is supervising.
#[derive(Debug, Clone)]
pub struct ShardHandle {
    /// The id the shard runs as.
    pub shard: ShardId,
    /// Where the shard's QUIC endpoint answered.
    pub endpoint: SocketAddr,
    /// The launcher's private id for the process (pid, container id, …).
    pub process: String,
}

/// Why a spawn or poll failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    /// The launcher could not start the shard at all.
    Spawn(String),
    /// The shard was started but never became reachable.
    Unreachable(ShardId),
    /// The launcher cannot answer this question.
    Unsupported(String),
    /// The launcher is not available in this build (the deferred Docker stub).
    Unavailable(String),
}

impl core::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LaunchError::Spawn(why) => write!(f, "spawn failed: {why}"),
            LaunchError::Unreachable(s) => write!(f, "{s} never became reachable"),
            LaunchError::Unsupported(what) => write!(f, "launcher does not support: {what}"),
            LaunchError::Unavailable(why) => write!(f, "launcher unavailable: {why}"),
        }
    }
}

impl std::error::Error for LaunchError {}

/// Whether a supervised shard is still alive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Poll {
    /// Alive and (as far as the launcher can tell) healthy.
    Alive,
    /// Exited or unreachable; the orchestrator reassigns its regions (H15).
    Dead,
}

/// The seam between the orchestrator and the process layer.
///
/// Native `impl Future` in trait (Rust 1.100, edition 2021 — no `async-trait`
/// dependency). The returned futures are `+ Send` so the orchestrator can
/// `tokio::spawn` a call; the trait is not dyn-compatible, so the orchestrator
/// is generic over `L: Launcher`, which also makes a `MockLauncher` in the
/// acceptance tests a plain struct.
pub trait Launcher: Send + Sync {
    /// Start a shard and return only once it is reachable.
    ///
    /// # Errors
    ///
    /// [`LaunchError::Spawn`] if the process cannot start,
    /// [`LaunchError::Unreachable`] if it starts but never binds.
    fn spawn(
        &self,
        spec: SpawnSpec,
    ) -> impl std::future::Future<Output = Result<ShardHandle, LaunchError>> + Send;

    /// Stop a shard cleanly (the orchestrator has already moved its regions).
    ///
    /// # Errors
    ///
    /// Any launcher-specific failure; the orchestrator logs and continues —
    /// a shard that will not die is a leak, not a correctness bug.
    fn kill(
        &self,
        shard: ShardId,
    ) -> impl std::future::Future<Output = Result<(), LaunchError>> + Send;

    /// Is the shard still alive? The liveness half of H15.
    ///
    /// # Errors
    ///
    /// [`LaunchError::Unsupported`] for a launcher with no liveness probe.
    fn poll(
        &self,
        shard: ShardId,
    ) -> impl std::future::Future<Output = Result<Poll, LaunchError>> + Send;
}
