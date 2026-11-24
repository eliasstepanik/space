//! The [`Launcher`] seam: how a shard process comes to exist (plan §3.4).
//!
//! The orchestrator never spawns a process or talks to a daemon directly; it
//! asks a `Launcher`. The impls:
//!
//! * [`local::LocalLauncher`] — a real in-process [`space_server::shard::Shard`]
//!   with a [`space_server::net::ShardNet`] per shard, in the caller's tokio
//!   runtime, started through the very function `space-server --orchestrator`
//!   runs. The sandbox-testable impl: no process and no Docker daemon is
//!   required, and it is what makes the whole lifecycle verifiable end to end.
//! * [`process::ProcessLauncher`] — one `space-server` process per shard: the
//!   executable lying next to the orchestrator's own, on a host port of the
//!   configured range, logging to a file of its own.
//! * [`docker::DockerLauncher`] — one container per shard, created, started,
//!   stopped and removed through the Docker Engine's socket (Linux; elsewhere
//!   it says so and refuses to start).
//!
//! # A launch has no shard id
//!
//! Every shard starts the same way, launched or started by hand: **without a
//! region and without an id**, and is given both when it says hello
//! (`docs/plan-sharding.md` §2). So a launcher is not told which shard it
//! starts — it is told to start *a* shard ([`SpawnSpec`]) — and what it
//! supervises is named by the orchestrator's [`LaunchId`], not by a shard id.
//! The orchestrator learns which shard a launch became from the
//! [`ShardHandle`]: the id itself when the launcher knows it (the in-process
//! one reads the welcome), otherwise the address the shard listens on, which
//! is the address its hello advertises.
//!
//! The trait is deliberately small and *asynchronous*. The orchestrator's loop
//! runs each `spawn` in a task of its own — a launcher may only return once
//! its shard has registered, and the registration is a `ShardHello` that only
//! that loop can answer.

use space_protocol::shard::ShardId;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

pub mod docker;
pub mod local;
pub mod process;

/// The orchestrator's name for one launch: a shard it asked a launcher to
/// start. Counted up from 1 by the orchestrator and never reused while it
/// runs; it means nothing to anybody else (it is not a shard id and never
/// goes on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaunchId(pub u64);

impl core::fmt::Display for LaunchId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "launch {}", self.0)
    }
}

/// Everything the launcher needs to bring one shard up.
///
/// No shard id and no region: the shard registers without either and the
/// orchestrator assigns both (see the module docs).
#[derive(Clone)]
pub struct SpawnSpec {
    /// The orchestrator's name for this launch.
    pub launch: LaunchId,
    /// The cluster id, forwarded so the shard's `ShardHello` matches.
    pub cluster: String,
    /// The orchestrator's control address the shard dials back to.
    pub control_addr: SocketAddr,
    /// The host UDP port this shard listens on, from the config's port range
    /// (0: any port — the in-process launcher binds an ephemeral one).
    pub host_port: u16,
    /// The cluster secret the shard presents in its hello (empty: none). A
    /// launcher hands it over **through the environment or in memory, never on
    /// a command line**, and never logs it.
    pub secret: String,
}

impl core::fmt::Debug for SpawnSpec {
    /// Everything but the secret.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpawnSpec")
            .field("launch", &self.launch)
            .field("cluster", &self.cluster)
            .field("control_addr", &self.control_addr)
            .field("host_port", &self.host_port)
            .field("secret", &!self.secret.is_empty())
            .finish()
    }
}

/// A shard process the launcher is supervising.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardHandle {
    /// The launch this is.
    pub launch: LaunchId,
    /// The id the shard registered as, when the launcher knows it (the
    /// in-process launcher reads the welcome). `None`: the orchestrator finds
    /// the shard by `endpoint`.
    pub shard: Option<ShardId>,
    /// Where the shard listens — the address its hello advertises.
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
    Unreachable(LaunchId),
    /// The launcher cannot answer this question.
    Unsupported(String),
    /// The launcher cannot run here (the Docker launcher without a socket it
    /// can use, a process launcher without its executable).
    Unavailable(String),
}

impl core::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LaunchError::Spawn(why) => write!(f, "spawn failed: {why}"),
            LaunchError::Unreachable(l) => write!(f, "the shard of {l} never became reachable"),
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
    /// Exited, stopped or unknown; the orchestrator gives its regions to a
    /// replacement (H15).
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
    /// Start a shard.
    ///
    /// Returns once the launcher has done its part: the in-process launcher
    /// when the shard has registered (it then knows the shard's id), the
    /// process launcher as soon as the process runs (the shard registers by
    /// itself a moment later, from [`ShardHandle::endpoint`]).
    ///
    /// # Errors
    ///
    /// [`LaunchError::Spawn`] if the shard cannot be started (or, in process,
    /// was refused for good or stopped while it was starting).
    fn spawn(
        &self,
        spec: SpawnSpec,
    ) -> impl std::future::Future<Output = Result<ShardHandle, LaunchError>> + Send;

    /// Stop what `launch` started — also while it is still starting. Stopping
    /// something that is not (or no longer) running is not an error.
    ///
    /// # Errors
    ///
    /// Any launcher-specific failure; the orchestrator logs and continues —
    /// a shard that will not die is a leak, not a correctness bug.
    fn kill(
        &self,
        launch: LaunchId,
    ) -> impl std::future::Future<Output = Result<(), LaunchError>> + Send;

    /// Is what `launch` started still alive? The liveness half of H15: the
    /// orchestrator's run loop asks this for everything it launched, twice a
    /// second.
    ///
    /// # Errors
    ///
    /// [`LaunchError::Unsupported`] for a launcher with no liveness probe.
    fn poll(
        &self,
        launch: LaunchId,
    ) -> impl std::future::Future<Output = Result<Poll, LaunchError>> + Send;

    /// The exit status of what `launch` started, once it has been seen to
    /// have exited and while the launcher still knows it (until
    /// [`Launcher::kill`] reaps it): `Some(0)` is a shard that stopped because
    /// it was told to (`ShardRetire`), anything else one that died. `None`
    /// when the launcher cannot tell (it still runs, it was never seen, or the
    /// launcher has no such notion — the default).
    fn exit_code(&self, launch: LaunchId) -> impl std::future::Future<Output = Option<i32>> + Send {
        let _ = launch;
        async { None }
    }

    /// Stop and remove **everything** this launcher started, within a
    /// bounded time: the orchestrator is shutting down (a signal). The
    /// default does nothing — a launcher whose shards end with the process
    /// (in-process ones) or are stopped when it is dropped needs no more.
    fn shutdown(&self) -> impl std::future::Future<Output = ()> + Send {
        async {}
    }
}

/// The address a shard on this host dials to reach `control`.
///
/// A launched shard runs on the orchestrator's own host, so a wildcard bind
/// address (`0.0.0.0:7900`, which cannot be dialled — on Windows not at all)
/// means "loopback, that port".
#[must_use]
pub fn dialable(control: SocketAddr) -> SocketAddr {
    if !control.ip().is_unspecified() {
        return control;
    }
    let loopback = match control {
        SocketAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
        SocketAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
    };
    SocketAddr::new(loopback, control.port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wildcard_control_address_is_dialled_on_loopback() {
        let wildcard: SocketAddr = "0.0.0.0:7900".parse().unwrap();
        assert_eq!(dialable(wildcard), "127.0.0.1:7900".parse().unwrap());
        let wildcard6: SocketAddr = "[::]:7900".parse().unwrap();
        assert_eq!(dialable(wildcard6), "[::1]:7900".parse().unwrap());
        let concrete: SocketAddr = "10.0.0.2:7900".parse().unwrap();
        assert_eq!(dialable(concrete), concrete, "a real address is kept");
    }

    #[test]
    fn a_spawn_spec_never_prints_the_secret() {
        let spec = SpawnSpec {
            launch: LaunchId(3),
            cluster: "c".into(),
            control_addr: "127.0.0.1:7900".parse().unwrap(),
            host_port: 7_870,
            secret: "do-not-print-this".into(),
        };
        let printed = format!("{spec:?}");
        assert!(!printed.contains("do-not-print"), "{printed}");
        assert!(printed.contains("launch"), "{printed}");
    }
}
