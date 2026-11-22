//! [`OrchestratorConfig`]: the declared topology and the tuning knobs.
//!
//! Everything here is a *declaration*, not a runtime value: the orchestrator
//! reconciles the live cluster against it at startup (H15) and every field is
//! pure data so a test can build one without touching the filesystem or a
//! socket.

use std::net::SocketAddr;
use std::path::PathBuf;

/// Simulation rate the whole cluster runs at, in ticks per second.
///
/// Every dwell timer and deadline in this crate is expressed in **ticks**, never
/// in wall-clock time, so a test can drive the orchestrator with `step(n)`
/// instead of sleeping (plan §3.6: H14's determinism depends on this).
pub const TICK_HZ: u64 = 30;

/// How long an epoch change leads its activation tick, in ticks (2 s).
///
/// `activation_tick` is `max(observer ticks) + EPOCH_LEAD_TICKS`: the shards run
/// independent loops and drift, which is exactly why Phase 2 split
/// `activation_tick` from `local_activation_tick`.
pub const EPOCH_LEAD_TICKS: u64 = 60;

/// How long to wait for every required `EpochAck` before aborting, in ticks (3 s).
pub const EPOCH_ACK_TIMEOUT_TICKS: u64 = 90;

/// Minimum ticks between two epoch commits (3 s).
///
/// One of the three flapping dampers (plan R6c); the others are
/// [`crate::autoscale::SCALE_COOLDOWN_TICKS`] and the hysteresis band.
pub const MIN_EPOCH_INTERVAL_TICKS: u64 = 90;

/// How often a shard reports its load, in ticks (1 s).
///
/// A shard that misses these beats is declared dead (H15).
pub const LOAD_REPORT_INTERVAL_TICKS: u64 = 30;

/// How long a shard's last load report is believed, in ticks (3 s: three missed
/// beats).
///
/// The autoscaler is fed the latest report per shard on every tick. Without an
/// expiry a shard that died or was removed kept "reporting" its last load for
/// ever, and was split or merged on it again and again. Three intervals of
/// silence is long enough to ride out a late report and short enough that a
/// stale one never completes a dwell on its own.
pub const REPORT_EXPIRY_TICKS: u64 = 3 * LOAD_REPORT_INTERVAL_TICKS;

/// Which launcher the binary uses.
///
/// `Local` runs shards in-process / as subprocesses (sandbox-testable);
/// `Docker` is the bollard-backed impl (plan §3.4, a later slice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LauncherKind {
    /// In-process / subprocess launcher.
    #[default]
    Local,
    /// Docker launcher (feature-gated, later slice).
    Docker,
}

/// The orchestrator's configuration.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Cluster id; a `ShardHello` naming another cluster is refused.
    pub cluster: String,
    /// Depth the root region is declared at.
    ///
    /// **This must leave split headroom.** A depth-64 region is one cell and
    /// `octree::children` returns `None` there, so a depth-64 root can never
    /// split — Phase 2's test convention (`DEPTH = 64`) is therefore *not*
    /// inherited here. The default 40 is ≈ 2.7 × 10¹⁰ m per edge, far above the
    /// 1024 km split floor (plan §3.2).
    pub root_depth: u8,
    /// Per-axis index of the root region's origin node.
    pub root_index: [i64; 3],
    /// Where the control plane listens.
    pub bind: SocketAddr,
    /// The address shards dial to reach the control port, when it differs from
    /// [`Self::bind`] (`None` = same as `bind`). A container-side orchestrator
    /// typically binds `0.0.0.0:7900`, which is not dialable; shards need the
    /// service name / host address instead. See [`Self::control_addr`].
    pub control_addr: Option<SocketAddr>,
    /// Path of the persisted shard-id counter (H9).
    pub state_path: PathBuf,
    /// First id the allocator hands out when no counter file exists yet.
    ///
    /// A config value (default 1) so a test can prove "restart resumes strictly
    /// above the last id" without a long run.
    pub first_shard_id: u64,
    /// Inclusive lower bound of the host UDP port range the launcher publishes.
    pub host_port_range: (u16, u16),
    /// Which launcher impl the binary picks.
    pub launcher: LauncherKind,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            cluster: "space-local".into(),
            root_depth: Self::DEFAULT_ROOT_DEPTH,
            root_index: [0, 0, 0],
            bind: SocketAddr::from(([127, 0, 0, 1], 7_900)),
            control_addr: None,
            state_path: PathBuf::from("space-orchestrator.id"),
            first_shard_id: 1,
            host_port_range: (7_870, 7_899),
            launcher: LauncherKind::default(),
        }
    }
}

impl OrchestratorConfig {
    /// Default root depth: enough headroom above the split floor that the root
    /// can be split many times (see [`Self::root_depth`]).
    pub const DEFAULT_ROOT_DEPTH: u8 = 40;

    /// The address launched shards dial: `control_addr` if set, else `bind`.
    #[must_use]
    pub fn control_addr(&self) -> SocketAddr {
        self.control_addr.unwrap_or(self.bind)
    }

    /// The root region node.
    #[must_use]
    pub fn root_node(&self) -> space_world::NodeRef {
        space_world::NodeRef::new(self.root_depth, self.root_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_world::octree;

    #[test]
    fn control_addr_defaults_to_bind_and_can_be_overridden() {
        let mut cfg = OrchestratorConfig::default();
        assert_eq!(cfg.control_addr(), cfg.bind);
        let dial = SocketAddr::from(([10, 0, 0, 2], 7_900));
        cfg.bind = SocketAddr::from(([0, 0, 0, 0], 7_900));
        cfg.control_addr = Some(dial);
        assert_eq!(
            cfg.control_addr(),
            dial,
            "shards dial control_addr, not bind"
        );
    }

    #[test]
    fn the_default_root_has_split_headroom() {
        // The whole point of DEFAULT_ROOT_DEPTH: children() must exist and the
        // node must be splittable under topology's rules.
        let cfg = OrchestratorConfig::default();
        let root = cfg.root_node();
        assert_eq!(root.depth, OrchestratorConfig::DEFAULT_ROOT_DEPTH);
        assert!(
            octree::children(root).is_some(),
            "the root must be splittable"
        );
        assert!(
            crate::topology::can_split(root),
            "the default root must pass the split gate"
        );
    }

    #[test]
    fn a_depth_64_root_would_never_split() {
        // The Phase-2 test convention, stated as a regression guard: depth 64 is
        // one cell, `children` is None, so it can never be split.
        let deep = space_world::NodeRef::new(64, [0, 0, 0]);
        assert!(octree::children(deep).is_none());
        assert!(!crate::topology::can_split(deep));
    }
}
