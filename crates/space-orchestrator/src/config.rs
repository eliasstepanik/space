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

/// How often a shard reports its load, in ticks (1 s;
/// `space_server::load::REPORT_INTERVAL_TICKS` is the shard's side of it).
///
/// (A shard is declared dead — H15 — when its process exits or its control
/// link closes, not for a missed report.)
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

/// How long a shard that was told to stop ([`space_protocol::ShardRetire`])
/// is given to end by itself before its link is closed and its process or
/// container stopped, in ticks (5 s).
pub const RETIRE_GRACE_TICKS: u64 = 5 * TICK_HZ;

/// How long a shard the orchestrator started may take to register before the
/// launch is given up, in ticks (30 s).
///
/// Generous on purpose: a shard that has reached the orchestrator and is told
/// to wait its turn (one map change at a time) is still "starting".
pub const LAUNCH_REGISTER_TIMEOUT_TICKS: u64 = 30 * TICK_HZ;

/// How long a region that was asked for (a split, a carve) waits for the shard
/// that is to own it before the request is given up, in ticks (60 s).
pub const WANTED_TIMEOUT_TICKS: u64 = 60 * TICK_HZ;

/// How often the run loop asks the launcher whether the shards it started are
/// still alive, in ticks (twice a second).
pub const POLL_INTERVAL_TICKS: u64 = TICK_HZ / 2;

/// The pause after a launch that failed (the launcher could not start the
/// shard, or it exited or never registered), in ticks (1 s). It doubles with
/// every further failure in a row, up to [`LAUNCH_BACKOFF_MAX_TICKS`], so a
/// shard that cannot start is not started thirty times a second.
pub const LAUNCH_BACKOFF_TICKS: u64 = TICK_HZ;

/// The longest pause between two launches after failures, in ticks (30 s).
pub const LAUNCH_BACKOFF_MAX_TICKS: u64 = 30 * TICK_HZ;

/// Which launcher the binary uses.
///
/// `Local` runs shards inside the orchestrator's own process (tests, and a
/// single-container deployment); `Process` starts the `space-server` executable
/// lying next to the orchestrator's own, one process per shard; `Docker`
/// starts one container per shard through the Docker Engine's socket
/// ([`crate::launcher::docker`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LauncherKind {
    /// In-process launcher.
    #[default]
    Local,
    /// One `space-server` process per shard.
    Process,
    /// One container per shard, through the Docker socket.
    Docker,
}

/// A secret that never shows in a `Debug` print: the cluster secret and the
/// peer token are held in one. The type is [`space_net::Secret`], the same one
/// a shard holds its peer token in.
pub use space_net::Secret;

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
    /// Where the **player port** listens (`--player-bind`); `None` — the
    /// default — opens none. It is the one address players type: a client's
    /// hello there is answered with the shard that owns the scene's spawn
    /// point ([`crate::player`]). Not a control surface: it needs no token on
    /// any bind, and nothing said on it reaches the cluster's state.
    pub player_bind: Option<SocketAddr>,
    /// Path of the persisted shard-id counter (H9).
    pub state_path: PathBuf,
    /// First id the allocator hands out when no counter file exists yet.
    ///
    /// A config value (default 1) so a test can prove "restart resumes strictly
    /// above the last id" without a long run.
    pub first_shard_id: u64,
    /// The host UDP ports the `process` launcher hands its shards, one each:
    /// first and last, both inclusive (`--host-port-range`).
    ///
    /// **Keep it out of the operating system's dynamic port range** (49152 up
    /// on Windows, 32768 up on Linux). A shard dials the orchestrator and its
    /// neighbours from sockets the system gives a dynamic port on the wildcard
    /// address; a shard that later binds `127.0.0.1:<that port>` takes over
    /// their datagrams, and the first shard stops hearing the orchestrator.
    pub host_port_range: (u16, u16),
    /// Which launcher impl the binary picks.
    pub launcher: LauncherKind,
    /// The most regions the map may hold. A split that would add one more is
    /// refused, whoever asks for it — the autoscaler, an operator or the admin
    /// verb (`--max-regions`, default [`Self::DEFAULT_MAX_REGIONS`]).
    ///
    /// Every region is a shard to launch, a control link to serve and a row
    /// the control loop walks for each snapshot it publishes, so the map must
    /// not be able to grow without bound: an operator (or a looping script)
    /// asking for split after split would otherwise slow the very loop that
    /// answers hellos and meets epoch deadlines.
    pub max_regions: usize,
    /// The scene the cluster shows until an operator chooses another one
    /// (`--scene`). Only used when no scene has been persisted yet: a scene
    /// chosen at run time is saved next to the id counter and wins on restart
    /// (see [`crate::scene`]).
    pub initial_scene: space_protocol::Scene,
    /// How many shards the orchestrator keeps running by itself
    /// (`--min-shards`, default 0 = it starts none: every shard is started by
    /// hand, or by a split). While fewer are live or starting it launches
    /// one; with 4 on the demo area that is the root's shard plus three layout
    /// cells — the four quadrants that meet at the scene.
    pub min_shards: usize,
    /// The most shards the cluster may have (`--max-shards`, default
    /// [`Self::DEFAULT_MAX_SHARDS`]). A shard that registers when there are
    /// that many is refused for good, and nothing is launched or split beyond
    /// it.
    pub max_shards: usize,
    /// The `space-server` executable the `process` launcher starts
    /// (`--shard-exe`); `None` = the one lying next to the orchestrator's own
    /// executable.
    pub shard_exe: Option<PathBuf>,
    /// Where the `process` launcher writes one log file per shard, and the pid
    /// file of what it started (`--shard-log-dir`, default: the working
    /// directory).
    pub shard_log_dir: PathBuf,
    /// The cluster secret an operator configured (`--cluster-secret`, its file
    /// or environment forms), if any. With one, every `ShardHello` must carry
    /// it; without one, [`crate::boot::boot`] generates one for the shards the
    /// orchestrator starts itself, and a hello needs it only when the control
    /// port is not on loopback ([`Self::secret_required`]).
    pub cluster_secret: Option<Secret>,
    /// The cluster's tick origin, µs of UTC since the Unix epoch
    /// (`--tick-origin`; default
    /// [`SceneClock::DEFAULT_ORIGIN_US`](space_protocol::SceneClock::DEFAULT_ORIGIN_US)):
    /// the instant of tick 0, handed to every shard in its welcome so all of
    /// them count the same ticks, a replacement started later included.
    pub tick_origin_us: i64,
    /// The autoscaler's settings: when a loaded shard is relieved by a new
    /// one, when an idle one is merged away, and how long it is given to
    /// hand everything over (`--shard-capacity`, `--scale-out-load`,
    /// `--scale-out-after`, `--scale-in-load`, `--scale-in-after`,
    /// `--scale-cooldown`, `--drain-timeout`; [`crate::autoscale`]).
    pub scale: crate::autoscale::ScaleConfig,
    /// What the launcher starts shards from, in words, for the operator view
    /// (the `docker` launcher: the image and the network; empty otherwise).
    pub launcher_note: String,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            cluster: "space-local".into(),
            root_depth: Self::DEFAULT_ROOT_DEPTH,
            root_index: [0, 0, 0],
            bind: SocketAddr::from(([127, 0, 0, 1], 7_900)),
            control_addr: None,
            player_bind: None,
            state_path: PathBuf::from("space-orchestrator.id"),
            first_shard_id: 1,
            host_port_range: (7_870, 7_899),
            launcher: LauncherKind::default(),
            max_regions: Self::DEFAULT_MAX_REGIONS,
            initial_scene: space_protocol::Scene::DEFAULT,
            min_shards: 0,
            max_shards: Self::DEFAULT_MAX_SHARDS,
            tick_origin_us: space_protocol::SceneClock::DEFAULT_ORIGIN_US,
            shard_exe: None,
            shard_log_dir: PathBuf::from("."),
            cluster_secret: None,
            scale: crate::autoscale::ScaleConfig::default(),
            launcher_note: String::new(),
        }
    }
}

impl OrchestratorConfig {
    /// Default root depth: enough headroom above the split floor that the root
    /// can be split many times (see [`Self::root_depth`]).
    pub const DEFAULT_ROOT_DEPTH: u8 = 40;

    /// Default for [`Self::max_regions`]: far more shards than one host runs
    /// (the default host port range holds 30), and few enough that everything
    /// the control loop does per region stays negligible.
    pub const DEFAULT_MAX_REGIONS: usize = 64;

    /// Default for [`Self::max_shards`]: the root's shard and the seven layout
    /// cells around the scene ([`crate::topology::layout_cells`]).
    pub const DEFAULT_MAX_SHARDS: usize = 8;

    /// Shortest cluster secret accepted when the control port is not on
    /// loopback, in bytes (the web UI's rule for its token).
    pub const MIN_SECRET_LEN: usize = 16;

    /// Whether a `ShardHello` must carry the cluster secret: when one was
    /// configured, and always when the control port is reachable from another
    /// machine. On a loopback bind with no configured secret a hello needs
    /// none — the rule the web UI and the client harness follow.
    #[must_use]
    pub fn secret_required(&self) -> bool {
        self.cluster_secret.is_some() || !self.bind.ip().is_loopback()
    }

    /// Why this configuration must not start, if it must not: the control
    /// port is not on loopback and no shard could present a secret — none was
    /// configured (or it is too short to be one) and the orchestrator starts
    /// no shards it could hand a generated one to.
    ///
    /// # Errors
    ///
    /// The reason, for the operator.
    pub fn check_secret(&self) -> Result<(), String> {
        if self.bind.ip().is_loopback() {
            return Ok(());
        }
        match &self.cluster_secret {
            Some(secret) if secret.expose().len() < Self::MIN_SECRET_LEN => Err(format!(
                "the cluster secret is too short for the non-loopback control address {}: use \
                 at least {} characters",
                self.bind,
                Self::MIN_SECRET_LEN
            )),
            Some(_) => Ok(()),
            None if self.min_shards > 0 => Ok(()),
            None => Err(format!(
                "the control port will not listen on {}, which is not a loopback address, \
                 without a cluster secret: set one (SPACE_CLUSTER_SECRET, on the orchestrator \
                 and on every shard), or let the orchestrator start the shards itself \
                 (--min-shards 1 or more: it then generates a secret and hands it to them), or \
                 bind 127.0.0.1",
                self.bind
            )),
        }
    }

    /// The most shards the orchestrator can keep running by itself: the root's
    /// shard and one per layout cell ([`crate::topology::layout_cells`]). A
    /// shard that registers when every cell has an owner is refused for good —
    /// there is no region left to give it.
    #[must_use]
    pub fn most_shards_it_can_place(&self) -> usize {
        let root = self.root_node();
        // (A root of one cell has no cell to give away: none lies inside it.)
        1 + crate::topology::layout_cells(root)
            .into_iter()
            .filter(|cell| crate::topology::is_inside(*cell, root))
            .count()
    }

    /// Why `--min-shards` cannot be kept with this configuration, if it
    /// cannot: it asks for more shards than `--max-shards` or `--max-regions`
    /// allow, or than there are regions to give them.
    ///
    /// Refused at startup because the alternative never ends: the shard that
    /// is one too many is launched, refused for good, exits, and is launched
    /// again.
    ///
    /// # Errors
    ///
    /// The reason, for the operator.
    pub fn check_shard_counts(&self) -> Result<(), String> {
        let min = self.min_shards;
        if min > self.max_shards {
            return Err(format!(
                "--min-shards {min} is more than --max-shards {}",
                self.max_shards
            ));
        }
        let placeable = self.most_shards_it_can_place();
        if min > placeable {
            return Err(format!(
                "--min-shards {min} is more than the {placeable} shards the orchestrator can \
                 give a region by itself (the root's shard and one per layout cell around the \
                 scene): one more would be started, refused and started again, for ever"
            ));
        }
        if min > self.max_regions {
            return Err(format!(
                "--min-shards {min} is more than --max-regions {}: every shard needs a region \
                 of its own",
                self.max_regions
            ));
        }
        Ok(())
    }

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

    /// The control port follows the web UI's rule: no credentials on loopback,
    /// and no reachable bind without a secret somebody can present.
    ///
    /// Mutation this catches: a `0.0.0.0` control port that takes any hello
    /// (the state before: "the loopback default is the only mitigation").
    #[test]
    fn a_non_loopback_control_port_needs_a_secret_a_shard_can_present() {
        let open: SocketAddr = "0.0.0.0:7900".parse().unwrap();
        let mut cfg = OrchestratorConfig::default();
        assert!(!cfg.secret_required(), "loopback, nothing configured");
        assert_eq!(cfg.check_secret(), Ok(()));

        cfg.bind = open;
        assert!(cfg.secret_required());
        let err = cfg.check_secret().expect_err("nobody could present one");
        assert!(err.contains("without a cluster secret"), "{err}");

        // A generated secret is fine when the orchestrator starts the shards
        // (and hands it to them).
        cfg.min_shards = 4;
        assert_eq!(cfg.check_secret(), Ok(()));
        cfg.min_shards = 0;

        cfg.cluster_secret = Some(Secret::new("short"));
        let err = cfg.check_secret().expect_err("too short");
        assert!(err.contains("too short"), "{err}");
        cfg.cluster_secret = Some(Secret::new("a-long-enough-cluster-secret"));
        assert_eq!(cfg.check_secret(), Ok(()));

        // A configured secret is required on loopback too.
        cfg.bind = OrchestratorConfig::default().bind;
        assert!(cfg.secret_required());
        assert_eq!(cfg.check_secret(), Ok(()));
    }

    #[test]
    fn a_secret_is_never_printed() {
        let cfg = OrchestratorConfig {
            cluster_secret: Some(Secret::new("do-not-print-this-one")),
            ..OrchestratorConfig::default()
        };
        let printed = format!("{cfg:?}");
        assert!(!printed.contains("do-not-print"), "{printed}");
        assert!(printed.contains("<redacted>"), "{printed}");
        assert_eq!(format!("{:?}", Secret::default()), "Secret(<none>)");
    }

    /// `--min-shards` is a promise the orchestrator keeps by launching: one
    /// it cannot keep must not start, because the shard that is one too many
    /// would be launched, refused for good and launched again without end.
    ///
    /// Mutation this catches: checking only `min <= max` (the state before:
    /// `--min-shards 9 --max-shards 9` started, and the ninth shard had no
    /// layout cell to be given).
    #[test]
    fn min_shards_must_fit_the_limits_and_the_layout() {
        let with = |min_shards, max_shards, max_regions| OrchestratorConfig {
            min_shards,
            max_shards,
            max_regions,
            ..OrchestratorConfig::default()
        };
        assert_eq!(OrchestratorConfig::default().most_shards_it_can_place(), 8);
        for min in 0..=8 {
            assert_eq!(with(min, 8, 64).check_shard_counts(), Ok(()), "{min}");
        }
        let err = with(9, 9, 64).check_shard_counts().expect_err("no ninth");
        assert!(
            err.contains("--min-shards 9") && err.contains("8 shards"),
            "{err}"
        );
        let err = with(5, 4, 64).check_shard_counts().expect_err("above max");
        assert!(err.contains("more than --max-shards 4"), "{err}");
        let err = with(4, 8, 3).check_shard_counts().expect_err("no room");
        assert!(err.contains("more than --max-regions 3"), "{err}");
        // A root of one cell has nothing to give away: one shard at most.
        let one_cell = OrchestratorConfig {
            root_depth: 64,
            min_shards: 2,
            ..OrchestratorConfig::default()
        };
        assert_eq!(one_cell.most_shards_it_can_place(), 1);
        assert!(one_cell.check_shard_counts().is_err());
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
