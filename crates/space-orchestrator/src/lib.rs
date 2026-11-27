//! Cluster orchestrator for `space` (Phase 3, H9–H15).
//!
//! The orchestrator is the control plane that owns the *map*: it allocates shard
//! ids, decides when a region is too hot or too cold, publishes every ownership
//! change through the epoch protocol, and launches the shards that carry the
//! regions. It never simulates anything and carries no game traffic — a shard
//! learns the new map from the orchestrator, and a client learns it from the
//! shard it is already talking to (`Redirect`/`Welcome` carry the map). What
//! it does for a client is point the way: its **player port** ([`player`]) is
//! the one address players type, and answers a hello with the shard a new
//! player starts on (`JoinRedirect`), then closes.
//!
//! # The derived guarantees H9–H15
//!
//! Phase 2's H1–H8 say what a single handoff must never do. H9–H15 say what must
//! hold when **the map itself** changes — when a shard is created or destroyed
//! under the players. Each is a testable property; the mechanism column names
//! the module that provides it.
//!
//! | # | guarantee | mechanism |
//! |---|-----------|-----------|
//! | **H9** | Shard ids are allocated by the orchestrator, are strictly monotonic, and are never reused across a split, a merge, a crash or an orchestrator restart. | [`ids::PersistentShardIdAllocator`]: persist-before-handout over [`space_server::alloc::ShardIdAllocator`]. |
//! | **H10** | A map change is published through the epoch protocol only: prepare → ack → commit. No shard observes a map that is not some committed epoch. | [`epoch::EpochPublisher`]: one outstanding prepare, sent to and acked by every shard of the map and every new owner, activation from the max observer tick, abort on deadline leaves the published map byte-identical. |
//! | **H11** | A carve (a split is the carve of one child octant) or a merge never leaves a position owned by two shards or by none. | [`topology`]: additive carves of one node inside an existing region — a child octant or a single cell many levels down — merges that remove exactly that region so ownership falls back to the region around it, [`topology::validate`] on every produced map. |
//! | **H12** | An epoch change cannot strand a handoff. | One epoch change at a time ([`epoch`]), never two bumps inside the handoff timeout for the same region pair. |
//! | **H13** | A split or merge is invisible to a connected client. | Phase 2's make-before-break, unchanged: the new map reaches the client through `Redirect`. A merge only ever removes a shard whose own load report says it has no players, and kills it only once the merge has committed ([`orchestrator`]). |
//! | **H14** | The autoscaler is deterministic and never flaps. | [`autoscale`]: pure function of `(reports, tick, prior state)`, dwell in ticks, hysteresis band, per-shard cooldown, one action in flight. |
//! | **H15** | After a shard dies its regions are owned by a live shard again, and nothing else changes until they are; a dead shard's id is never reused. What the dead shard simulated is gone: the scene's bodies are built once, afresh, and its players join again. | A shard is dead when its launcher says its process is ([`launcher::Launcher::poll`], asked twice a second) or its control link has closed ([`orchestrator::Orchestrator::declare_dead`]). The orchestrator keeps `--min-shards` shards running, so a replacement is launched; the shard that registers next is given the dead one's regions under a **new id** ([`ids`]) through one epoch change that every *live* shard acks and the dead one is not asked for ([`epoch::EpochPublisher::propose_without`]). Until it commits no other change is proposed. When it has, the scene's revision is raised once ([`scene::SceneStore::raise_above`]), so every shard drops the scene's bodies and the spawner builds them all again. A shard that only lost its orchestrator stops itself (`space_server::orchestrated`), so no second owner survives. |
//!
//! # Shape of the code
//!
//! [`ids`], [`topology`], [`epoch`] and [`autoscale`] are **pure and
//! synchronous** — like `Shard::tick`, they are a function of their inputs and
//! are testable without a single socket. The async layer (`control.rs`'s QUIC
//! front end and `orchestrator.rs`'s event loop) only moves messages.
//!
//! # Modules
//!
//! * [`config`] — [`config::OrchestratorConfig`] and the tuning constants
//! * [`ids`] — the persistent, monotonic, never-reused
//!   [`ShardId`](space_protocol::ShardId) allocator
//! * [`topology`] — carve/split/merge geometry over `space_world::octree`
//! * [`epoch`] — the two-phase epoch publication state machine
//! * [`autoscale`] — the load metric and the split/merge decision
//! * [`launcher`] — the [`launcher::Launcher`] seam, [`launcher::local::LocalLauncher`]
//!   (shards inside this process) and [`launcher::process::ProcessLauncher`]
//!   (one `space-server` process per shard); [`launcher::docker`] runs each shard in a container
//! * [`scene`] — the cluster's scene, its revision and its persistence (not an
//!   epoch: see below)
//! * [`control`] — the control-plane message handlers and the control port's
//!   QUIC front end (shards only)
//! * [`player`] — the player port: the one address players type, answered
//!   with the shard that owns the scene's spawn point
//! * [`orchestrator`] — [`orchestrator::Orchestrator`], the tie-in point
//! * [`events`] — the bounded recent-events log (pure)
//! * [`snapshot`] — [`snapshot::Snapshot`], everything the orchestrator knows
//!   as plain data (pure)
//! * [`ops`] — the operator's way into the run loop: read a snapshot, offer a
//!   command
//! * [`webui`] — the built-in web UI: an HTTP/1.1 server on tokio that renders
//!   the snapshot as HTML and JSON and turns form posts into operator commands
//!
//! # The operator view and H9–H15
//!
//! The web UI is a *reader* of [`snapshot::Snapshot`]s and a *sender* of
//! [`ops::OperatorCommand`]s; it holds no reference to the orchestrator. A
//! split or merge asked for by an operator is carried out by
//! [`orchestrator::Orchestrator::handle_operator`] on the run loop, through the
//! same planning, epoch proposal and launcher steps as an autoscaler decision,
//! so H9–H13 and H15 hold for it unchanged. The one new input to H14 is the
//! operator's pause switch: the autoscaler is still a pure function of its
//! reports — while paused it is simply not fed, and its dwell timers are
//! cleared so nothing fires on resume without a fresh dwell.
//!
//! # The scene and H9–H15
//!
//! The scene (which world clients draw around the ships) is cluster state the
//! orchestrator owns and an operator can change at run time. It is **not** part
//! of the map and does not travel through the epoch protocol: a change is
//! persisted, told to every connected shard at once (`SceneSet`), confirmed by
//! each (`SceneAck`) and re-offered once a second to a shard that has not
//! confirmed. It neither waits for an epoch in flight nor delays one, and it
//! touches no id, no region and no publisher state — so H9–H15 are exactly
//! what they were. See [`scene`] and [`space_protocol::scene`].
//!
//! # What this crate is not yet
//!
//! Load reports are
//! received and acted on, but no shard sends one yet, so the autoscaler only
//! acts in tests and a merge needs its report said on the shard's behalf
//! (`docs/plan-sharding.md` §3).

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod autoscale;
pub mod boot;
pub mod config;
pub mod control;
pub mod epoch;
pub mod events;
pub mod ids;
pub mod launcher;
pub mod ops;
pub mod orchestrator;
pub mod player;
pub mod scene;
pub mod snapshot;
pub mod topology;
pub mod webui;

pub use autoscale::{load_metric, Autoscaler, ScaleAction};
pub use config::OrchestratorConfig;
pub use control::ControlPlane;
pub use epoch::EpochPublisher;
pub use ids::PersistentShardIdAllocator;
pub use orchestrator::Orchestrator;
