//! Cluster orchestrator for `space` (Phase 3, H9–H15).
//!
//! The orchestrator is the control plane that owns the *map*: it allocates shard
//! ids, decides when a region is too hot or too cold, publishes every ownership
//! change through the epoch protocol, and launches the shards that carry the
//! regions. It never simulates anything and never reaches a client — a shard
//! learns the new map from the orchestrator, and a client learns it from the
//! shard it is already talking to (`Redirect`/`Welcome` carry the map).
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
//! | **H10** | A map change is published through the epoch protocol only: prepare → ack → commit. No shard observes a map that is not some committed epoch. | [`epoch::EpochPublisher`]: one outstanding prepare, activation from the max observer tick, abort on deadline leaves the published map byte-identical. |
//! | **H11** | A split or merge never leaves a position owned by two shards or by none. | [`topology`]: additive one-octant splits, exact-inverse merges, [`topology::validate`] on every produced map. |
//! | **H12** | An epoch change cannot strand a handoff. | One epoch change at a time ([`epoch`]), never two bumps inside the handoff timeout for the same region pair. |
//! | **H13** | A split or merge is invisible to a connected client. | Phase 2's make-before-break, unchanged: the new map reaches the client through `Redirect`. A merge only ever removes a shard whose own load report says it has no players, and kills it only once the merge has committed ([`orchestrator`]). |
//! | **H14** | The autoscaler is deterministic and never flaps. | [`autoscale`]: pure function of `(reports, tick, prior state)`, dwell in ticks, hysteresis band, per-shard cooldown, one action in flight. |
//! | **H15** | The world converges back to its declared topology after a shard dies; a dead shard's id is never reused. | `launcher::Launcher::poll` + reassignment through the epoch protocol (wired in a later slice). |
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
//! * [`topology`] — split/merge geometry over `space_world::octree`
//! * [`epoch`] — the two-phase epoch publication state machine
//! * [`autoscale`] — the load metric and the split/merge decision
//! * [`launcher`] — the [`launcher::Launcher`] seam and [`launcher::local::LocalLauncher`]
//!   (the in-process impl; `launcher::docker` is a later slice)
//! * [`scene`] — the cluster's scene, its revision and its persistence (not an
//!   epoch: see below)
//! * [`control`] — the control-plane message handlers (QUIC front end later)
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
//! The Docker launcher (`launcher/docker.rs`), the QUIC front end, the event
//! loop, `main.rs` and the H9–H15 acceptance suite are the following Phase-3
//! slices (plan §4 P3.8–P3.13). The [`launcher::local::LocalLauncher`] and the
//! shard-side epoch runtime are in place; everything else here is the foundation
//! they build on.

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
