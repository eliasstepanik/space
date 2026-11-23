//! Wire protocol for `space` (Phase 1, task 1.3).
//!
//! One crate, four consumers: the client, a shard server, the orchestrator and
//! the MCP servers all speak the [`Message`]s defined here, so the wire format
//! is defined exactly once. It builds on `space-world` ([`space_world::NodeRef`]
//! — a shard region is an octree node) and `space-ecs` ([`space_ecs::ComponentId`]
//! — the component id inside an [`EntitySnapshot`]).
//!
//! # Codec
//!
//! [`bincode`] 2.x through its `serde` bridge, behind one function pair,
//! [`encode_frame`] / [`decode_frame`]. It is pure Rust like the rest of the
//! workspace (no C code), reuses the `serde` derive the messages already carry,
//! and is compact (varint integers). A hand-rolled codec would be a second
//! format to test and to keep bug-compatible across four consumers, and
//! bincode 2.0.1 is the released 2.x, so there is no alpha/beta churn. The
//! bincode `Config` is pinned here ([`codec`]), which makes the byte layout a
//! property of *this* crate rather than of a dependency default.
//!
//! A frame is `len:u32 (BE) | protocol:u32 (BE) | payload`, where `len` counts
//! the payload only. [`MAX_FRAME_SIZE`] is 1 MiB: larger frames are rejected
//! before the payload is looked at, and decoding runs under a bincode limit of
//! the same size so a forged container length cannot blow up an allocation.
//!
//! # Versioning
//!
//! [`PROTOCOL_VERSION`] travels twice: in every frame header and in
//! [`Hello::protocol`]. A header mismatch is [`CodecError::VersionMismatch`]
//! (the frame is dropped before it is decoded); a [`Hello`] carrying a different
//! version is answered with [`Reject`] / [`RejectReason::VersionMismatch`], so
//! the client learns *why* it was turned away.
//!
//! # Modules
//!
//! * [`guid`] — [`EntityGuid`], the shard-stable entity identity
//! * [`shard`] — [`ShardId`], [`ShardMap`], [`ShardRegion`], epoch handshake
//! * [`messages`] — the message types and the [`Message`] enum
//! * [`control`] — orchestrator load reports, directory and admin messages
//! * [`scene`] — the scene catalogue ([`Scene`], [`SceneId`]) and the
//!   revisioned [`SceneState`] the scene messages carry
//! * [`codec`] — [`encode_frame`], [`decode_frame`], [`CodecError`]
//! * [`snapshot`] — [`SnapshotBuilder`], the per-client byte budget
//!
//! Acceptance tests live in `tests/acceptance.rs`.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod codec;
pub mod control;
pub mod guid;
pub mod messages;
pub mod pos;
pub mod scene;
pub mod shard;
pub mod snapshot;

pub use codec::{
    check_hello, decode_frame, encode_frame, try_encode_frame, CodecError, FRAME_HEADER_LEN,
    MAX_FRAME_SIZE, PROTOCOL_VERSION,
};
pub use control::{AdminCommand, AdminReply, DirectoryQuery, DirectoryReply, LoadReport};
pub use guid::EntityGuid;
pub use messages::{
    Delta, EntitySnapshot, GhostUpdate, HandoffAbort, HandoffAccept, HandoffCommit, HandoffOffer,
    HandoffReason, Hello, InputTick, IslandKey, Message, NeighbourStatus, PlayerInput, Redirect,
    Reject, RejectReason, ShardHello, ShardWelcome, Snapshot, StaleEpoch, Welcome,
};
pub use pos::{decode_world_pos, encode_world_pos, WORLD_POS_LEN};
pub use scene::{Scene, SceneAck, SceneClock, SceneId, SceneState, UnknownScene};
pub use shard::{
    AbortReason, EpochAbort, EpochAck, EpochCommit, EpochPrepare, ShardId, ShardMap, ShardRegion,
};
pub use snapshot::{entity_snapshot, SnapshotBuilder, DEFAULT_BUDGET};
