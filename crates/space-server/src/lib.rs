//! Authoritative shard server for `space` (Phase 2, tasks 2.2–2.4).
//!
//! A shard owns one octree region of the world and runs the authoritative
//! [`SimWorld`] for it at [`TICK_HZ`]: it accepts client connections, spawns a
//! ship per player, applies tick-keyed inputs, and replicates the result back to
//! every client that can see it. One process, one region, one simulation.
//!
//! # Nothing on the wire is trusted as an identity
//!
//! A client never names an ECS entity. It names an [`EntityGuid`], and the shard
//! resolves that guid through its own [`GuidIndex`] / [`SimWorld::entity_of`]
//! before touching a row — the `entity_of` pattern. An input for a guid this
//! shard does not simulate is dropped, and an input for a guid another session
//! owns is dropped too ([`Shard::apply_input`]). That is what makes the input
//! path safe against a hostile client: there is no code path where a wire integer
//! is used as an index into the world.
//!
//! # The tick loop
//!
//! [`Shard::tick`] is one simulation step and everything derived from it, in a
//! fixed order:
//!
//! 1. drain the inputs received since the last tick into [`SimWorld::apply_inputs`]
//!    (which buffers them by their own tick — see its docs);
//! 2. [`SimWorld::step`];
//! 3. refresh the interest grid from the new positions;
//! 4. detect shard-boundary crossings and start a handoff ([`crate::handoff`]);
//! 5. build one frame per client ([`crate::replicate::build_frame`]) — every
//!    tick, even when nothing changed: the frame is also the client's clock.
//!
//! The loop itself lives in the front end ([`net`]): tick, then
//! [`net::ShardNet::flush`], which never waits on a queue, so neither a slow
//! client nor a dead neighbour can stop the simulation; and
//! [`net::supervise`] ends the process if the tick loop ever dies.
//!
//! # Seamless handoff
//!
//! When a player ship crosses out of this shard's region, ownership moves to the
//! shard that owns the destination **without the client noticing**: the two shards
//! run the three-phase offer/accept/commit exchange from [`space_protocol`], the
//! client is [`Redirect`]ed to the new shard and keeps receiving from the old one
//! until the activation tick (make-before-break), and the entity's guid is
//! unchanged so the client's own ship object survives the move. The full protocol
//! and the guarantees it provides are documented in [`crate::handoff`].
//!
//! # Modules
//!
//! * [`config`] — [`ShardConfig`] and the tuning constants
//! * [`alloc`] — monotonic, never-reused [`ShardId`] allocation
//! * [`replicate`] — interest grid, change-tick deltas, the frame budget
//! * [`handoff`] — the seamless handoff state machine
//! * [`shard`] — [`Shard`], the state the tick loop drives
//! * [`net`] — the quinn front end that turns frames into [`Shard`] calls
//!
//! Acceptance tests live in `tests/acceptance.rs`.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod alloc;
pub mod config;
pub mod epoch;
pub mod handoff;
pub mod net;
pub mod orchestrated;
pub mod replicate;
pub mod shard;

pub use alloc::ShardIdAllocator;
pub use config::ShardConfig;
pub use epoch::EpochParticipant;
pub use shard::{Shard, ShardStats};
