//! Shared ship-flight simulation for `space` (Phase 1, task 1.4).
//!
//! One crate, two callers: the shard server runs it as the **authoritative**
//! simulation, and a client runs the *same* code to **predict** between
//! snapshots ([`prediction`]). Both paths call [`step_ship`] with the same
//! arguments, so a client's prediction and the server's authority agree to the
//! last bit — which is what makes reconciliation a small correction instead of a
//! teleport.
//!
//! # Determinism policy
//!
//! The contract is **bit-identical state on the same platform, from the same
//! inputs**. Cross-platform bit-identity is *not* promised.
//!
//! What that buys, and how it is enforced here:
//!
//! * **Fixed-step integration only.** The simulation advances in whole
//!   [`SimTick`]s of [`dt()`] = 1/30 s ([`TICK_HZ`]). There is no variable
//!   timestep, no accumulator and no wall-clock read anywhere in the crate: the
//!   tick counter is the only notion of time. A run is therefore a pure function
//!   of its initial state and its input script.
//! * **No order that depends on a hash map.** Nothing in this crate iterates a
//!   [`std::collections::HashMap`] where the order is observable. Inputs are
//!   sorted by entity inside [`SimWorld::apply_inputs`] before they are applied,
//!   and [`SimWorld::state_hash`] walks the entity list **sorted by**
//!   [`space_ecs::Entity`], so the hash is a function of the state and not of
//!   any container's iteration order.
//! * **Float-op policy.** All ship arithmetic is `f32`, in a fixed, documented
//!   order (see [`physics`]): the same sequence of machine operations, with no
//!   runtime-only constants, no `f64`↔`f32` surprises beyond the documented
//!   widening inside [`space_world::WorldPos`], and no vectorised reduction
//!   whose order a compiler could change. Every transcendental used by the
//!   physics is a plain `sqrt`/`sin`/`cos` on the same libm.
//!
//!   **Same-platform bit-identity is the contract; cross-platform bit-identity
//!   is NOT promised.** Different compilers, different libm implementations,
//!   different `-C target-cpu` settings (FMA contraction!) or a different
//!   architecture can produce different low bits, because IEEE-754 only
//!   requires each individual operation to be correctly rounded — the *sequence*
//!   of operations is fixed here, but the rounding of a transcendental or a
//!   contracted `a*b+c` is a property of the toolchain, not of this crate. The
//!   [`space_world`] generators are the exception: they hash with integers only
//!   and *are* bit-identical everywhere. Anything that has to survive a
//!   client/server platform mismatch is reconciled numerically (see
//!   [`prediction::PredictedShip::reconcile`]) rather than trusted to be equal.
//!
//! # Modules
//!
//! * [`tick`] — [`SimTick`], [`TICK_HZ`], [`dt()`]
//! * [`components`] — [`ShipState`], [`ShipControls`], [`Owner`],
//!   [`ShardAuthority`] and the replication registry
//! * [`params`] — [`ShipParams`]
//! * [`physics`] — [`step_ship`] and the quaternion helpers
//! * [`sim_world`] — [`SimWorld`], the ticked simulation world + state hash
//! * [`orbits`] — [`orbits::rail_pos`], planets on rails (never simulated)
//! * [`prediction`] — [`PredictedShip`], client prediction + reconciliation
//!
//! Acceptance tests live in `tests/acceptance.rs`.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod components;
pub mod orbits;
pub mod params;
pub mod physics;
pub mod prediction;
pub mod sim_world;
pub mod tick;
pub mod wire;

pub use components::{Owner, ShardAuthority, ShipControls, ShipState};
pub use orbits::rail_pos;
pub use params::ShipParams;
pub use physics::step_ship;
pub use prediction::PredictedShip;
pub use sim_world::SimWorld;
pub use tick::{dt, SimTick, TICK_HZ};
