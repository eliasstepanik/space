//! Shared simulation for `space`: ship flight and rigid-body physics.
//!
//! One crate, two callers: the shard server runs it as the **authoritative**
//! simulation, and a client runs the *same* code to **predict** between
//! snapshots ([`prediction`]). Both paths step the same functions with the
//! same arguments, so a client's prediction and the server's authority agree
//! to the last bit — which is what makes reconciliation a small correction
//! instead of a teleport.
//!
//! # Physics
//!
//! Every simulated entity is a rigid body ([`rigid`], on `rapier3d-f64`):
//! ships, crates, balls, doors, chains. What a body is — dynamic or kinematic,
//! mass, damping, material, collision shape (primitives, convex hulls,
//! compounds, triangle meshes from a [`shape::ShapeCatalog`]) — is its
//! [`Body`]; its motion is still its [`ShipState`], so motion replicates and
//! hands off as before. Springs ([`spring`]) are forces between two entities
//! or an entity and a world point; rigid joints ([`articulation`]) live
//! inside one entity or tie it to the static world. Gravity ([`gravity`]) is
//! a pure function of position and time: uniform zones and large bodies.
//!
//! # Large worlds
//!
//! The world spans the whole `i64` cell space of [`space_world::WorldPos`],
//! far beyond what one floating-point frame resolves. The physics therefore
//! runs in **bubbles**: local frames anchored at cell-aligned positions, one
//! per group of bodies that can touch ([`rigid`] has the rules). Bodies far
//! apart are in different bubbles and cost each other nothing; bubbles merge
//! before their bodies can touch, split when they drift apart and re-anchor
//! when their bodies move away, all as deterministic functions of the state.
//! Local coordinates are always derived from the [`space_world::WorldPos`]
//! and the anchor and written back every tick, so the [`space_world::WorldPos`]
//! stays the one truth, and precision is the same anywhere: free flight is
//! integrated on the [`space_world::WorldPos`] itself, contacts in a frame a
//! few kilometres across (`f64` resolution better than 1e-12 m), gravity and
//! springs from exact cell deltas.
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
//!   [`ShardAuthority`], [`components::Ghost`] and the replication registry
//! * [`params`] — [`ShipParams`]
//! * [`physics`] — the ship flight model ([`step_ship`]) and quaternion helpers
//! * [`body`] — [`Body`] (id 4) and [`Visual`] (id 5)
//! * [`shape`] — [`ShapeRef`] and the [`shape::ShapeCatalog`]
//! * [`spring`] — [`Springs`] (id 6), force-based springs and ropes
//! * [`articulation`] — [`Articulation`] (id 7) and [`PartStates`] (id 8)
//! * [`gravity`] — [`gravity::GravityField`]: zones and large bodies
//! * [`motion`] — [`motion::OrbitPath`]: fixed orbits as a function of scene time
//! * [`rigid`] — the rigid-body world: physics bubbles, statics, islands
//! * [`sim_world`] — [`SimWorld`], the ticked simulation world + state hash
//! * [`orbits`] — [`orbits::rail_pos`], planets on rails (never simulated)
//! * [`prediction`] — [`PredictedShip`] and [`prediction::PredictedPhysics`],
//!   client prediction + reconciliation
//!
//! Acceptance tests live in `tests/acceptance.rs` and `tests/physics.rs`.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod articulation;
pub mod body;
mod codec;
pub mod components;
pub mod gravity;
pub mod motion;
pub mod orbits;
pub mod params;
pub mod physics;
pub mod prediction;
pub mod rigid;
pub mod shape;
pub mod sim_world;
pub mod spring;
pub mod tick;
pub mod wire;

pub use articulation::{
    Articulation, JointDef, JointKind, JointTarget, Motor, PartDef, PartStates,
};
pub use body::{Body, BodyKind, CollisionGroups, MassSpec, PosResidual, Visual};
pub use components::{Ghost, Owner, ShardAuthority, ShipControls, ShipState};
pub use gravity::{GravityField, GravityZone, LargeBody, WorldPoint};
pub use orbits::rail_pos;
pub use params::ShipParams;
pub use physics::step_ship;
pub use prediction::{PredictedPhysics, PredictedShip};
pub use rigid::{
    BubbleId, BubbleInfo, ContactEvent, Island, IslandQuery, PhysicsError, StaticCollider,
};
pub use shape::{AssetId, LocalPose, ShapeCatalog, ShapeRef};
pub use sim_world::{BodySpawn, SimWorld};
pub use spring::{Spring, SpringMode, SpringTarget, Springs};
pub use tick::{dt, SimTick, TICK_HZ};
