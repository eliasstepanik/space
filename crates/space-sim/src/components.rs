//! The ship components and the replication registry.
//!
//! Four components, with the ids fixed here (ids are part of the wire format:
//! an [`space_protocol::EntitySnapshot`] carries them, so they are appended,
//! never renumbered):
//!
//! | id | component        | replicated |
//! |----|------------------|------------|
//! | 0  | [`ShipState`]    | yes        |
//! | 1  | [`ShipControls`] | yes        |
//! | 2  | [`Owner`]        | yes        |
//! | 3  | [`ShardAuthority`]| no        |
//!
//! [`ShipState`] is the only component the physics writes; [`ShipControls`]
//! holds the input a tick's step consumes, [`Owner`] maps the ship back to the
//! player session that drives it, and [`ShardAuthority`] marks the entities this
//! shard simulates authoritatively (a ghost replica carries a foreign
//! [`space_protocol::ShardId`] and is stepped predictively, if at all).

use space_ecs::ComponentDef;
use space_world::WorldPos;

/// The simulated state of a ship: where it is, how fast it moves, how it is
/// oriented and how fast it spins.
///
/// All fields are `f32` except the position, which is a
/// [`space_world::WorldPos`] so a ship far from the origin keeps millimetre
/// resolution. `quat` is a unit quaternion in `(w, x, y, z)` order — `w` first,
/// matching the convention used by [`crate::physics::quat_rotate`] and by most
/// graphics stacks that consume it.
///
/// The struct is `Copy` on purpose: [`crate::physics::step_ship`] and
/// [`crate::prediction`] both work on owned copies, which is what keeps the
/// prediction replay free of aliasing with the authoritative state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShipState {
    /// Position in the world.
    pub pos: WorldPos,
    /// Linear velocity, metres/second, world frame.
    pub vel: [f32; 3],
    /// Orientation as a unit quaternion, `(w, x, y, z)`.
    pub quat: [f32; 4],
    /// Angular velocity, radians/second, **body** frame.
    pub ang_vel: [f32; 3],
}

impl ShipState {
    /// A ship at `pos`, at rest, identity-oriented.
    #[must_use]
    pub const fn at_rest(pos: WorldPos) -> Self {
        Self {
            pos,
            vel: [0.0; 3],
            quat: [1.0, 0.0, 0.0, 0.0],
            ang_vel: [0.0; 3],
        }
    }

    /// The identity state at the world origin.
    pub const IDENTITY: Self = Self::at_rest(WorldPos::ZERO);
}

impl Default for ShipState {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// The control input currently applied to a ship.
///
/// Mirrors the payload of an [`space_protocol::PlayerInput`] minus the routing
/// fields: the server writes the newest input for the entity into this
/// component, and [`crate::SimWorld::step`] feeds it to
/// [`crate::step_ship`]. Keeping it a separate component (rather than a field of
/// [`ShipState`]) is what lets a client replicate the *state* without also
/// replicating the input stream, and lets a ship exist with no controller at all
/// (a derelict coasts on its [`ShipState::vel`] alone).
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShipControls {
    /// Thrust in entity-local newtons, `(x, y, z)`.
    pub thrust: [f32; 3],
    /// Torque in entity-local newton-metres, `(x, y, z)`.
    pub torque: [f32; 3],
}

/// The player session that drives a ship.
///
/// A session id is a `u64` handed out by the shard in
/// [`space_protocol::Welcome::session`]. Storing it on the entity is what lets
/// the server reject an [`space_protocol::PlayerInput`] whose `entity` guid is
/// driven by a different session than the connection that sent it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Owner(pub u64);

/// Marker: this shard simulates the entity authoritatively.
///
/// It is a *marker* in the sense that its presence is the information — the
/// payload is the owning [`space_protocol::ShardId`], so a replica that arrived
/// through a handoff or a ghost update carries a foreign id and is skipped by
/// the local physics pass. Not replicated: ownership is a property of *this*
/// process's view, and the shard map / handoff messages are the authority on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShardAuthority(pub u32);

impl ComponentDef for ShipState {
    const ID: u16 = 0;
    type Data = Self;
}

impl ComponentDef for ShipControls {
    const ID: u16 = 1;
    type Data = Self;
}

impl ComponentDef for Owner {
    const ID: u16 = 2;
    type Data = Self;
}

impl ComponentDef for ShardAuthority {
    const ID: u16 = 3;
    type Data = Self;
}

mod registry {
    use super::{Owner, ShardAuthority, ShipControls, ShipState};

    space_ecs::register_components!(
        (0, ShipState, replicated = true),
        (1, ShipControls, replicated = true),
        (2, Owner, replicated = true),
        (3, ShardAuthority),
    );

    /// The macro's `const` is module-private; re-expose it as a `pub(crate)`
    /// item in the same module so the crate can publish it as a `static`.
    ///
    /// It has to stay a `const` (not a `static`) so the `static` below can be
    /// initialised from it, which is what makes the registry a `'static`
    /// reference `World::with_registry` can keep.
    #[allow(
        clippy::large_const_arrays,
        reason = "the registry must be a const so it can be re-exported as the crate's static"
    )]
    pub(crate) const REGISTRY: [::core::option::Option<space_ecs::ComponentInfo>;
        space_ecs::MAX_COMPONENTS] = COMPONENT_REGISTRY;
}

/// The replication registry built by
/// [`register_components!`](space_ecs::register_components) for this crate's
/// components, indexed by [`space_ecs::ComponentId`].
///
/// It is a `static` (not just a `const`) because
/// [`World::with_registry`](space_ecs::World::with_registry) keeps the
/// reference; [`SimWorld::new`](crate::SimWorld::new) passes it in, so query
/// panics report component names.
pub static COMPONENT_REGISTRY: [::core::option::Option<space_ecs::ComponentInfo>;
    space_ecs::MAX_COMPONENTS] = registry::REGISTRY;

/// The component ids this crate registers, in id order.
pub const IDS: [u16; 4] = [
    <ShipState as ComponentDef>::ID,
    <ShipControls as ComponentDef>::ID,
    <Owner as ComponentDef>::ID,
    <ShardAuthority as ComponentDef>::ID,
];

#[cfg(test)]
mod tests {
    use super::{Owner, ShardAuthority, ShipControls, ShipState, COMPONENT_REGISTRY, IDS};
    use space_ecs::ComponentDef;

    #[test]
    fn registry_matches_the_component_ids() {
        assert_eq!(IDS, [0, 1, 2, 3]);
        assert_eq!(ShipState::ID, 0);
        assert_eq!(ShipControls::ID, 1);
        assert_eq!(Owner::ID, 2);
        assert_eq!(ShardAuthority::ID, 3);

        assert_eq!(COMPONENT_REGISTRY[0].unwrap().name, "ShipState");
        assert!(COMPONENT_REGISTRY[0].unwrap().replicated);
        assert!(COMPONENT_REGISTRY[1].unwrap().replicated);
        assert!(COMPONENT_REGISTRY[2].unwrap().replicated);
        // The authority marker is local-only.
        assert!(!COMPONENT_REGISTRY[3].unwrap().replicated);
        for id in IDS {
            assert!(COMPONENT_REGISTRY[usize::from(id)].is_some());
        }
        assert!(COMPONENT_REGISTRY[4].is_none());
    }

    #[test]
    fn identity_state_is_a_resting_ship() {
        let s = ShipState::IDENTITY;
        assert_eq!(s.vel, [0.0; 3]);
        assert_eq!(s.ang_vel, [0.0; 3]);
        assert_eq!(s.quat, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(s, ShipState::default());
    }
}
