//! The rigid-body components: [`Body`] (id 4) and [`Visual`] (id 5).
//!
//! A simulated entity's **motion** is always its [`crate::ShipState`] (pose,
//! velocity, orientation, spin), so replicating or handing off motion needs no
//! new message. What a physics body *is* — its kind, mass, damping, material
//! and collision shape — is [`Body`]; which model a client draws for it is
//! [`Visual`].
//!
//! # `Body` wire layout (version 1, big endian, floats as raw IEEE-754 bits)
//!
//! ```text
//! version         u8   = 1
//! kind            u8   0 dynamic | 1 kinematic
//! mass mode       u8   0 density (kg/m³) | 1 total mass (kg)
//! mass value      f32  density 1e-3..=1e5, mass 1e-3..=1e9
//! linear_damping  f32  0..=10   (1/s)
//! angular_damping f32  0..=10   (1/s)
//! gravity_scale   f32  -10..=10
//! friction        f32  0..=10
//! restitution     f32  0..=1
//! flags           u8   bit 0 CCD, bit 1 rotation locked; other bits must be 0
//! groups          u32 memberships, u32 filter
//! shape           ShapeRef (see below)
//! ```
//!
//! `ShapeRef` = tag `u8` then: `1` ball `radius f32`; `2` cuboid `half 3×f32`;
//! `3` capsule / `4` cylinder / `5` cone `half_height f32, radius f32`;
//! `6` hull `n u8 (4..=64), n × 3×f32`; `7` compound `n u8 (1..=16),
//! n × (pose 7×f32 [translation, quaternion w x y z], child shape)` — no
//! nesting, no assets, at most [`MAX_TOTAL_HULL_POINTS`] hull points in all;
//! `8` asset `id u32`. Bounds: see [`crate::shape`].
//!
//! Decoding is **total**: wrong version, unknown tag, a non-finite or
//! out-of-range number, a count over its cap, a degenerate hull, a reserved
//! flag bit or a trailing byte is an error, never a panic, and nothing is
//! allocated before its count has been checked. The largest valid encoding is
//! under 4 KiB.
//!
//! # `Visual` wire layout (version 1)
//!
//! `version u8 = 1, model u32` — 5 bytes. The id names a model in the
//! client's table (`space_scene::visual`); the *size* of what is drawn comes
//! from the body's shape.

use space_ecs::ComponentDef;
use space_protocol::CodecError;

use crate::codec::{invalid, Reader, Writer};
use crate::params::ShipParams;
use crate::shape::{decode_shape, encode_shape, LocalPose, ShapeRef};

/// Most hull points one shape (all compound children together) may carry.
pub const MAX_TOTAL_HULL_POINTS: usize = 256;

/// Wire version of [`Body`].
pub const BODY_WIRE_VERSION: u8 = 1;
/// Wire version of [`Visual`].
pub const VISUAL_WIRE_VERSION: u8 = 1;

/// How a body moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    /// Moved by forces, gravity and contacts.
    Dynamic,
    /// Moves with its own velocity ([`crate::ShipState::vel`] /
    /// `ang_vel`), unaffected by forces and contacts, and pushes dynamic
    /// bodies out of its way (a moving platform, a lift).
    Kinematic,
}

/// How a body's mass is given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MassSpec {
    /// Density in kg/m³; mass and inertia follow from the shape's volume.
    Density(f32),
    /// Total mass in kg, distributed over the shape like a uniform density.
    Mass(f32),
}

/// Which bodies a body collides with: two bodies touch only when each one's
/// `memberships` intersects the other's `filter` (rapier's interaction
/// groups).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionGroups {
    /// The groups the body is in.
    pub memberships: u32,
    /// The groups it collides with.
    pub filter: u32,
}

impl CollisionGroups {
    /// Everything collides with everything.
    pub const ALL: Self = Self {
        memberships: u32::MAX,
        filter: u32::MAX,
    };

    /// The group bit ships ([`Body::ship`]) are in, and nothing else is.
    pub const SHIP_BIT: u32 = 1 << 1;

    /// A ship's groups: a member of [`CollisionGroups::SHIP_BIT`] only,
    /// colliding with everything — other ships included — so a static or
    /// body whose `filter` leaves that bit out lets ships pass through and
    /// stops everything else.
    ///
    /// Ships are solid against each other: a shard gives every player who
    /// joins a distinct free spawn point (`space-server`'s spawn search, with
    /// [`crate::SimWorld::overlaps`]), so no two ships start inside each
    /// other. (Before that existed ships passed through each other, because
    /// every player was spawned at one point.)
    pub const SHIP: Self = Self {
        memberships: Self::SHIP_BIT,
        filter: u32::MAX,
    };

    /// Collides with everything except ships.
    pub const NOT_SHIPS: Self = Self {
        memberships: u32::MAX,
        filter: !Self::SHIP_BIT,
    };
}

impl Default for CollisionGroups {
    fn default() -> Self {
        Self::ALL
    }
}

/// A rigid body: kind, mass, damping, material and collision shape. The
/// motion lives in the entity's [`crate::ShipState`]. Replicated (id 4).
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    /// Dynamic or kinematic.
    pub kind: BodyKind,
    /// Mass, as a density or a total.
    pub mass: MassSpec,
    /// Linear damping, 1/s: `a -= linear_damping · v`. Ignored for ships
    /// (their drag is [`ShipParams::drag_lin`]).
    pub linear_damping: f32,
    /// Angular damping, 1/s. Ignored for ships.
    pub angular_damping: f32,
    /// Multiplier of the gravity at the body's position (zones and large
    /// bodies, see [`crate::gravity`]).
    pub gravity_scale: f32,
    /// Coulomb friction coefficient (combined with the other side's by the
    /// average rule).
    pub friction: f32,
    /// Restitution, 0 (no bounce) ..= 1 (elastic).
    pub restitution: f32,
    /// Continuous collision detection: a fast body does not tunnel through a
    /// thin wall in one tick.
    pub ccd: bool,
    /// Contacts and springs never turn the body; it turns only by its own
    /// angular velocity (and, for a ship, its pilot's torque). Ships use it so
    /// a bump never tumbles the camera.
    pub lock_rotation: bool,
    /// Collision filtering.
    pub groups: CollisionGroups,
    /// The collision shape, in the body frame.
    pub shape: ShapeRef,
}

impl Body {
    /// A dynamic body of `shape` with density 1000 kg/m³ (water), light
    /// damping, full gravity and an ordinary material.
    #[must_use]
    pub fn dynamic(shape: ShapeRef) -> Self {
        Self {
            kind: BodyKind::Dynamic,
            mass: MassSpec::Density(1000.0),
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            friction: 0.6,
            restitution: 0.1,
            ccd: false,
            lock_rotation: false,
            groups: CollisionGroups::ALL,
            shape,
        }
    }

    /// A kinematic body of `shape`.
    #[must_use]
    pub fn kinematic(shape: ShapeRef) -> Self {
        Self {
            kind: BodyKind::Kinematic,
            ..Self::dynamic(shape)
        }
    }

    /// The shape of a player's ship: the placeholder character a client draws
    /// for a player, as a capsule from 2 cm above its feet
    /// ([`crate::ShipState::pos`] is the pilot's eye, 1.7 m above the feet) to
    /// 0.1 m above the eye, 0.3 m in radius.
    #[must_use]
    pub fn ship_shape() -> ShapeRef {
        ShapeRef::Compound {
            parts: vec![(
                LocalPose::at([0.0, -0.79, 0.0]),
                ShapeRef::Capsule {
                    half_height: 0.59,
                    radius: 0.3,
                },
            )],
        }
    }

    /// The body [`crate::SimWorld::spawn_ship`] gives a ship flown with
    /// `params`: dynamic, `params.mass` kg, no damping of its own (the flight
    /// model's drag applies), full gravity, CCD on, rotation locked against
    /// contacts, and no friction of its own: a ship standing on a floor in
    /// gravity slides on half the floor's friction (the average rule), so its
    /// 6 m/s² of thrust still moves it (at 0.5 the demo area's floor held it
    /// to 0.6 m/s²), and its brake assist stops it.
    #[must_use]
    pub fn ship(params: &ShipParams) -> Self {
        Self {
            kind: BodyKind::Dynamic,
            mass: MassSpec::Mass(params.mass),
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            friction: 0.0,
            restitution: 0.0,
            ccd: true,
            lock_rotation: true,
            groups: CollisionGroups::SHIP,
            shape: Self::ship_shape(),
        }
    }

    /// Check every bound of the wire layout (the decoder is the validator).
    ///
    /// # Errors
    /// [`CodecError::InvalidPayload`] naming the first violated bound.
    pub fn validate(&self) -> Result<(), CodecError> {
        decode_body(&encode_body(self)).map(|_| ())
    }
}

/// Which model a client draws for an entity. Replicated (id 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Visual(pub u32);

/// Wire version of [`PosResidual`].
pub const RESIDUAL_WIRE_VERSION: u8 = 1;
/// Largest residual component accepted on decode, metres (a residual is at
/// most half the `f32` resolution of a `WorldPos` offset, 31 µm).
pub const MAX_RESIDUAL: f32 = 1.0e-3;

/// The part of a body's position finer than its [`crate::ShipState::pos`]
/// can hold, per part (part 0 first), metres. Replicated (id 10).
///
/// A [`space_world::WorldPos`] stores its offset inside the 1024 m cell as an
/// `f32`, which resolves 0.06 µm near a cell's centre but 61 µm at its edge.
/// Rounding the position into it every tick would move a body by up to half
/// that per tick, systematically at a constant velocity (3 cm over a 1e7 m
/// straight flight). So a body (not a ship, see below) carries the rounding
/// remainder here: its exact position is `pos + residual`, the physics
/// integrates that, and the `WorldPos` stays the rounded value everyone reads.
/// Because the residual is a component, a handoff carries it and a body
/// continues *exactly*.
///
/// Ships have none: a client predicts its ship with [`crate::step_ship`] from
/// the replicated [`crate::ShipState`] alone, so a ship in free flight is
/// stepped on its `WorldPos` exactly as that function steps it.
///
/// Wire layout (version 1): `version u8, count u8 (1..=16), count × 3×f32`,
/// each value finite with `|v| <= MAX_RESIDUAL`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PosResidual(pub Vec<[f32; 3]>);

impl ComponentDef for PosResidual {
    const ID: u16 = 10;
    type Data = Self;
}

/// Encode a [`PosResidual`].
#[must_use]
pub fn encode_residual(r: &PosResidual) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(RESIDUAL_WIRE_VERSION);
    w.u8(u8::try_from(r.0.len()).unwrap_or(u8::MAX));
    for v in &r.0 {
        w.f32s(v);
    }
    w.into_bytes()
}

/// Decode a [`PosResidual`].
///
/// # Errors
/// [`CodecError::InvalidPayload`] for a wrong version, a count outside
/// `1..=16`, a value out of range or a wrong length.
pub fn decode_residual(bytes: &[u8]) -> Result<PosResidual, CodecError> {
    let mut r = Reader::new(bytes, "PosResidual");
    r.version(RESIDUAL_WIRE_VERSION)?;
    let n = r.count_u8("residuals", 16)?;
    if n == 0 {
        return Err(invalid("PosResidual: no part"));
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.f32s_in::<3>("residual", -MAX_RESIDUAL, MAX_RESIDUAL)?);
    }
    r.finish()?;
    Ok(PosResidual(out))
}

impl ComponentDef for Body {
    const ID: u16 = 4;
    type Data = Self;
}

impl ComponentDef for Visual {
    const ID: u16 = 5;
    type Data = Self;
}

/// Encode a [`Body`] (see the [module docs](self) for the layout).
#[must_use]
pub fn encode_body(body: &Body) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(BODY_WIRE_VERSION);
    w.u8(match body.kind {
        BodyKind::Dynamic => 0,
        BodyKind::Kinematic => 1,
    });
    match body.mass {
        MassSpec::Density(d) => {
            w.u8(0);
            w.f32(d);
        }
        MassSpec::Mass(m) => {
            w.u8(1);
            w.f32(m);
        }
    }
    w.f32(body.linear_damping);
    w.f32(body.angular_damping);
    w.f32(body.gravity_scale);
    w.f32(body.friction);
    w.f32(body.restitution);
    w.u8(u8::from(body.ccd) | (u8::from(body.lock_rotation) << 1));
    w.u32(body.groups.memberships);
    w.u32(body.groups.filter);
    encode_shape(&mut w, &body.shape);
    w.into_bytes()
}

/// Decode a [`Body`].
///
/// # Errors
/// [`CodecError::InvalidPayload`] for anything outside the layout's bounds.
pub fn decode_body(bytes: &[u8]) -> Result<Body, CodecError> {
    let mut r = Reader::new(bytes, "Body");
    r.version(BODY_WIRE_VERSION)?;
    let kind = match r.u8()? {
        0 => BodyKind::Dynamic,
        1 => BodyKind::Kinematic,
        k => return Err(invalid(format!("Body: unknown kind {k}"))),
    };
    let mass = match r.u8()? {
        0 => MassSpec::Density(r.f32_in("density", 1.0e-3, 1.0e5)?),
        1 => MassSpec::Mass(r.f32_in("mass", 1.0e-3, 1.0e9)?),
        m => return Err(invalid(format!("Body: unknown mass mode {m}"))),
    };
    let linear_damping = r.f32_in("linear_damping", 0.0, 10.0)?;
    let angular_damping = r.f32_in("angular_damping", 0.0, 10.0)?;
    let gravity_scale = r.f32_in("gravity_scale", -10.0, 10.0)?;
    let friction = r.f32_in("friction", 0.0, 10.0)?;
    let restitution = r.f32_in("restitution", 0.0, 1.0)?;
    let flags = r.u8()?;
    if flags & !0b11 != 0 {
        return Err(invalid(format!(
            "Body: reserved flag bits set ({flags:#b})"
        )));
    }
    let groups = CollisionGroups {
        memberships: r.u32()?,
        filter: r.u32()?,
    };
    let shape = decode_shape(&mut r)?;
    if total_hull_points(&shape) > MAX_TOTAL_HULL_POINTS {
        return Err(invalid(format!(
            "Body: more than {MAX_TOTAL_HULL_POINTS} hull points in one shape"
        )));
    }
    r.finish()?;
    Ok(Body {
        kind,
        mass,
        linear_damping,
        angular_damping,
        gravity_scale,
        friction,
        restitution,
        ccd: flags & 1 != 0,
        lock_rotation: flags & 2 != 0,
        groups,
        shape,
    })
}

fn total_hull_points(shape: &ShapeRef) -> usize {
    match shape {
        ShapeRef::ConvexHull { points } => points.len(),
        ShapeRef::Compound { parts } => parts.iter().map(|(_, s)| total_hull_points(s)).sum(),
        _ => 0,
    }
}

/// Encode a [`Visual`] (5 bytes).
#[must_use]
pub fn encode_visual(visual: Visual) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(VISUAL_WIRE_VERSION);
    w.u32(visual.0);
    w.into_bytes()
}

/// Decode a [`Visual`].
///
/// # Errors
/// [`CodecError::InvalidPayload`] for a wrong version or length.
pub fn decode_visual(bytes: &[u8]) -> Result<Visual, CodecError> {
    let mut r = Reader::new(bytes, "Visual");
    r.version(VISUAL_WIRE_VERSION)?;
    let v = Visual(r.u32()?);
    r.finish()?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::{decode_body, decode_visual, encode_body, encode_visual, Body, BodyKind, Visual};
    use crate::shape::{LocalPose, ShapeRef};
    use crate::ShipParams;

    #[test]
    fn bodies_round_trip() {
        let mut crate_body = Body::dynamic(ShapeRef::cuboid([0.8; 3]));
        crate_body.ccd = true;
        crate_body.groups.filter = 0b1010;
        let mut platform = Body::kinematic(ShapeRef::Compound {
            parts: vec![
                (
                    LocalPose::at([0.0, 0.0, 0.0]),
                    ShapeRef::cuboid([4.0, 0.2, 4.0]),
                ),
                (
                    LocalPose::yawed([0.0, 1.0, 0.0], 0.5),
                    ShapeRef::Ball { radius: 0.5 },
                ),
            ],
        });
        platform.mass = super::MassSpec::Mass(500.0);
        for body in [crate_body, platform, Body::ship(&ShipParams::PLACEHOLDER)] {
            let bytes = encode_body(&body);
            assert!(bytes.len() < 4096);
            assert_eq!(decode_body(&bytes).unwrap(), body);
        }
        assert_eq!(Body::ship(&ShipParams::PLACEHOLDER).kind, BodyKind::Dynamic);
        assert_eq!(
            decode_visual(&encode_visual(Visual(42))).unwrap(),
            Visual(42)
        );
        assert_eq!(encode_visual(Visual(42)).len(), 5);
        let r = super::PosResidual(vec![[1.0e-5, -2.0e-6, 0.0], [0.0; 3]]);
        assert_eq!(
            super::decode_residual(&super::encode_residual(&r)).unwrap(),
            r
        );
        assert!(
            super::decode_residual(&super::encode_residual(&super::PosResidual(vec![]))).is_err()
        );
        let big = super::PosResidual(vec![[0.5, 0.0, 0.0]]);
        assert!(super::decode_residual(&super::encode_residual(&big)).is_err());
    }

    #[test]
    fn hostile_bodies_are_rejected() {
        let good = encode_body(&Body::dynamic(ShapeRef::Ball { radius: 0.5 }));
        // Every truncation fails; so does a trailing byte.
        for len in 0..good.len() {
            assert!(decode_body(&good[..len]).is_err(), "truncated at {len}");
        }
        let mut long = good.clone();
        long.push(0);
        assert!(decode_body(&long).is_err());
        // Version, kind, mass mode, flags.
        for (offset, value) in [(0usize, 2u8), (1, 7), (2, 9)] {
            let mut b = good.clone();
            b[offset] = value;
            assert!(decode_body(&b).is_err(), "byte {offset} = {value}");
        }
        let flags_at = 1 + 1 + 1 + 4 + 5 * 4;
        let mut b = good.clone();
        b[flags_at] = 0b100;
        assert!(decode_body(&b).is_err(), "reserved flag");
        // A NaN anywhere in the numbers.
        for field in 0..6 {
            let mut b = good.clone();
            let at = 3 + field * 4;
            b[at..at + 4].copy_from_slice(&f32::NAN.to_bits().to_be_bytes());
            assert!(decode_body(&b).is_err(), "NaN in field {field}");
        }
        // Out-of-range values.
        let mut restitution = Body::dynamic(ShapeRef::Ball { radius: 0.5 });
        restitution.restitution = 1.5;
        assert!(restitution.validate().is_err());
        let mut damping = Body::dynamic(ShapeRef::Ball { radius: 0.5 });
        damping.linear_damping = -1.0;
        assert!(damping.validate().is_err());
        // Too many hull points across a compound.
        let hull = ShapeRef::ConvexHull {
            points: (0..64)
                .map(|i| {
                    let t = i as f32 * 0.4;
                    [t.cos(), (i % 7) as f32 * 0.3 - 1.0, t.sin()]
                })
                .collect(),
        };
        let fat = Body::dynamic(ShapeRef::Compound {
            parts: vec![(LocalPose::IDENTITY, hull); 5],
        });
        assert!(fat.validate().is_err());
        // Visual: wrong version and length.
        assert!(decode_visual(&[2, 0, 0, 0, 1]).is_err());
        assert!(decode_visual(&[1, 0, 0, 1]).is_err());
        assert!(decode_visual(&[1, 0, 0, 0, 1, 0]).is_err());
    }
}
