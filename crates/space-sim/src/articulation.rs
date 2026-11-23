//! Articulated objects: [`Articulation`] (component id 7) and [`PartStates`]
//! (component id 8).
//!
//! # One entity, many rigid parts
//!
//! Rigid joints (fixed, ball, hinge, slider) are solver constraints: the two
//! sides must be solved together, in one physics world, on one shard. So a
//! joint is only ever
//!
//! * **inside one entity** — between two of its parts (a door on its frame, a
//!   chain of links, a seesaw plank on its own pivot body), or
//! * **between one of its parts and the static world** — a fixed anchor pose
//!   in world coordinates ([`JointTarget::World`]),
//!
//! never between two separately owned entities. An entity with an
//! [`Articulation`] is simulated as several rigid bodies: **part 0 is the
//! entity itself** (its [`crate::Body`] and its [`crate::ShipState`]), parts
//! `1..` are listed in [`Articulation::parts`], and their motion is
//! [`PartStates`] (`PartStates.0[i - 1]` is part `i`). Because every part's
//! motion is a component of the one entity, the entity's component set *is*
//! the whole object: a handoff moves it atomically and a replica shows every
//! part.
//!
//! # Joint frames
//!
//! A joint connects a frame on part `a` ([`JointDef::frame_a`], in that part's
//! body frame) with a frame on `b` (a part's body frame, or relative to the
//! world anchor pose). The joint holds the two frames' origins together and,
//! depending on the kind, their orientations:
//!
//! | kind | free motion |
//! |---|---|
//! | [`JointKind::Fixed`] | none |
//! | [`JointKind::Spherical`] | rotation about the shared origin |
//! | [`JointKind::Revolute`] | rotation about the frames' local **X** axis, optional angle limits (radians) and velocity motor |
//! | [`JointKind::Prismatic`] | translation along the frames' local **X** axis, optional limits (metres) and velocity motor |
//!
//! # Wire layouts (version 1)
//!
//! `Articulation`: `version u8, nparts u8 (0..=15)`, per part `len u16, Body
//! (len bytes, see crate::body), visual u32`; `njoints u8 (0..=16)`, per joint
//! `a u8, frame_a pose (7×f32), target u8 (0 part: index u8 | 1 world: cell
//! 3×i64, local 3×f32, rotation 4×f32), frame_b pose, kind u8 (0 fixed,
//! 1 spherical, 2 revolute, 3 prismatic)`, and for revolute / prismatic
//! `has_limits u8, [min f32, max f32], has_motor u8, [target_velocity f32,
//! max_force f32]`. Part indices must name a part (`0..=nparts`), a joint may
//! not join a part to itself, limits need `min <= max`, the whole encoding is
//! at most [`MAX_ARTICULATION_BYTES`].
//!
//! `PartStates`: `version u8, count u8 (0..=15)`, then `count` ×
//! [`crate::wire`] `ShipState` (76 bytes each, validated like any other).

use space_ecs::ComponentDef;
use space_protocol::CodecError;
use space_world::{WorldPos, CELL_HALF};

use crate::body::{decode_body, encode_body, Body, Visual};
use crate::codec::{check_quat, invalid, Reader, Writer};
use crate::shape::{decode_pose, encode_pose, LocalPose};
use crate::wire::{decode_ship_state, encode_ship_state, SHIP_STATE_LEN};
use crate::ShipState;

/// Most parts an articulated entity has besides its root (part 0).
pub const MAX_EXTRA_PARTS: usize = 15;
/// Most joints one articulation may have.
pub const MAX_JOINTS: usize = 16;
/// Largest encoded [`Articulation`], bytes.
pub const MAX_ARTICULATION_BYTES: usize = 16 * 1024;
/// Wire version of [`Articulation`].
pub const ARTICULATION_WIRE_VERSION: u8 = 1;
/// Wire version of [`PartStates`].
pub const PART_STATES_WIRE_VERSION: u8 = 1;

/// One extra rigid part (parts `1..`) of an articulated entity.
#[derive(Clone, Debug, PartialEq)]
pub struct PartDef {
    /// The part's body (kind, mass, material, shape).
    pub body: Body,
    /// What a client draws for the part.
    pub visual: Visual,
}

/// What the second side of a joint is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointTarget {
    /// Another part of the same entity (0 = the root).
    Part(u8),
    /// A fixed pose of the static world.
    World {
        /// Where the anchor is.
        pos: WorldPos,
        /// Its orientation `(w, x, y, z)`.
        rotation: [f32; 4],
    },
}

/// A velocity motor on a joint's free axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motor {
    /// Target velocity (rad/s or m/s).
    pub target_velocity: f32,
    /// Largest force or torque the motor applies.
    pub max_force: f32,
}

/// The kind of joint, see the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind {
    /// Welded.
    Fixed,
    /// Ball and socket.
    Spherical,
    /// Hinge about local X.
    Revolute {
        /// Angle limits `[min, max]`, radians.
        limits: Option<[f32; 2]>,
        /// Velocity motor.
        motor: Option<Motor>,
    },
    /// Slider along local X.
    Prismatic {
        /// Travel limits `[min, max]`, metres.
        limits: Option<[f32; 2]>,
        /// Velocity motor.
        motor: Option<Motor>,
    },
}

/// One joint. See the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointDef {
    /// The first part (0 = root).
    pub a: u8,
    /// The joint frame on `a`.
    pub frame_a: LocalPose,
    /// The second side.
    pub b: JointTarget,
    /// The joint frame on `b` (relative to the anchor pose for a world target).
    pub frame_b: LocalPose,
    /// What the joint lets move.
    pub kind: JointKind,
}

/// An articulated entity's extra parts and its joints. Replicated (id 7).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Articulation {
    /// Parts `1..=parts.len()` (part 0 is the entity's own [`Body`]).
    pub parts: Vec<PartDef>,
    /// The joints.
    pub joints: Vec<JointDef>,
}

/// The motion of parts `1..` of an articulated entity (part 0's motion is the
/// entity's [`ShipState`]). Replicated (id 8).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PartStates(pub Vec<ShipState>);

impl ComponentDef for Articulation {
    const ID: u16 = 7;
    type Data = Self;
}

impl ComponentDef for PartStates {
    const ID: u16 = 8;
    type Data = Self;
}

fn encode_limits_motor(w: &mut Writer, limits: Option<[f32; 2]>, motor: Option<Motor>) {
    match limits {
        Some(l) => {
            w.u8(1);
            w.f32s(&l);
        }
        None => w.u8(0),
    }
    match motor {
        Some(m) => {
            w.u8(1);
            w.f32(m.target_velocity);
            w.f32(m.max_force);
        }
        None => w.u8(0),
    }
}

type LimitsMotor = (Option<[f32; 2]>, Option<Motor>);

fn decode_limits_motor(r: &mut Reader<'_>) -> Result<LimitsMotor, CodecError> {
    let limits = if r.bool("has_limits")? {
        let l = r.f32s_in::<2>("limits", -1.0e4, 1.0e4)?;
        if l[0] > l[1] {
            return Err(invalid(format!(
                "Articulation: limits {l:?} have min > max"
            )));
        }
        Some(l)
    } else {
        None
    };
    let motor = if r.bool("has_motor")? {
        Some(Motor {
            target_velocity: r.f32_in("motor.target_velocity", -1.0e4, 1.0e4)?,
            max_force: r.f32_in("motor.max_force", 0.0, 1.0e12)?,
        })
    } else {
        None
    };
    Ok((limits, motor))
}

/// Encode an [`Articulation`].
#[must_use]
pub fn encode_articulation(a: &Articulation) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(ARTICULATION_WIRE_VERSION);
    w.u8(u8::try_from(a.parts.len()).unwrap_or(u8::MAX));
    for p in &a.parts {
        let body = encode_body(&p.body);
        w.u16(u16::try_from(body.len()).unwrap_or(u16::MAX));
        w.bytes(&body);
        w.u32(p.visual.0);
    }
    w.u8(u8::try_from(a.joints.len()).unwrap_or(u8::MAX));
    for j in &a.joints {
        w.u8(j.a);
        encode_pose(&mut w, &j.frame_a);
        match j.b {
            JointTarget::Part(p) => {
                w.u8(0);
                w.u8(p);
            }
            JointTarget::World { pos, rotation } => {
                w.u8(1);
                for c in pos.cell {
                    w.i64(c);
                }
                w.f32s(&pos.local);
                w.f32s(&rotation);
            }
        }
        encode_pose(&mut w, &j.frame_b);
        match j.kind {
            JointKind::Fixed => w.u8(0),
            JointKind::Spherical => w.u8(1),
            JointKind::Revolute { limits, motor } => {
                w.u8(2);
                encode_limits_motor(&mut w, limits, motor);
            }
            JointKind::Prismatic { limits, motor } => {
                w.u8(3);
                encode_limits_motor(&mut w, limits, motor);
            }
        }
    }
    w.into_bytes()
}

/// Decode an [`Articulation`].
///
/// # Errors
/// [`CodecError::InvalidPayload`] for anything outside the layout's bounds.
pub fn decode_articulation(bytes: &[u8]) -> Result<Articulation, CodecError> {
    if bytes.len() > MAX_ARTICULATION_BYTES {
        return Err(invalid(format!(
            "Articulation: {} bytes exceed the cap {MAX_ARTICULATION_BYTES}",
            bytes.len()
        )));
    }
    let mut r = Reader::new(bytes, "Articulation");
    r.version(ARTICULATION_WIRE_VERSION)?;
    let nparts = r.count_u8("parts", MAX_EXTRA_PARTS)?;
    let mut parts = Vec::with_capacity(nparts);
    for _ in 0..nparts {
        let len = usize::from(r.u16()?);
        let body = decode_body(r.bytes(len)?)?;
        let visual = Visual(r.u32()?);
        parts.push(PartDef { body, visual });
    }
    let njoints = r.count_u8("joints", MAX_JOINTS)?;
    let mut joints = Vec::with_capacity(njoints);
    for _ in 0..njoints {
        let a = r.u8()?;
        let frame_a = decode_pose(&mut r)?;
        let b = match r.u8()? {
            0 => JointTarget::Part(r.u8()?),
            1 => {
                let cell = [r.i64()?, r.i64()?, r.i64()?];
                let local = r.f32s_in::<3>("anchor.local", -CELL_HALF, CELL_HALF)?;
                if local.contains(&CELL_HALF) {
                    return Err(invalid("Articulation: anchor.local must be < CELL_HALF"));
                }
                let rotation = r.f32s_in::<4>("anchor.rotation", -2.0, 2.0)?;
                check_quat("Articulation anchor", rotation)?;
                JointTarget::World {
                    pos: WorldPos { cell, local },
                    rotation,
                }
            }
            t => return Err(invalid(format!("Articulation: unknown joint target {t}"))),
        };
        let frame_b = decode_pose(&mut r)?;
        let kind = match r.u8()? {
            0 => JointKind::Fixed,
            1 => JointKind::Spherical,
            2 => {
                let (limits, motor) = decode_limits_motor(&mut r)?;
                JointKind::Revolute { limits, motor }
            }
            3 => {
                let (limits, motor) = decode_limits_motor(&mut r)?;
                JointKind::Prismatic { limits, motor }
            }
            k => return Err(invalid(format!("Articulation: unknown joint kind {k}"))),
        };
        let parts_total = nparts + 1;
        if usize::from(a) >= parts_total {
            return Err(invalid(format!(
                "Articulation: joint part {a} does not exist"
            )));
        }
        if let JointTarget::Part(p) = b {
            if usize::from(p) >= parts_total || p == a {
                return Err(invalid(format!(
                    "Articulation: joint {a}-{p} names a missing part or the same part twice"
                )));
            }
        }
        joints.push(JointDef {
            a,
            frame_a,
            b,
            frame_b,
            kind,
        });
    }
    r.finish()?;
    Ok(Articulation { parts, joints })
}

/// Encode [`PartStates`].
#[must_use]
pub fn encode_part_states(s: &PartStates) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(PART_STATES_WIRE_VERSION);
    w.u8(u8::try_from(s.0.len()).unwrap_or(u8::MAX));
    for state in &s.0 {
        w.bytes(&encode_ship_state(state));
    }
    w.into_bytes()
}

/// Decode [`PartStates`].
///
/// # Errors
/// [`CodecError::InvalidPayload`] for a wrong version, a count over the cap,
/// a wrong length or an invalid state.
pub fn decode_part_states(bytes: &[u8]) -> Result<PartStates, CodecError> {
    let mut r = Reader::new(bytes, "PartStates");
    r.version(PART_STATES_WIRE_VERSION)?;
    let n = r.count_u8("part states", MAX_EXTRA_PARTS)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(decode_ship_state(r.bytes(SHIP_STATE_LEN)?)?);
    }
    r.finish()?;
    Ok(PartStates(out))
}

#[cfg(test)]
mod tests {
    use super::{
        decode_articulation, decode_part_states, encode_articulation, encode_part_states,
        Articulation, JointDef, JointKind, JointTarget, Motor, PartDef, PartStates,
        MAX_EXTRA_PARTS,
    };
    use crate::body::{Body, Visual};
    use crate::shape::{LocalPose, ShapeRef};
    use crate::ShipState;
    use space_world::WorldPos;

    fn chain() -> Articulation {
        let link = || PartDef {
            body: Body::dynamic(ShapeRef::Capsule {
                half_height: 0.2,
                radius: 0.05,
            }),
            visual: Visual(5),
        };
        Articulation {
            parts: vec![link(), link()],
            joints: vec![
                JointDef {
                    a: 0,
                    frame_a: LocalPose::at([0.0, 0.25, 0.0]),
                    b: JointTarget::World {
                        pos: WorldPos::new([0, 0, 0], [1.0, 2.0, 3.0]),
                        rotation: [1.0, 0.0, 0.0, 0.0],
                    },
                    frame_b: LocalPose::IDENTITY,
                    kind: JointKind::Spherical,
                },
                JointDef {
                    a: 0,
                    frame_a: LocalPose::at([0.0, -0.25, 0.0]),
                    b: JointTarget::Part(1),
                    frame_b: LocalPose::at([0.0, 0.25, 0.0]),
                    kind: JointKind::Revolute {
                        limits: Some([-0.5, 0.5]),
                        motor: Some(Motor {
                            target_velocity: 1.0,
                            max_force: 10.0,
                        }),
                    },
                },
                JointDef {
                    a: 1,
                    frame_a: LocalPose::IDENTITY,
                    b: JointTarget::Part(2),
                    frame_b: LocalPose::IDENTITY,
                    kind: JointKind::Prismatic {
                        limits: None,
                        motor: None,
                    },
                },
            ],
        }
    }

    #[test]
    fn articulations_round_trip_and_reject_hostile_input() {
        let a = chain();
        let bytes = encode_articulation(&a);
        assert_eq!(decode_articulation(&bytes).unwrap(), a);
        for len in 0..bytes.len() {
            assert!(decode_articulation(&bytes[..len]).is_err(), "len {len}");
        }
        // A joint to a missing part, to itself, inverted limits, too many parts.
        let mut bad = chain();
        bad.joints[1].b = JointTarget::Part(7);
        assert!(decode_articulation(&encode_articulation(&bad)).is_err());
        let mut bad = chain();
        bad.joints[1].b = JointTarget::Part(0);
        assert!(decode_articulation(&encode_articulation(&bad)).is_err());
        let mut bad = chain();
        bad.joints[1].kind = JointKind::Revolute {
            limits: Some([1.0, -1.0]),
            motor: None,
        };
        assert!(decode_articulation(&encode_articulation(&bad)).is_err());
        let mut bad = chain();
        bad.parts = vec![bad.parts[0].clone(); MAX_EXTRA_PARTS + 1];
        bad.joints.clear();
        assert!(decode_articulation(&encode_articulation(&bad)).is_err());
        assert!(decode_articulation(&vec![1u8; 20_000]).is_err());

        let states = PartStates(vec![ShipState::IDENTITY; 3]);
        let bytes = encode_part_states(&states);
        assert_eq!(bytes.len(), 2 + 3 * 76);
        assert_eq!(decode_part_states(&bytes).unwrap(), states);
        assert!(decode_part_states(&bytes[..bytes.len() - 1]).is_err());
        let mut nan = states;
        nan.0[1].vel[0] = f32::NAN;
        assert!(decode_part_states(&encode_part_states(&nan)).is_err());
    }
}
