//! Springs (component id 6): force elements between an entity and another
//! entity or a fixed point of the world.
//!
//! # Forces, not constraints
//!
//! A spring is evaluated from the **start-of-tick states of its two ends**
//! and applied as a force, never as a solver constraint. That is deliberate:
//! a force computed from both ends' states can be computed on two different
//! shards (or in two different physics bubbles, see [`crate::rigid`]) from
//! each other's state, so a spring works across shard and bubble borders
//! without the two ends having to be solved together. The rule that makes it
//! symmetric: **whoever simulates an end applies the force to it** — the
//! world simulating the spring's owner applies `F` to the owner, and the
//! world simulating the other entity applies `-F` to that one, reading the
//! owner's [`Springs`] from its replica (phase 2 ships it with the ghost).
//!
//! # The force
//!
//! With `pA`, `pB` the world attachment points (computed from exact
//! [`WorldPos`] deltas, so the ends may be any distance apart and anywhere in
//! the world), `d = pB − pA`, `L = |d|`, `n = d / L` and `v_rel` the relative
//! velocity of the attachment points:
//!
//! ```text
//! f = stiffness · (L − rest_length) + damping · (v_rel · n)
//! F_A = clamp(f) · n        F_B = −F_A        (torque r × F at each end)
//! ```
//!
//! [`SpringMode::Rope`] pulls only: `f = 0` while `L <= rest_length`, and
//! never pushes (`f ≥ 0`). `max_force` (if non-zero) caps `|f|`.
//!
//! # Stability
//!
//! An explicit force at a fixed 1/30 s step is stable only below a
//! stiffness of the order of `m / dt²`. Rather than let a hostile or careless
//! value explode a body, the evaluation clamps the *effective* coefficients:
//! `stiffness ≤ m_eff / dt²` and `damping ≤ m_eff / dt`, where `m_eff` is the
//! reduced mass of the dynamic ends (a fixed or kinematic end counts as
//! infinitely heavy). The clamp keeps one oscillation at six or more ticks; a
//! spring within it (every spring of the demo area) is evaluated as written.
//!
//! # Wire layout (version 1)
//!
//! ```text
//! version u8 = 1, count u8 (0..=8), then per spring:
//!   part u8                     the part of this entity (0 = the root body)
//!   anchor 3×f32                attachment in that part's body frame, |c| ≤ 1e4
//!   target u8                   0 world | 1 entity
//!     world : cell 3×i64, local 3×f32 in [-512, 512)
//!     entity: shard u32, serial u64, part u8, anchor 3×f32
//!   rest_length f32             0..=1e6 m
//!   stiffness f32               0..=1e9 N/m
//!   damping f32                 0..=1e9 N·s/m
//!   max_force f32               0 (no cap) ..=1e12 N
//!   mode u8                     0 spring | 1 rope
//! ```

use space_ecs::ComponentDef;
use space_protocol::{CodecError, EntityGuid, ShardId};
use space_world::{WorldPos, CELL_HALF};

use crate::codec::{invalid, Reader, Writer};

/// Most springs one entity may carry.
pub const MAX_SPRINGS: usize = 8;
/// Wire version of [`Springs`].
pub const SPRINGS_WIRE_VERSION: u8 = 1;

/// What the far end of a spring is attached to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpringTarget {
    /// A fixed point of the world.
    World(WorldPos),
    /// A point on another entity.
    Entity {
        /// The other entity.
        guid: EntityGuid,
        /// Which of its parts (0 = its root body).
        part: u8,
        /// Attachment in that part's body frame, metres.
        anchor: [f32; 3],
    },
}

/// Push-and-pull spring or pull-only rope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpringMode {
    /// Pushes when compressed, pulls when stretched.
    Spring,
    /// Pulls only, and only when stretched past its rest length.
    Rope,
}

/// One spring. See the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    /// The part of this entity the spring is attached to (0 = root).
    pub part: u8,
    /// Attachment in that part's body frame, metres.
    pub anchor: [f32; 3],
    /// The other end.
    pub target: SpringTarget,
    /// Length at which the spring exerts no force, metres.
    pub rest_length: f32,
    /// Newtons per metre of extension.
    pub stiffness: f32,
    /// Newton-seconds per metre of extension rate.
    pub damping: f32,
    /// Largest force, newtons; `0` = no cap.
    pub max_force: f32,
    /// Spring or rope.
    pub mode: SpringMode,
}

/// The springs an entity owns (at most [`MAX_SPRINGS`]). Replicated (id 6).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Springs(pub Vec<Spring>);

impl ComponentDef for Springs {
    const ID: u16 = 6;
    type Data = Self;
}

/// One end of a spring as the force evaluation sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndState {
    /// The body origin.
    pub pos: WorldPos,
    /// Orientation `(w, x, y, z)`.
    pub quat: [f32; 4],
    /// Velocity of the centre of mass, m/s, world frame.
    pub vel: [f64; 3],
    /// Angular velocity, rad/s, **world** frame.
    pub ang_vel: [f64; 3],
    /// Centre of mass in the body frame, metres.
    pub local_com: [f64; 3],
    /// Mass, kg; `f64::INFINITY` for a fixed, kinematic or ghost end.
    pub mass: f64,
    /// The part of the position finer than `pos` holds (see
    /// [`crate::PosResidual`]): the end's exact origin is `pos + residual`.
    pub residual: [f64; 3],
}

impl EndState {
    /// A fixed point of the world.
    #[must_use]
    pub fn fixed(pos: WorldPos) -> Self {
        Self {
            pos,
            quat: [1.0, 0.0, 0.0, 0.0],
            vel: [0.0; 3],
            ang_vel: [0.0; 3],
            local_com: [0.0; 3],
            mass: f64::INFINITY,
            residual: [0.0; 3],
        }
    }
}

/// What a spring does this tick: the force on each end and where it acts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringForce {
    /// Force on the spring's owner (end A), newtons, world frame.
    pub force: [f64; 3],
    /// Lever arm of end A: attachment point minus centre of mass, world frame.
    pub arm_a: [f64; 3],
    /// Lever arm of end B (the force on B is `-force`).
    pub arm_b: [f64; 3],
    /// The current length, metres.
    pub length: f64,
}

/// Rotate `v` by the unit quaternion `q` (`f64`).
#[must_use]
pub fn rotate(q: [f32; 4], v: [f64; 3]) -> [f64; 3] {
    let (w, x, y, z) = (
        f64::from(q[0]),
        f64::from(q[1]),
        f64::from(q[2]),
        f64::from(q[3]),
    );
    let tx = 2.0 * (y * v[2] - z * v[1]);
    let ty = 2.0 * (z * v[0] - x * v[2]);
    let tz = 2.0 * (x * v[1] - y * v[0]);
    [
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Evaluate `spring` between end `a` (its owner, attachment `spring.anchor`)
/// and end `b` (attachment `anchor_b` in b's body frame), with the stability
/// clamp of the [module docs](self) for a step of `dt` seconds.
#[must_use]
pub fn evaluate(
    spring: &Spring,
    a: &EndState,
    b: &EndState,
    anchor_b: [f32; 3],
    dt: f64,
) -> SpringForce {
    let anchor_a = spring.anchor.map(f64::from);
    let anchor_b = anchor_b.map(f64::from);
    // Lever arms (attachment − centre of mass) and attachment offsets from
    // each origin, world frame.
    let off_a = rotate(a.quat, anchor_a);
    let off_b = rotate(b.quat, anchor_b);
    let com_a = rotate(a.quat, a.local_com);
    let com_b = rotate(b.quat, b.local_com);
    let arm_a = [
        off_a[0] - com_a[0],
        off_a[1] - com_a[1],
        off_a[2] - com_a[2],
    ];
    let arm_b = [
        off_b[0] - com_b[0],
        off_b[1] - com_b[1],
        off_b[2] - com_b[2],
    ];
    // d = pB − pA from the exact origin delta.
    let origin = WorldPos::delta(a.pos, b.pos);
    let d: [f64; 3] =
        core::array::from_fn(|k| origin[k] + (b.residual[k] - a.residual[k]) + off_b[k] - off_a[k]);
    let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let none = SpringForce {
        force: [0.0; 3],
        arm_a,
        arm_b,
        length,
    };
    if !length.is_finite() || length < 1.0e-9 {
        return none;
    }
    let n = [d[0] / length, d[1] / length, d[2] / length];
    let ext = length - f64::from(spring.rest_length);
    if spring.mode == SpringMode::Rope && ext <= 0.0 {
        return none;
    }
    let wa = cross(a.ang_vel, arm_a);
    let wb = cross(b.ang_vel, arm_b);
    let v_rel = [
        b.vel[0] + wb[0] - a.vel[0] - wa[0],
        b.vel[1] + wb[1] - a.vel[1] - wa[1],
        b.vel[2] + wb[2] - a.vel[2] - wa[2],
    ];
    let rate = v_rel[0] * n[0] + v_rel[1] * n[1] + v_rel[2] * n[2];
    // Stability clamp on the effective coefficients.
    let m_eff = if a.mass.is_finite() && b.mass.is_finite() {
        a.mass * b.mass / (a.mass + b.mass)
    } else if a.mass.is_finite() {
        a.mass
    } else {
        b.mass
    };
    let (mut k, mut c) = (f64::from(spring.stiffness), f64::from(spring.damping));
    if m_eff.is_finite() && m_eff > 0.0 {
        k = k.min(m_eff / (dt * dt));
        c = c.min(m_eff / dt);
    }
    let mut f = k * ext + c * rate;
    if spring.mode == SpringMode::Rope {
        f = f.max(0.0);
    }
    if spring.max_force > 0.0 {
        let cap = f64::from(spring.max_force);
        f = f.clamp(-cap, cap);
    }
    SpringForce {
        force: [f * n[0], f * n[1], f * n[2]],
        arm_a,
        arm_b,
        length,
    }
}

fn encode_worldpos(w: &mut Writer, p: WorldPos) {
    for c in p.cell {
        w.i64(c);
    }
    w.f32s(&p.local);
}

fn decode_worldpos(r: &mut Reader<'_>) -> Result<WorldPos, CodecError> {
    let cell = [r.i64()?, r.i64()?, r.i64()?];
    let local = r.f32s_in::<3>("world.local", -CELL_HALF, CELL_HALF)?;
    if local.contains(&CELL_HALF) {
        return Err(invalid("Springs: world.local must be < CELL_HALF"));
    }
    Ok(WorldPos { cell, local })
}

/// Encode [`Springs`] (see the [module docs](self)).
#[must_use]
pub fn encode_springs(springs: &Springs) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(SPRINGS_WIRE_VERSION);
    w.u8(u8::try_from(springs.0.len()).unwrap_or(u8::MAX));
    for s in &springs.0 {
        w.u8(s.part);
        w.f32s(&s.anchor);
        match s.target {
            SpringTarget::World(p) => {
                w.u8(0);
                encode_worldpos(&mut w, p);
            }
            SpringTarget::Entity { guid, part, anchor } => {
                w.u8(1);
                w.u32(guid.shard.0);
                w.u64(guid.serial);
                w.u8(part);
                w.f32s(&anchor);
            }
        }
        w.f32(s.rest_length);
        w.f32(s.stiffness);
        w.f32(s.damping);
        w.f32(s.max_force);
        w.u8(match s.mode {
            SpringMode::Spring => 0,
            SpringMode::Rope => 1,
        });
    }
    w.into_bytes()
}

/// Decode [`Springs`].
///
/// # Errors
/// [`CodecError::InvalidPayload`] for anything outside the layout's bounds.
pub fn decode_springs(bytes: &[u8]) -> Result<Springs, CodecError> {
    let mut r = Reader::new(bytes, "Springs");
    r.version(SPRINGS_WIRE_VERSION)?;
    let n = r.count_u8("springs", MAX_SPRINGS)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let part = r.u8()?;
        let anchor = r.f32s_in::<3>("anchor", -1.0e4, 1.0e4)?;
        let target = match r.u8()? {
            0 => SpringTarget::World(decode_worldpos(&mut r)?),
            1 => {
                let shard = ShardId(r.u32()?);
                let serial = r.u64()?;
                let part = r.u8()?;
                let anchor = r.f32s_in::<3>("target.anchor", -1.0e4, 1.0e4)?;
                SpringTarget::Entity {
                    guid: EntityGuid::new(shard, serial),
                    part,
                    anchor,
                }
            }
            t => return Err(invalid(format!("Springs: unknown target {t}"))),
        };
        let rest_length = r.f32_in("rest_length", 0.0, 1.0e6)?;
        let stiffness = r.f32_in("stiffness", 0.0, 1.0e9)?;
        let damping = r.f32_in("damping", 0.0, 1.0e9)?;
        let max_force = r.f32_in("max_force", 0.0, 1.0e12)?;
        let mode = match r.u8()? {
            0 => SpringMode::Spring,
            1 => SpringMode::Rope,
            m => return Err(invalid(format!("Springs: unknown mode {m}"))),
        };
        out.push(Spring {
            part,
            anchor,
            target,
            rest_length,
            stiffness,
            damping,
            max_force,
            mode,
        });
    }
    r.finish()?;
    Ok(Springs(out))
}

#[cfg(test)]
mod tests {
    use super::{
        decode_springs, encode_springs, evaluate, EndState, Spring, SpringMode, SpringTarget,
        Springs, MAX_SPRINGS,
    };
    use space_protocol::{EntityGuid, ShardId};
    use space_world::WorldPos;

    fn sample() -> Springs {
        Springs(vec![
            Spring {
                part: 0,
                anchor: [0.0, 0.5, 0.0],
                target: SpringTarget::World(WorldPos::new([1, -2, 3], [4.0, 5.0, -6.0])),
                rest_length: 2.0,
                stiffness: 40.0,
                damping: 0.5,
                max_force: 0.0,
                mode: SpringMode::Spring,
            },
            Spring {
                part: 2,
                anchor: [0.1, 0.0, 0.0],
                target: SpringTarget::Entity {
                    guid: EntityGuid::new(ShardId(3), 99),
                    part: 1,
                    anchor: [0.0, 0.0, -0.3],
                },
                rest_length: 1.0,
                stiffness: 10.0,
                damping: 0.0,
                max_force: 100.0,
                mode: SpringMode::Rope,
            },
        ])
    }

    #[test]
    fn springs_round_trip_and_reject_hostile_input() {
        let s = sample();
        let bytes = encode_springs(&s);
        assert_eq!(decode_springs(&bytes).unwrap(), s);
        for len in 0..bytes.len() {
            assert!(decode_springs(&bytes[..len]).is_err());
        }
        let mut extra = bytes.clone();
        extra.push(1);
        assert!(decode_springs(&extra).is_err());
        let too_many = Springs(vec![s.0[0]; MAX_SPRINGS + 1]);
        assert!(decode_springs(&encode_springs(&too_many)).is_err());
        let mut bad = s.clone();
        bad.0[0].stiffness = -1.0;
        assert!(decode_springs(&encode_springs(&bad)).is_err());
        let mut bad = s.clone();
        bad.0[0].rest_length = f32::INFINITY;
        assert!(decode_springs(&encode_springs(&bad)).is_err());
        let mut bad = s;
        bad.0[0].target = SpringTarget::World(WorldPos {
            cell: [0; 3],
            local: [512.0, 0.0, 0.0],
        });
        assert!(decode_springs(&encode_springs(&bad)).is_err());
    }

    #[test]
    fn force_is_hookes_law_and_ropes_only_pull() {
        let spring = Spring {
            part: 0,
            anchor: [0.0; 3],
            target: SpringTarget::World(WorldPos::ZERO),
            rest_length: 1.0,
            stiffness: 10.0,
            damping: 0.0,
            max_force: 0.0,
            mode: SpringMode::Spring,
        };
        let mut a = EndState::fixed(WorldPos::ZERO);
        a.mass = 100.0;
        // B is 3 m away along +x: stretched by 2 m, A is pulled toward B.
        let b = EndState::fixed(WorldPos::from_metres([3000, 0, 0]));
        let f = evaluate(&spring, &a, &b, [0.0; 3], 1.0 / 30.0);
        assert!((f.force[0] - 20.0).abs() < 1e-9 && f.length == 3.0);
        // Compressed: pushes apart; a rope does nothing.
        let near = EndState::fixed(WorldPos::from_metres([500, 0, 0]));
        let f = evaluate(&spring, &a, &near, [0.0; 3], 1.0 / 30.0);
        assert!((f.force[0] + 5.0).abs() < 1e-9);
        let rope = Spring {
            mode: SpringMode::Rope,
            ..spring
        };
        assert_eq!(
            evaluate(&rope, &a, &near, [0.0; 3], 1.0 / 30.0).force,
            [0.0; 3]
        );
        // Exact across a large distance: 1e12 m apart, still Hooke's law.
        let far_a = EndState {
            pos: WorldPos::from_metres([1_000_000_000_000_000, 0, 0]),
            ..a
        };
        let far_b = EndState::fixed(WorldPos::from_metres([1_000_000_000_003_000, 0, 0]));
        let f = evaluate(&spring, &far_a, &far_b, [0.0; 3], 1.0 / 30.0);
        assert!((f.force[0] - 20.0).abs() < 1e-9, "{:?}", f.force);
        // The stability clamp caps an absurd stiffness at m / dt².
        let stiff = Spring {
            stiffness: 1.0e9,
            ..spring
        };
        let f = evaluate(&stiff, &a, &b, [0.0; 3], 1.0 / 30.0);
        assert!(
            (f.force[0] - 100.0 * 900.0 * 2.0).abs() < 1e-6,
            "{:?}",
            f.force
        );
    }
}
