//! The ship integrator.
//!
//! [`step_ship`] advances one ship by exactly one [`crate::tick::dt()`] step.
//! It is a pure function of `(state, input, params)`: no globals, no clock, no
//! allocation, and the arithmetic order is fixed by the source below. Both the
//! server's authoritative pass and the client's prediction replay call it, which
//! is the whole reason the two agree.
//!
//! # Integration scheme: semi-implicit (symplectic) Euler
//!
//! Velocities are advanced first, then positions with the **new** velocity:
//!
//! ```text
//! a_lin = (R(q) · thrust · params.thrust) / mass  −  drag_lin · v
//! v    += a_lin · dt
//! a_ang = (torque · params.torque) / mass  −  (drag_ang / mass) · ω
//! ω    += a_ang · dt
//! q     = normalise(q ⊗ [1, ω·dt/2])
//! p     = p.offset(v · dt)
//! ```
//!
//! * `R(q) · thrust` is [`quat_rotate`] — the input arrives in the ship's own
//!   frame (the player pushes "forward"), the state stores world-frame velocity.
//! * Both drag terms are **linear damping** (`−k·v`), so the closed form the
//!   acceptance test checks against is
//!   `v(t) = (F/m)/k · (1 − e^{−k t})`, with terminal speed
//!   `params.thrust / (mass · drag_lin)`.
//! * Semi-implicit Euler (rather than explicit) is chosen because it is stable
//!   at `dt` = 1/30 for these parameters and its energy drift is bounded, and
//!   because it costs the same: one extra assignment.
//!
//! # Quaternion update — the exact formula
//!
//! `q ⊗ [w, x, y, z]` is Hamilton product, with `q` stored `(w, x, y, z)`. The
//! rotation over one step is the axis-angle quaternion of the body-frame rate
//! `ω` through `dt`, `exp([0, ω·dt/2])`; the **small-angle** form used here
//! replaces `cos(|ω|dt/2) → 1` and `sin(|ω|dt/2)/|ω| → dt/2`, giving exactly
//!
//! ```text
//! dq  = q ⊗ [1, ωx·dt/2, ωy·dt/2, ωz·dt/2]
//! q'  = dq / |dq|
//! ```
//!
//! The error per step is `O(|ω|³dt³)` in the angle (≈ 5e-7 rad at the
//! placeholder's 3.3 rad/s cap), and the renormalisation keeps `|q| = 1` so the
//! error does not accumulate as a scale. The *first-order* kinematic form
//! `q += ½ q ⊗ [0, ω dt]` is the same expression divided by `|dq| ≈ 1`, so the
//! two agree to the renormalisation.
//!
//! # Order of operations (this is the contract, do not reorder casually)
//!
//! 1. Clamp the input (see [`clamp_unit`]).
//! 2. Linear acceleration with the orientation from the **start** of the tick,
//!    then `v += a·dt`.
//! 3. Angular acceleration, then `ω += α·dt`.
//! 4. Orientation from the **new** `ω`.
//! 5. Position from the **new** `v`, through
//!    [`WorldPos::from_f64_metres_around`] so a ship far from the origin keeps
//!    its precision. The displacement `v · dt` is formed in `f64` (an `f32`
//!    times the `f32` tick widened to `f64` is exact), so the only rounding of
//!    the step is the final one into the `f32` local of the new [`WorldPos`].
//!    (Before the physics core the product was formed in `f32`, which added a
//!    relative error of ~6e-8 of the displacement per tick; at 1 km/s that
//!    was 2 µm per tick, always in the same direction.)
//!
//! [`ship_velocity_step`] (steps 1–3 plus the settling snap) and
//! [`ship_pose_step`] (steps 4–5) are the two halves of [`step_ship`]; the
//! rigid-body world ([`crate::rigid`]) calls them separately so a ship in
//! contact takes its velocity change from the flight model and its position
//! from the contact solver, while a ship in free flight is stepped *exactly*
//! as [`step_ship`] steps it.
//!
//! Steps 2 and 3 are independent, so a future parallel scheduler may run them
//! in either order; step 4 deliberately uses the post-torque `ω` so a ship
//! begins turning on the same tick it was commanded.
//!
//! [`WorldPos::from_f64_metres_around`]: space_world::WorldPos::from_f64_metres_around

use space_protocol::PlayerInput;
use space_world::WorldPos;

use crate::params::ShipParams;
use crate::tick::dt;
use crate::ShipState;

/// Hamilton product `a ⊗ b` of quaternions stored `(w, x, y, z)`.
#[must_use]
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Rotate the vector `v` by the unit quaternion `q` (`q v q*`).
#[must_use]
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    // t = 2 * cross(q.xyz, v); v' = v + w*t + cross(q.xyz, t)
    let [qw, qx, qy, qz] = q;
    let tx = 2.0 * (qy * v[2] - qz * v[1]);
    let ty = 2.0 * (qz * v[0] - qx * v[2]);
    let tz = 2.0 * (qx * v[1] - qy * v[0]);
    [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    ]
}

/// Squared length of `v`.
#[must_use]
pub fn length_squared(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Speed / spin rate below which a component is snapped to exactly `0.0`.
///
/// # Settling rule (documented policy)
///
/// Both drag terms are **linear** (`a -= k·v`), so a coasting ship's velocity
/// decays geometrically toward zero but, in `f32`, never *reaches* it: it stalls
/// in the subnormals and keeps changing its low bits every tick. A naive
/// integrator would therefore stamp a ship that has, for every practical purpose,
/// stopped forever — replicating it on the wire indefinitely (see
/// [`crate::SimWorld::step`], which only writes a row whose bits actually
/// changed).
///
/// To make a stopped ship *bit-stable*, [`step_ship`] snaps to exactly `0.0`:
///
/// * the **linear velocity** when its length is below `SETTLE_EPS` (1e-4 m/s —
///   a tenth of a millimetre per second); and
/// * the **angular velocity** when its length is below `SETTLE_EPS` (1e-4 rad/s).
///
/// # The snap has an input dead zone
///
/// The snap is applied *after* the integration step and unconditionally, so it
/// also erases a velocity a small input has just produced. From rest, one tick of
/// thrust changes the speed by `|thrust_local| · params.thrust / mass · dt` ≈
/// `0.2 · |thrust_local|` m/s at full stick, so any stick deflection below
/// `SETTLE_EPS / 0.2` ≈ **5e-4 of full** leaves the ship exactly where it started
/// — the same dead zone applies to the angular term (`≈ 7.5e-4` of full stick,
/// from `params.torque / mass · dt` ≈ 0.133 rad/s at full).
///
/// This is deliberate and harmless: an axis that small is far below what an
/// analogue stick resolves in practice, and the rule is a pure function of
/// `(state, input, params)` — client and server compute the *same* bits, so the
/// dead zone can never cause a desync (see the determinism note below). It is
/// *not* true that a driven ship is never snapped: what is true is that a ship
/// driven at any appreciable stick deflection is never snapped, because its
/// velocity leaves the `SETTLE_EPS` ball after one tick.
///
/// Once the angular velocity is exactly zero the ship is not turning, so the
/// orientation update is skipped entirely rather than re-normalised: `f32`
/// normalisation is **not** a guaranteed fixed point (each pass multiplies by
/// `1/|q| ≈ 1` and can flip a low bit), so re-normalising a settled quaternion
/// every tick would drift its bits forever and stamp the row forever. A
/// quaternion that is *not* already unit within [`SETTLE_QUAT_EPS`] (e.g. one
/// that arrived on a handoff payload) is normalised once and then left alone.
///
/// The rule is a pure function of `(state, input, params)` — no clock, no
/// globals — so it preserves the crate's determinism contract: the same inputs
/// produce the same bits.
pub const SETTLE_EPS: f32 = 1e-4;

/// Tolerance on `|q|² − 1` within which a settled quaternion is left untouched
/// (see [`SETTLE_EPS`]). A quaternion that has just been normalised is unit to
/// within `f32` rounding (≈ 1e-7), so this is comfortably satisfied and the
/// quaternion becomes a fixed point.
pub const SETTLE_QUAT_EPS: f32 = 1e-5;

/// Snap each component of `v` to `0.0` when the vector's length is below
/// [`SETTLE_EPS`]; otherwise return it unchanged.
#[must_use]
fn settle(v: [f32; 3]) -> [f32; 3] {
    if length_squared(v) < SETTLE_EPS * SETTLE_EPS {
        [0.0; 3]
    } else {
        v
    }
}

/// Scale `v` down to length ≤ 1 if it is longer; a zero vector is returned
/// unchanged.
///
/// # Input clamping (documented policy)
///
/// An input axis vector is treated as a *direction with magnitude*, and its
/// magnitude is clamped to **1.0** — a diagonal keypress (`[1,1,0]`) therefore
/// produces the same total thrust as a single axis, not √2 times it, which is
/// what players expect and what keeps the reachable speed set bounded. The
/// vector is scaled, never truncated per axis, so directions survive. Inputs
/// that are not finite (a corrupt packet) are zeroed: a NaN entering the
/// integrator would poison the state permanently, and dropping the input is the
/// only safe reading of it.
#[must_use]
pub fn clamp_unit(v: [f32; 3]) -> [f32; 3] {
    if !v.iter().all(|x| x.is_finite()) {
        return [0.0; 3];
    }
    let len2 = length_squared(v);
    if len2 > 1.0 {
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        v
    }
}

/// Divide a quaternion's components by its length, in place.
///
/// A zero-length quaternion is left as the identity: it cannot be normalised, and
/// the identity is the least surprising recovery.
pub fn normalize_quat(q: &mut [f32; 4]) {
    let len2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if len2 == 0.0 || !len2.is_finite() {
        *q = [1.0, 0.0, 0.0, 0.0];
        return;
    }
    let inv = 1.0 / len2.sqrt();
    for c in q.iter_mut() {
        *c *= inv;
    }
}

/// Advance `state` by one fixed tick under `input`.
///
/// `input.thrust` and `input.torque` are read as entity-local direction ×
/// magnitude and clamped to unit length (see [`clamp_unit`]); the `tick` and
/// `entity` fields of the [`PlayerInput`] are routing metadata and are ignored
/// here — the caller (see [`crate::SimWorld::apply_inputs`]) decides which
/// entity and which tick an input belongs to.
///
/// See the module docs for the exact equations and the order of operations.
pub fn step_ship(state: &mut ShipState, input: &PlayerInput, params: &ShipParams) {
    ship_velocity_step(state, input.thrust, input.torque, params, [0.0; 3]);
    ship_pose_step(state);
}

/// The velocity half of [`step_ship`]: thrust, torque, both drags and the
/// settling snap, plus an external acceleration `ext` (gravity, springs) in
/// m/s², world frame.
///
/// `ext` is added to the linear acceleration of an axis only where it is not
/// zero, so with `ext = [0; 3]` this is bit for bit the velocity update
/// [`step_ship`] always made (adding `+0.0` would turn a `-0.0` into `+0.0`).
pub fn ship_velocity_step(
    state: &mut ShipState,
    thrust: [f32; 3],
    torque: [f32; 3],
    params: &ShipParams,
    ext: [f32; 3],
) {
    let dt = dt();
    let inv_mass = 1.0 / params.mass;

    let thrust_local = clamp_unit(thrust);
    let torque_local = clamp_unit(torque);

    // 1. Linear: thrust along the ship's current orientation, minus linear
    //    damping proportional to velocity.
    let thrust_world = quat_rotate(state.quat, thrust_local);
    let thrust_scale = params.thrust * inv_mass;
    for axis in 0..3 {
        let mut accel = thrust_world[axis] * thrust_scale - params.drag_lin * state.vel[axis];
        if ext[axis] != 0.0 {
            accel += ext[axis];
        }
        state.vel[axis] += accel * dt;
    }

    // 2. Angular: same shape, with the angular damping scaled by mass so the
    //    terminal rate is `torque / drag_ang` (see ShipParams).
    let torque_scale = params.torque * inv_mass;
    let ang_damping = params.drag_ang * inv_mass;
    for axis in 0..3 {
        let accel = torque_local[axis] * torque_scale - ang_damping * state.ang_vel[axis];
        state.ang_vel[axis] += accel * dt;
    }

    // Settling: snap a slow-enough linear / angular velocity to exactly 0.0 so a
    // stopped ship becomes bit-stable (see [`SETTLE_EPS`]).
    state.vel = settle(state.vel);
    state.ang_vel = settle(state.ang_vel);
}

/// The ship's lift engines ([`ShipParams::lift`]): the extra acceleration,
/// m/s² in the world frame, that the input `thrust` (ship frame, clamped like
/// [`ship_velocity_step`] clamps it) gets in the gravity `gravity` (m/s²,
/// world frame) at the ship.
///
/// Only the part of the input that pushes *against* the pull counts: with
/// `ĝ = g / |g|` and `u = max(0, −(R(q)·thrust) · ĝ)` the result is
/// `−u · lift · g` — straight up against the pull, `u · lift · |g|` strong.
/// Pushing sideways or with the pull adds nothing, and in weightless space
/// (`g = 0`) nothing at all, so a ship in free flight is stepped exactly as
/// [`step_ship`] steps it. With `lift = 1` a ship that pushes straight up
/// climbs at `thrust / mass` whatever the pull; a ship that pushes nothing
/// falls like any body.
///
/// The rigid-body world ([`crate::rigid`]) adds this to the gravity it hands
/// [`ship_velocity_step`]; the server and a client's prediction call the same
/// function with the same numbers.
#[must_use]
pub fn lift_accel(
    quat: [f32; 4],
    thrust: [f32; 3],
    params: &ShipParams,
    gravity: [f64; 3],
) -> [f64; 3] {
    let g2 = gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2];
    if params.lift == 0.0 || !(g2 > 0.0 && g2.is_finite()) {
        return [0.0; 3];
    }
    let t = quat_rotate(quat, clamp_unit(thrust));
    let along =
        f64::from(t[0]) * gravity[0] + f64::from(t[1]) * gravity[1] + f64::from(t[2]) * gravity[2];
    // `along` is `t · g`; against the pull it is negative.
    let u = (-along / g2.sqrt()).max(0.0);
    if u == 0.0 {
        return [0.0; 3];
    }
    let k = -u * f64::from(params.lift);
    [k * gravity[0], k * gravity[1], k * gravity[2]]
}

/// The pose half of [`step_ship`]: orientation from the (new) angular
/// velocity, then position from the (new) velocity.
pub fn ship_pose_step(state: &mut ShipState) {
    quat_step(state);
    position_step(state);
}

/// Orientation from the body-frame angular velocity, by the small-angle rule
/// of the [module docs](self) (and the settled-quaternion rule of
/// [`SETTLE_EPS`]).
pub fn quat_step(state: &mut ShipState) {
    let dt = dt();
    // 3. Orientation: small-angle update from the *new* body-frame rate.
    if state.ang_vel == [0.0; 3] {
        // Not turning: leave the quaternion alone rather than re-normalising it —
        // f32 normalisation is not a fixed point and would drift the bits every
        // tick (see [`SETTLE_EPS`]). A quaternion that is not already unit within
        // [`SETTLE_QUAT_EPS`] (e.g. one that arrived on a handoff payload) is
        // normalised once and then left alone.
        let norm2 = state.quat[0] * state.quat[0]
            + state.quat[1] * state.quat[1]
            + state.quat[2] * state.quat[2]
            + state.quat[3] * state.quat[3];
        if (norm2 - 1.0).abs() >= SETTLE_QUAT_EPS {
            let mut q = state.quat;
            normalize_quat(&mut q);
            state.quat = q;
        }
    } else {
        let half = 0.5 * dt;
        let mut q = quat_mul(
            state.quat,
            [
                1.0,
                state.ang_vel[0] * half,
                state.ang_vel[1] * half,
                state.ang_vel[2] * half,
            ],
        );
        normalize_quat(&mut q);
        state.quat = q;
    }
}

/// Position from the velocity: `pos += v · dt`, the displacement formed in
/// `f64` (exact) and added in the frame of the ship's own cell.
pub fn position_step(state: &mut ShipState) {
    let dt = f64::from(dt());
    // 4. Position: new velocity, through WorldPos so precision is independent of
    //    distance from the origin.
    state.pos = WorldPos::from_f64_metres_around(
        state.pos,
        [
            f64::from(state.vel[0]) * dt,
            f64::from(state.vel[1]) * dt,
            f64::from(state.vel[2]) * dt,
        ],
    );
}

#[cfg(test)]
mod tests {
    use super::{clamp_unit, length_squared, normalize_quat, quat_mul, quat_rotate, step_ship};
    use crate::params::ShipParams;
    use crate::ShipState;
    use space_protocol::{EntityGuid, PlayerInput, ShardId};

    fn input(thrust: [f32; 3], torque: [f32; 3]) -> PlayerInput {
        PlayerInput {
            entity: EntityGuid::new(ShardId(1), 1),
            thrust,
            torque,
            tick: 0,
        }
    }

    #[test]
    fn quat_identity_is_neutral() {
        let id = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(quat_mul(id, id), id);
        assert_eq!(quat_rotate(id, [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn quat_rotate_matches_axis_rotations() {
        // 90° about +z maps +x to +y.
        let s = (std::f32::consts::FRAC_PI_4).sin();
        let c = (std::f32::consts::FRAC_PI_4).cos();
        let q = [c, 0.0, 0.0, s];
        let v = quat_rotate(q, [1.0, 0.0, 0.0]);
        assert!((v[0] - 0.0).abs() < 1e-6 && (v[1] - 1.0).abs() < 1e-6);
        // A unit quaternion rotated by a unit quaternion stays unit.
        let r = quat_mul(q, q);
        assert!((length_squared([r[1], r[2], r[3]]) + r[0] * r[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn clamp_unit_scales_but_does_not_truncate() {
        assert_eq!(clamp_unit([0.5, 0.0, 0.0]), [0.5, 0.0, 0.0]);
        let d = clamp_unit([1.0, 1.0, 0.0]);
        assert!((length_squared(d) - 1.0).abs() < 1e-6);
        assert!((d[0] - d[1]).abs() < 1e-9, "direction must survive");
        assert_eq!(clamp_unit([0.0, 0.0, 0.0]), [0.0; 3]);
        assert_eq!(clamp_unit([f32::NAN, 1.0, 0.0]), [0.0; 3]);
        assert_eq!(clamp_unit([f32::INFINITY, 0.0, 0.0]), [0.0; 3]);
    }

    #[test]
    fn normalize_recovers_a_degenerate_quat() {
        let mut q = [0.0, 0.0, 0.0, 0.0];
        normalize_quat(&mut q);
        assert_eq!(q, [1.0, 0.0, 0.0, 0.0]);
        let mut q = [2.0, 0.0, 0.0, 0.0];
        normalize_quat(&mut q);
        assert_eq!(q, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn step_ship_is_pure_and_finite() {
        let p = ShipParams::PLACEHOLDER;
        let mut a = ShipState::IDENTITY;
        let mut b = ShipState::IDENTITY;
        let inp = input([1.0, 0.3, -0.2], [0.1, 0.9, 0.0]);
        for _ in 0..100 {
            step_ship(&mut a, &inp, &p);
            step_ship(&mut b, &inp, &p);
        }
        assert_eq!(a, b, "same inputs must give the same state");
        assert!(a.vel.iter().all(|v| v.is_finite()));
        assert!(a.ang_vel.iter().all(|v| v.is_finite()));
        let norm = a.quat[0] * a.quat[0]
            + a.quat[1] * a.quat[1]
            + a.quat[2] * a.quat[2]
            + a.quat[3] * a.quat[3];
        assert!((norm - 1.0).abs() < 1e-5, "quat drifted: {norm}");
    }

    /// The lift engines push against the pull only, by `lift · |g|` at full
    /// stick, and do nothing in weightless space or for a push along the pull.
    #[test]
    fn lift_pushes_against_gravity_only() {
        use super::lift_accel;
        let p = ShipParams::PLACEHOLDER;
        let id = [1.0, 0.0, 0.0, 0.0];
        let g = [0.0, -9.81, 0.0];
        assert_eq!(lift_accel(id, [0.0, 1.0, 0.0], &p, [0.0; 3]), [0.0; 3]);
        let up = lift_accel(id, [0.0, 1.0, 0.0], &p, g);
        assert!(
            (up[1] - 9.81).abs() < 1e-9 && up[0] == 0.0 && up[2] == 0.0,
            "{up:?}"
        );
        assert_eq!(lift_accel(id, [1.0, 0.0, 0.0], &p, g), [0.0; 3], "sideways");
        assert_eq!(
            lift_accel(id, [0.0, -1.0, 0.0], &p, g),
            [0.0; 3],
            "with the pull"
        );
        assert_eq!(lift_accel(id, [0.0; 3], &p, g), [0.0; 3]);
        // A diagonal input is clamped first: half of it pushes up.
        let diagonal = lift_accel(id, [1.0, 1.0, 0.0], &p, g);
        assert!((diagonal[1] - 9.81 * std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        // A ship without lift engines gets nothing.
        let none = ShipParams { lift: 0.0, ..p };
        assert_eq!(lift_accel(id, [0.0, 1.0, 0.0], &none, g), [0.0; 3]);
        // Gravity in any direction (the side of a planetoid pulls along -X,
        // its far side along a diagonal): the push against it is answered,
        // straight against the pull, by lift times the pull's strength.
        let cases: [([f64; 3], [f64; 3]); 3] = [
            ([-4.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 0.0, 3.0], [0.0, 0.0, -1.0]),
            ([2.0, -2.0, 1.0], [-2.0, 2.0, -1.0]),
        ];
        for (g, against) in cases {
            let strength = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            let a = lift_accel(id, against.map(|c: f64| c as f32), &p, g);
            let along: f64 = (0..3).map(|k| a[k] * -g[k] / strength).sum();
            let push = against.map(|c: f64| c / against.iter().map(|x| x * x).sum::<f64>().sqrt());
            let input = push
                .iter()
                .zip(g)
                .map(|(t, gk)| -t * gk / strength)
                .sum::<f64>()
                .min(1.0);
            assert!((along - input * strength).abs() < 1e-5, "{g:?}: {a:?}");
            // ...and only against it: nothing across.
            let across: Vec<f64> = (0..3).map(|k| a[k] - along * -g[k] / strength).collect();
            assert!(across.iter().all(|c| c.abs() < 1e-9), "{g:?}: {a:?}");
            assert_eq!(lift_accel(id, push.map(|c| -c as f32), &p, g), [0.0; 3]);
        }
        // A turned ship: its input is in its own frame.
        let turned = [
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
        ];
        // 90° about +Z turns the ship's +X onto world +Y: pushing "forward
        // along x" in a -Y pull is pushing up.
        let a = lift_accel(turned, [1.0, 0.0, 0.0], &p, [0.0, -9.81, 0.0]);
        assert!((a[1] - 9.81).abs() < 1e-5, "{a:?}");
    }

    #[test]
    fn zero_input_from_rest_does_nothing() {
        let mut s = ShipState::IDENTITY;
        let before = s;
        step_ship(&mut s, &input([0.0; 3], [0.0; 3]), &ShipParams::PLACEHOLDER);
        assert_eq!(s, before);
    }

    /// The settling rule (see [`SETTLE_EPS`]): a ship that thrusts + torques then
    /// coasts snaps its linear and angular velocity to exactly `0.0`, and its
    /// quaternion stops drifting (no re-normalisation once `ω == 0`), so the state
    /// becomes a bit-exact fixed point. Determinism: the same input script gives
    /// the same bits.
    #[test]
    fn coasting_snaps_velocities_to_zero_and_freezes_the_quat() {
        let p = ShipParams::PLACEHOLDER;
        let drive = input([1.0, 0.3, -0.2], [0.1, 0.9, 0.0]);
        let idle = input([0.0; 3], [0.0; 3]);

        let mut s = ShipState::IDENTITY;
        for _ in 0..5 {
            step_ship(&mut s, &drive, &p);
        }
        assert_ne!(s.quat, [1.0, 0.0, 0.0, 0.0], "torque must turn the ship");
        for _ in 0..10_000 {
            step_ship(&mut s, &idle, &p);
        }
        assert_eq!(s.vel, [0.0; 3], "linear velocity must snap to exactly 0");
        assert_eq!(
            s.ang_vel, [0.0; 3],
            "angular velocity must snap to exactly 0"
        );
        // A further coast step changes nothing at all — the state is bit-stable.
        let frozen = s;
        step_ship(&mut s, &idle, &p);
        assert_eq!(s, frozen, "a settled ship must be a bit-exact fixed point");

        // Determinism: an identical run from the same start gives the same bits.
        let mut t = ShipState::IDENTITY;
        for _ in 0..5 {
            step_ship(&mut t, &drive, &p);
        }
        for _ in 0..10_000 {
            step_ship(&mut t, &idle, &p);
        }
        assert_eq!(t, frozen, "settling must be deterministic");
    }
}
