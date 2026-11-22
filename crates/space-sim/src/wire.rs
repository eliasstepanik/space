//! Lossless wire codec for [`ShipState`].
//!
//! A [`ShipState`] is the one replicated component the physics writes, so its
//! encoding is what a `Delta`/`Snapshot` actually carries per ship. The codec is
//! **lossless**: a decoded state is bit-identical to the encoded one, which is
//! what lets the change-tick replication model treat "re-encoded to the same
//! bytes" as "did not change".
//!
//! # Layout (big endian, fixed width)
//!
//! ```text
//! pos     : WorldPos  (36 B — see [`space_protocol::pos`])
//! vel     : f32 × 3   (12 B, raw bits)
//! quat    : f32 × 4   (16 B, raw bits)
//! ang_vel : f32 × 3   (12 B, raw bits)
//! ```
//!
//! Total: 76 bytes. Every `f32` is written as its IEEE-754 bits (not a varint,
//! not a decimal), so `-0.0`, every finite subnormal and every value the
//! physics can produce survive the round trip exactly. The `pos` field reuses
//! [`space_protocol::pos`], which validates its `local` on decode.
//!
//! # Validation on decode
//!
//! A decoded `ShipState` is untrusted input (a
//! [`space_protocol::HandoffCommit::state`] arrives from another shard, a
//! component payload from a client), and it goes straight into the integrator.
//! Decoding therefore rejects, with
//! [`space_protocol::CodecError::InvalidPayload`], every state the integrator
//! could not have produced and could be poisoned by:
//!
//! * a **non-finite** component anywhere — a NaN never leaves a float state;
//! * a quaternion that is **not close to unit** (`|q|²` outside
//!   [`QUAT_NORM2_MIN`]`..=`[`QUAT_NORM2_MAX`]). [`crate::step_ship`] rotates the
//!   thrust by the stored quaternion *before* it re-normalises it, and that
//!   rotation is quadratic in `|q|`: `q = [0, 2e19, 0, 0]` with one tick of
//!   thrust overflows `f32` to `−inf`, the position becomes NaN on the same
//!   tick and the velocity on the next;
//! * a velocity component beyond [`MAX_SPEED`] or an angular-velocity component
//!   beyond [`MAX_SPIN`] — finite, but absurd. Near `f32::MAX` the angular
//!   damping term `(drag_ang / mass) · ω` overflows to `±inf` and `ω` is NaN two
//!   ticks later; a linear velocity that large carries the position across the
//!   whole cell space in one step.
//!
//! The check is a pure *accept or reject*: a state that passes is returned
//! bit-for-bit, and [`crate::step_ship`] itself is untouched, so the server and
//! a client that both hold a valid state still integrate it identically. The
//! bounds are far outside anything the flight model reaches (the placeholder
//! ship tops out near 120 m/s and 3.3 rad/s, and keeps `|q|²` within about
//! `1e-6` of 1), so no state produced by the simulation is ever rejected.

use serde::{Deserialize, Serialize};

use crate::ShipState;

/// Encoded size of a [`ShipState`]: `WorldPos (36) + vel (12) + quat (16) +
/// ang_vel (12)` = 76 bytes.
pub const SHIP_STATE_LEN: usize = 36 + 12 + 16 + 12;

/// Smallest squared norm a decoded quaternion may have (see the
/// [module docs](self)).
///
/// The physics keeps `|q|² = 1` to within `f32` rounding; the tolerance here is
/// wide so a hand-written or slightly drifted quaternion still decodes (the
/// integrator normalises it on its first step) while a degenerate or huge one —
/// the kind that scales the thrust rotation — does not.
pub const QUAT_NORM2_MIN: f32 = 0.5;

/// Largest squared norm a decoded quaternion may have (see [`QUAT_NORM2_MIN`]).
pub const QUAT_NORM2_MAX: f32 = 2.0;

/// Largest magnitude a decoded linear-velocity component may have, in m/s.
///
/// Three times the speed of light: nothing the flight model can reach, and
/// small enough that every product the integrator forms with it stays finite.
pub const MAX_SPEED: f32 = 1.0e9;

/// Largest magnitude a decoded angular-velocity component may have, in rad/s.
pub const MAX_SPIN: f32 = 1.0e6;

/// The wire form of a [`ShipState`].
///
/// Floats are carried as raw bit patterns so the encoding is lossless and
/// independent of any float formatting.
#[derive(Serialize, Deserialize)]
struct ShipStateWire {
    #[serde(with = "space_protocol::pos")]
    pos: space_world::WorldPos,
    vel_bits: [u32; 3],
    quat_bits: [u32; 4],
    ang_vel_bits: [u32; 3],
}

impl ShipStateWire {
    fn from_state(state: &ShipState) -> Self {
        Self {
            pos: state.pos,
            vel_bits: to_bits(&state.vel),
            quat_bits: to_bits(&state.quat),
            ang_vel_bits: to_bits(&state.ang_vel),
        }
    }

    fn into_state(self) -> Result<ShipState, space_protocol::CodecError> {
        let vel = from_bits(&self.vel_bits);
        let quat = from_bits(&self.quat_bits);
        let ang_vel = from_bits(&self.ang_vel_bits);
        // A `ShipState` decoded off the wire is *untrusted* input: a
        // [`space_protocol::HandoffCommit::state`] is produced by another shard,
        // and a ghost/delta payload by a client. A non-finite velocity,
        // quaternion or angular velocity would poison the integrator permanently
        // (a NaN never leaves a float state), so it is rejected here rather than
        // trusted. `pos.local` is validated by the `pos` codec, but re-check it
        // here too so this function is self-contained about finiteness.
        for (name, v) in [
            ("vel", vel.as_slice()),
            ("quat", quat.as_slice()),
            ("ang_vel", ang_vel.as_slice()),
            ("pos.local", self.pos.local.as_slice()),
        ] {
            for (axis, &f) in v.iter().enumerate() {
                if !f.is_finite() {
                    return Err(space_protocol::CodecError::InvalidPayload(format!(
                        "ShipState {name}[{axis}] is not finite: {f}"
                    )));
                }
            }
        }
        // Finite is not enough: a finite state can still drive the integrator to
        // `inf`/NaN within a tick or two (see the module docs). Reject what the
        // integrator could never have produced. Nothing is clamped or
        // normalised here — a state that passes is returned bit-for-bit.
        for (name, v, limit) in [
            ("vel", vel.as_slice(), MAX_SPEED),
            ("ang_vel", ang_vel.as_slice(), MAX_SPIN),
        ] {
            for (axis, &f) in v.iter().enumerate() {
                if f.abs() > limit {
                    return Err(space_protocol::CodecError::InvalidPayload(format!(
                        "ShipState {name}[{axis}] = {f} is beyond the ±{limit} limit"
                    )));
                }
            }
        }
        // Every component is finite, so the sum is a number; it is `inf` (and so
        // out of range) when a component is large enough to overflow its square.
        let norm2 = quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3];
        if !(QUAT_NORM2_MIN..=QUAT_NORM2_MAX).contains(&norm2) {
            return Err(space_protocol::CodecError::InvalidPayload(format!(
                "ShipState quat {quat:?} is not a unit quaternion (|q|² = {norm2}, \
                 accepted {QUAT_NORM2_MIN}..={QUAT_NORM2_MAX})"
            )));
        }
        Ok(ShipState {
            pos: self.pos,
            vel,
            quat,
            ang_vel,
        })
    }
}

fn to_bits<const N: usize>(arr: &[f32; N]) -> [u32; N] {
    core::array::from_fn(|i| arr[i].to_bits())
}

fn from_bits<const N: usize>(arr: &[u32; N]) -> [f32; N] {
    core::array::from_fn(|i| f32::from_bits(arr[i]))
}

/// Encode a [`ShipState`] with the pinned protocol config.
///
/// This is the payload a replication pass places in an
/// [`space_protocol::EntitySnapshot`] for the `ShipState` component id.
#[must_use]
pub fn encode_ship_state(state: &ShipState) -> Vec<u8> {
    space_protocol::codec::encode_raw(&ShipStateWire::from_state(state))
}

/// Decode a [`ShipState`] previously produced by [`encode_ship_state`].
///
/// # Errors
///
/// [`space_protocol::CodecError::InvalidPayload`] when the bytes are not a valid
/// encoding: wrong length, a `pos.local` that fails validation (see
/// [`space_protocol::pos`]), or a state the integrator could not have produced —
/// a non-finite component, a quaternion far from unit, an absurd velocity (see
/// the [module docs](self)).
pub fn decode_ship_state(bytes: &[u8]) -> Result<ShipState, space_protocol::CodecError> {
    let (wire, read) = space_protocol::codec::decode_raw::<ShipStateWire>(bytes)?;
    if read != bytes.len() {
        return Err(space_protocol::CodecError::InvalidPayload(format!(
            "{} of {} bytes were consumed",
            read,
            bytes.len()
        )));
    }
    wire.into_state()
}

/// Serialize a [`ShipState`] field (usable via `#[serde(with = "...")]`).
pub fn serialize<S: serde::Serializer>(state: &ShipState, s: S) -> Result<S::Ok, S::Error> {
    ShipStateWire::from_state(state).serialize(s)
}

/// Deserialize a [`ShipState`] field, rejecting an invalid state (see
/// [`decode_ship_state`]).
pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<ShipState, D::Error> {
    ShipStateWire::deserialize(d)?
        .into_state()
        .map_err(serde::de::Error::custom)
}
