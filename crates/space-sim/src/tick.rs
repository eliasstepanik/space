//! Fixed-step clock for the shared simulation.
//!
//! The simulation advances in whole [`SimTick`]s of a fixed length ([`TICK_HZ`]
//! Hz, [`dt()`] seconds). Nothing in `space-sim` reads a wall clock: the tick
//! counter is the only notion of time, which is what lets the server and the
//! client integrate the same ship with the same sequence of calls.

/// One step of the simulation.
pub type SimTick = u64;

/// Simulation rate, in ticks per second.
pub const TICK_HZ: u32 = 30;

/// Seconds per tick, as a `const` so the physics step never divides at runtime.
///
/// `1.0 / 30.0` is not representable in binary floating point; the value is the
/// correctly rounded `f32` of the exact quotient, computed once at compile time
/// and used identically by every caller.
pub const DT: f32 = 1.0 / (TICK_HZ as f32);

/// The fixed timestep, in seconds (`1 / `[`TICK_HZ`] = 1/30 ≈ 0.0333).
#[must_use]
pub const fn dt() -> f32 {
    DT
}

/// Whole ticks in `seconds` (rounded to nearest).
#[must_use]
pub const fn ticks_in(seconds: f64) -> SimTick {
    (seconds * (TICK_HZ as f64) + 0.5) as SimTick
}

/// `tick` converted to seconds since the simulation epoch, as `f64`.
///
/// The division is done in `f64` so that even large tick counters keep full
/// precision; the result is a pure function of `tick`.
#[must_use]
pub const fn seconds_of(tick: SimTick) -> f64 {
    tick as f64 / TICK_HZ as f64
}

#[cfg(test)]
mod tests {
    use super::{dt, seconds_of, ticks_in, SimTick, DT, TICK_HZ};

    #[test]
    fn clock_constants_are_consistent() {
        assert_eq!(TICK_HZ, 30);
        assert_eq!(dt(), DT);
        assert!((dt() - 1.0 / 30.0).abs() < 1e-9);
        assert_eq!(ticks_in(1.0), 30);
        assert_eq!(seconds_of(30), 1.0);
        let t: SimTick = 10_000;
        assert!((seconds_of(t) - 333.333_333_333_333_3).abs() < 1e-9);
    }
}
