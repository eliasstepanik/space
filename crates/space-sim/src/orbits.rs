//! Planets and belt bodies on rails — positions that never need simulating.
//!
//! A [`space_world::StarSystem`] already derives its planets' orbital elements
//! from an integer seed and computes their positions as a pure function of time
//! ([`space_world::StarSystem::planet_pos`]), so a ship's view of "where is the
//! second planet" is a lookup, not a simulation. This module is the thin
//! deterministic bridge from the simulation's [`SimTick`] clock to that
//! function; it holds no state at all.
//!
//! # Body indexing
//!
//! [`rail_pos`] takes one flat `body_index`:
//!
//! | index | body |
//! |-------|------|
//! | `0` | the system's star mesh |
//! | `1 ..= planet_count` | planet `body_index - 1` |
//! | `planet_count + 1 ..` | belt body `body_index - planet_count - 1` |
//!
//! so a caller can enumerate every rail body of a system with one counter and
//! [`body_count`].
//!
//! # Determinism
//!
//! [`rail_pos`] is a pure function of `(system, body_index, tick)`: two
//! [`space_world::StarSystem`] instances built from the same seed and star
//! position return bit-identical positions for the same arguments, and the
//! orbit closes exactly over one period when queried in seconds
//! ([`rail_pos_seconds`]). Queried in *ticks* the orbit closes up to the tick
//! quantisation of the period — a period is rarely a whole number of 1/30 s
//! ticks — which is why [`period_ticks`] returns a `f64` and the acceptance test
//! checks tick-based periodicity with a tolerance derived from that
//! quantisation.

use space_world::{StarSystem, WorldPos};

use crate::tick::{seconds_of, TICK_HZ};

/// Index of the star within a system's rail bodies.
pub const STAR_BODY: u32 = 0;

/// Total number of rail bodies in `system` (star + planets + belt bodies).
#[must_use]
pub fn body_count(system: &StarSystem) -> u32 {
    1 + system
        .planet_count()
        .try_into()
        .unwrap_or(u32::MAX - 1_000_000)
        + system.belt_body_count()
}

/// Position of rail body `body_index` at `tick`.
///
/// See the [module docs](self) for the index layout. The time parameter is
/// [`seconds_of`]`(tick)`, so the result is a pure function of the tick.
///
/// # Panics
/// If `body_index` is `>= `[`body_count`]`(system)`.
#[must_use]
pub fn rail_pos(system: &StarSystem, body_index: u32, tick: u64) -> WorldPos {
    rail_pos_seconds(system, body_index, seconds_of(tick))
}

/// Position of rail body `body_index` at `t` seconds since the system epoch.
///
/// The seconds form is exposed because it is *exactly* periodic:
/// `rail_pos_seconds(s, i, t) == rail_pos_seconds(s, i, t + period)` for the
/// body's period, whereas the tick form is periodic only up to the tick
/// quantisation of that period.
///
/// # Panics
/// If `body_index` is `>= `[`body_count`]`(system)`.
#[must_use]
pub fn rail_pos_seconds(system: &StarSystem, body_index: u32, t: f64) -> WorldPos {
    let planets = system
        .planet_count()
        .try_into()
        .unwrap_or(u32::MAX - 1_000_000);
    if body_index == STAR_BODY {
        return system.star_mesh_position();
    }
    let body = body_index - 1;
    if body < planets {
        return system.planet_pos(body, t);
    }
    let belt = body - planets;
    assert!(
        belt < system.belt_body_count(),
        "rail_pos: body_index {body_index} is past the end of the belt ({} belt bodies)",
        system.belt_body_count()
    );
    system.belt_body_pos(belt, t)
}

/// The orbital period of rail body `body_index`, in ticks (see [`period`]).
#[must_use]
pub fn period_ticks(system: &StarSystem, body_index: u32) -> f64 {
    period(system, body_index) * f64::from(TICK_HZ)
}

/// The orbital period of rail body `body_index`, in seconds.
///
/// The star has no period and reports `0.0`.
///
/// # Panics
/// If `body_index` is `>= `[`body_count`]`(system)`.
#[must_use]
pub fn period(system: &StarSystem, body_index: u32) -> f64 {
    let planets = system
        .planet_count()
        .try_into()
        .unwrap_or(u32::MAX - 1_000_000);
    if body_index == STAR_BODY {
        return 0.0;
    }
    let body = body_index - 1;
    if body < planets {
        return system.planets[usize::try_from(body).unwrap()].period;
    }
    // Belt bodies ride a circular orbit at their scattered radius; recompute it
    // exactly the way `belt_body_pos` does, from the same hashes.
    let belt = body - planets;
    belt_period(system, belt)
}

/// Period of belt body `i`, in seconds.
///
/// `space_world::StarSystem::belt_body_pos` derives the body's radius and
/// period internally and exposes neither, so both are recovered here from the
/// generator's own output instead of duplicating its hashes (a second source of
/// truth would silently drift):
///
/// * the radius is the **in-plane** distance from the star to the body at
///   `t = 0` — `belt_body_pos` places the body at `[r cos θ, r sin θ, z]`, so
///   ignoring `z` recovers `r` exactly;
/// * the system's `mu = GM` is recovered from planet 0, whose semi-major axis
///   and period the system publishes, via Kepler's third law.
fn belt_period(system: &StarSystem, i: u32) -> f64 {
    let p0 = system.belt_body_pos(i, 0.0);
    let d = WorldPos::delta(system.star(), p0);
    let r = (d[0] * d[0] + d[1] * d[1]).max(1.0).sqrt();
    let pl = &system.planets[0];
    let mu = 4.0
        * std::f64::consts::PI
        * std::f64::consts::PI
        * pl.semi_major_axis
        * pl.semi_major_axis
        * pl.semi_major_axis
        / (pl.period * pl.period);
    2.0 * std::f64::consts::PI * ((r * r * r) / mu).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{
        belt_period, body_count, period, period_ticks, rail_pos, rail_pos_seconds, STAR_BODY,
    };
    use space_world::{StarSystem, WorldPos};

    #[test]
    fn body_count_covers_star_planets_and_belt() {
        let sys = StarSystem::new(7, WorldPos::ZERO);
        assert_eq!(
            body_count(&sys),
            1 + sys.planet_count() as u32 + sys.belt_body_count()
        );
        assert_eq!(rail_pos(&sys, STAR_BODY, 0), sys.star_mesh_position());
    }

    #[test]
    fn rail_pos_is_pure_across_instances() {
        let a = StarSystem::new(1234, WorldPos::ZERO);
        let b = StarSystem::new(1234, WorldPos::ZERO);
        assert_eq!(a, b);
        for body in [0, 1, 2, 5, 12] {
            let body = body % body_count(&a);
            for tick in [0u64, 1, 999, 1_000_000, 40_000_000_000] {
                assert_eq!(
                    rail_pos(&a, body, tick),
                    rail_pos(&b, body, tick),
                    "body {body} tick {tick}"
                );
            }
        }
    }

    #[test]
    fn rail_pos_is_periodic_in_seconds() {
        let sys = StarSystem::new(7, WorldPos::ZERO);
        for body in 1..=(sys.planet_count() as u32) {
            let p = period(&sys, body);
            assert!(p > 0.0);
            assert_eq!(
                rail_pos_seconds(&sys, body, 0.0),
                rail_pos_seconds(&sys, body, p),
                "planet {body} must close its orbit exactly"
            );
        }
    }

    #[test]
    fn rail_pos_is_periodic_in_ticks_within_quantisation() {
        let sys = StarSystem::new(7, WorldPos::ZERO);
        for body in 1..=(sys.planet_count() as u32) {
            let ticks = period_ticks(&sys, body).round() as u64;
            let a = rail_pos(&sys, body, 0);
            let b = rail_pos(&sys, body, ticks);
            let err = WorldPos::distance(a, b);
            // One tick of quantisation is at most 1/30 s of a period of hours,
            // so the closure error is a tiny fraction of the orbit's radius.
            let radius = sys.planets[usize::try_from(body - 1).unwrap()].semi_major_axis;
            assert!(
                err < radius * 1e-4,
                "planet {body} did not close: {err} m over a {radius} m orbit"
            );
        }
    }

    #[test]
    fn belt_bodies_are_reachable_and_periodic() {
        let sys = StarSystem::new(7, WorldPos::ZERO);
        let first_belt = 1 + sys.planet_count() as u32;
        let p = period(&sys, first_belt);
        assert!(p > 0.0 && p.is_finite());
        assert_eq!(belt_period(&sys, 0), p);
        // The measured period is consistent with the body's own motion: after a
        // full period the body is back where it started.
        let a = rail_pos_seconds(&sys, first_belt, 0.0);
        let b = rail_pos_seconds(&sys, first_belt, p);
        let r = WorldPos::distance(sys.star(), a);
        assert!(
            WorldPos::distance(a, b) < r * 1e-6,
            "belt body did not close"
        );
        assert!(body_count(&sys) > first_belt);
    }

    #[test]
    fn out_of_range_body_index_panics() {
        let sys = StarSystem::new(7, WorldPos::ZERO);
        let too_big = body_count(&sys);
        let panicked = std::panic::catch_unwind(|| rail_pos(&sys, too_big, 0)).is_err();
        assert!(panicked, "body_index {too_big} must be rejected");
    }
}
