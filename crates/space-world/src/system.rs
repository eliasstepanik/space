//! Deterministic star systems: one star, `N` planets on rails, one belt.
//!
//! [`StarSystem::new(seed, star_pos)`] derives everything — planet count,
//! orbital elements, belt body count — from `hash(seed, index)`, so the same
//! seed and star position always produce the same system. Planets move on
//! Keplerian rails: [`StarSystem::planet_pos`] is a pure function of the
//! orbital elements and the time parameter `t`, exactly periodic with the
//! orbital period.

use crate::hash::{hash_unit, mix, splitmix64};
use crate::pos::WorldPos;

/// One planet of a [`StarSystem`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetInfo {
    /// Index of the planet within its system, `0..planet_count`.
    pub index: u32,
    /// Semi-major axis, in metres.
    pub semi_major_axis: f64,
    /// Orbital eccentricity, in `[0, 0.4]`.
    pub eccentricity: f64,
    /// Mean anomaly at `t = 0`, in radians.
    pub phase: f64,
    /// Orbital period, in seconds.
    pub period: f64,
    /// Planet radius, in metres.
    pub radius: f64,
}

/// A deterministic star system.
///
/// ```
/// use space_world::{StarSystem, WorldPos};
///
/// let sys = StarSystem::new(7, WorldPos::ZERO);
/// let p0 = sys.planet_pos(0, 0.0);
/// assert_eq!(p0, sys.planet_pos(0, 0.0)); // deterministic
/// assert_eq!(p0, sys.planet_pos(0, sys.planets[0].period)); // periodic
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct StarSystem {
    seed: u64,
    star: WorldPos,
    /// Position of the primary star mesh (the system barycentre plus a small
    /// deterministic offset), always the first element.
    star_mesh_pos: WorldPos,
    /// The planets, ordered by index.
    pub planets: Vec<PlanetInfo>,
    /// Number of asteroid-belt bodies.
    pub belt_count: u32,
    /// Semi-major axis of the belt, in metres.
    belt_axis: f64,
}

/// Gravitational parameter used to derive periods from semi-major axes,
/// `mu = GM` in m^3/s^2 (tuned so inner planets have periods of hours, which
/// keeps the sandbox lively).
const SYSTEM_MU: f64 = 3.986_004_418e14;

/// Minimum / maximum planet semi-major axis, in metres.
const AXIS_MIN: f64 = 7.0e6;
const AXIS_MAX: f64 = 4.0e10;

impl StarSystem {
    /// Generate the system of the star at `star`, from `seed`.
    #[must_use]
    pub fn new(seed: u64, star: WorldPos) -> Self {
        let planet_count = (mix(seed, &[0x51_53]) % 9) as u32 + 1; // 1..=9
        let mut planets = Vec::with_capacity(planet_count as usize);
        for i in 0..planet_count {
            let h1 = mix(seed, &[u64::from(i) + 1, 0xA1]);
            let h2 = mix(seed, &[u64::from(i) + 1, 0xA2]);
            let h3 = mix(seed, &[u64::from(i) + 1, 0xA3]);
            // Log-uniform spacing between AXIS_MIN and AXIS_MAX.
            let u = hash_unit(h1);
            let a = AXIS_MIN * (AXIS_MAX / AXIS_MIN).powf(u);
            let e = hash_unit(h2) * 0.4;
            let phase = hash_unit(h3) * std::f64::consts::TAU;
            let period = 2.0 * std::f64::consts::PI * (a * a * a / SYSTEM_MU).sqrt();
            let radius = 1.0e5 + hash_unit(splitmix64(h1 ^ 0xB0)) * 7.0e7;
            planets.push(PlanetInfo {
                index: i,
                semi_major_axis: a,
                eccentricity: e,
                phase,
                period,
                radius,
            });
        }
        let belt_count = (mix(seed, &[0xBE17]) % 2000) as u32 + 1000; // 1000..=2999
        let belt_axis = AXIS_MIN * (AXIS_MAX / AXIS_MIN).powf(hash_unit(mix(seed, &[0xBE18])));
        let star_mesh_pos = star;
        Self {
            seed,
            star,
            star_mesh_pos,
            planets,
            belt_count,
            belt_axis,
        }
    }

    /// The seed the system was generated from.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The star's world position (the system barycentre).
    #[must_use]
    pub const fn star(&self) -> WorldPos {
        self.star
    }

    /// Position of the primary star mesh.
    #[must_use]
    pub const fn star_mesh_position(&self) -> WorldPos {
        self.star_mesh_pos
    }

    /// Number of planets.
    #[must_use]
    pub fn planet_count(&self) -> usize {
        self.planets.len()
    }

    /// Position of planet `index` at time `t` (seconds since the system epoch).
    ///
    /// The orbit lies in the system's `z = 0` plane through the star. Solves
    /// Kepler's equation by a fixed number of Newton iterations, so the result
    /// depends only on the orbital elements and `t` — never on history — and is
    /// exactly periodic with `planet.period`.
    #[must_use]
    pub fn planet_pos(&self, index: u32, t: f64) -> WorldPos {
        let p = &self.planets[index as usize % self.planets.len()];
        let mean = {
            // Reduce `t` modulo the period first: the trigonometric argument
            // stays small, so the result is exactly periodic and
            // platform-stable even for huge `t`.
            let tt = t % p.period;
            p.phase + (std::f64::consts::TAU * tt) / p.period
        };
        let e = p.eccentricity;
        // Kepler: E - e sin E = M. 8 Newton steps converge for e < 0.4.
        let mut ecc = mean;
        for _ in 0..8 {
            let f = ecc - e * ecc.sin() - mean;
            let fp = 1.0 - e * ecc.cos();
            ecc -= f / fp;
        }
        let a = p.semi_major_axis;
        let x = a * (ecc.cos() - e);
        let y = a * ((1.0 - e * e).sqrt()) * ecc.sin();
        WorldPos::from_f64_metres_around(self.star, [x, y, 0.0])
    }

    /// Number of asteroid-belt bodies.
    #[must_use]
    pub const fn belt_body_count(&self) -> u32 {
        self.belt_count
    }

    /// Position of belt body `i` at time `t` (seconds).
    ///
    /// Bodies ride a circular orbit at the belt radius with a per-body radial
    /// scatter; the angle advances at the Keplerian rate for the belt radius,
    /// so the call is deterministic and periodic like [`StarSystem::planet_pos`].
    #[must_use]
    pub fn belt_body_pos(&self, i: u32, t: f64) -> WorldPos {
        let h = mix(self.seed, &[u64::from(i), 0x8E17]);
        let h2 = splitmix64(h ^ 0x5A7);
        let r = self.belt_axis * (0.9 + hash_unit(h) * 0.2);
        let phase = hash_unit(h2) * std::f64::consts::TAU;
        let period = 2.0 * std::f64::consts::PI * (r * r * r / SYSTEM_MU).sqrt();
        // A *re-mixed* draw. `hash_unit(h2 ^ 0x9)` would be the phase draw
        // again: xor-ing two low bits moves `h2` by at most 9, so
        // `hash_unit` (a remainder) moves by at most 9e-6 and every body's
        // height would be a function of its starting angle — a one-turn helix
        // instead of a scattered belt.
        let z = (hash_unit(splitmix64(h2 ^ 0x9)) - 0.5) * 1.0e6;
        // Reduce `t` modulo the period first so the trigonometric argument
        // stays small and platform-stable for large `t`.
        let tt = t % period;
        let ang = phase + (std::f64::consts::TAU * tt) / period;
        WorldPos::from_f64_metres_around(self.star, [r * ang.cos(), r * ang.sin(), z])
    }
}

#[cfg(test)]
mod tests {
    use super::StarSystem;
    use crate::pos::WorldPos;

    /// A belt body's height above the orbital plane must be independent of
    /// where on the ring it starts.
    ///
    /// Both are drawn from the same hash; the height used to be drawn from that
    /// hash with two low bits flipped, which moves a `hash_unit` draw by at most
    /// 9e-6 — so `height = phase ± 9e-6` for *every* body and the belt was a
    /// one-turn helix. With independent draws, a body lands within 1e-3 of that
    /// line by chance about twice in a thousand.
    #[test]
    fn belt_height_is_independent_of_orbital_phase() {
        const BODIES: u32 = 1000;
        for seed in [0x5EED_u64, 7, 0xDEAD_BEEF] {
            let star = WorldPos::ZERO;
            let sys = StarSystem::new(seed, star);
            assert!(sys.belt_body_count() >= BODIES);
            let mut on_the_helix = 0u32;
            for i in 0..BODIES {
                let d = WorldPos::delta(star, sys.belt_body_pos(i, 0.0));
                // The starting angle as a fraction of a turn and the height as
                // a fraction of the belt's 1000 km thickness: both in [0, 1].
                let phase =
                    d[1].atan2(d[0]).rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU;
                let height = d[2] / 1.0e6 + 0.5;
                assert!((0.0..=1.0).contains(&height), "body {i}: height {height}");
                // Distance on the unit circle, so 0.999 and 0.001 are close.
                let gap = (phase - height).abs();
                if gap.min(1.0 - gap) < 1e-3 {
                    on_the_helix += 1;
                }
            }
            assert!(
                on_the_helix < BODIES / 20,
                "seed {seed:#x}: {on_the_helix}/{BODIES} belt bodies have height == phase"
            );
        }
    }
}
