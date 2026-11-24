//! Analytic motion: fixed circular orbits as a pure function of **scene
//! time**, for the planets and moons of a scene.
//!
//! Nothing here is simulated. A body on an [`OrbitPath`] is where the path
//! says at a scene time — microseconds since the Unix epoch, mapped from a
//! shard's ticks by its [`SceneClock`] — so a shard, a second shard and a
//! client agree on where the Moon is as far as they agree on the time.
//!
//! A path is a sum of [`Circle`]s, each a circular orbit of fixed radius and
//! period, tilted about the X axis, with a weight (`+1` or `-1`). That covers
//! a frame that rides along with one planet: in an Earth-centred frame the Sun
//! is `-earth(t)`, Mars is `mars(t) - earth(t)` and the Moon is `moon(t)`.
//!
//! The phase is computed from the time **modulo the period** in integers, so
//! an orbit is as precise in 2026 as at the epoch: the angle is
//! `phase + 2π · ((t mod P) + frac) / P`.

use space_protocol::SceneClock;

use crate::tick::SimTick;

/// One circular orbit term of an [`OrbitPath`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle {
    /// Orbit radius, metres.
    pub radius: f64,
    /// Time for one lap, scene-clock microseconds (> 0).
    pub period_us: i64,
    /// The angle at scene time 0, radians.
    pub phase: f64,
    /// The orbit plane's tilt about the X axis, radians (0: the X-Z plane).
    pub tilt: f64,
    /// `+1` or `-1` (the term is added or subtracted).
    pub weight: f64,
}

impl Circle {
    /// The angle at scene time `t_us` plus `frac` µs.
    #[must_use]
    pub fn angle(&self, t_us: i64, frac: f64) -> f64 {
        let p = self.period_us.max(1);
        let into = t_us.rem_euclid(p) as f64 + frac;
        self.phase + std::f64::consts::TAU * (into / p as f64)
    }

    /// The term's offset at scene time `t_us` plus `frac` µs, metres. Angle 0
    /// is `-Z` (straight ahead of a fresh player), increasing towards `+X`.
    #[must_use]
    pub fn offset(&self, t_us: i64, frac: f64) -> [f64; 3] {
        let a = self.angle(t_us, frac);
        let (x, z) = (self.radius * a.sin(), -self.radius * a.cos());
        let (st, ct) = self.tilt.sin_cos();
        // Turned about X: z → (y, z).
        [
            self.weight * x,
            self.weight * (-z * st),
            self.weight * (z * ct),
        ]
    }

    /// Orbital speed, m/s of scene time.
    #[must_use]
    pub fn speed(&self) -> f64 {
        std::f64::consts::TAU * self.radius / (self.period_us.max(1) as f64 * 1.0e-6)
    }
}

/// A body's motion: the sum of its circles (see the [module docs](self)).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct OrbitPath {
    /// The terms, summed in order.
    pub terms: Vec<Circle>,
}

impl OrbitPath {
    /// A path of these terms.
    #[must_use]
    pub fn new(terms: Vec<Circle>) -> Self {
        Self { terms }
    }

    /// The offset at scene time `t_us`, metres.
    #[must_use]
    pub fn offset(&self, t_us: i64) -> [f64; 3] {
        self.offset_frac(t_us, 0.0)
    }

    /// The offset at scene time `t_us` plus `frac` µs (drawing between
    /// microseconds), metres.
    #[must_use]
    pub fn offset_frac(&self, t_us: i64, frac: f64) -> [f64; 3] {
        let mut out = [0.0; 3];
        for c in &self.terms {
            let o = c.offset(t_us, frac);
            for k in 0..3 {
                out[k] += o[k];
            }
        }
        out
    }

    /// How far the path moves from `from_us` to `to_us`, metres.
    #[must_use]
    pub fn displacement(&self, from_us: i64, to_us: i64) -> [f64; 3] {
        if from_us == to_us {
            return [0.0; 3];
        }
        let (a, b) = (self.offset(from_us), self.offset(to_us));
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    }

    /// Whether the path moves at all.
    #[must_use]
    pub fn is_moving(&self) -> bool {
        self.terms
            .iter()
            .any(|c| c.radius != 0.0 && c.weight != 0.0)
    }
}

/// Scene time at `tick` by `clock` (see [`SceneClock::time_us_at`]).
#[must_use]
pub fn scene_time(clock: &SceneClock, tick: SimTick) -> i64 {
    clock.time_us_at(tick)
}

#[cfg(test)]
mod tests {
    use super::{Circle, OrbitPath};

    fn circle(radius: f64, period_us: i64, weight: f64) -> Circle {
        Circle {
            radius,
            period_us,
            phase: 0.0,
            tilt: 0.0,
            weight,
        }
    }

    #[test]
    fn a_circle_goes_round_once_per_period() {
        let c = circle(10.0, 1_000_000, 1.0);
        let at = |t: i64| c.offset(t, 0.0);
        let close = |a: [f64; 3], b: [f64; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-9);
        assert!(close(at(0), [0.0, 0.0, -10.0]));
        assert!(close(at(250_000), [10.0, 0.0, 0.0]));
        assert!(close(at(500_000), [0.0, 0.0, 10.0]));
        assert!(close(at(1_000_000), at(0)));
        // Far from the epoch, exactly as precise.
        assert_eq!(at(1_763_900_000_000_000 + 250_000), at(250_000));
        assert!(close(at(-750_000), at(250_000)));
        assert!((c.speed() - std::f64::consts::TAU * 10.0).abs() < 1e-9);
    }

    /// The protocol's scene clock counts in this crate's ticks.
    #[test]
    fn the_scene_clock_counts_in_simulation_ticks() {
        assert_eq!(
            space_protocol::SceneClock::TICKS_PER_SECOND,
            i64::from(crate::tick::TICK_HZ)
        );
    }

    #[test]
    fn terms_add_up_and_a_tilt_lifts_the_plane() {
        let p = OrbitPath::new(vec![circle(5.0, 4_000, 1.0), circle(5.0, 4_000, -1.0)]);
        assert_eq!(p.offset(1_234), [0.0; 3]);
        assert_eq!(p.displacement(7, 7), [0.0; 3]);
        let mut tilted = circle(1.0, 4_000, 1.0);
        tilted.tilt = std::f64::consts::FRAC_PI_2;
        let o = tilted.offset(0, 0.0);
        assert!((o[1] - 1.0).abs() < 1e-12 && o[2].abs() < 1e-12, "{o:?}");
        assert!(!OrbitPath::default().is_moving());
    }
}
