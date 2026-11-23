//! Gravity: a pure, shared function of position and time.
//!
//! The server and a client's prediction must agree on the pull at every
//! point, so gravity is not a component and not state: it is a
//! [`GravityField`] built from the scene definition and evaluated as
//! [`GravityField::at`]`(position, tick)`. Two sources add up:
//!
//! * **Zones** ([`GravityZone`]) — a volume (box, oriented box or sphere) with
//!   a constant acceleration inside and an optional soft edge over which it
//!   fades to zero at the boundary. The demo area's "gravity room" is one;
//!   outside every zone space is weightless.
//! * **Large bodies** ([`LargeBody`]) — point masses `GM` at a fixed place,
//!   riding a planet's rails ([`crate::orbits::rail_pos`]) or on a fixed orbit
//!   of scene time ([`BodyPlace::Orbit`], [`crate::motion`]); inside the body's
//!   radius the pull falls off linearly to the centre (a uniform ball), so the
//!   field is finite everywhere. A body has a **reach** (its sphere of
//!   influence): over the outer [`REACH_FADE`] of it the pull fades smoothly
//!   to zero, and beyond it the pull is exactly `0.0`, so a small planetoid
//!   next to a weightless place leaves that place weightless to the bit. A
//!   reach of `f64::INFINITY` is the plain inverse square everywhere.
//!
//! # Moving bodies and scene time
//!
//! A body on an orbit is where its [`crate::motion::OrbitPath`] says at the
//! **scene time** of the tick, which the field's [`SceneClock`]
//! ([`GravityField::clock`]) maps from ticks. A body moving thousands of
//! kilometres a second (the demo area's Moon in its Earth-centred frame) would
//! leave every ship near it behind, so inside its reach a moving body **carries**
//! what is near it along ([`GravityField::carry`]): each tick a point is moved
//! by the body's own displacement over the tick, weighted like the pull (fully
//! inside `(1 − REACH_FADE) · reach`, fading to nothing at the reach). A ship
//! hovering over the Moon stays over the same spot; one landed on it stays
//! landed. There is no fictitious force: a ship that leaves the reach keeps
//! only its own velocity. Several bodies' carries add up (reaches are chosen
//! not to overlap).
//!
//! Every distance is an exact [`WorldPos::delta`] (an `i64` cell difference
//! plus the `f32` offsets, in `f64`), never a coordinate of a physics bubble,
//! so a planet one astronomical unit away pulls as precisely as a zone around
//! the corner, wherever in the world the body is.
//!
//! A body's acceleration is `gravity_scale × field.at(pos)`; the field is
//! sampled at the body's origin once per tick.

use std::sync::Arc;

use space_protocol::SceneClock;
use space_world::{StarSystem, WorldPos};

use crate::motion::OrbitPath;
use crate::orbits::rail_pos;
use crate::spring::rotate;
use crate::tick::SimTick;

/// A world point given exactly: a [`WorldPos`] plus an `f64` offset in
/// metres. Scene definitions place their statics and zones this way so a
/// point is as precise as an `f64` offset from the scene's anchor, not
/// rounded to the `f32` local of a `WorldPos`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPoint {
    /// The reference position.
    pub origin: WorldPos,
    /// Metres from it.
    pub offset: [f64; 3],
}

impl WorldPoint {
    /// `origin` itself.
    #[must_use]
    pub const fn at(origin: WorldPos) -> Self {
        Self {
            origin,
            offset: [0.0; 3],
        }
    }

    /// `offset` metres from `origin`.
    #[must_use]
    pub const fn new(origin: WorldPos, offset: [f64; 3]) -> Self {
        Self { origin, offset }
    }

    /// `p − self`, metres (exact cell delta, then the offsets).
    #[must_use]
    pub fn delta_to(&self, p: WorldPos) -> [f64; 3] {
        let d = WorldPos::delta(self.origin, p);
        [
            d[0] - self.offset[0],
            d[1] - self.offset[1],
            d[2] - self.offset[2],
        ]
    }

    /// The nearest [`WorldPos`] (rounded to its `f32` local).
    #[must_use]
    pub fn to_world_pos(&self) -> WorldPos {
        WorldPos::from_f64_metres_around(self.origin, self.offset)
    }
}

/// The volume of a [`GravityZone`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZoneShape {
    /// A box with these half extents, metres, in the zone's frame.
    Box {
        /// Half the edge lengths.
        half_extents: [f64; 3],
    },
    /// A sphere.
    Sphere {
        /// Radius, metres.
        radius: f64,
    },
}

/// A volume of constant gravity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GravityZone {
    /// The volume's centre.
    pub center: WorldPoint,
    /// The volume's orientation `(w, x, y, z)` (identity: axis-aligned).
    pub rotation: [f32; 4],
    /// Box or sphere.
    pub shape: ZoneShape,
    /// The acceleration inside, m/s², world frame.
    pub gravity: [f64; 3],
    /// Over the last `soft_edge` metres inside the boundary the pull fades
    /// linearly to zero at the boundary; `0` = a hard edge.
    pub soft_edge: f64,
}

impl GravityZone {
    /// How strongly the zone acts at `p`: `0` outside, `1` deeper inside than
    /// the soft edge.
    #[must_use]
    pub fn strength(&self, p: WorldPos) -> f64 {
        self.strength_at(p, [0.0; 3])
    }

    /// [`GravityZone::strength`] at the point `offset` metres from `p`.
    #[must_use]
    pub fn strength_at(&self, p: WorldPos, offset: [f64; 3]) -> f64 {
        let c = self.center.delta_to(p);
        let d = [c[0] + offset[0], c[1] + offset[1], c[2] + offset[2]];
        let depth = match self.shape {
            ZoneShape::Sphere { radius } => {
                radius - (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            }
            ZoneShape::Box { half_extents } => {
                let inv = [
                    self.rotation[0],
                    -self.rotation[1],
                    -self.rotation[2],
                    -self.rotation[3],
                ];
                let l = rotate(inv, d);
                (0..3)
                    .map(|i| half_extents[i] - l[i].abs())
                    .fold(f64::INFINITY, f64::min)
            }
        };
        // A NaN depth (non-finite input) counts as outside.
        if depth.is_nan() || depth <= 0.0 {
            0.0
        } else if self.soft_edge > 0.0 {
            (depth / self.soft_edge).min(1.0)
        } else {
            1.0
        }
    }
}

/// Where a [`LargeBody`] is.
#[derive(Clone, Debug, PartialEq)]
pub enum BodyPlace {
    /// A fixed point.
    Fixed(WorldPoint),
    /// Body `index` of a star system on rails ([`crate::orbits`] indexing:
    /// 0 = the star, then planets, then belt bodies), at the tick's time.
    Rail {
        /// The system (shared; building one is not free).
        system: Arc<StarSystem>,
        /// The body index.
        index: u32,
    },
    /// On a fixed orbit: `base` plus the path's offset at the scene time
    /// ([`crate::motion`]).
    Orbit {
        /// The point the path is relative to.
        base: WorldPoint,
        /// The path.
        path: OrbitPath,
    },
}

/// The share of a [`LargeBody::reach`] over which the pull fades out: from
/// `(1 − REACH_FADE) · reach` to `reach` the inverse square is multiplied by a
/// smoothstep that falls from 1 to 0.
pub const REACH_FADE: f64 = 0.25;

/// A point mass.
#[derive(Clone, Debug, PartialEq)]
pub struct LargeBody {
    /// Name, for logs and tests.
    pub name: String,
    /// Standard gravitational parameter `GM`, m³/s².
    pub gm: f64,
    /// Radius, metres: inside it the pull falls off linearly to the centre.
    pub radius: f64,
    /// Where it is.
    pub place: BodyPlace,
    /// Its sphere of influence, metres from the centre: the pull fades out
    /// over the last [`REACH_FADE`] of it and is exactly zero beyond.
    /// `f64::INFINITY`: no limit.
    pub reach: f64,
}

impl LargeBody {
    /// The body's centre at `tick` (an orbit by the default clock: tick 0 at
    /// scene time 0; a [`GravityField`] uses its own [`GravityField::clock`]).
    #[must_use]
    pub fn center(&self, tick: SimTick) -> WorldPoint {
        self.center_at(tick, SceneClock::default().time_us_at(tick))
    }

    /// The body's centre at `tick`, an orbit at scene time `time_us`.
    #[must_use]
    pub fn center_at(&self, tick: SimTick, time_us: i64) -> WorldPoint {
        self.center_frac(tick, time_us, 0.0)
    }

    /// [`LargeBody::center_at`] `frac` µs later (drawing between ticks).
    #[must_use]
    pub fn center_frac(&self, tick: SimTick, time_us: i64, frac: f64) -> WorldPoint {
        match &self.place {
            BodyPlace::Fixed(p) => *p,
            BodyPlace::Rail { system, index } => WorldPoint::at(rail_pos(system, *index, tick)),
            BodyPlace::Orbit { base, path } => {
                let o = path.offset_frac(time_us, frac);
                WorldPoint::new(
                    base.origin,
                    [
                        base.offset[0] + o[0],
                        base.offset[1] + o[1],
                        base.offset[2] + o[2],
                    ],
                )
            }
        }
    }

    /// Whether the body moves with scene time (and so carries what is near
    /// it, see the [module docs](self)).
    #[must_use]
    pub fn is_orbiting(&self) -> bool {
        matches!(&self.place, BodyPlace::Orbit { path, .. } if path.is_moving())
    }

    /// The acceleration it gives at `p` at `tick`, m/s².
    #[must_use]
    pub fn accel(&self, p: WorldPos, tick: SimTick) -> [f64; 3] {
        self.accel_at(p, [0.0; 3], tick)
    }

    /// [`LargeBody::accel`] at the point `offset` metres from `p`.
    #[must_use]
    pub fn accel_at(&self, p: WorldPos, offset: [f64; 3], tick: SimTick) -> [f64; 3] {
        self.accel_at_time(p, offset, tick, SceneClock::default().time_us_at(tick))
    }

    /// [`LargeBody::accel_at`] with the body where it is at scene time
    /// `time_us`.
    #[must_use]
    pub fn accel_at_time(
        &self,
        p: WorldPos,
        offset: [f64; 3],
        tick: SimTick,
        time_us: i64,
    ) -> [f64; 3] {
        let c = self.center_at(tick, time_us).delta_to(p);
        let d = [c[0] + offset[0], c[1] + offset[1], c[2] + offset[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let r = r2.sqrt();
        let fade = self.fade(r);
        if fade == 0.0 {
            return [0.0; 3];
        }
        let k = if r >= self.radius && r > 0.0 {
            -self.gm / (r2 * r)
        } else if self.radius > 0.0 {
            -self.gm / (self.radius * self.radius * self.radius)
        } else {
            0.0
        };
        // Inside the fade only: an unlimited body is the inverse square to
        // the bit.
        let k = if fade < 1.0 { k * fade } else { k };
        [k * d[0], k * d[1], k * d[2]]
    }

    /// How much of the pull is left `r` metres from the centre: 1 inside
    /// `(1 − REACH_FADE) · reach`, a smoothstep down to 0 at the reach, 0
    /// beyond it (and for a distance that is not a number).
    #[must_use]
    pub fn fade(&self, r: f64) -> f64 {
        if !self.reach.is_finite() {
            return if r.is_nan() { 0.0 } else { 1.0 };
        }
        if r.is_nan() || r >= self.reach {
            return 0.0;
        }
        let inner = self.reach * (1.0 - REACH_FADE);
        if r <= inner {
            return 1.0;
        }
        let t = (self.reach - r) / (self.reach - inner);
        t * t * (3.0 - 2.0 * t)
    }
}

/// Every source of gravity of a scene. See the [module docs](self).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GravityField {
    /// Uniform zones.
    pub zones: Vec<GravityZone>,
    /// Point masses.
    pub bodies: Vec<LargeBody>,
    /// The mapping from ticks to scene time that orbiting bodies move by
    /// (the default: tick 0 at scene time 0). A [`crate::SimWorld`] keeps it
    /// ([`crate::SimWorld::set_scene_clock`]).
    pub clock: SceneClock,
}

impl GravityField {
    /// No gravity anywhere.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// `true` when there is no source at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.zones.is_empty() && self.bodies.is_empty()
    }

    /// The acceleration at `p` at `tick`, m/s² (zones first, then bodies, in
    /// list order — a fixed summation order).
    #[must_use]
    pub fn at(&self, p: WorldPos, tick: SimTick) -> [f64; 3] {
        self.at_offset(p, [0.0; 3], tick)
    }

    /// Whether any body moves with scene time.
    #[must_use]
    pub fn has_orbits(&self) -> bool {
        self.bodies.iter().any(LargeBody::is_orbiting)
    }

    /// [`GravityField::at`] the point `offset` metres from `p` (a body's
    /// exact position is its `WorldPos` plus its [`crate::PosResidual`]).
    #[must_use]
    pub fn at_offset(&self, p: WorldPos, offset: [f64; 3], tick: SimTick) -> [f64; 3] {
        self.at_time(p, offset, tick, self.clock.time_us_at(tick))
    }

    /// [`GravityField::at_offset`] with the orbiting bodies where they are at
    /// scene time `time_us`.
    #[must_use]
    pub fn at_time(&self, p: WorldPos, offset: [f64; 3], tick: SimTick, time_us: i64) -> [f64; 3] {
        let mut g = [0.0f64; 3];
        for z in &self.zones {
            let s = z.strength_at(p, offset);
            if s > 0.0 {
                for i in 0..3 {
                    g[i] += s * z.gravity[i];
                }
            }
        }
        for b in &self.bodies {
            let a = b.accel_at_time(p, offset, tick, time_us);
            for i in 0..3 {
                g[i] += a[i];
            }
        }
        g
    }

    /// How far the orbiting bodies carry the point `offset` metres from `p`
    /// from scene time `from_us` to `to_us`, metres (see the [module
    /// docs](self)): each moving body's displacement, weighted by its
    /// [`LargeBody::fade`] at the point's distance from it at `from_us`.
    /// Exactly zero away from every moving body.
    #[must_use]
    pub fn carry(
        &self,
        p: WorldPos,
        offset: [f64; 3],
        tick: SimTick,
        from_us: i64,
        to_us: i64,
    ) -> [f64; 3] {
        let mut out = [0.0f64; 3];
        if from_us == to_us {
            return out;
        }
        for b in &self.bodies {
            let BodyPlace::Orbit { path, .. } = &b.place else {
                continue;
            };
            if !path.is_moving() {
                continue;
            }
            let c = b.center_at(tick, from_us).delta_to(p);
            let d = [c[0] + offset[0], c[1] + offset[1], c[2] + offset[2]];
            let w = b.fade((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            if w == 0.0 {
                continue;
            }
            let m = path.displacement(from_us, to_us);
            for k in 0..3 {
                out[k] += w * m[k];
            }
        }
        out
    }

    /// Distance from `p` to the nearest surface of the field, metres: a large
    /// body's sphere, or a zone's volume (0 inside one). `f64::INFINITY` for
    /// an empty field. What fast travel scales its speed limit by
    /// ([`crate::physics::fast_travel_limit`]).
    #[must_use]
    pub fn surface_distance(
        &self,
        p: WorldPos,
        offset: [f64; 3],
        tick: SimTick,
        time_us: i64,
    ) -> f64 {
        self.nearest_surface(p, offset, tick, time_us).0
    }

    /// [`GravityField::surface_distance`], and the unit vector from that
    /// surface towards the point (zero where the point is inside a zone or a
    /// body).
    #[must_use]
    pub fn nearest_surface(
        &self,
        p: WorldPos,
        offset: [f64; 3],
        tick: SimTick,
        time_us: i64,
    ) -> (f64, [f64; 3]) {
        let mut best = (f64::INFINITY, [0.0; 3]);
        for b in &self.bodies {
            let c = b.center_at(tick, time_us).delta_to(p);
            let d = [c[0] + offset[0], c[1] + offset[1], c[2] + offset[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let gap = (r - b.radius).max(0.0);
            if gap < best.0 {
                best = (gap, unit(d));
            }
        }
        for z in &self.zones {
            let c = z.center.delta_to(p);
            let d = [c[0] + offset[0], c[1] + offset[1], c[2] + offset[2]];
            let (gap, out) = match z.shape {
                ZoneShape::Sphere { radius } => {
                    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    ((r - radius).max(0.0), unit(d))
                }
                ZoneShape::Box { half_extents } => {
                    let inv = [
                        z.rotation[0],
                        -z.rotation[1],
                        -z.rotation[2],
                        -z.rotation[3],
                    ];
                    let l = rotate(inv, d);
                    let e: [f64; 3] = core::array::from_fn(|i| {
                        (l[i].abs() - half_extents[i]).max(0.0) * l[i].signum()
                    });
                    (
                        (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt(),
                        unit(rotate(z.rotation, e)),
                    )
                }
            };
            if gap < best.0 {
                best = (gap, out);
            }
        }
        best
    }
}

/// `v` scaled to unit length; zero for a zero (or non-finite) vector.
fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 0.0 && l.is_finite() {
        v.map(|c| c / l)
    } else {
        [0.0; 3]
    }
}

#[cfg(test)]
mod tests {
    use super::{BodyPlace, GravityField, GravityZone, LargeBody, WorldPoint, ZoneShape};
    use space_world::WorldPos;

    fn room() -> GravityZone {
        GravityZone {
            center: WorldPoint::new(WorldPos::ZERO, [0.0, 5.0, 0.0]),
            rotation: [1.0, 0.0, 0.0, 0.0],
            shape: ZoneShape::Box {
                half_extents: [4.0, 5.0, 4.0],
            },
            gravity: [0.0, -9.81, 0.0],
            soft_edge: 0.5,
        }
    }

    #[test]
    fn zones_have_an_inside_a_soft_edge_and_an_outside() {
        let f = GravityField {
            zones: vec![room()],
            bodies: vec![],
            ..GravityField::default()
        };
        let at = |x: f64, y: f64| {
            f.at(
                WorldPos::from_f64_metres_around(WorldPos::ZERO, [x, y, 0.0]),
                0,
            )
        };
        assert_eq!(at(0.0, 5.0), [0.0, -9.81, 0.0]);
        assert_eq!(at(5.0, 5.0), [0.0; 3], "outside");
        let edge = at(3.75, 5.0)[1];
        assert!(
            (edge + 9.81 * 0.5).abs() < 1e-6,
            "half way through the soft edge: {edge}"
        );
        // An oriented box: turned 90° about Y, the long axis swaps.
        let mut turned = room();
        turned.shape = ZoneShape::Box {
            half_extents: [10.0, 5.0, 1.0],
        };
        let (s, c) = (
            std::f32::consts::FRAC_PI_4.sin(),
            std::f32::consts::FRAC_PI_4.cos(),
        );
        turned.rotation = [c, 0.0, s, 0.0];
        let p = WorldPos::from_f64_metres_around(WorldPos::ZERO, [0.0, 5.0, 8.0]);
        assert!(turned.strength(p) > 0.9);
        let q = WorldPos::from_f64_metres_around(WorldPos::ZERO, [8.0, 5.0, 0.0]);
        assert_eq!(turned.strength(q), 0.0);
    }

    #[test]
    fn a_point_mass_pulls_by_the_inverse_square_far_away_and_exactly() {
        let earth = LargeBody {
            name: "Earth".into(),
            gm: 3.986_004_418e14,
            radius: 6.371e6,
            // 1 AU from the world origin.
            place: BodyPlace::Fixed(WorldPoint::new(
                WorldPos::ZERO,
                [1.495_978_707e11, 0.0, 0.0],
            )),
            reach: f64::INFINITY,
        };
        let f = GravityField {
            zones: vec![],
            bodies: vec![earth],
            ..GravityField::default()
        };
        // On the surface: 9.82 m/s² toward the centre.
        let surface = WorldPos::from_f64_metres_around(
            WorldPos::ZERO,
            [1.495_978_707e11 + 6.371e6, 0.0, 0.0],
        );
        let g = f.at(surface, 0);
        assert!((g[0] + 9.82).abs() < 0.01, "{g:?}");
        // At the centre: zero, not a singularity.
        let centre = WorldPos::from_f64_metres_around(WorldPos::ZERO, [1.495_978_707e11, 0.0, 0.0]);
        assert!(f.at(centre, 0).iter().all(|v| v.abs() < 1e-3));
    }

    /// A body with a reach pulls by the inverse square well inside it, fades
    /// smoothly (monotonically, continuously) over the outer quarter and is
    /// exactly zero — positive zero, every component — at and beyond it; an
    /// unlimited body is the plain inverse square to the bit.
    #[test]
    fn a_reach_fades_the_pull_to_exactly_zero() {
        let rock = |reach: f64| LargeBody {
            name: "planetoid".into(),
            gm: 64.0,
            radius: 4.0,
            place: BodyPlace::Fixed(WorldPoint::new(WorldPos::ZERO, [0.0, 10.0, 0.0])),
            reach,
        };
        let limited = rock(16.0);
        let free = rock(f64::INFINITY);
        let at = |x: f64| WorldPos::from_f64_metres_around(WorldPos::ZERO, [x, 10.0, 0.0]);
        // Inside: identical to the unlimited body.
        for x in [4.0, 6.0, 9.0, 11.9] {
            assert_eq!(limited.accel(at(x), 0), free.accel(at(x), 0), "{x}");
        }
        assert!(
            (free.accel(at(4.0), 0)[0] + 4.0).abs() < 1e-6,
            "4 m/s² on the surface"
        );
        // The fade: below the inverse square, falling, continuous.
        let mut last = limited.accel(at(12.0), 0)[0].abs();
        for i in 1..=40 {
            let x = 12.0 + 4.0 * f64::from(i) / 40.0;
            let a = limited.accel(at(x), 0)[0].abs();
            assert!(a <= last + 1e-12, "not falling at {x}");
            assert!(last - a < 0.03, "a jump at {x}: {last} -> {a}");
            last = a;
        }
        // At and beyond the reach: exactly zero, in every direction.
        for p in [
            [16.0, 10.0, 0.0],
            [0.0, 30.0, 0.0],
            [-100.0, 10.0, 5.0],
            [1.0e9, 0.0, 0.0],
        ] {
            let a = limited.accel(WorldPos::from_f64_metres_around(WorldPos::ZERO, p), 0);
            assert!(a.iter().all(|c| c.to_bits() == 0), "{p:?}: {a:?}");
        }
        assert_eq!(limited.fade(f64::NAN), 0.0);
        assert_eq!(free.fade(1.0e300), 1.0);
        // A field of a zone and a limited body: beyond both, the zero vector
        // to the bit.
        let f = GravityField {
            zones: vec![room()],
            bodies: vec![limited],
            ..GravityField::default()
        };
        let g = f.at(
            WorldPos::from_f64_metres_around(WorldPos::ZERO, [-30.0, 0.0, 0.0]),
            7,
        );
        assert!(g.iter().all(|c| c.to_bits() == 0), "{g:?}");
    }
}
