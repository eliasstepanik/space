//! The solar system around the demo area.
//!
//! The demo area **rides along with Earth**: the scene uses an Earth-centred
//! frame in which Earth stands still at [`EARTH_CENTER`] (eight Earth radii
//! from the spawn point, ahead and above: big in the sky, its gravity far out
//! of reach), the Moon circles Earth, and the Sun and the other planets are at
//! `heliocentric(t) − heliocentric_earth(t)`. Every orbit is a fixed circle
//! ([`space_sim::motion`]) with Kepler's period for its radius, run
//! [`SPEEDUP`] times faster than real time — one Earth year per real hour, a
//! lunar month in about four and a half minutes — as a pure function of the
//! scene clock (`space_protocol::SceneClock`). Nothing is simulated: where a
//! planet is depends on the time only.
//!
//! Every body is a solid ball at its real size (a static collider that moves
//! with it, [`space_sim::StaticCollider::motion`]) and pulls with its real
//! `GM` — but only within its **reach** ([`SkyBody::reach`]), so the demo area
//! is exactly as weightless as before:
//!
//! | body | reach | why |
//! |---|---|---|
//! | Earth | 6 Earth radii (38,226 km) | ends 12,700 km before the demo area (8 radii out) |
//! | Moon | 10 Moon radii (17,374 km) | a sphere of influence to arrive in and land |
//! | Sun, planets | 10 radii each | far from each other and from Earth's neighbourhood |
//!
//! Inside its reach a moving body carries ships and bodies along with it
//! ([`space_sim::gravity`], "Moving bodies and scene time"), so a ship can
//! hover over the Moon — 9,000 km/s in this frame — and land on it. There is
//! no fictitious force: the frame is Earth's, and Earth does not move in it.
//!
//! The longitudes at scene time 0 (the Unix epoch) are the client's old fixed
//! layout, not an ephemeris.

use std::sync::OnceLock;

use space_sim::motion::{Circle, OrbitPath};

/// Orbits run this many times faster than real time: 365.25 × 24, one Earth
/// year per real hour.
pub const SPEEDUP: f64 = 8766.0;

/// One astronomical unit, metres.
pub const AU: f64 = 1.495_978_707e11;
/// The Sun's radius, metres.
pub const SUN_RADIUS: f64 = 6.9634e8;
/// Earth's radius, metres.
pub const EARTH_RADIUS: f64 = 6.371e6;
/// The Moon's radius, metres.
pub const MOON_RADIUS: f64 = 1.7374e6;
/// The Moon's distance from Earth, metres.
pub const MOON_DISTANCE: f64 = 3.844e8;
/// The Moon's sidereal month, days.
pub const MOON_MONTH_DAYS: f64 = 27.321_661;
/// The Moon's orbit's tilt, degrees.
pub const MOON_TILT_DEG: f64 = 5.14;
/// The Sun's `GM`.
pub const SUN_GM: f64 = 1.327_124_400_18e20;
/// Earth's `GM`.
pub const EARTH_GM: f64 = 3.986_004_418e14;
/// The Moon's `GM`.
pub const MOON_GM: f64 = 4.904_869_5e12;

/// How far from Earth's centre the demo area is, in Earth radii.
pub const DEMO_DISTANCE_RADII: f64 = 8.0;
/// Earth's reach, in Earth radii (well short of [`DEMO_DISTANCE_RADII`]).
pub const EARTH_REACH_RADII: f64 = 6.0;
/// Every other body's reach, in its radii.
pub const REACH_RADII: f64 = 10.0;

/// Where Earth's centre is, metres from the spawn point (the scene anchor):
/// [`DEMO_DISTANCE_RADII`] Earth radii away, ahead (−Z), a little to the right
/// and above the horizon, so it fills a good part of the view from the spawn.
/// (`normalize(0.3, 0.27, −0.91) × 8 R⊕`, written out; a test checks it.)
pub const EARTH_CENTER: [f64; 3] = [
    15_359_674.756_899_627,
    13_823_707.281_209_666,
    -46_591_013.429_262_206,
];

/// Name, mean radius (m), mean distance from the Sun (m), longitude at scene
/// time 0 (degrees) and `GM` (m³/s²) of the eight planets.
pub const PLANETS: [(&str, f64, f64, f64, f64); 8] = [
    ("Mercury", 2.4397e6, 5.791e10, 40.0, 2.2032e13),
    ("Venus", 6.0518e6, 1.0821e11, 130.0, 3.248_59e14),
    ("Earth", EARTH_RADIUS, AU, 0.0, EARTH_GM),
    ("Mars", 3.3895e6, 2.2794e11, 310.0, 4.282_837e13),
    ("Jupiter", 6.9911e7, 7.7857e11, 20.0, 1.266_865_34e17),
    ("Saturn", 5.8232e7, 1.43353e12, 345.0, 3.793_118_7e16),
    ("Uranus", 2.5362e7, 2.87246e12, 330.0, 5.793_939e15),
    ("Neptune", 2.4622e7, 4.49506e12, 10.0, 6.836_529e15),
];

/// A body of the solar system, as the demo area has it.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyBody {
    /// `"Sun"`, `"Earth"`, `"Moon"`, ...
    pub name: &'static str,
    /// Radius, metres.
    pub radius: f64,
    /// `GM`, m³/s².
    pub gm: f64,
    /// Its reach, metres (see the [module docs](self)).
    pub reach: f64,
    /// Its motion relative to Earth's centre (none for Earth).
    pub path: Option<OrbitPath>,
}

impl SkyBody {
    /// Its centre at scene time `t_us` (+ `frac` µs), metres from the spawn
    /// point.
    #[must_use]
    pub fn center(&self, t_us: i64, frac: f64) -> [f64; 3] {
        let o = self
            .path
            .as_ref()
            .map_or([0.0; 3], |p| p.offset_frac(t_us, frac));
        core::array::from_fn(|k| EARTH_CENTER[k] + o[k])
    }

    /// Its speed relative to Earth, m/s, at scene time `t_us`.
    #[must_use]
    pub fn speed(&self, t_us: i64) -> f64 {
        let Some(p) = &self.path else { return 0.0 };
        let d = p.displacement(t_us, t_us + 1_000);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() / 1.0e-3
    }
}

/// The period of a circular orbit of radius `a` around `gm`, scene-clock µs
/// (Kepler's third law, sped up by [`SPEEDUP`]).
#[must_use]
pub fn period_us(a: f64, gm: f64) -> i64 {
    let real = std::f64::consts::TAU * (a * a * a / gm).sqrt();
    (real * 1.0e6 / SPEEDUP).round() as i64
}

fn heliocentric(distance: f64, longitude_deg: f64, weight: f64) -> Circle {
    Circle {
        radius: distance,
        period_us: period_us(distance, SUN_GM),
        phase: longitude_deg.to_radians(),
        tilt: 0.0,
        weight,
    }
}

/// The ten bodies: the Sun, the eight planets and the Moon, in that order.
#[must_use]
pub fn sky_bodies() -> &'static [SkyBody] {
    static BODIES: OnceLock<Vec<SkyBody>> = OnceLock::new();
    BODIES.get_or_init(|| {
        let earth = heliocentric(AU, 0.0, -1.0);
        let mut out = vec![SkyBody {
            name: "Sun",
            radius: SUN_RADIUS,
            gm: SUN_GM,
            reach: REACH_RADII * SUN_RADIUS,
            path: Some(OrbitPath::new(vec![earth])),
        }];
        for (name, radius, distance, longitude, gm) in PLANETS {
            let (reach, path) = if name == "Earth" {
                (EARTH_REACH_RADII * radius, None)
            } else {
                (
                    REACH_RADII * radius,
                    Some(OrbitPath::new(vec![
                        heliocentric(distance, longitude, 1.0),
                        earth,
                    ])),
                )
            };
            out.push(SkyBody {
                name,
                radius,
                gm,
                reach,
                path,
            });
        }
        out.push(SkyBody {
            name: "Moon",
            radius: MOON_RADIUS,
            gm: MOON_GM,
            reach: REACH_RADII * MOON_RADIUS,
            path: Some(OrbitPath::new(vec![Circle {
                radius: MOON_DISTANCE,
                period_us: (MOON_MONTH_DAYS * 86_400.0e6 / SPEEDUP).round() as i64,
                phase: 0.0,
                tilt: MOON_TILT_DEG.to_radians(),
                weight: 1.0,
            }])),
        });
        out
    })
}

/// The body named `name`.
#[must_use]
pub fn sky_body(name: &str) -> Option<&'static SkyBody> {
    sky_bodies().iter().find(|b| b.name == name)
}
