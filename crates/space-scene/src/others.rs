//! The physics of the scenes other than the demo area. Their numbers are the
//! ones `space-client/src/scenes.rs` draws them with (copied; phase 3 makes
//! the client read them from here).
//!
//! * **Materials**: the chart is something to fly around, so its spheres,
//!   the glass and lamp spheres and the floor slab are solid; nothing is
//!   spawned and there is no gravity (players keep flying as before).
//! * **Solar**: the Sun, the eight planets and the Moon are solid balls at
//!   their real sizes and where the client draws them, and they pull with
//!   their real `GM`. The scene's anchor is its spawn point, four Earth radii
//!   from Earth's centre on the sunward side: a ship there feels Earth's
//!   0.61 m/s² (the Sun adds 0.006 m/s², the Moon 3e-5 m/s²). The planets do
//!   not orbit (the client draws them fixed), so the pull is from fixed
//!   points. Nothing is spawned.
//! * **Galaxy**: the spawn cube (2 m half extent, 12 m behind the spawn
//!   point) is solid. Stars are light-years apart and drawn by a level-of-
//!   detail selector, not bodies: no gravity, nothing spawned.

use space_sim::body::{CollisionGroups, Visual};
use space_sim::shape::ShapeRef;

use crate::{visual, GravityDef, StaticDef};

fn solid(name: &'static str, shape: ShapeRef, offset: [f64; 3]) -> StaticDef {
    StaticDef {
        name,
        shape,
        offset,
        rotation: [1.0, 0.0, 0.0, 0.0],
        friction: 0.6,
        restitution: 0.1,
        visual: Visual(visual::NONE),
        groups: CollisionGroups::ALL,
    }
}

/// How far below the spawn point the floors are (the client's `EYE_HEIGHT`).
const EYE_HEIGHT: f64 = 1.7f32 as f64;
/// Distance from the spawn point to the plane of the chart's spheres.
const CHART_DISTANCE: f64 = 8.0;

/// The material chart's statics.
#[must_use]
pub fn materials_statics() -> Vec<StaticDef> {
    let mut out = Vec::new();
    for y in [1.45, 0.2, -1.05] {
        for column in 0..7 {
            let x = (f64::from(column) - 3.0) * 1.25;
            out.push(solid(
                "chart sphere",
                ShapeRef::Ball { radius: 0.5 },
                [x, y, -CHART_DISTANCE],
            ));
        }
    }
    out.push(solid(
        "floor",
        ShapeRef::Cuboid {
            half_extents: [100.0, 100.0, 100.0],
        },
        [0.0, -100.0 - EYE_HEIGHT, -CHART_DISTANCE],
    ));
    out.push(solid(
        "glass",
        ShapeRef::Ball { radius: 0.75 },
        [2.2, -0.9, -4.6],
    ));
    out.push(solid(
        "lamp",
        ShapeRef::Ball { radius: 0.4 },
        [-2.4, -1.25, -4.8],
    ));
    out
}

/// One astronomical unit, metres.
pub const AU: f64 = 1.495_978_707e11;
/// The Sun's radius, metres.
pub const SUN_RADIUS: f64 = 6.9634e8;
const EARTH_RADIUS: f64 = 6.371e6;
const MOON_RADIUS: f64 = 1.7374e6;
const MOON_DISTANCE: f64 = 3.844e8;

/// Name, mean radius (m), mean distance from the Sun (m), longitude (degrees)
/// and `GM` (m³/s²) of the eight planets. Sizes, distances and longitudes are
/// the client's; `GM` the IAU values.
pub const PLANETS: [(&str, f64, f64, f64, f64); 8] = [
    ("Mercury", 2.4397e6, 5.791e10, 40.0, 2.2032e13),
    ("Venus", 6.0518e6, 1.0821e11, 130.0, 3.248_59e14),
    ("Earth", 6.371e6, AU, 0.0, 3.986_004_418e14),
    ("Mars", 3.3895e6, 2.2794e11, 310.0, 4.282_837e13),
    ("Jupiter", 6.9911e7, 7.7857e11, 20.0, 1.266_865_34e17),
    ("Saturn", 5.8232e7, 1.43353e12, 345.0, 3.793_118_7e16),
    ("Uranus", 2.5362e7, 2.87246e12, 330.0, 5.793_939e15),
    ("Neptune", 2.4622e7, 4.49506e12, 10.0, 6.836_529e15),
];
/// The Sun's `GM`.
pub const SUN_GM: f64 = 1.327_124_400_18e20;
/// The Moon's `GM`.
pub const MOON_GM: f64 = 4.904_869_5e12;

fn heliocentric(distance: f64, longitude_deg: f64) -> [f64; 3] {
    let a = longitude_deg.to_radians();
    [distance * a.sin(), 0.0, -distance * a.cos()]
}

/// The solar scene's spawn point, Sun-centred: four Earth radii sunward of
/// Earth. Everything below is given relative to it.
#[must_use]
pub fn solar_spawn() -> [f64; 3] {
    let e = heliocentric(AU, 0.0);
    [e[0], e[1], e[2] + 4.0 * EARTH_RADIUS]
}

fn from_spawn(p: [f64; 3]) -> [f64; 3] {
    let s = solar_spawn();
    [p[0] - s[0], p[1] - s[1], p[2] - s[2]]
}

/// `(name, radius, centre relative to the spawn point, GM)` of the ten bodies.
#[must_use]
pub fn solar_bodies() -> Vec<(&'static str, f64, [f64; 3], f64)> {
    let mut out = vec![("Sun", SUN_RADIUS, from_spawn([0.0; 3]), SUN_GM)];
    for (name, radius, distance, longitude, gm) in PLANETS {
        out.push((
            name,
            radius,
            from_spawn(heliocentric(distance, longitude)),
            gm,
        ));
    }
    let e = heliocentric(AU, 0.0);
    let moon = [
        e[0] + MOON_DISTANCE * 0.62,
        e[1] + MOON_DISTANCE * 0.18,
        e[2] - MOON_DISTANCE * 0.76,
    ];
    out.push(("Moon", MOON_RADIUS, from_spawn(moon), MOON_GM));
    out
}

/// The solar system's statics: its ten bodies as balls.
#[must_use]
pub fn solar_statics() -> Vec<StaticDef> {
    solar_bodies()
        .into_iter()
        .map(|(name, radius, center, _)| {
            solid(
                name,
                ShapeRef::Ball {
                    radius: radius as f32,
                },
                center,
            )
        })
        .collect()
}

/// The solar system's gravity: its ten bodies as point masses.
#[must_use]
pub fn solar_gravity() -> Vec<GravityDef> {
    solar_bodies()
        .into_iter()
        .map(|(name, radius, center, gm)| GravityDef::Body {
            // The real system: no limit (their pull everywhere is the
            // inverse square, as before reaches existed).
            reach: f64::INFINITY,
            name,
            gm,
            radius,
            center,
        })
        .collect()
}

/// The galaxy scene's statics: the spawn cube.
#[must_use]
pub fn galaxy_statics() -> Vec<StaticDef> {
    vec![solid(
        "spawn cube",
        ShapeRef::Cuboid {
            half_extents: [2.0; 3],
        },
        [0.0, 0.0, 12.0],
    )]
}
