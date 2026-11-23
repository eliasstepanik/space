//! The physics of the scenes other than the demo area, and of the demo area's sky. Their numbers are the
//! ones `space-client/src/scenes.rs` draws them with (copied; phase 3 makes
//! the client read them from here).
//!
//! * **Materials**: the chart is something to fly around, so its spheres,
//!   the glass and lamp spheres and the floor slab are solid; nothing is
//!   spawned and there is no gravity (players keep flying as before).
//! * **The demo area's sky** ([`sky_statics`], [`sky_gravity`]): the Sun,
//!   the eight planets and the Moon of [`crate::solar`] as solid balls moving
//!   on their orbits, pulling within their reaches (the old `solar` scene,
//!   merged into the demo area).
//! * **Galaxy**: the spawn cube (2 m half extent, 12 m behind the spawn
//!   point) is solid. Stars are light-years apart and drawn by a level-of-
//!   detail selector, not bodies: no gravity, nothing spawned.

use space_sim::body::{CollisionGroups, Visual};
use space_sim::shape::ShapeRef;

use crate::{solar, visual, GravityDef, StaticDef};

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
        motion: None,
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

/// The demo area's solar system as statics: its ten bodies as balls at their
/// real sizes, moving with their orbits ([`crate::solar`]).
#[must_use]
pub fn sky_statics() -> Vec<StaticDef> {
    solar::sky_bodies()
        .iter()
        .map(|b| StaticDef {
            motion: b.path.clone(),
            ..solid(
                b.name,
                ShapeRef::Ball {
                    radius: b.radius as f32,
                },
                solar::EARTH_CENTER,
            )
        })
        .collect()
}

/// The demo area's solar system's gravity: its ten bodies as point masses
/// with their reaches ([`crate::solar`]).
#[must_use]
pub fn sky_gravity() -> Vec<GravityDef> {
    solar::sky_bodies()
        .iter()
        .map(|b| match &b.path {
            None => GravityDef::Body {
                reach: b.reach,
                name: b.name,
                gm: b.gm,
                radius: b.radius,
                center: solar::EARTH_CENTER,
            },
            Some(path) => GravityDef::Orbit {
                reach: b.reach,
                name: b.name,
                gm: b.gm,
                radius: b.radius,
                base: solar::EARTH_CENTER,
                path: path.clone(),
            },
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
