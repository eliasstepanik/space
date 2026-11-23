//! The demo area's physics: the stations ported as static colliders with the
//! client's exact dimensions (`space-client/src/demo_area.rs`), the Kestrel on
//! its apron, and the physics showcase.
//!
//! Scene space, as in the client: the anchor is the spawn point (a player's
//! eye), +Y up, a fresh player looks down -Z, the floor's top is
//! [`FLOOR`] = −1.7 m. Seen from above (x right, z down the page = towards
//! the player):
//!
//! ```text
//!  z=-40 .. -20  [ the Kestrel on its apron (x -15..15) — a static triangle mesh ]
//!  z=-17         shadow wall      material chart            ship pad
//!  z=-11         lamps            beacon      texture crates
//!  z= -9.8                                    ┌─ GRAVITY ROOM (x 10..19, z -9.8..6.2) ─┐
//!  z= -7  glass corner                mirror  │  concave bowl, 3 balls rolling          │
//!  z= -2  ┌ ZERO-G CORNER ┐                   │  5-crate stack · 3-2-1 pyramid · ramp   │
//!  z=  0  │ 4 floating     │  spawn            │                          with 3 balls  │
//!  z=  3  │ crates, a      │                   │  seesaw · gantry: 2 spring pendulums,   │
//!  z=  5  │ spring pair,   │                   │           a hanging chain               │
//!         │ a hinged door  │                   └─────────────────────────────────────────┘
//!         └ (x -19..-10)   ┘
//! ```
//!
//! Up and to the left of the spawn point, beyond the floor's corner, hangs a
//! **planetoid** ([`PLANETOID_CENTER`]): a static ball 4 m in radius that
//! pulls with large-body gravity — 4 m/s² on its surface, the inverse square
//! out from it, faded to exactly zero at its reach of 16 m — with three crates
//! resting on its surface and three balls in circular orbits around it (7, 8.5
//! and 10 m out, in three planes). Its reach ends well short of the Kestrel,
//! the stations, the zero-g corner and the spawn point, which stay weightless
//! to the bit. A player lands on it like on any ground and lifts off with the
//! lift engines.
//!
//! The gravity room is the only other place with weight (9.81 m/s² downwards, a
//! 0.25 m soft edge, reaching 1 m under the floor so things resting on it
//! feel full gravity); the rest of the area, the spawn point included, stays
//! weightless so players can still fly. Low curbs keep rolling balls inside
//! the room; players walk in and out over them (they do not stop ships), and
//! a ship's lift engines (`ShipParams::lift`) let it climb out of the gravity
//! as easily as it flies anywhere else. The whole showcase is within 20 m of the spawn point, clear of
//! the stations, and nearly 500 m from the shard border 512 m away in +X.

use space_sim::articulation::{Articulation, JointDef, JointKind, JointTarget, PartDef};
use space_sim::body::{Body, CollisionGroups, MassSpec, Visual};
use space_sim::shape::{LocalPose, ShapeRef};
use space_sim::spring::{Spring, SpringMode, SpringTarget};
use space_world::WorldPos;

use crate::{visual, Anchor, BodyDef, GravityDef, StaticDef, BOWL, KESTREL};

/// The height of the floor's top, scene space: `-EYE_HEIGHT`, the client's
/// `f32` 1.7 widened (−1.7000000476837158), so every height here is the
/// client's to the bit.
pub const FLOOR: f64 = -(1.7f32 as f64);
/// The top of the Kestrel's apron.
pub const KESTREL_APRON_TOP: f64 = FLOOR + 0.12;
/// Half the apron's size along X and Z.
pub const KESTREL_APRON_HALF: [f64; 2] = [15.0, 10.25];
/// How far the Kestrel's origin is above the ground it stands on.
pub const GEAR_DROP: f64 = 3.3;
/// The Kestrel's berth: origin (scene space) and heading (radians about +Y),
/// as `space-client`'s `KESTREL_BERTH` (`Berth::landed(0, apron top, -30.5,
/// 105°)`).
pub const KESTREL_POSITION: [f64; 3] = [0.0, KESTREL_APRON_TOP + GEAR_DROP, -30.5];
/// The Kestrel's heading.
pub const KESTREL_YAW: f32 = 105.0 * std::f32::consts::PI / 180.0;
/// Whether the Kestrel's hull stops ships. The client's design lets a player
/// walk up to and *into* the ship (its interior is modelled); the collider is
/// the closed exterior proxy, so it stops crates and balls but lets ships
/// (players) through. Flip to `true` to make the hull solid for players too.
pub const KESTREL_BLOCKS_SHIPS: bool = false;

/// The gravity room: centre and half extents (scene space).
pub const GRAVITY_ROOM_CENTER: [f64; 3] = [14.5, FLOOR + 3.5, -1.8];
/// Half extents of the gravity room.
pub const GRAVITY_ROOM_HALF: [f64; 3] = [4.5, 4.5, 8.0];
/// The bowl's lowest point (scene space).
pub const BOWL_AT: [f64; 3] = [14.5, FLOOR + 0.02, -6.9];
/// The planetoid's centre (scene space): up and to the left of the spawn
/// point, 53 m out, in the view of a newcomer (about 30° left, 10° up).
pub const PLANETOID_CENTER: [f64; 3] = [-27.0, 10.0, -46.0];
/// The planetoid's radius, metres.
pub const PLANETOID_RADIUS: f64 = 4.0;
/// The pull on the planetoid's surface, m/s².
pub const PLANETOID_SURFACE_GRAVITY: f64 = 4.0;
/// The planetoid's `GM`: surface gravity × radius².
pub const PLANETOID_GM: f64 = PLANETOID_SURFACE_GRAVITY * PLANETOID_RADIUS * PLANETOID_RADIUS;
/// The planetoid's reach: its pull fades out over the outer quarter of it
/// and is exactly zero beyond.
pub const PLANETOID_REACH: f64 = 16.0;
/// The radii of the orbits of the three balls circling the planetoid, metres
/// from its centre (inside the unfaded part of the reach, 12 m).
pub const PLANETOID_ORBITS: [f64; 3] = [7.0, 8.5, 10.0];

/// The bowl's sphere radius and depth (its rim is 1.2 m above its bottom).
pub const BOWL_RADIUS: f64 = 3.0;
/// See [`BOWL_RADIUS`].
pub const BOWL_DEPTH: f64 = 1.2;

fn yaw(radians: f32) -> [f32; 4] {
    let (s, c) = (radians * 0.5).sin_cos();
    [c, 0.0, s, 0.0]
}

fn about_x(radians: f32) -> [f32; 4] {
    let (s, c) = (radians * 0.5).sin_cos();
    [c, s, 0.0, 0.0]
}

/// The client's `rotation_y(yaw)` applied to `p`.
fn turn_y(radians: f32, p: [f64; 3]) -> [f64; 3] {
    let (s, c) = (f64::from(radians).sin(), f64::from(radians).cos());
    [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]]
}

/// A joint frame whose local X (the hinge / slide axis) is world +Y.
const X_TO_Y: [f32; 4] = [
    std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    0.0,
    std::f32::consts::FRAC_1_SQRT_2,
];
/// A joint frame whose local X is world +Z.
const X_TO_Z: [f32; 4] = [
    std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    -std::f32::consts::FRAC_1_SQRT_2,
    0.0,
];

fn cuboid(half: [f32; 3]) -> ShapeRef {
    ShapeRef::Cuboid { half_extents: half }
}

fn ball(radius: f32) -> ShapeRef {
    ShapeRef::Ball { radius }
}

/// A static the client already draws (visual 0).
fn drawn(name: &'static str, shape: ShapeRef, at: [f64; 3], rotation: [f32; 4]) -> StaticDef {
    StaticDef {
        name,
        shape,
        offset: at,
        rotation,
        friction: 0.6,
        restitution: 0.1,
        visual: Visual(visual::NONE),
        groups: CollisionGroups::ALL,
        motion: None,
    }
}

/// A static the client does not draw yet (phase 3 draws it from `visual`).
fn new_static(
    name: &'static str,
    shape: ShapeRef,
    at: [f64; 3],
    rotation: [f32; 4],
    v: u32,
) -> StaticDef {
    StaticDef {
        visual: Visual(v),
        ..drawn(name, shape, at, rotation)
    }
}

/// The client's `sample_ship` as boxes (centre, half extents), model units.
const SHIP_BOXES: [([f32; 3], [f32; 3]); 6] = [
    ([0.0, 0.0, 0.0], [0.5, 0.35, 1.6]),
    ([0.0, -0.05, -1.9], [0.3, 0.2, 0.35]),
    ([0.0, -0.1, 0.5], [1.9, 0.06, 0.6]),
    ([0.0, 0.6, 1.2], [0.05, 0.4, 0.4]),
    ([0.0, 0.4, -0.7], [0.3, 0.12, 0.45]),
    ([0.0, 0.0, 1.7], [0.3, 0.2, 0.12]),
];

/// The placeholder ship model at `scale` as a compound of boxes.
#[must_use]
pub fn ship_prop_shape(scale: f32) -> ShapeRef {
    ShapeRef::Compound {
        parts: SHIP_BOXES
            .iter()
            .map(|(c, h)| {
                (
                    LocalPose::at([c[0] * scale, c[1] * scale, c[2] * scale]),
                    cuboid([h[0] * scale, h[1] * scale, h[2] * scale]),
                )
            })
            .collect(),
    }
}

/// The placeholder character (1.8 m, origin between the feet) as a capsule.
#[must_use]
pub fn character_shape() -> ShapeRef {
    ShapeRef::Compound {
        parts: vec![(
            LocalPose::at([0.0, 0.9, 0.0]),
            ShapeRef::Capsule {
                half_height: 0.6,
                radius: 0.3,
            },
        )],
    }
}

/// The demo area's static colliders, anchor-relative. The first ones are
/// the client's stations, ported with its exact numbers; the last ones are
/// the showcase's own (non-zero visuals).
#[must_use]
pub fn statics() -> Vec<StaticDef> {
    let f = FLOOR;
    let id = [1.0, 0.0, 0.0, 0.0];
    let mut s = vec![drawn(
        "floor",
        cuboid([19.0, 0.1, 19.0]),
        [0.0, f - 0.1, -10.0],
        id,
    )];
    // People (characters: a capsule from the feet up).
    for (x, z, lift) in [
        (-1.9, -3.6, 0.0),
        (1.4, -6.6, 0.0),
        (5.6, -15.0, 0.0),
        (4.4, -8.6, 0.02),
        (-9.6, -14.8, 0.0),
    ] {
        s.push(drawn("character", character_shape(), [x, f + lift, z], id));
    }
    // Glass corner.
    s.push(drawn(
        "uv panel",
        cuboid([1.6, 1.6, 0.08]),
        [-6.4, f + 1.6, -7.9],
        id,
    ));
    s.push(drawn(
        "checker panel",
        cuboid([0.08, 1.6, 1.6]),
        [-8.08, f + 1.6, -6.3],
        id,
    ));
    s.push(drawn(
        "striped pillar",
        cuboid([0.22, 1.3, 0.22]),
        [-4.4, f + 1.3, -7.9],
        id,
    ));
    for [x, z] in [[-5.8, -6.6], [-7.0, -5.2]] {
        s.push(drawn(
            "pedestal",
            cuboid([0.4, 0.5, 0.4]),
            [x, f + 0.5, z],
            id,
        ));
    }
    s.push(drawn(
        "glass cube",
        cuboid([0.5; 3]),
        [-5.8, f + 1.5, -6.6],
        id,
    ));
    s.push(drawn(
        "glass sphere",
        ball(0.55),
        [-7.0, f + 1.55, -5.2],
        id,
    ));
    // Mirror floor and what stands on it.
    s.push(drawn(
        "mirror",
        // A slab let into the floor: top 2 cm proud of it, underside inside it.
        cuboid([3.0, 0.1, 3.0]),
        [6.0, f - 0.08, -7.0],
        id,
    ));
    s.push(drawn(
        "mirror ship",
        ship_prop_shape(0.55),
        [6.9, f + 1.0, -7.4],
        yaw(0.7),
    ));
    s.push(drawn("red ball", ball(0.4), [4.3, f + 0.42, -5.2], id));
    s.push(drawn("chrome ball", ball(0.6), [2.3, f + 0.6, -4.3], id));
    // Texture crates, three wide and two high.
    for i in 0..6 {
        let (column, row) = (f64::from(i % 3), f64::from(i / 3));
        s.push(drawn(
            "texture crate",
            cuboid([0.9; 3]),
            [5.2 + 1.9 * column, f + 0.9 + 1.8 * row, -11.4],
            id,
        ));
    }
    // Material chart: two steps and their spheres.
    for (half_height, z) in [(0.25f64, -15.2f64), (0.8, -16.4)] {
        s.push(drawn(
            "chart step",
            cuboid([4.5, half_height as f32, 0.6]),
            [0.0, f + half_height, z],
            id,
        ));
        for column in 0..7 {
            let x = (f64::from(column) - 3.0) * 1.25;
            s.push(drawn(
                "chart sphere",
                ball(0.45),
                [x, f + 2.0 * half_height + 0.45, z],
                id,
            ));
        }
    }
    // Shadow theatre.
    s.push(drawn(
        "shadow wall",
        cuboid([4.4, 2.2, 0.15]),
        [-9.4, f + 2.2, -17.4],
        yaw(0.35),
    ));
    s.push(drawn(
        "theatre ship",
        ship_prop_shape(0.4),
        [-7.0, f + 1.9, -15.2],
        yaw(-0.5),
    ));
    s.push(drawn(
        "theatre pillar",
        cuboid([0.18, 1.1, 0.18]),
        [-11.9, f + 1.1, -15.4],
        id,
    ));
    for (x, z) in [(-11.4, -11.4), (-9.2, -10.8), (-7.0, -11.2)] {
        s.push(drawn("lamp", ball(0.12), [x, f + 0.55, z], id));
    }
    // Beacon and the ship pad.
    s.push(drawn(
        "beacon pedestal",
        cuboid([0.3, 0.45, 0.3]),
        [-2.6, f + 0.45, -10.2],
        id,
    ));
    s.push(drawn(
        "beacon",
        cuboid([0.35; 3]),
        [-2.6, f + 1.25, -10.2],
        id,
    ));
    s.push(drawn(
        "ship pad",
        cuboid([3.2, 0.12, 3.2]),
        [14.0, f + 0.12, -15.5],
        id,
    ));
    s.push(drawn(
        "pad ship",
        ship_prop_shape(1.0),
        [14.0, f + 1.5, -15.5],
        yaw(0.9),
    ));
    s.push(drawn("key lamp", ball(0.14), [0.8, 4.4, -6.8], id));
    // The Kestrel's apron, its stripes, the floodlight mast and lamp.
    let [hx, hz] = KESTREL_APRON_HALF;
    let (kx, kz) = (KESTREL_POSITION[0], KESTREL_POSITION[2]);
    s.push(drawn(
        "apron",
        cuboid([hx as f32, 0.15, hz as f32]),
        [kx, KESTREL_APRON_TOP - 0.15, kz],
        id,
    ));
    let (width, inset) = (0.3f32, 0.3f64);
    for side in [-1.0, 1.0] {
        s.push(drawn(
            "apron stripe",
            cuboid([hx as f32, 0.02, width]),
            [kx, KESTREL_APRON_TOP, kz + side * (hz - inset)],
            id,
        ));
        s.push(drawn(
            "apron stripe",
            cuboid([width, 0.02, hz as f32 - 2.0 * width]),
            [kx + side * (hx - inset), KESTREL_APRON_TOP, kz],
            id,
        ));
    }
    const MAST: f64 = 4.2;
    let lamp = {
        let t = turn_y(
            KESTREL_YAW,
            [-10.0, f64::from((MAST - GEAR_DROP) as f32), 0.5],
        );
        [
            KESTREL_POSITION[0] + t[0],
            KESTREL_POSITION[1] + t[1],
            KESTREL_POSITION[2] + t[2],
        ]
    };
    s.push(drawn("flood lamp", ball(0.16), lamp, id));
    s.push(drawn(
        "mast",
        cuboid([0.06, (MAST / 2.0) as f32, 0.06]),
        [lamp[0], lamp[1] - MAST / 2.0 - 0.16, lamp[2]],
        id,
    ));
    s.push(StaticDef {
        groups: if KESTREL_BLOCKS_SHIPS {
            CollisionGroups::ALL
        } else {
            CollisionGroups::NOT_SHIPS
        },
        ..drawn(
            "kestrel",
            ShapeRef::Asset(KESTREL),
            KESTREL_POSITION,
            yaw(KESTREL_YAW),
        )
    });

    // ---- the showcase's own statics ----
    let [cx, _, cz] = GRAVITY_ROOM_CENTER;
    let [rx, _, rz] = GRAVITY_ROOM_HALF;
    // The curbs keep the balls in the room; players walk in and out over
    // them (a 25 cm step a ship's capsule cannot climb).
    let curb = |name, half: [f32; 3], at| StaticDef {
        groups: CollisionGroups::NOT_SHIPS,
        ..new_static(name, cuboid(half), at, id, visual::CURB)
    };
    s.push(curb(
        "curb west",
        [0.05, 0.125, rz as f32],
        [cx - rx, f + 0.125, cz],
    ));
    s.push(curb(
        "curb east",
        [0.05, 0.125, rz as f32],
        [cx + rx - 0.05, f + 0.125, cz],
    ));
    s.push(curb(
        "curb north",
        [rx as f32, 0.125, 0.05],
        [cx, f + 0.125, cz + rz],
    ));
    s.push(curb(
        "curb south",
        [rx as f32, 0.125, 0.05],
        [cx, f + 0.125, cz - rz],
    ));
    s.push(new_static(
        "bowl",
        ShapeRef::Asset(BOWL),
        BOWL_AT,
        id,
        visual::BOWL,
    ));
    let (ramp_at, ramp_rot) = ramp();
    s.push(new_static(
        "ramp",
        cuboid([0.5, 0.05, 2.0]),
        ramp_at,
        ramp_rot,
        visual::RAMP,
    ));
    s.push(new_static(
        "fulcrum",
        cuboid([0.15, 0.275, 0.3]),
        [12.2, f + 0.275, 3.5],
        id,
        visual::FULCRUM,
    ));
    for x in [13.9, 18.6] {
        s.push(new_static(
            "gantry post",
            cuboid([0.08, 2.5, 0.08]),
            [x, f + 2.5, 3.5],
            id,
            visual::GANTRY,
        ));
    }
    s.push(new_static(
        "gantry beam",
        cuboid([2.43, 0.08, 0.08]),
        [16.25, f + 5.0, 3.5],
        id,
        visual::GANTRY,
    ));
    s.push(new_static(
        "door post",
        cuboid([0.06, 1.1, 0.06]),
        [-11.0, f + 1.1, 5.0],
        id,
        visual::DOOR_POST,
    ));
    s.push(StaticDef {
        friction: 0.8,
        restitution: 0.05,
        ..new_static(
            "planetoid",
            ball(PLANETOID_RADIUS as f32),
            PLANETOID_CENTER,
            id,
            visual::PLANETOID,
        )
    });
    s
}

/// The unit vector `v` normalised.
fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / l)
}

/// The rotation `(w, x, y, z)` that turns +Y onto the unit vector `n`.
fn y_onto(n: [f64; 3]) -> [f32; 4] {
    let c = n[1].clamp(-1.0, 1.0);
    // Axis +Y × n, normalised; angle acos(n.y).
    let axis = [n[2], 0.0, -n[0]];
    let l = (axis[0] * axis[0] + axis[2] * axis[2]).sqrt();
    if l < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let half = c.acos() * 0.5;
    let (s, w) = half.sin_cos();
    [
        w as f32,
        (axis[0] / l * s) as f32,
        0.0,
        (axis[2] / l * s) as f32,
    ]
}

/// A point on the planetoid's surface (scene space) `along` metres out from
/// the surface, in the direction `polar` radians from straight up and
/// `azimuth` radians round from +X; and the surface's outward normal there.
fn on_planetoid(polar: f64, azimuth: f64, along: f64) -> ([f64; 3], [f64; 3]) {
    let n = [
        polar.sin() * azimuth.cos(),
        polar.cos(),
        polar.sin() * azimuth.sin(),
    ];
    let r = PLANETOID_RADIUS + along;
    let c = PLANETOID_CENTER;
    ([c[0] + n[0] * r, c[1] + n[1] * r, c[2] + n[2] * r], n)
}

/// The planetoid's bodies: three crates resting on its surface (35° from
/// its top, a third of a turn apart, each standing square on the ground under
/// it) and three balls in circular orbits — the speed `√(GM/r)` across the
/// radius, in three different planes.
fn planetoid_bodies() -> Vec<BodyDef> {
    let mut out = Vec::new();
    for k in 0..3 {
        let azimuth = 0.5 + f64::from(k) * std::f64::consts::TAU / 3.0;
        let (at, n) = on_planetoid(35.0f64.to_radians(), azimuth, 0.25 + 0.005);
        let mut b = crate_body(0.5);
        b.friction = 0.8;
        out.push(BodyDef {
            quat: y_onto(n),
            ..at_rest("planetoid crate", at, b, visual::CRATE)
        });
    }
    // Each orbit: the plane spanned by `u` and `w` (orthonormal), starting
    // `phase` radians round from `u`.
    let planes: [([f64; 3], [f64; 3], f64); 3] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0),
        (
            [1.0, 0.0, 0.0],
            [0.0, 50.0f64.to_radians().sin(), 50.0f64.to_radians().cos()],
            2.1,
        ),
        ([0.0, 0.0, 1.0], unit([1.0, 1.6, 0.0]), 4.2),
    ];
    for (r, (u, w, phase)) in PLANETOID_ORBITS.iter().zip(planes) {
        let v = (PLANETOID_GM / r).sqrt();
        let (s, c) = phase.sin_cos();
        let dir = [0, 1, 2].map(|k| c * u[k] + s * w[k]);
        let tangent = [0, 1, 2].map(|k| -s * u[k] + c * w[k]);
        let p = PLANETOID_CENTER;
        let mut b = ball_body(0.3);
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
        b.restitution = 0.3;
        out.push(BodyDef {
            vel: tangent.map(|t| (t * v) as f32),
            ..at_rest(
                "moon",
                [0, 1, 2].map(|k| p[k] + dir[k] * r),
                b,
                visual::BALL,
            )
        });
    }
    out
}

/// The ramp: centre and rotation (15° about X, high end towards -Z).
fn ramp() -> ([f64; 3], [f32; 4]) {
    ([17.6, FLOOR + 0.57, -1.6], about_x(15.0f32.to_radians()))
}

/// Where a ball of `radius` rests on the ramp's top face at `along` metres
/// (ramp-local z) from the ramp's centre, 1 cm clear of it.
fn on_ramp(along: f64, radius: f64) -> [f64; 3] {
    let (at, _) = ramp();
    let phi = 15.0f64.to_radians();
    let (y, z) = (0.05 + radius + 0.01, along);
    [
        at[0],
        at[1] + y * phi.cos() - z * phi.sin(),
        at[2] + y * phi.sin() + z * phi.cos(),
    ]
}

/// Where a ball of `radius` rests on the bowl's inside at horizontal
/// distance `rho` from its axis, bearing `theta`, 1 cm clear of it.
fn in_bowl(rho: f64, theta: f64, radius: f64) -> [f64; 3] {
    let y = BOWL_RADIUS - (BOWL_RADIUS * BOWL_RADIUS - rho * rho).sqrt();
    // Inward normal: towards the sphere's centre (0, R, 0).
    let n = [-rho / BOWL_RADIUS, (BOWL_RADIUS - y) / BOWL_RADIUS];
    let d = radius + 0.01;
    let (r, h) = (rho + n[0] * d, y + n[1] * d);
    [
        BOWL_AT[0] + r * theta.cos(),
        BOWL_AT[1] + h,
        BOWL_AT[2] + r * theta.sin(),
    ]
}

fn crate_body(size: f32) -> Body {
    let mut b = Body::dynamic(ShapeRef::cuboid([size; 3]));
    b.mass = MassSpec::Density(500.0);
    b.friction = 0.7;
    b.restitution = 0.05;
    b
}

fn ball_body(radius: f32) -> Body {
    let mut b = Body::dynamic(ball(radius));
    b.mass = MassSpec::Density(800.0);
    b.friction = 0.6;
    b.restitution = 0.2;
    b.angular_damping = 0.2;
    b.linear_damping = 0.05;
    b
}

fn at_rest(name: &'static str, offset: [f64; 3], body: Body, v: u32) -> BodyDef {
    BodyDef {
        name,
        offset,
        quat: [1.0, 0.0, 0.0, 0.0],
        vel: [0.0; 3],
        ang_vel: [0.0; 3],
        body,
        visual: Visual(v),
        springs: Vec::new(),
        articulation: None,
    }
}

/// The showcase's bodies, anchor-relative (see the [module docs](self)).
#[must_use]
pub fn bodies() -> Vec<BodyDef> {
    let f = FLOOR;
    let mut out = Vec::new();
    // Gravity room: a stack of five 0.6 m crates.
    for i in 0..5 {
        out.push(at_rest(
            "stack crate",
            [12.0, f + 0.3 + 0.6 * f64::from(i), -1.6],
            crate_body(0.6),
            visual::CRATE,
        ));
    }
    // A 3-2-1 pyramid of 0.5 m crates.
    for (x, level) in [
        (14.0, 0),
        (14.52, 0),
        (15.04, 0),
        (14.26, 1),
        (14.78, 1),
        (14.52, 2),
    ] {
        out.push(at_rest(
            "pyramid crate",
            [x, f + 0.25 + 0.5 * f64::from(level), -1.6],
            crate_body(0.5),
            visual::CRATE,
        ));
    }
    // Three balls at the top of the ramp.
    for along in [-1.7, -1.2, -0.7] {
        out.push(at_rest(
            "ramp ball",
            on_ramp(along, 0.2),
            ball_body(0.2),
            visual::BALL,
        ));
    }
    // Three balls on the bowl's inner slope.
    for k in 0..3 {
        let theta = f64::from(k) * std::f64::consts::TAU / 3.0 + 0.3;
        out.push(at_rest(
            "bowl ball",
            in_bowl(1.6, theta, 0.25),
            ball_body(0.25),
            visual::BALL,
        ));
    }
    // The seesaw: a plank on a hinge (about Z) to the static world just above
    // the fulcrum, and a crate dropped on one end.
    let pivot = [12.2, f + 0.6, 3.5];
    let mut plank = crate_body(1.0);
    plank.shape = cuboid([1.5, 0.04, 0.3]);
    plank.mass = MassSpec::Density(600.0);
    out.push(BodyDef {
        articulation: Some((
            Articulation {
                parts: Vec::new(),
                joints: vec![JointDef {
                    a: 0,
                    frame_a: LocalPose {
                        translation: [0.0, -0.045, 0.0],
                        rotation: X_TO_Z,
                    },
                    b: JointTarget::World {
                        pos: WorldPos::ZERO,
                        rotation: [1.0, 0.0, 0.0, 0.0],
                    },
                    frame_b: LocalPose {
                        translation: [0.0; 3],
                        rotation: X_TO_Z,
                    },
                    kind: JointKind::Revolute {
                        limits: Some([-0.3, 0.3]),
                        motor: None,
                    },
                }],
            },
            vec![Anchor::World(pivot)],
            Vec::new(),
        )),
        ..at_rest(
            "seesaw plank",
            [pivot[0], pivot[1] + 0.045, pivot[2]],
            plank,
            visual::PLANK,
        )
    });
    out.push(at_rest(
        "seesaw crate",
        [11.0, f + 1.6, 3.5],
        crate_body(0.4),
        visual::CRATE,
    ));
    // Two spring pendulums under the gantry: one swinging (released 30° out),
    // one bobbing (released below its rest point).
    let pendulum = |name, anchor: [f64; 3], start: [f64; 3]| {
        let mut b = ball_body(0.2);
        b.mass = MassSpec::Mass(4.0);
        b.linear_damping = 0.0;
        BodyDef {
            springs: vec![(
                Spring {
                    part: 0,
                    anchor: [0.0; 3],
                    target: SpringTarget::World(WorldPos::ZERO),
                    rest_length: 1.5,
                    stiffness: 200.0,
                    damping: 0.5,
                    max_force: 0.0,
                    mode: SpringMode::Spring,
                },
                Anchor::World(anchor),
            )],
            ..at_rest(name, start, b, visual::PENDULUM_BOB)
        }
    };
    out.push(pendulum(
        "swinging pendulum",
        [14.8, f + 4.9, 3.5],
        [14.8, f + 4.9 - 1.7 * 0.866, 3.5 + 0.85],
    ));
    out.push(pendulum(
        "bobbing pendulum",
        [16.0, f + 4.9, 3.5],
        [16.0, f + 2.9, 3.5],
    ));
    // A chain of six links hanging from the gantry, released 30° out so it
    // swings.
    out.push(chain([17.6, f + 4.9, 3.5], 6));
    // ---- zero-g corner ----
    for (x, y, z) in [
        (-15.5, 0.0, -5.5),
        (-13.2, 0.6, -4.0),
        (-17.0, 1.1, -1.5),
        (-12.5, -0.4, -0.5),
    ] {
        let mut b = crate_body(0.7);
        b.linear_damping = 0.02;
        b.angular_damping = 0.02;
        out.push(at_rest("floating crate", [x, y, z], b, visual::CRATE));
    }
    // Two crates on a spring, released stretched: they oscillate.
    let mut pair = crate_body(0.5);
    pair.linear_damping = 0.0;
    out.push(BodyDef {
        springs: vec![(
            Spring {
                part: 0,
                anchor: [0.25, 0.0, 0.0],
                target: SpringTarget::World(WorldPos::ZERO),
                rest_length: 1.5,
                stiffness: 40.0,
                damping: 0.2,
                max_force: 0.0,
                mode: SpringMode::Spring,
            },
            Anchor::Body("spring pair b", [-0.25, 0.0, 0.0]),
        )],
        ..at_rest(
            "spring pair a",
            [-16.0, 0.3, 3.0],
            pair.clone(),
            visual::CRATE,
        )
    });
    out.push(at_rest(
        "spring pair b",
        [-13.0, 0.3, 3.0],
        pair,
        visual::CRATE,
    ));
    // A door on a vertical hinge beside its post.
    let mut door = crate_body(1.0);
    door.shape = cuboid([0.5, 1.0, 0.03]);
    door.mass = MassSpec::Density(300.0);
    door.linear_damping = 0.05;
    door.angular_damping = 0.3;
    let hinge = [-11.08, f + 1.1, 5.0];
    out.push(BodyDef {
        articulation: Some((
            Articulation {
                parts: Vec::new(),
                joints: vec![JointDef {
                    a: 0,
                    frame_a: LocalPose {
                        translation: [0.5, 0.0, 0.0],
                        rotation: X_TO_Y,
                    },
                    b: JointTarget::World {
                        pos: WorldPos::ZERO,
                        rotation: [1.0, 0.0, 0.0, 0.0],
                    },
                    frame_b: LocalPose {
                        translation: [0.0; 3],
                        rotation: X_TO_Y,
                    },
                    kind: JointKind::Revolute {
                        limits: Some([-1.7, 1.7]),
                        motor: None,
                    },
                }],
            },
            vec![Anchor::World(hinge)],
            Vec::new(),
        )),
        ..at_rest(
            "door",
            [hinge[0] - 0.5, hinge[1], hinge[2]],
            door,
            visual::DOOR,
        )
    });
    // ---- the planetoid ----
    out.extend(planetoid_bodies());
    out
}

/// A chain of `links` capsule links (0.5 m apart) hanging from `top`, tilted
/// 30° towards +Z.
fn chain(top: [f64; 3], links: usize) -> BodyDef {
    let link = || {
        let mut b = Body::dynamic(ShapeRef::Capsule {
            half_height: 0.15,
            radius: 0.05,
        });
        b.mass = MassSpec::Density(2000.0);
        b.angular_damping = 0.3;
        b.linear_damping = 0.05;
        b
    };
    let tilt = -30.0f32.to_radians();
    let dir = [0.0, -f64::from(tilt.cos()), -f64::from(tilt.sin())];
    let center = |i: usize| {
        let d = 0.25 + 0.5 * i as f64;
        [
            top[0] + dir[0] * d,
            top[1] + dir[1] * d,
            top[2] + dir[2] * d,
        ]
    };
    let quat = about_x(tilt);
    let mut joints = vec![JointDef {
        a: 0,
        frame_a: LocalPose::at([0.0, 0.25, 0.0]),
        b: JointTarget::World {
            pos: WorldPos::ZERO,
            rotation: [1.0, 0.0, 0.0, 0.0],
        },
        frame_b: LocalPose::IDENTITY,
        kind: JointKind::Spherical,
    }];
    let mut anchors = vec![Anchor::World(top)];
    for i in 1..links {
        joints.push(JointDef {
            a: (i - 1) as u8,
            frame_a: LocalPose::at([0.0, -0.25, 0.0]),
            b: JointTarget::Part(i as u8),
            frame_b: LocalPose::at([0.0, 0.25, 0.0]),
            kind: JointKind::Spherical,
        });
        anchors.push(Anchor::Part);
    }
    let parts = (1..links)
        .map(|_| PartDef {
            body: link(),
            visual: Visual(visual::CHAIN_LINK),
        })
        .collect();
    let part_states = (1..links).map(|i| (center(i), quat)).collect();
    BodyDef {
        quat,
        articulation: Some((Articulation { parts, joints }, anchors, part_states)),
        ..at_rest("chain", center(0), link(), visual::CHAIN_LINK)
    }
}

/// The demo area's gravity: the gravity room and the planetoid.
#[must_use]
pub fn gravity() -> Vec<GravityDef> {
    vec![
        GravityDef::Zone {
            center: GRAVITY_ROOM_CENTER,
            half_extents: GRAVITY_ROOM_HALF,
            gravity: [0.0, -9.81, 0.0],
            soft_edge: 0.25,
        },
        GravityDef::Body {
            reach: PLANETOID_REACH,
            name: "planetoid",
            gm: PLANETOID_GM,
            radius: PLANETOID_RADIUS,
            center: PLANETOID_CENTER,
        },
    ]
}
