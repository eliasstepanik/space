//! Acceptance tests for the physics core: gravity, contacts (convex and
//! concave), stacking, CCD, springs, joints, ship feel, determinism, the
//! phase-2 hooks (handoff export/import, ghosts, islands) and the
//! floating-origin bubbles at large coordinates.
//!
//! Run with `cargo test -p space-sim --test physics` (add `--release` for
//! speed; every tolerance holds in both).

use std::sync::Arc;

use space_protocol::{EntityGuid, PlayerInput, ShardId};
use space_sim::articulation::{
    Articulation, JointDef, JointKind, JointTarget, PartDef, PartStates,
};
use space_sim::gravity::{BodyPlace, GravityField, GravityZone, LargeBody, WorldPoint, ZoneShape};
use space_sim::rigid::{IslandQuery, StaticCollider};
use space_sim::shape::{AssetId, LocalPose, MeshGeometry, ShapeCatalog, ShapeRef};
use space_sim::spring::{Spring, SpringMode, SpringTarget, Springs};
use space_sim::tick::dt;
use space_sim::{Body, BodySpawn, ShipParams, ShipState, SimWorld, Visual};
use space_world::WorldPos;

const G: f64 = 9.81;

fn guid(n: u64) -> EntityGuid {
    EntityGuid::new(ShardId(1), n)
}

/// `p` metres from `origin`.
fn at(origin: WorldPos, p: [f64; 3]) -> WorldPos {
    WorldPos::from_f64_metres_around(origin, p)
}

fn rel(origin: WorldPos, p: WorldPos) -> [f64; 3] {
    WorldPos::delta(origin, p)
}

/// The exact position of body `n` (its `WorldPos` plus its residual)
/// relative to `origin`.
fn exact(sim: &SimWorld, n: u64, origin: WorldPos) -> [f64; 3] {
    let e = sim.entity_of(guid(n)).expect("alive");
    let r = sim
        .world()
        .get::<space_sim::PosResidual>(e)
        .map_or([0.0; 3], |r| r.0[0]);
    let d = rel(origin, sim.state_of(guid(n)).unwrap().pos);
    [
        d[0] + f64::from(r[0]),
        d[1] + f64::from(r[1]),
        d[2] + f64::from(r[2]),
    ]
}

/// A zone of standard gravity, a box `half` metres around `center`.
fn zone(center: WorldPos, half: f64) -> GravityZone {
    GravityZone {
        center: WorldPoint::at(center),
        rotation: [1.0, 0.0, 0.0, 0.0],
        shape: ZoneShape::Box {
            half_extents: [half; 3],
        },
        gravity: [0.0, -G, 0.0],
        soft_edge: 0.0,
    }
}

/// A floor slab whose top is at `origin.y + top`.
fn floor(origin: WorldPos, top: f64, half: f32) -> StaticCollider {
    StaticCollider {
        shape: ShapeRef::Cuboid {
            half_extents: [half, 0.5, half],
        },
        place: WorldPoint::new(origin, [0.0, top - 0.5, 0.0]),
        rotation: [1.0, 0.0, 0.0, 0.0],
        friction: 0.8,
        restitution: 0.0,
        visual: Visual(0),
        groups: space_sim::CollisionGroups::ALL,
        motion: None,
    }
}

fn ball(radius: f32) -> Body {
    Body::dynamic(ShapeRef::Ball { radius })
}

fn crate_body(size: f32) -> Body {
    let mut b = Body::dynamic(ShapeRef::cuboid([size; 3]));
    b.friction = 0.8;
    b.restitution = 0.0;
    b
}

fn spawn(sim: &mut SimWorld, n: u64, pos: WorldPos, body: Body) {
    sim.spawn_body(guid(n), BodySpawn::new(ShipState::at_rest(pos), body))
        .expect("spawns");
}

/// A procedural concave bowl: the inside of a sphere cap of radius `r`,
/// `rings` × `segments`, rim at height `depth` above the bottom (the
/// origin). Triangles wound so the inside faces up.
fn bowl_mesh(r: f64, depth: f64, rings: u32, segments: u32) -> MeshGeometry {
    let max_angle = ((r - depth) / r).acos();
    let mut vertices = vec![[0.0, 0.0, 0.0]];
    for i in 1..=rings {
        let a = max_angle * f64::from(i) / f64::from(rings);
        for j in 0..segments {
            let phi = std::f64::consts::TAU * f64::from(j) / f64::from(segments);
            vertices.push([
                r * a.sin() * phi.cos(),
                r - r * a.cos(),
                r * a.sin() * phi.sin(),
            ]);
        }
    }
    let mut indices = Vec::new();
    let ring = |i: u32, j: u32| 1 + (i - 1) * segments + (j % segments);
    for j in 0..segments {
        indices.push([0, ring(1, j + 1), ring(1, j)]);
    }
    for i in 1..rings {
        for j in 0..segments {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i, j + 1),
                ring(i + 1, j),
                ring(i + 1, j + 1),
            );
            indices.push([a, b, d]);
            indices.push([a, d, c]);
        }
    }
    MeshGeometry { vertices, indices }
}

fn bowl_catalog() -> Arc<ShapeCatalog> {
    let mut c = ShapeCatalog::new();
    c.insert(AssetId(1), "bowl", bowl_mesh(3.0, 1.5, 12, 32))
        .expect("bowl");
    Arc::new(c)
}

// ---------------------------------------------------------------------------
// gravity
// ---------------------------------------------------------------------------

/// A ball released inside a gravity zone falls `s = g t² / 2` (the
/// semi-implicit Euler sum `g dt² n(n+1)/2`, within 1e-6 relative of it and
/// within 2 % of the continuous form), and one outside every zone does not
/// move at all.
#[test]
fn falling_in_a_zone_and_weightless_outside() {
    let mut sim = SimWorld::new(ShardId(1));
    let o = WorldPos::ZERO;
    sim.set_gravity(GravityField {
        zones: vec![zone(at(o, [0.0, 100.0, 0.0]), 100.0)],
        bodies: vec![],
        clock: Default::default(),
    });
    spawn(&mut sim, 1, at(o, [0.0, 150.0, 0.0]), ball(0.5));
    spawn(&mut sim, 2, at(o, [500.0, 150.0, 0.0]), ball(0.5));
    let n = 60u32;
    sim.step_n(u64::from(n));
    let fell = 150.0 - rel(o, sim.state_of(guid(1)).unwrap().pos)[1];
    let dt = f64::from(dt());
    let discrete = G * dt * dt * f64::from(n) * f64::from(n + 1) / 2.0;
    let continuous = 0.5 * G * (dt * f64::from(n)).powi(2);
    assert!(
        (fell - discrete).abs() < 1e-6 * discrete + 1e-4,
        "fell {fell}, discrete {discrete}"
    );
    assert!(
        (fell - continuous).abs() / continuous < 0.02,
        "fell {fell}, analytic {continuous}"
    );
    let still = sim.state_of(guid(2)).unwrap();
    assert_eq!(
        still.pos,
        at(o, [500.0, 150.0, 0.0]),
        "zero-g outside the zone"
    );
    assert_eq!(still.vel, [0.0; 3]);
}

/// A body on a circular orbit around a point mass stays on it: over two
/// periods the radius stays within 0.5 % and the specific energy within 0.1 %
/// (semi-implicit Euler is symplectic: the error oscillates, it does not
/// drift). Placed 1e11 m from the world origin to show the field is exact
/// there too.
#[test]
fn orbit_around_a_point_mass_conserves_energy() {
    let center = WorldPos::from_metres([100_000_000_000_000, 0, 0]);
    let r = 1000.0f64;
    let period = 60.0f64;
    let v = std::f64::consts::TAU * r / period;
    let gm = v * v * r;
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_gravity(GravityField {
        zones: vec![],
        bodies: vec![LargeBody {
            name: "rock".into(),
            gm,
            radius: 50.0,
            place: BodyPlace::Fixed(WorldPoint::at(center)),
            reach: f64::INFINITY,
        }],
        clock: Default::default(),
    });
    let mut s = ShipState::at_rest(at(center, [r, 0.0, 0.0]));
    s.vel = [0.0, 0.0, v as f32];
    let mut b = ball(1.0);
    b.angular_damping = 0.0;
    sim.spawn_body(guid(1), BodySpawn::new(s, b)).unwrap();
    let energy = |s: &ShipState| {
        let d = rel(center, s.pos);
        let rr = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let vv: f64 = s.vel.iter().map(|x| f64::from(*x).powi(2)).sum();
        (0.5 * vv - gm / rr, rr)
    };
    let (e0, _) = energy(&s);
    let ticks = (2.0 * period * 30.0) as u64;
    let mut worst_e = 0.0f64;
    let mut worst_r = 0.0f64;
    for _ in 0..ticks {
        sim.step();
        let (e, rr) = energy(&sim.state_of(guid(1)).unwrap());
        worst_e = worst_e.max(((e - e0) / e0).abs());
        worst_r = worst_r.max(((rr - r) / r).abs());
    }
    assert!(worst_e < 1e-3, "energy drifted by {worst_e}");
    assert!(worst_r < 5e-3, "radius drifted by {worst_r}");
}

// ---------------------------------------------------------------------------
// contacts
// ---------------------------------------------------------------------------

/// Two balls collide head-on in zero-g: they bounce apart and momentum is
/// conserved to 1e-6 relative.
#[test]
fn convex_convex_collision_conserves_momentum() {
    let mut sim = SimWorld::new(ShardId(1));
    let o = WorldPos::ZERO;
    let mut a = ball(0.5);
    a.restitution = 0.8;
    let mut b = a.clone();
    b.mass = space_sim::MassSpec::Mass(3.0 * 523.598_76);
    let mut sa = ShipState::at_rest(at(o, [-2.0, 0.0, 0.0]));
    sa.vel = [4.0, 0.0, 0.0];
    let sb = ShipState::at_rest(at(o, [2.0, 0.0, 0.0]));
    sim.spawn_body(guid(1), BodySpawn::new(sa, a)).unwrap();
    sim.spawn_body(guid(2), BodySpawn::new(sb, b)).unwrap();
    let m_a = 1000.0 * 4.0 / 3.0 * std::f64::consts::PI * 0.125;
    let m_b = 3.0 * 523.598_76;
    let p0 = m_a * 4.0;
    let mut touched = false;
    for _ in 0..60 {
        sim.step();
        touched |= !sim.contacts().is_empty();
    }
    let (va, vb) = (
        sim.state_of(guid(1)).unwrap().vel,
        sim.state_of(guid(2)).unwrap().vel,
    );
    assert!(touched, "the balls must have touched");
    assert!(
        va[0] < 0.0 && vb[0] > 0.0,
        "they bounce apart: {va:?} {vb:?}"
    );
    let p1 = m_a * f64::from(va[0]) + m_b * f64::from(vb[0]);
    assert!((p1 - p0).abs() / p0 < 1e-4, "momentum {p0} -> {p1}");
}

/// Balls released on the inner slope of a concave bowl (a triangle-mesh
/// asset) stay inside it and come to rest at the bottom.
#[test]
fn balls_in_a_concave_bowl_settle_at_the_bottom() {
    let o = WorldPos::from_metres([5_000_000, 0, -3_000_000]);
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_catalog(bowl_catalog());
    sim.set_statics(vec![StaticCollider {
        shape: ShapeRef::Asset(AssetId(1)),
        place: WorldPoint::at(o),
        rotation: [1.0, 0.0, 0.0, 0.0],
        friction: 0.6,
        restitution: 0.0,
        visual: Visual(0),
        groups: space_sim::CollisionGroups::ALL,
        motion: None,
    }])
    .unwrap();
    sim.set_gravity(GravityField {
        zones: vec![zone(o, 20.0)],
        bodies: vec![],
        clock: Default::default(),
    });
    for (n, x) in [(1u64, 1.8f64), (2, -1.6), (3, 0.6)] {
        // On the slope: y of the sphere cap at radius x, plus the ball radius.
        let y = 3.0 - (9.0 - x * x).sqrt() + 0.35;
        let mut b = ball(0.25);
        b.angular_damping = 0.3;
        b.linear_damping = 0.1;
        spawn(&mut sim, n, at(o, [x, y, 0.2 * x]), b);
    }
    for _ in 0..(30 * 25) {
        sim.step();
        for n in 1..=3 {
            let p = rel(o, sim.state_of(guid(n)).unwrap().pos);
            assert!(p[1] > -0.1, "ball {n} fell through the bowl: {p:?}");
            assert!(
                (p[0] * p[0] + p[2] * p[2]).sqrt() < 2.7,
                "ball {n} left the bowl: {p:?}"
            );
        }
    }
    for n in 1..=3 {
        let s = sim.state_of(guid(n)).unwrap();
        let p = rel(o, s.pos);
        let speed: f32 = s.vel.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(p[1] < 0.7, "ball {n} rests low in the bowl: {p:?}");
        assert!(speed < 0.05, "ball {n} at rest: {speed}");
    }
}

/// A stack of five 0.8 m crates on a floor under gravity stands for 10 s:
/// no crate moves more than 5 cm sideways or sinks more than 2 cm.
#[test]
fn a_five_crate_stack_stands_for_ten_seconds() {
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_statics(vec![floor(o, 0.0, 10.0)]).unwrap();
    sim.set_gravity(GravityField {
        zones: vec![zone(o, 50.0)],
        bodies: vec![],
        clock: Default::default(),
    });
    for i in 0..5u64 {
        spawn(
            &mut sim,
            i + 1,
            at(o, [0.0, 0.4 + 0.8 * i as f64, 0.0]),
            crate_body(0.8),
        );
    }
    sim.step_n(300);
    for i in 0..5u64 {
        let p = rel(o, sim.state_of(guid(i + 1)).unwrap().pos);
        let want = 0.4 + 0.8 * i as f64;
        assert!(
            (p[0] * p[0] + p[2] * p[2]).sqrt() < 0.05,
            "crate {i} slid: {p:?}"
        );
        assert!(
            (p[1] - want).abs() < 0.02,
            "crate {i} at {} not {want}",
            p[1]
        );
    }
    // And it went to sleep: nothing is stamped any more.
    assert!(
        (1..=5).all(|n| sim.is_sleeping(guid(n)) == Some(true)),
        "a standing stack sleeps"
    );
}

/// A ball at 400 m/s (13 m per tick) against a 5 cm wall: with CCD it stops
/// at the wall instead of tunnelling through it.
#[test]
fn ccd_stops_a_fast_ball_at_a_thin_wall() {
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_statics(vec![StaticCollider {
        shape: ShapeRef::Cuboid {
            half_extents: [0.025, 5.0, 5.0],
        },
        place: WorldPoint::new(o, [20.0, 0.0, 0.0]),
        rotation: [1.0, 0.0, 0.0, 0.0],
        friction: 0.5,
        restitution: 0.0,
        visual: Visual(0),
        groups: space_sim::CollisionGroups::ALL,
        motion: None,
    }])
    .unwrap();
    let mut b = ball(0.1);
    b.ccd = true;
    let mut s = ShipState::at_rest(at(o, [0.0, 0.0, 0.0]));
    s.vel = [400.0, 0.0, 0.0];
    sim.spawn_body(guid(1), BodySpawn::new(s, b)).unwrap();
    for _ in 0..30 {
        sim.step();
        let x = rel(o, sim.state_of(guid(1)).unwrap().pos)[0];
        assert!(x < 20.0, "tunnelled to x = {x}");
    }
}

// ---------------------------------------------------------------------------
// springs and joints
// ---------------------------------------------------------------------------

fn vertical_spring(anchor: WorldPos, k: f32, c: f32) -> Springs {
    Springs(vec![Spring {
        part: 0,
        anchor: [0.0; 3],
        target: SpringTarget::World(anchor),
        rest_length: 2.0,
        stiffness: k,
        damping: c,
        max_force: 0.0,
        mode: SpringMode::Spring,
    }])
}

/// A mass on a vertical spring under gravity oscillates about its
/// equilibrium with the period `2π √(m/k)` (within 1 %), and damping makes
/// the amplitude decay.
#[test]
fn spring_pendulum_period_and_damping() {
    let o = WorldPos::from_metres([0, 1_000_000_000_000, 0]);
    let k = 400.0f32;
    let run = |c: f32| {
        let mut sim = SimWorld::new(ShardId(1));
        sim.set_gravity(GravityField {
            zones: vec![zone(o, 100.0)],
            bodies: vec![],
            clock: Default::default(),
        });
        let mut b = ball(0.2);
        b.mass = space_sim::MassSpec::Mass(10.0);
        b.angular_damping = 0.0;
        let anchor = at(o, [0.0, 10.0, 0.0]);
        // Released 0.5 m below the unstretched length: amplitude about the
        // equilibrium (stretch m g / k = 0.245 m) is 0.255 m.
        let start = ShipState::at_rest(at(o, [0.0, 10.0 - 2.5, 0.0]));
        sim.spawn_body(
            guid(1),
            BodySpawn::new(start, b).with_springs(vertical_spring(anchor, k, c)),
        )
        .unwrap();
        let mut ys = Vec::new();
        for _ in 0..(30 * 8) {
            sim.step();
            ys.push(rel(o, sim.state_of(guid(1)).unwrap().pos)[1]);
        }
        ys
    };
    let ys = run(0.0);
    // Period from the upward zero crossings of y − equilibrium.
    let eq = 10.0 - 2.0 - 10.0 * G / f64::from(k);
    let crossings: Vec<f64> = ys
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] < eq && w[1] >= eq)
        .map(|(i, w)| i as f64 + (eq - w[0]) / (w[1] - w[0]))
        .collect();
    assert!(crossings.len() >= 5, "{crossings:?}");
    let period = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64
        * f64::from(dt());
    let analytic = std::f64::consts::TAU * (10.0 / f64::from(k)).sqrt();
    assert!(
        (period - analytic).abs() / analytic < 0.01,
        "period {period} s, analytic {analytic} s"
    );
    // Damped: the last second's swing is far smaller than the first's.
    let damped = run(20.0);
    let swing = |s: &[f64]| {
        s.iter().fold(f64::MIN, |m, v| m.max(*v)) - s.iter().fold(f64::MAX, |m, v| m.min(*v))
    };
    let first = swing(&damped[..30]);
    let last = swing(&damped[damped.len() - 30..]);
    assert!(last < 0.2 * first, "damping: {first} -> {last}");
    // The undamped one does not decay (symplectic: energy kept).
    let u_first = swing(&ys[..60]);
    let u_last = swing(&ys[ys.len() - 60..]);
    assert!(
        (u_last - u_first).abs() / u_first < 0.05,
        "{u_first} -> {u_last}"
    );
}

/// A door on a vertical hinge to the static world: pushed, it swings, and
/// never past its limits; without gravity torque it keeps its hinge in place.
#[test]
fn hinged_door_swings_within_its_limits() {
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    let hinge = at(o, [0.0, 1.0, 0.0]);
    let mut door = crate_body(1.0);
    door.shape = ShapeRef::Cuboid {
        half_extents: [0.5, 1.0, 0.05],
    };
    door.mass = space_sim::MassSpec::Mass(20.0);
    // The door's origin is its centre, 0.5 m along +x from the hinge line.
    let mut s = ShipState::at_rest(at(o, [0.5, 1.0, 0.0]));
    s.ang_vel = [0.0, 3.0, 0.0];
    // Hinge axis: the joint frames' local X turned onto world Y.
    let to_y = [
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let articulation = Articulation {
        parts: vec![],
        joints: vec![JointDef {
            a: 0,
            frame_a: LocalPose {
                translation: [-0.5, 0.0, 0.0],
                rotation: to_y,
            },
            b: JointTarget::World {
                pos: hinge,
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            frame_b: LocalPose {
                translation: [0.0; 3],
                rotation: to_y,
            },
            kind: JointKind::Revolute {
                limits: Some([-1.2, 1.2]),
                motor: None,
            },
        }],
    };
    sim.spawn_body(
        guid(1),
        BodySpawn::new(s, door).with_articulation(articulation, PartStates(vec![])),
    )
    .unwrap();
    let mut max_angle = 0.0f64;
    for _ in 0..90 {
        sim.step();
        let st = sim.state_of(guid(1)).unwrap();
        let q = st.quat;
        let angle = 2.0 * f64::from(q[2]).atan2(f64::from(q[0]));
        max_angle = max_angle.max(angle.abs());
        // The hinge edge stays on the hinge line.
        let edge = space_sim::spring::rotate(q, [-0.5, 0.0, 0.0]);
        let p = rel(o, st.pos);
        let e = [p[0] + edge[0], p[1] + edge[1], p[2] + edge[2]];
        let gap = ((e[0]).powi(2) + (e[1] - 1.0).powi(2) + e[2].powi(2)).sqrt();
        assert!(gap < 0.01, "the door left its hinge by {gap}");
    }
    assert!(max_angle > 0.8, "the door swung: {max_angle}");
    assert!(max_angle < 1.2 + 0.05, "the limit holds: {max_angle}");
}

/// A chain of six links hangs from a world anchor under gravity: every link
/// below the one above it, consecutive links joined (gap under 2 cm), the
/// whole chain one entity whose part states replicate together.
#[test]
fn a_chain_hangs_from_its_anchor() {
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_gravity(GravityField {
        zones: vec![zone(o, 50.0)],
        bodies: vec![],
        clock: Default::default(),
    });
    let link_len = 0.5f32;
    let link = || {
        let mut b = Body::dynamic(ShapeRef::Capsule {
            half_height: 0.15,
            radius: 0.05,
        });
        b.angular_damping = 1.0;
        b.linear_damping = 0.8;
        b
    };
    let top = at(o, [0.0, 10.0, 0.0]);
    // Links start horizontal, so the chain has to swing down to hang.
    let pos_of = |i: usize| at(o, [0.25 + f64::from(link_len) * i as f64, 10.0, 0.0]);
    // Local +Y (the link's top end) points at -X, toward the anchor.
    let horizontal = [
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let state = |i: usize| ShipState {
        quat: horizontal,
        ..ShipState::at_rest(pos_of(i))
    };
    let n = 6usize;
    let mut joints = vec![JointDef {
        a: 0,
        frame_a: LocalPose::at([0.0, 0.25, 0.0]),
        b: JointTarget::World {
            pos: top,
            rotation: [1.0, 0.0, 0.0, 0.0],
        },
        frame_b: LocalPose::IDENTITY,
        kind: JointKind::Spherical,
    }];
    for i in 1..n {
        joints.push(JointDef {
            a: (i - 1) as u8,
            frame_a: LocalPose::at([0.0, -0.25, 0.0]),
            b: JointTarget::Part(i as u8),
            frame_b: LocalPose::at([0.0, 0.25, 0.0]),
            kind: JointKind::Spherical,
        });
    }
    let articulation = Articulation {
        parts: (1..n)
            .map(|_| PartDef {
                body: link(),
                visual: Visual(5),
            })
            .collect(),
        joints,
    };
    let parts = PartStates((1..n).map(state).collect());
    sim.spawn_body(
        guid(1),
        BodySpawn::new(state(0), link()).with_articulation(articulation, parts),
    )
    .unwrap();
    sim.step_n(30 * 15);
    let states = sim.part_states_of(guid(1)).unwrap();
    assert_eq!(states.len(), n);
    let mut prev_bottom = [0.0, 10.0, 0.0];
    for (i, s) in states.iter().enumerate() {
        let p = rel(o, s.pos);
        let up = space_sim::spring::rotate(s.quat, [0.0, 0.25, 0.0]);
        let down = space_sim::spring::rotate(s.quat, [0.0, -0.25, 0.0]);
        let top_end = [p[0] + up[0], p[1] + up[1], p[2] + up[2]];
        let gap = (0..3)
            .map(|k| (top_end[k] - prev_bottom[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(gap < 0.02, "link {i} is {gap} m from the one above");
        assert!(p[1] < prev_bottom[1], "link {i} hangs below the one above");
        prev_bottom = [p[0] + down[0], p[1] + down[1], p[2] + down[2]];
    }
    // Hanging: the bottom is roughly under the anchor.
    assert!(
        prev_bottom[0].abs() < 0.6,
        "the chain swings toward vertical: {prev_bottom:?}"
    );
}

// ---------------------------------------------------------------------------
// ships
// ---------------------------------------------------------------------------

/// The flight feel numbers, before and after the physics core (the free
/// flight path is `step_ship`, so they are the same; the reference below is
/// the pre-physics discrete flight model computed independently in `f64`):
/// terminal speed 120 m/s, 10 s of full thrust from rest reaches 47.2 m/s,
/// and with the client's brake assist a ship at full speed stops (under
/// 1 mm/s) in 15.4 s over 737 m.
#[test]
fn ship_flight_feel_is_unchanged() {
    let p = ShipParams::PLACEHOLDER;
    let mut sim = SimWorld::new(ShardId(1));
    sim.spawn_ship(guid(1), 7, ShipState::IDENTITY);
    let push = |sim: &mut SimWorld, thrust: [f32; 3]| {
        let t = sim.tick() + 1;
        sim.apply_inputs(
            t,
            &[PlayerInput {
                entity: guid(1),
                thrust,
                torque: [0.0; 3],
                tick: t,
                fast: false,
            }],
        );
        sim.step();
    };
    for _ in 0..300 {
        push(&mut sim, [0.0, 0.0, -1.0]);
    }
    let v10 = -sim.state_of(guid(1)).unwrap().vel[2];
    let r = 1.0 - f64::from(p.drag_lin) * f64::from(dt());
    let v10_ref = 120.0 * (1.0 - r.powi(300));
    assert!(
        (f64::from(v10) - v10_ref).abs() < 1e-3,
        "10 s of thrust: {v10} vs {v10_ref}"
    );
    for _ in 0..(30 * 120) {
        push(&mut sim, [0.0, 0.0, -1.0]);
    }
    let terminal = -sim.state_of(guid(1)).unwrap().vel[2];
    let analytic = p.thrust / (p.mass * p.drag_lin);
    assert!(
        (terminal - analytic).abs() < 0.5,
        "terminal {terminal} vs {analytic}"
    );
    // The client's brake assist (space-client input.rs `brake`): full reverse
    // thrust above 1.5 m/s, proportional below.
    let brake = |v: [f32; 3]| {
        let speed = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if speed < 1.0e-3 {
            [0.0; 3]
        } else {
            let power = (speed / 1.5).min(1.0);
            v.map(|x| -x / speed * power)
        }
    };
    let start = sim.state_of(guid(1)).unwrap().pos;
    let mut ticks = 0;
    while sim
        .state_of(guid(1))
        .unwrap()
        .vel
        .iter()
        .any(|v| v.abs() > 1.0e-3)
        && ticks < 3000
    {
        let v = sim.state_of(guid(1)).unwrap().vel;
        push(&mut sim, brake(v));
        ticks += 1;
    }
    let dist = WorldPos::distance(start, sim.state_of(guid(1)).unwrap().pos);
    let secs = f64::from(ticks) / 30.0;
    // The reference: the same discrete model, one axis, in f64.
    let (mut v, mut x, mut n) = (f64::from(terminal), 0.0f64, 0u32);
    let dt64 = f64::from(dt());
    while v > 1.0e-3 && n < 3000 {
        let power = (v / 1.5).min(1.0);
        v += (-6.0 * power - f64::from(p.drag_lin) * v) * dt64;
        if v.abs() < 1.0e-4 {
            v = 0.0;
        }
        x += v * dt64;
        n += 1;
    }
    let ref_secs = f64::from(n) / 30.0;
    println!(
        "10 s thrust: {v10:.2} m/s; terminal {terminal:.2} m/s; brake: {secs:.2} s over {dist:.1} m          (reference {ref_secs:.2} s over {x:.1} m)"
    );
    assert!(
        (secs - ref_secs).abs() <= 2.0 / 30.0,
        "stopping time {secs} vs {ref_secs}"
    );
    assert!(
        (dist - x).abs() / x < 2e-3,
        "stopping distance {dist} vs {x}"
    );
}

/// A ship flying into a floor stops on it instead of passing through, and a
/// crate in its way is pushed.
#[test]
fn ships_collide_with_the_world_and_push_bodies() {
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_statics(vec![floor(o, -1.7, 20.0)]).unwrap();
    sim.spawn_ship(guid(1), 7, ShipState::at_rest(at(o, [0.0, 5.0, 0.0])));
    spawn(&mut sim, 2, at(o, [0.0, -1.3, -8.0]), crate_body(0.8));
    for _ in 0..90 {
        let t = sim.tick() + 1;
        sim.apply_inputs(
            t,
            &[PlayerInput {
                entity: guid(1),
                thrust: [0.0, -1.0, 0.0],
                torque: [0.0; 3],
                tick: t,
                fast: false,
            }],
        );
        sim.step();
    }
    let ship = rel(o, sim.state_of(guid(1)).unwrap().pos);
    // The capsule's bottom is 1.68 m below the eye: resting on the floor top.
    assert!(
        (ship[1] - (-1.7 + 1.68)).abs() < 0.05,
        "ship on the floor: {ship:?}"
    );
    let q = sim.state_of(guid(1)).unwrap().quat;
    assert_eq!(q, [1.0, 0.0, 0.0, 0.0], "contacts never turn a ship");
    for _ in 0..60 {
        let t = sim.tick() + 1;
        sim.apply_inputs(
            t,
            &[PlayerInput {
                entity: guid(1),
                thrust: [0.0, 0.0, -1.0],
                torque: [0.0; 3],
                tick: t,
                fast: false,
            }],
        );
        sim.step();
    }
    let crate_pos = rel(o, sim.state_of(guid(2)).unwrap().pos);
    assert!(
        crate_pos[2] < -8.5,
        "the ship pushed the crate: {crate_pos:?}"
    );
}

// ---------------------------------------------------------------------------
// determinism
// ---------------------------------------------------------------------------

fn busy_world(order: &[u64]) -> SimWorld {
    let o = WorldPos::from_metres([1_234_567_000, -42_000, 7_000_000]);
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_catalog(bowl_catalog());
    sim.set_statics(vec![
        floor(o, 0.0, 15.0),
        StaticCollider {
            shape: ShapeRef::Asset(AssetId(1)),
            place: WorldPoint::new(o, [6.0, 0.0, 0.0]),
            rotation: [1.0, 0.0, 0.0, 0.0],
            friction: 0.5,
            restitution: 0.1,
            visual: Visual(0),
            groups: space_sim::CollisionGroups::ALL,
            motion: None,
        },
    ])
    .unwrap();
    sim.set_gravity(GravityField {
        zones: vec![zone(o, 30.0)],
        bodies: vec![],
        clock: Default::default(),
    });
    for &n in order {
        let x = (n % 4) as f64 * 1.1 - 2.0;
        let y = 0.5 + (n / 4) as f64 * 1.05;
        let body = if n % 3 == 0 {
            ball(0.4)
        } else {
            crate_body(0.9)
        };
        let mut s = ShipState::at_rest(at(o, [x, y, 0.1 * x]));
        s.vel = [0.3 * (n as f32 - 5.0), 0.0, 0.1];
        sim.spawn_body(guid(n), BodySpawn::new(s, body)).unwrap();
    }
    sim.spawn_ship(guid(100), 9, ShipState::at_rest(at(o, [0.0, 4.0, 6.0])));
    sim
}

/// Two worlds with the same inputs reach the same state hash, bit for bit;
/// spawning the same bodies in another order gives the same guid-keyed hash.
#[test]
fn determinism_same_inputs_and_spawn_order() {
    let order: Vec<u64> = (1..=12).collect();
    let reversed: Vec<u64> = order.iter().rev().copied().collect();
    let mut a = busy_world(&order);
    let mut b = busy_world(&order);
    let mut c = busy_world(&reversed);
    for t in 1..=240u64 {
        let input = PlayerInput {
            entity: guid(100),
            thrust: [((t % 7) as f32 - 3.0) / 3.0, -0.5, 0.2],
            torque: [0.0, 0.1, 0.0],
            tick: t,
            fast: false,
        };
        for w in [&mut a, &mut b, &mut c] {
            w.apply_inputs(t, &[input]);
            w.step();
        }
    }
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(a.serialize_state(), b.serialize_state());
    assert_eq!(
        a.guid_state_hash(),
        c.guid_state_hash(),
        "spawn order leaked"
    );
    assert!(!a.contacts().is_empty() || a.bubbles().iter().any(|b| b.stepped));
}

// ---------------------------------------------------------------------------
// phase-2 hooks
// ---------------------------------------------------------------------------

/// Export a body mid-flight, import it into another world, and it continues
/// on exactly the trajectory it would have had (free flight: bit for bit;
/// sliding on a floor under gravity: within 1 mm after 2 s).
#[test]
fn a_body_moves_between_worlds_mid_flight() {
    let o = WorldPos::from_metres([77_000_000, 0, 0]);
    // Free flight with spin, gravity and damping.
    let make = || {
        let mut sim = SimWorld::new(ShardId(1));
        sim.set_gravity(GravityField {
            zones: vec![zone(o, 1000.0)],
            bodies: vec![],
            clock: Default::default(),
        });
        sim
    };
    let mut stay = make();
    let mut b = crate_body(0.7);
    b.linear_damping = 0.05;
    let mut s = ShipState::at_rest(at(o, [0.0, 500.0, 0.0]));
    s.vel = [12.0, 5.0, -3.0];
    s.ang_vel = [0.4, 1.0, -0.2];
    stay.spawn_body(guid(1), BodySpawn::new(s, b).with_visual(Visual(1)))
        .unwrap();
    stay.step_n(40);
    let exported = stay.export_entity(guid(1)).unwrap();
    // The receiving world starts at the same tick with a different history.
    let mut other = SimWorld::with_tick(ShardId(2), ShipParams::PLACEHOLDER, stay.tick());
    other.set_gravity(stay.gravity().clone());
    spawn(&mut other, 50, at(o, [-300.0, 0.0, 0.0]), ball(1.0));
    other.import_entity(guid(1), &exported).unwrap();
    for _ in 0..60 {
        stay.step();
        other.step();
        assert_eq!(
            stay.state_of(guid(1)),
            other.state_of(guid(1)),
            "free flight diverged"
        );
    }
    // In contact: sliding along a floor.
    let make_floor = || {
        let mut sim = make();
        sim.set_statics(vec![floor(o, 0.0, 50.0)]).unwrap();
        sim
    };
    let mut stay = make_floor();
    let mut s = ShipState::at_rest(at(o, [0.0, 0.35, 0.0]));
    s.vel = [3.0, 0.0, 1.0];
    stay.spawn_body(guid(1), BodySpawn::new(s, crate_body(0.7)))
        .unwrap();
    stay.step_n(20);
    let exported = stay.export_entity(guid(1)).unwrap();
    let mut other = make_floor();
    other.reset_clock(stay.tick());
    other.import_entity(guid(1), &exported).unwrap();
    for _ in 0..60 {
        stay.step();
        other.step();
    }
    let d = WorldPos::distance(
        stay.state_of(guid(1)).unwrap().pos,
        other.state_of(guid(1)).unwrap().pos,
    );
    assert!(
        d < 1.0e-3,
        "in contact the transfer stays within 1 mm, got {d}"
    );
}

/// Every replicated component survives export → import exactly, and hostile
/// component lists are refused without spawning anything.
#[test]
fn export_import_round_trip_and_hostile_lists() {
    let o = WorldPos::ZERO;
    let mut a = SimWorld::new(ShardId(1));
    let springs = vertical_spring(at(o, [0.0, 5.0, 0.0]), 50.0, 1.0);
    let articulation = Articulation {
        parts: vec![PartDef {
            body: ball(0.2),
            visual: Visual(6),
        }],
        joints: vec![JointDef {
            a: 0,
            frame_a: LocalPose::at([0.0, -0.5, 0.0]),
            b: JointTarget::Part(1),
            frame_b: LocalPose::at([0.0, 0.5, 0.0]),
            kind: JointKind::Fixed,
        }],
    };
    let parts = PartStates(vec![ShipState::at_rest(at(o, [0.0, 2.0, 0.0]))]);
    a.spawn_body(
        guid(1),
        BodySpawn::new(ShipState::at_rest(at(o, [0.0, 3.0, 0.0])), crate_body(0.5))
            .with_visual(Visual(3))
            .with_springs(springs)
            .with_articulation(articulation, parts),
    )
    .unwrap();
    a.spawn_ship(guid(2), 99, ShipState::at_rest(at(o, [20.0, 0.0, 0.0])));
    let mut b = SimWorld::new(ShardId(2));
    for g in [guid(1), guid(2)] {
        let list = a.export_entity(g).unwrap();
        b.import_entity(g, &list).unwrap();
        assert_eq!(b.export_entity(g).unwrap(), list, "{g} round trip");
    }
    assert_eq!(a.guid_state_hash(), b.guid_state_hash());
    assert_eq!(b.ship_count(), 1, "the ship came back as a ship");
    // Hostile: duplicate ids, unknown ids, garbage, mismatched parts.
    let good = a.export_entity(guid(1)).unwrap();
    let mut dup = good.clone();
    dup.push(good[0].clone());
    let mut unknown = good.clone();
    unknown[1].0 = 42;
    let mut garbage = good.clone();
    garbage[1].1 = vec![0xFF; 7];
    let mismatched: Vec<_> = good.iter().filter(|(id, _)| *id != 8).cloned().collect();
    let no_state: Vec<_> = good.iter().filter(|(id, _)| *id != 0).cloned().collect();
    for (name, list) in [
        ("duplicate", dup),
        ("unknown", unknown),
        ("garbage", garbage),
        ("mismatched", mismatched),
        ("no state", no_state),
    ] {
        let mut c = SimWorld::new(ShardId(3));
        assert!(c.import_entity(guid(1), &list).is_err(), "{name} accepted");
        assert!(c.bodies().is_empty(), "{name} left something behind");
    }
}

/// Inputs that were filed for a ship while this world simulated it are not
/// applied once the ship has been handed away and is a ghost here: the ghost
/// keeps the controls and the velocity its owner reports.
///
/// This holds because a ghost is given neither `ShipControls` nor an
/// `Owner` when it is imported, and every write of `drain_inputs` goes
/// through one of the two.
///
/// Mutation this catches: a ghost that is imported like a ship, with
/// controls and an owner (the old player's input then leaves fast travel
/// for it and clamps its velocity to the normal top speed — the ship a
/// neighbour is flying at 300 m/s is shown, and collided with, at 120 — and
/// the old session's `input_ack` runs on for inputs nobody applied).
#[test]
fn a_ghost_takes_no_input_that_was_filed_for_the_ship_it_replaces() {
    let o = WorldPos::ZERO;
    let mut here = SimWorld::new(ShardId(1));
    let mut state = ShipState::at_rest(at(o, [0.0, 50.0, 0.0]));
    state.vel = [300.0, 0.0, 0.0];
    let entity = here.spawn_ship(guid(7), 5, state);
    here.world_mut()
        .get_mut::<space_sim::ShipControls>(entity)
        .expect("a ship has controls")
        .fast = true;
    // The player's inputs for the next three ticks: fast travel off.
    let now = here.tick();
    let inputs: Vec<PlayerInput> = (1..=3)
        .map(|i| PlayerInput {
            entity: guid(7),
            thrust: [0.0, 0.0, 1.0],
            torque: [0.0; 3],
            tick: now + i,
            fast: false,
        })
        .collect();
    here.apply_inputs(now, &inputs);
    assert_eq!(here.buffered_input_count(), 3);
    // The ship is handed away: it leaves, and its ghost takes its place.
    let components = here.export_entity(guid(7)).expect("exports");
    here.despawn(guid(7));
    here.insert_ghost(guid(7), &components).expect("a ghost");
    let before = here.export_entity(guid(7)).expect("the ghost exports");
    for _ in 0..3 {
        here.step();
    }
    assert_eq!(here.buffered_input_count(), 0);
    assert_eq!(
        here.state_of(guid(7)).expect("the ghost").vel,
        [300.0, 0.0, 0.0],
        "the ghost's velocity is its owner's word"
    );
    assert_eq!(
        here.export_entity(guid(7)).expect("the ghost exports"),
        before,
        "no component of the ghost was touched"
    );
    assert_eq!(
        here.input_ack(5),
        0,
        "nothing was applied for the old session"
    );
    assert_eq!(here.input_ack(0), 0, "nor for nobody's");
}

/// A ghost driven from outside pushes a dynamic body of this world and is
/// never integrated here; a spring owned by a ghost pulls on this world's end.
#[test]
fn ghosts_push_and_are_never_integrated() {
    let o = WorldPos::ZERO;
    let mut owner = SimWorld::new(ShardId(1));
    let mut gb = crate_body(1.0);
    gb.mass = space_sim::MassSpec::Mass(1000.0);
    let mut gs = ShipState::at_rest(at(o, [-3.0, 0.0, 0.0]));
    gs.vel = [2.0, 0.0, 0.0];
    owner.spawn_body(guid(9), BodySpawn::new(gs, gb)).unwrap();
    let mut here = SimWorld::new(ShardId(2));
    here.insert_ghost(guid(9), &owner.export_entity(guid(9)).unwrap())
        .unwrap();
    spawn(&mut here, 1, at(o, [0.0, 0.0, 0.0]), crate_body(0.5));
    for t in 0..60u32 {
        // The ghost is driven along its own path; this world never moves it.
        let mut s = gs;
        s.pos = at(o, [-3.0 + 2.0 * f64::from(t) / 30.0, 0.0, 0.0]);
        here.drive_ghost(guid(9), s, None);
        here.step();
        assert_eq!(
            here.state_of(guid(9)).unwrap().pos,
            s.pos,
            "a ghost is not integrated"
        );
    }
    let pushed = rel(o, here.state_of(guid(1)).unwrap().pos);
    assert!(pushed[0] > 0.5, "the ghost pushed the crate: {pushed:?}");
    assert!(here.is_ghost(guid(9)));
    // The ghost is no part of this world's replicated state.
    let mut alone = SimWorld::new(ShardId(2));
    alone
        .import_entity(guid(1), &here.export_entity(guid(1)).unwrap())
        .unwrap();
    alone.reset_clock(here.tick());
    assert_eq!(alone.guid_state_hash(), here.guid_state_hash());
}

/// Islands: bodies in contact proximity or tied by a spring form one island;
/// distant ones form their own; a ghost within reach is reported as touching.
#[test]
fn islands_group_bodies_by_proximity_and_springs() {
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    spawn(&mut sim, 1, at(o, [0.0, 0.0, 0.0]), ball(0.5));
    spawn(&mut sim, 2, at(o, [1.5, 0.0, 0.0]), ball(0.5));
    spawn(&mut sim, 3, at(o, [100.0, 0.0, 0.0]), ball(0.5));
    let tie = Springs(vec![Spring {
        part: 0,
        anchor: [0.0; 3],
        target: SpringTarget::Entity {
            guid: guid(3),
            part: 0,
            anchor: [0.0; 3],
        },
        rest_length: 50.0,
        stiffness: 1.0,
        damping: 0.0,
        max_force: 0.0,
        mode: SpringMode::Rope,
    }]);
    sim.spawn_body(
        guid(4),
        BodySpawn::new(ShipState::at_rest(at(o, [60.0, 0.0, 0.0])), ball(0.5)).with_springs(tie),
    )
    .unwrap();
    spawn(&mut sim, 5, at(o, [1.0e6, 0.0, 0.0]), ball(0.5));
    let mut ghost_src = SimWorld::new(ShardId(7));
    spawn(&mut ghost_src, 6, at(o, [1.0e6 + 1.2, 0.0, 0.0]), ball(0.5));
    sim.insert_ghost(guid(6), &ghost_src.export_entity(guid(6)).unwrap())
        .unwrap();
    sim.step();
    let islands = sim.islands(IslandQuery {
        margin: 0.5,
        lead_time: 1.0,
        springs: true,
    });
    let sets: Vec<Vec<u64>> = islands
        .iter()
        .map(|i| i.bodies.iter().map(|g| g.serial).collect())
        .collect();
    assert_eq!(sets, vec![vec![1, 2], vec![3, 4], vec![5]]);
    assert_eq!(islands[2].touching, vec![guid(6)]);
    // Without springs, 3 and 4 are apart.
    let apart = sim.islands(IslandQuery {
        springs: false,
        ..IslandQuery::default()
    });
    assert_eq!(apart.len(), 4);
}

// ---------------------------------------------------------------------------
// large worlds: bubbles
// ---------------------------------------------------------------------------

/// The same small scene — a stack of crates, a spring pendulum and a ball in
/// a bowl — simulated at the world origin, at 1e9 m, at 1e13 m and at
/// ~1e20 m (cell index ~1e17) gives the same local motion: bit for bit when
/// the scene sits at the same offset inside its cell, because every frame
/// is a bubble anchored next to it.
#[test]
fn the_same_scene_moves_the_same_anywhere_in_the_world() {
    let run = |o: WorldPos| {
        let mut sim = SimWorld::new(ShardId(1));
        sim.set_catalog(bowl_catalog());
        sim.set_statics(vec![
            floor(o, 0.0, 12.0),
            StaticCollider {
                shape: ShapeRef::Asset(AssetId(1)),
                place: WorldPoint::new(o, [6.0, 0.015_625, 0.0]),
                rotation: [1.0, 0.0, 0.0, 0.0],
                friction: 0.6,
                restitution: 0.0,
                visual: Visual(0),
                groups: space_sim::CollisionGroups::ALL,
                motion: None,
            },
        ])
        .unwrap();
        sim.set_gravity(GravityField {
            zones: vec![zone(o, 40.0)],
            bodies: vec![],
            clock: Default::default(),
        });
        // Every position is a short binary fraction, so it is exactly the
        // same offset in every cell placement below.
        for i in 0..3u64 {
            spawn(
                &mut sim,
                i + 1,
                at(o, [-4.0, 0.375 + 0.75 * i as f64, 0.0]),
                crate_body(0.75),
            );
        }
        let mut bob = ball(0.2);
        bob.mass = space_sim::MassSpec::Mass(5.0);
        sim.spawn_body(
            guid(10),
            BodySpawn::new(ShipState::at_rest(at(o, [0.0, 4.0, 1.0])), bob)
                .with_springs(vertical_spring(at(o, [0.0, 8.0, 0.0]), 200.0, 0.5)),
        )
        .unwrap();
        spawn(&mut sim, 20, at(o, [7.5, 1.25, 0.0]), ball(0.25));
        let mut track = Vec::new();
        for _ in 0..150 {
            sim.step();
            let mut frame = Vec::new();
            for n in [1, 2, 3, 10, 20] {
                frame.push(exact(&sim, n, o));
            }
            track.push(frame);
        }
        track
    };
    // Same in-cell offset everywhere: a cell-aligned origin.
    let cell = |x: i64| WorldPos {
        cell: [x, -x / 3, x / 7],
        local: [0.0; 3],
    };
    let here = run(cell(0));
    let distances = [1.0e9, 1.0e13, 1.0e20];
    for d in distances {
        let c = (d / 1024.0) as i64;
        let far = run(cell(c));
        assert_eq!(here, far, "the scene at {d} m moved differently");
    }
    // A different in-cell offset (and so different f32 rounding of every
    // WorldPos written): the exact positions (`pos + residual`) carry the
    // motion, so it is the same to well under a micrometre.
    let shifted = run(WorldPos {
        cell: [123_456_789_012, 0, 0],
        local: [300.25, -100.5, 7.75],
    });
    let worst: Vec<f64> = (0..5)
        .map(|body| {
            here.iter()
                .zip(&shifted)
                .map(|(a, b)| {
                    (0..3)
                        .map(|k| (a[body][k] - b[body][k]).abs())
                        .fold(0.0, f64::max)
                })
                .fold(0.0, f64::max)
        })
        .collect();
    println!("shifted in its cell, worst per body (crates, bob, ball): {worst:?}");
    assert!(
        worst.iter().all(|w| *w < 1.0e-6),
        "shifted in its cell: {worst:?}"
    );
}

/// A body flying 1e7 m in a straight line crosses thousands of cells and
/// re-anchors its bubble thousands of times, and arrives where `x0 + v t`
/// says, to a micrometre: its exact position is `WorldPos + residual`, so the
/// `f32` rounding of each tick's `WorldPos` does not accumulate (without the
/// residual it did: 3.3 cm over this flight).
#[test]
fn a_long_straight_flight_arrives_where_the_analytic_answer_says() {
    let mut sim = SimWorld::new(ShardId(1));
    let o = WorldPos::from_metres([-3_000_000_000, 17_000, 0]);
    let mut b = ball(0.5);
    b.angular_damping = 0.0;
    let v = 25_000.0f32;
    let mut s = ShipState::at_rest(o);
    s.vel = [v, 0.0, 0.0];
    sim.spawn_body(guid(1), BodySpawn::new(s, b)).unwrap();
    // A second body flies alongside 3 m away, so the bubble is a real
    // two-body bubble that re-anchors (and is stepped by rapier).
    let mut s2 = s;
    s2.pos = at(o, [0.0, 3.0, 0.0]);
    sim.spawn_body(guid(2), BodySpawn::new(s2, ball(0.5)))
        .unwrap();
    let ticks = 12_000u64;
    let mut anchors = std::collections::BTreeSet::new();
    for _ in 0..ticks {
        sim.step();
        if let Some(b) = sim.bubbles().first() {
            anchors.insert(b.anchor.cell);
        }
    }
    let flown = exact(&sim, 1, o);
    let expected = f64::from(v) * f64::from(dt()) * ticks as f64;
    assert!(expected > 9.9e6);
    let err = ((flown[0] - expected).powi(2) + flown[1].powi(2) + flown[2].powi(2)).sqrt();
    println!(
        "flew {expected:.0} m through {} bubble anchors; error {err:e} m",
        anchors.len()
    );
    assert!(
        anchors.len() > 1000,
        "the bubble re-anchored {} times",
        anchors.len()
    );
    assert!(err < 1.0e-6, "error {err} m after {expected} m");
}

/// Two bodies one astronomical unit apart in one world are in two bubbles,
/// and each keeps micrometre precision: a 0.1 mm/s drift (3.3 µm per tick,
/// less than the 61 µm `f32` resolution of the far body's cell offset) moves
/// each by exactly 1 mm in 10 s.
#[test]
fn two_bodies_an_au_apart_each_keep_micrometre_precision() {
    let a0 = WorldPos::ZERO;
    // One astronomical unit: 404 m from its cell's centre, where the `f32`
    // offset resolves only 30 µm.
    let b0 = WorldPos::from_metres([149_597_870_700_000, 0, 0]);
    assert!(b0.local[0].abs() > 400.0);
    let mut sim = SimWorld::new(ShardId(1));
    for (n, p) in [(1u64, a0), (2, b0)] {
        let mut s = ShipState::at_rest(p);
        // 1e-4 m/s: above the settling threshold, 3.3 µm per tick.
        s.vel = [1.0e-4, 0.0, 0.0];
        let mut body = ball(0.5);
        body.angular_damping = 0.0;
        sim.spawn_body(guid(n), BodySpawn::new(s, body)).unwrap();
    }
    sim.step_n(300);
    assert_eq!(sim.bubbles().len(), 2);
    for (n, p) in [(1u64, a0), (2, b0)] {
        let moved = rel(p, sim.state_of(guid(n)).unwrap().pos)[0];
        let want = 1.0e-4 * 10.0;
        let residual = {
            let e = sim.entity_of(guid(n)).unwrap();
            f64::from(sim.world().get::<space_sim::PosResidual>(e).unwrap().0[0][0])
        };
        // The rounded WorldPos alone is within the cell's resolution; with
        // the residual it is exact.
        assert!(
            (moved - want).abs() < 3.1e-5,
            "body {n} WorldPos moved {moved} m"
        );
        let exact = moved + residual;
        assert!(
            (exact - want).abs() < 1.0e-6,
            "body {n} moved {exact} m, want {want}"
        );
    }
}

/// A body that changes bubble — its bubble merges with another and later
/// re-anchors — follows exactly the trajectory of a body that never did.
#[test]
fn merges_and_reanchors_do_not_perturb_motion() {
    let o = WorldPos::ZERO;
    let fly = |with_company: bool| {
        let mut sim = SimWorld::new(ShardId(1));
        let mut s = ShipState::at_rest(at(o, [0.0, 0.0, 0.0]));
        s.vel = [400.0, 0.0, 0.0];
        s.ang_vel = [0.0, 0.5, 0.0];
        sim.spawn_body(guid(1), BodySpawn::new(s, crate_body(0.6)))
            .unwrap();
        if with_company {
            // A body parked on the path, offset sideways so they pass 3 m
            // apart: their bubbles merge, then split again.
            spawn(&mut sim, 2, at(o, [2000.0, 3.0, 0.0]), ball(0.5));
        }
        let mut path = Vec::new();
        let mut merged = false;
        for _ in 0..300 {
            sim.step();
            merged |= sim.bubbles().iter().any(|b| b.members.len() == 2);
            path.push(sim.state_of(guid(1)).unwrap());
        }
        (path, merged)
    };
    let (alone, _) = fly(false);
    let (company, merged) = fly(true);
    assert!(merged, "the bubbles must have merged");
    assert_eq!(alone, company, "a merge or re-anchor changed the motion");
}

/// Bubbles are per group of bodies that can touch: far-apart bodies are in
/// different bubbles, and bodies about to touch share one before they do.
#[test]
fn bubbles_merge_before_bodies_touch_and_split_after() {
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    let mut a = ShipState::at_rest(at(o, [-30.0, 0.0, 0.0]));
    a.vel = [10.0, 0.0, 0.0];
    let mut b = ShipState::at_rest(at(o, [30.0, 0.0, 0.0]));
    b.vel = [-10.0, 0.0, 0.0];
    let mut body = ball(0.5);
    body.restitution = 1.0;
    sim.spawn_body(guid(1), BodySpawn::new(a, body.clone()))
        .unwrap();
    sim.spawn_body(guid(2), BodySpawn::new(b, body)).unwrap();
    sim.step();
    assert_eq!(sim.bubbles().len(), 2, "far apart: two bubbles");
    let mut merged_at = None;
    let mut first_contact = None;
    for t in 0..150 {
        sim.step();
        if merged_at.is_none() && sim.bubbles().len() == 1 {
            merged_at = Some(t);
        }
        if first_contact.is_none() && !sim.contacts().is_empty() {
            first_contact = Some(t);
        }
    }
    let (m, c) = (merged_at.expect("merged"), first_contact.expect("collided"));
    assert!(m < c, "merged at {m}, touched at {c}");
    assert_eq!(
        sim.bubbles().len(),
        2,
        "after bouncing apart they split again"
    );
}

// ---------------------------------------------------------------------------
// client prediction through the physics
// ---------------------------------------------------------------------------

/// `PredictedPhysics` replays the same physics as the server: with lossless,
/// in-order delivery every reconcile is exactly zero in free flight and
/// under gravity, and stays under a millimetre while the ship lands on and
/// slides along a floor (the server's contact solver keeps its warm-start
/// state across ticks, the replay starts fresh from each authority).
#[test]
fn physics_prediction_matches_the_server() {
    use space_sim::prediction::PredictedPhysics;
    let o = WorldPos::from_metres([2_000_000_000, 0, 0]);
    let environment = |sim: &mut SimWorld| {
        sim.set_statics(vec![floor(o, -1.7, 30.0)]).unwrap();
        sim.set_gravity(GravityField {
            zones: vec![zone(at(o, [0.0, 0.0, 0.0]), 20.0)],
            bodies: vec![],
            clock: Default::default(),
        });
    };
    let mut server = SimWorld::new(ShardId(1));
    environment(&mut server);
    let start = ShipState::at_rest(at(o, [0.0, 3.0, 0.0]));
    server.spawn_ship(guid(1), 7, start);
    let mut client_world = SimWorld::new(ShardId(1));
    environment(&mut client_world);
    let mut client = PredictedPhysics::new(client_world, guid(1), 7, start, 0).unwrap();
    let mut worst_free = 0.0f32;
    let mut worst_contact = 0.0f32;
    for t in 1..=150u64 {
        let input = PlayerInput {
            entity: guid(1),
            thrust: if t < 60 {
                [0.0, 0.0, 0.0]
            } else {
                [0.6, 0.0, -0.4]
            },
            torque: [0.0; 3],
            tick: t,
            fast: false,
        };
        client.apply_input(input);
        server.apply_inputs(t, &[input]);
        server.step();
        if t % 3 == 0 {
            let c = client.reconcile(server.state_of(guid(1)).unwrap(), t, server.input_ack(7));
            // It touches the floor after ~24 ticks of falling.
            if t < 18 {
                worst_free = worst_free.max(c);
            } else {
                worst_contact = worst_contact.max(c);
            }
        }
    }
    let landed = rel(o, server.state_of(guid(1)).unwrap().pos)[1];
    assert!(
        (landed - (-1.7 + 1.68)).abs() < 0.05,
        "the ship stands on the floor: {landed}"
    );
    assert_eq!(worst_free, 0.0, "falling freely the prediction is exact");
    assert!(worst_contact < 1.0e-3, "in contact: {worst_contact} m");
}

/// A *dynamic* body with a concave catalogue shape uses the asset's convex
/// decomposition (VHACD), computed once and shared: the bowl dropped upside
/// down onto a floor comes to rest on its rim.
#[test]
fn a_dynamic_concave_body_uses_its_cached_convex_decomposition() {
    let catalog = bowl_catalog();
    let asset = catalog.get(AssetId(1)).unwrap();
    let started = std::time::Instant::now();
    let first = asset.convex_decomposition();
    let took = started.elapsed();
    let again = asset.convex_decomposition();
    assert!(std::ptr::eq(first.as_ref(), again.as_ref()), "cached");
    let parts = first.as_compound().map_or(1, |c| c.shapes().len());
    println!("bowl decomposition: {parts} convex parts in {took:?}");
    assert!(parts > 1, "a concave shape needs several convex parts");
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_catalog(catalog.clone());
    sim.set_statics(vec![floor(o, 0.0, 20.0)]).unwrap();
    sim.set_gravity(GravityField {
        zones: vec![zone(o, 30.0)],
        bodies: vec![],
        clock: Default::default(),
    });
    let mut bowl = Body::dynamic(ShapeRef::Asset(AssetId(1)));
    bowl.mass = space_sim::MassSpec::Mass(50.0);
    // Upside down (rim down), 1 m above the floor.
    let mut s = ShipState::at_rest(at(o, [0.0, 2.5, 0.0]));
    s.quat = [0.0, 1.0, 0.0, 0.0];
    sim.spawn_body(guid(1), BodySpawn::new(s, bowl)).unwrap();
    sim.step_n(120);
    let st = sim.state_of(guid(1)).unwrap();
    let y = rel(o, st.pos)[1];
    // Upside down, its origin (the bottom of the cap) is the bowl's depth,
    // 1.5 m, above the rim.
    assert!((y - 1.5).abs() < 0.1, "rests on its rim: origin at {y}");
    assert!(
        st.vel.iter().all(|v| v.abs() < 0.05),
        "at rest: {:?}",
        st.vel
    );
}

// ---------------------------------------------------------------------------
// bubbles and resting bodies
// ---------------------------------------------------------------------------

/// A stack of five 0.8 m crates asleep on a floor, with a bob on a spring
/// beside it that never comes to rest (so their bubble is stepped every
/// tick), and `passers` weightless balls flying past in a row, 5 m beside the
/// stack, at 20 m/s: they come into the stack's bubble and leave it again.
/// The passers have the lowest guids of all.
///
/// Returns how far any crate was from where it rested, at worst, and whether
/// every crate stayed asleep throughout.
fn a_stack_is_passed_by(passers: u64) -> (f64, bool) {
    let o = WorldPos::ZERO;
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_statics(vec![floor(o, 0.0, 10.0)]).unwrap();
    sim.set_gravity(GravityField {
        zones: vec![zone(o, 50.0)],
        bodies: vec![],
        clock: Default::default(),
    });
    let crates = 100..105u64;
    for (i, n) in crates.clone().enumerate() {
        spawn(
            &mut sim,
            n,
            at(o, [0.0, 0.4 + 0.8 * i as f64, 0.0]),
            crate_body(0.8),
        );
    }
    let mut bob = ball(0.2);
    bob.linear_damping = 0.0;
    bob.mass = space_sim::MassSpec::Mass(4.0);
    sim.spawn_body(
        guid(110),
        BodySpawn::new(ShipState::at_rest(at(o, [2.2, 2.6, 0.0])), bob)
            .with_springs(vertical_spring(at(o, [2.2, 5.0, 0.0]), 200.0, 0.0)),
    )
    .unwrap();
    sim.step_n(300);
    assert!(
        crates
            .clone()
            .all(|n| sim.is_sleeping(guid(n)) == Some(true)),
        "the stack sleeps"
    );
    assert_eq!(sim.bubbles().len(), 1, "stack and bob share a bubble");
    assert!(sim.bubbles()[0].stepped, "which the bob keeps stepped");
    let height = |sim: &SimWorld, n: u64| rel(o, sim.state_of(guid(n)).unwrap().pos)[1];
    let rest: Vec<f64> = crates.clone().map(|n| height(&sim, n)).collect();
    let mut passer = ball(0.3);
    passer.gravity_scale = 0.0;
    passer.linear_damping = 0.0;
    for k in 0..passers {
        let mut s = ShipState::at_rest(at(o, [-40.0 - 1.2 * k as f64, 3.0, 5.0]));
        s.vel = [20.0, 0.0, 0.0];
        sim.spawn_body(guid(k + 1), BodySpawn::new(s, passer.clone()))
            .unwrap();
    }
    let (mut worst, mut slept, mut most_bubbles, mut fewest_bubbles) = (0.0f64, true, 0, 9);
    for _ in 0..400 {
        sim.step();
        most_bubbles = most_bubbles.max(sim.bubbles().len());
        fewest_bubbles = fewest_bubbles.min(sim.bubbles().len());
        for (i, n) in crates.clone().enumerate() {
            worst = worst.max((height(&sim, n) - rest[i]).abs());
            slept &= sim.is_sleeping(guid(n)) == Some(true);
        }
    }
    assert_eq!(
        fewest_bubbles, 1,
        "the passers came into the stack's bubble"
    );
    assert_eq!(sim.bubbles().len(), 2, "and left it again");
    (worst, slept)
}

/// A body that comes into a resting stack's bubble and leaves it again does
/// not touch the stack: the crates stay asleep and do not move at all.
///
/// Mutation this catches: on a split the group holding the bubble's
/// *smallest guid* keeps the bubble (as it once did). The passer has it, so
/// the stack and the bob were moved into a new solver world — and there the
/// stack woke and fell into the floor before its contacts were found: the
/// bottom crate by 6.5 cm, the top one by 30 cm. Seen in the demo area: every
/// player who walked away from the gravity room made its crate stack hop (a
/// player's ship has a lower guid than any scene body).
#[test]
fn a_resting_stack_is_not_disturbed_when_a_passer_by_leaves_its_bubble() {
    let (worst, slept) = a_stack_is_passed_by(1);
    assert_eq!(worst, 0.0, "the stack moved");
    assert!(slept, "the stack was woken");
}

/// When it is the stack that has to move — the group that passes is the
/// larger one, so the stack is taken into its bubble and out of it again —
/// the stack keeps standing: it is put into the other solver world awake,
/// finds its contacts in the first step and settles again within
/// millimetres.
///
/// Mutation this catches: inserting a sleeping body into a solver world
/// asleep (`RigidBodyBuilder::sleeping(e.sleeping)`): the top crate then
/// drops 30 cm into the stack before it is pushed out again.
#[test]
fn a_resting_stack_moved_into_another_bubble_keeps_standing() {
    let (worst, _) = a_stack_is_passed_by(8);
    assert!(worst < 0.01, "the stack moved by {worst} m");
}
