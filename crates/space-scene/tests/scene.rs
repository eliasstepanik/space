//! The scene definitions: they build, they match what the client draws, the
//! showcase starts with nothing overlapping and behaves as designed, the
//! Kestrel's collider is its GLB, and the anchor rule is the client's.

use std::net::SocketAddr;

use rapier3d_f64::parry::query;
use rapier3d_f64::parry::shape::SharedShape;
use rapier3d_f64::prelude::{Pose, Vector};
use space_protocol::{EntityGuid, Scene, ShardId, ShardMap, ShardRegion};
use space_scene::{demo, visual, BodyDef, StaticDef, KESTREL};
use space_sim::shape::quat_to_rot;
use space_sim::{SimWorld, Visual, WorldPoint};
use space_world::{NodeRef, WorldPos};

fn minter() -> impl FnMut() -> EntityGuid {
    let mut serial = 0u64;
    move || {
        serial += 1;
        EntityGuid::new(ShardId(9), serial)
    }
}

fn region(depth: u8, index: [i64; 3], shard: u32) -> ShardRegion {
    let addr: SocketAddr = "127.0.0.1:7777".parse().unwrap();
    ShardRegion::new(NodeRef::new(depth, index), ShardId(shard), addr)
}

/// The anchor is a point of the shallowest region (ties: lowest shard, then
/// node index) — `space-client`'s `link::map_anchor`, on the same maps its
/// own test uses.
#[test]
fn the_anchor_rule_is_the_clients() {
    let root = region(0, [0, 0, 0], 1);
    // The whole cell space: its centre corner is the lower corner of cell 0,
    // and the anchor 6 m, 512 m and 4 m from it.
    let off_the_corner = WorldPos {
        cell: [0; 3],
        local: [-506.0, 0.0, -508.0],
    };
    let single = ShardMap::new(1, vec![root]);
    assert_eq!(space_scene::scene_anchor(&single), Some(off_the_corner));
    for map in [
        single,
        ShardMap::new(2, vec![region(1, [0, 0, 0], 2), region(1, [-1, 0, 0], 3)]),
        ShardMap::new(3, vec![region(63, [5, 0, 0], 7), region(63, [0, 0, 0], 4)]),
        ShardMap::new(4, vec![region(64, [5, 0, 0], 7), region(64, [0, 0, 0], 4)]),
        ShardMap::new(5, vec![region(64, [1, 1, 1], 2), region(63, [0, 0, 0], 4)]),
    ] {
        assert_eq!(
            space_scene::scene_anchor(&map),
            space_client::link::map_anchor(&map),
            "{map:?}"
        );
    }
    assert_eq!(
        space_scene::scene_anchor(&ShardMap::new(1, Vec::new())),
        None
    );
}

/// Where the anchor lies in the root (`docs/plan-sharding.md` §1):
///
/// * a root of one cell (depth 64): the cell's centre;
/// * any larger root: its centre corner plus (6 m, 512 m, 4 m). With `k` the
///   corner's cell index the spawn point is strictly inside cell `[k,k,k]`,
///   no horizontal border lies within 512 m of the scene, the vertical ones
///   pass at scene x = −6 and z = −4, and the four cells that meet on that
///   line are `[k,k,k]`, `[k,k,k−1]`, `[k−1,k,k]` and `[k−1,k,k−1]`.
///
/// Mutations this catches: the old rule (the corner itself: borders through
/// the spawn point on all three planes, one of them at eye height); an
/// offset applied to a one-cell root too (a standalone shard's scene would
/// move off its cell centre and 6 m from nothing).
#[test]
fn the_anchor_lies_beside_and_above_the_corner_where_the_roots_cells_meet() {
    assert_eq!(space_scene::ANCHOR_OFFSET, [6.0, 512.0, 4.0]);
    // One cell: its centre, wherever the cell is.
    for cell in [[0i64, 0, 0], [1, 1, 1], [-7, 3, 9_765_625_000]] {
        let node = NodeRef::new(64, cell);
        let anchor = space_scene::anchor_in(node);
        assert_eq!((anchor.cell, anchor.local), (cell, [0.0; 3]));
        assert_eq!(
            space_scene::scene_anchor(&ShardMap::new(1, vec![region(64, cell, 1)])),
            Some(anchor)
        );
    }
    // More than one cell: off the centre corner, at every size from two
    // cells to the whole cell space, on both sides of the origin.
    for node in [
        NodeRef::new(63, [0, 0, 0]),
        NodeRef::new(63, [-1, 4, -9]),
        NodeRef::new(40, [0, 0, 0]),
        NodeRef::new(40, [3, -2, 5]),
        NodeRef::new(1, [-1, 0, -1]),
        NodeRef::ROOT,
    ] {
        let corner = node.center();
        assert_eq!(
            corner.local, [-512.0; 3],
            "{node:?}: the corner of eight cells"
        );
        let k = corner.cell;
        let anchor = space_scene::anchor_in(node);
        assert_eq!(
            space_scene::scene_anchor(&ShardMap::new(1, vec![region(node.depth, node.index, 1)])),
            Some(anchor),
            "{node:?}"
        );
        assert_eq!(
            WorldPos::delta(corner, anchor),
            [6.0, 512.0, 4.0],
            "{node:?}"
        );
        // Exact: whole metres in the cell's own frame.
        assert_eq!((anchor.cell, anchor.local), (k, [-506.0, 0.0, -508.0]));
        assert!(space_world::octree::contains(node, anchor));
        // The cell of a point `d` metres from the spawn point.
        let cell_at = |d: [f64; 3]| WorldPos::from_f64_metres_around(anchor, d).cell;
        // The four cells that meet on the line x = -6, z = -4, at every
        // height of the scene's cell layer.
        for y in [-511.9, -100.0, 0.0, 100.0, 511.9] {
            assert_eq!(cell_at([0.0, y, 0.0]), k, "the spawn point's cell");
            assert_eq!(cell_at([-5.99, y, -3.99]), k);
            assert_eq!(cell_at([-5.99, y, -4.01]), [k[0], k[1], k[2] - 1]);
            assert_eq!(cell_at([-6.01, y, -3.99]), [k[0] - 1, k[1], k[2]]);
            assert_eq!(cell_at([-6.01, y, -4.01]), [k[0] - 1, k[1], k[2] - 1]);
            // A border belongs to the cell above it.
            assert_eq!(cell_at([-6.0, y, -4.0]), k);
        }
        // No other vertical border for a kilometre: the cell reaches 1018 m
        // along +x and 1020 m along +z.
        assert_eq!(cell_at([1017.9, 0.0, 1019.9]), k);
        assert_eq!(cell_at([1018.0, 0.0, 0.0]), [k[0] + 1, k[1], k[2]]);
        assert_eq!(cell_at([0.0, 0.0, 1020.0]), [k[0], k[1], k[2] + 1]);
        // The horizontal borders are half a cell away, above and below.
        assert_eq!(cell_at([0.0, -512.0, 0.0]), k);
        assert_eq!(cell_at([0.0, -512.01, 0.0]), [k[0], k[1] - 1, k[2]]);
        assert_eq!(cell_at([0.0, 512.0, 0.0]), [k[0], k[1] + 1, k[2]]);
    }
    // The root decides, whoever owns the cell the anchor lies in and
    // whatever was carved out around it.
    let root = NodeRef::new(40, [0, 0, 0]);
    let k = root.center().cell[0];
    let anchor = space_scene::anchor_in(root);
    for regions in [
        vec![region(40, [0, 0, 0], 1), region(64, [k, k, k], 2)],
        vec![
            region(64, [k, k, k - 1], 2),
            region(64, [k - 1, k, k], 3),
            region(40, [0, 0, 0], 9),
            region(64, [k - 1, k, k - 1], 4),
            region(41, [1, 1, 1], 5),
        ],
    ] {
        assert_eq!(
            space_scene::scene_anchor(&ShardMap::new(7, regions)),
            Some(anchor)
        );
    }
    // Nothing of the demo area's showcase comes near a horizontal border:
    // every static and body of it is placed well inside the 512 m to the cell
    // layer's floor and ceiling. The demo area's **sky** (the solar system
    // merged into it: Earth, Moon, Sun, planets) is the other kind of static
    // — thousands of kilometres away and larger than any cell, so it spans
    // shard borders by nature. That is sound because a static is no entity:
    // every shard builds every static of the scene itself, at the same place
    // (this anchor) and moved by the same scene clock, and none is ever
    // handed over.
    let statics = space_scene::statics(Scene::DemoArea);
    let far = |s: &&StaticDef| s.offset.iter().map(|c| c * c).sum::<f64>().sqrt() > 1_000.0;
    let sky: Vec<&StaticDef> = statics.iter().filter(far).collect();
    assert_eq!(
        sky.len(),
        space_scene::solar::sky_bodies().len(),
        "what lies beyond a kilometre is exactly the sky's bodies"
    );
    for body in &sky {
        let distance = body.offset.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!(distance > 1.0e6, "a sky body {distance} m away");
    }
    let heights = statics
        .iter()
        .filter(|s| !far(s))
        .map(|s| s.offset[1])
        .chain(
            space_scene::body_defs(Scene::DemoArea)
                .iter()
                .map(|b| b.offset[1]),
        )
        .fold((f64::MAX, f64::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
    assert!(heights.0 > -400.0 && heights.1 < 400.0, "{heights:?}");
}

/// The Kestrel's collider is the LOD1 GLB: 11,585 triangles, its bounds those
/// the asset's README gives for the model (within 0.1 m: the proxy is a
/// decimated exterior).
#[test]
fn the_kestrel_collider_is_its_glb() {
    let catalog = space_scene::catalog();
    let asset = catalog.get(KESTREL).expect("in the catalogue");
    let mesh = asset.mesh();
    assert_eq!(mesh.indices.len(), 11_585);
    let (lo, hi) = mesh.bounds();
    let readme_lo = [-6.32, -3.3, -11.84];
    let readme_hi = [6.32, 2.3436, 11.955];
    for k in 0..3 {
        assert!((lo[k] - readme_lo[k]).abs() < 0.1, "min {lo:?}");
        assert!((hi[k] - readme_hi[k]).abs() < 0.1, "max {hi:?}");
    }
    // The parry triangle mesh built from it has the same bounds.
    let aabb = asset.trimesh().compute_local_aabb();
    assert!((aabb.mins.x - lo[0]).abs() < 1e-9 && (aabb.maxs.z - hi[2]).abs() < 1e-9);
}

fn static_shape(s: &StaticDef) -> SharedShape {
    s.shape
        .build_static(&space_scene::catalog())
        .expect("builds")
}

fn pose(offset: [f64; 3], rotation: [f32; 4]) -> Pose {
    Pose::from_parts(
        Vector::new(offset[0], offset[1], offset[2]),
        quat_to_rot(rotation),
    )
}

fn aabb(shape: &SharedShape, p: &Pose) -> ([f64; 3], [f64; 3]) {
    let a = shape.compute_aabb(p);
    (
        [a.mins.x, a.mins.y, a.mins.z],
        [a.maxs.x, a.maxs.y, a.maxs.z],
    )
}

/// Every station of the client's demo area within the area (models, boxes
/// and spheres; the backdrop excluded) has a static collider here with the
/// same origin and the same world bounds, and the Kestrel stands where the
/// client draws it. (Characters are capsules, so their bounds agree to 0.25 m.)
#[test]
fn the_demo_statics_match_what_the_client_draws() {
    use space_client::demo_area::{DemoArea, AREA_RADIUS};
    let area = DemoArea::shared();
    let statics: Vec<StaticDef> = demo::statics();
    let drawn: Vec<&StaticDef> = statics
        .iter()
        .filter(|s| s.visual == Visual(visual::NONE))
        .collect();
    let mut matched = vec![false; drawn.len()];
    let metres = |p: WorldPos| WorldPos::delta(WorldPos::ZERO, p);
    let find = |origin: [f64; 3], lo: [f64; 3], hi: [f64; 3], tol: f64, what: &str| {
        let found = drawn.iter().position(|s| {
            let near = (0..3).all(|k| (s.offset[k] - origin[k]).abs() < 1e-4);
            if !near {
                return false;
            }
            let (slo, shi) = aabb(&static_shape(s), &pose(s.offset, s.rotation));
            (0..3).all(|k| (slo[k] - lo[k]).abs() < tol && (shi[k] - hi[k]).abs() < tol)
        });
        assert!(
            found.is_some(),
            "no collider matches {what} at {origin:?} ({lo:?}..{hi:?})"
        );
        found.unwrap()
    };
    for prop in area.props() {
        let p = metres(prop.position);
        if (p[0] * p[0] + p[2] * p[2]).sqrt() > AREA_RADIUS {
            continue;
        }
        // The prop's world bounds: its mesh bounds, scaled, turned, moved.
        let (bmin, bmax) = (prop.mesh.bounds_min, prop.mesh.bounds_max);
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for corner in 0..8 {
            let c = [
                if corner & 1 == 0 { bmin[0] } else { bmax[0] },
                if corner & 2 == 0 { bmin[1] } else { bmax[1] },
                if corner & 4 == 0 { bmin[2] } else { bmax[2] },
            ];
            for row in 0..3 {
                let v: f32 = (0..3)
                    .map(|k| prop.rotation[row][k] * c[k] * prop.scale)
                    .sum();
                let w = p[row] + f64::from(v);
                lo[row] = lo[row].min(w);
                hi[row] = hi[row].max(w);
            }
        }
        let character = bmax[1] > 1.7 && bmax[1] < 1.9 && bmax[0] < 0.6;
        let i = find(p, lo, hi, if character { 0.25 } else { 2e-3 }, "a prop");
        matched[i] = true;
    }
    for body in area.bodies() {
        let p = metres(body.position);
        if (p[0] * p[0] + p[2] * p[2]).sqrt() > AREA_RADIUS {
            continue;
        }
        let r = f64::from(body.radius);
        let i = find(p, p.map(|c| c - r), p.map(|c| c + r), 2e-3, body.name);
        matched[i] = true;
    }
    // The Kestrel: same berth.
    let berth = area.kestrel();
    for (have, want) in berth.position.iter().zip(demo::KESTREL_POSITION) {
        assert!((have - want).abs() < 1e-9, "{have} vs {want}");
    }
    assert!((berth.yaw - demo::KESTREL_YAW).abs() < 1e-6);
    let kestrel = drawn
        .iter()
        .position(|s| s.name == "kestrel")
        .expect("kestrel");
    matched[kestrel] = true;
    let unmatched: Vec<&str> = drawn
        .iter()
        .zip(&matched)
        .filter(|(_, m)| !**m)
        .map(|(s, _)| s.name)
        .collect();
    assert!(
        unmatched.is_empty(),
        "colliders with nothing drawn: {unmatched:?}"
    );
}

fn body_shapes(d: &BodyDef) -> Vec<(SharedShape, Pose)> {
    let catalog = space_scene::catalog();
    let mut out = vec![(
        d.body.shape.build(&catalog, true).unwrap(),
        pose(d.offset, d.quat),
    )];
    if let Some((art, _, parts)) = &d.articulation {
        for (part, (offset, quat)) in art.parts.iter().zip(parts) {
            out.push((
                part.body.shape.build(&catalog, true).unwrap(),
                pose(*offset, *quat),
            ));
        }
    }
    out
}

/// No showcase body starts overlapping another body or a static (touching —
/// a crate on the floor, a crate on a crate — is fine: penetration under 1 mm).
#[test]
fn no_showcase_object_starts_intersecting_another() {
    let bodies = space_scene::body_defs(Scene::DemoArea);
    let statics = space_scene::statics(Scene::DemoArea);
    // (The sky's bodies where they are at scene time 0.)
    let static_shapes: Vec<(SharedShape, Pose, &str)> = statics
        .iter()
        .map(|s| {
            let o = s.motion.as_ref().map_or([0.0; 3], |m| m.offset(0));
            let at = [s.offset[0] + o[0], s.offset[1] + o[1], s.offset[2] + o[2]];
            (static_shape(s), pose(at, s.rotation), s.name)
        })
        .collect();
    let shapes: Vec<Vec<(SharedShape, Pose)>> = bodies.iter().map(body_shapes).collect();
    let depth = |a: &(SharedShape, Pose), b: (&SharedShape, &Pose)| {
        query::contact(&a.1, a.0.as_ref(), b.1, b.0.as_ref(), 0.0)
            .expect("supported pair")
            .map_or(0.0, |c| -c.dist)
    };
    for (i, bi) in shapes.iter().enumerate() {
        for part in bi {
            for (j, bj) in shapes.iter().enumerate().skip(i + 1) {
                for other in bj {
                    let d = depth(part, (&other.0, &other.1));
                    assert!(
                        d < 1e-3,
                        "{} overlaps {} by {d} m",
                        bodies[i].name,
                        bodies[j].name
                    );
                }
            }
            for (s, p, name) in &static_shapes {
                let d = depth(part, (s, p));
                assert!(
                    d < 1e-3,
                    "{} overlaps the static {name} by {d} m",
                    bodies[i].name
                );
            }
        }
    }
}

/// Every scene installs and spawns, anywhere in the world, and steps.
#[test]
fn every_scene_builds_and_steps_anywhere() {
    for scene in Scene::ALL {
        for anchor in [
            WorldPos::ZERO,
            WorldPos::from_metres([1_000_000_000_000_000_000, 7, -3]),
        ] {
            let mut sim = SimWorld::new(ShardId(9));
            space_scene::install_environment(&mut sim, scene, anchor).unwrap();
            let guids = space_scene::spawn_bodies(&mut sim, scene, anchor, &mut minter()).unwrap();
            assert_eq!(guids.len(), space_scene::body_defs(scene).len());
            sim.spawn_ship(
                EntityGuid::new(ShardId(9), 1000),
                1,
                space_sim::ShipState::at_rest(anchor),
            );
            sim.step_n(5);
            assert!(sim.statics().count() > 0, "{scene:?} has statics");
        }
    }
}

/// The showcase behaves as designed over 10 s: the stack stands, the bowl
/// keeps its balls, the ramp balls roll off, the pendulums and the chain
/// hang under the gantry, the seesaw tips to its limit under the crate, the
/// zero-g crates stay put, the spring pair oscillates, and a player at the
/// spawn point floats.
#[test]
fn the_showcase_works() {
    let anchor = WorldPos::from_metres([3_000_000, 0, -2_000_000]);
    let mut sim = SimWorld::new(ShardId(9));
    space_scene::install_environment(&mut sim, Scene::DemoArea, anchor).unwrap();
    let placed = space_scene::bodies(Scene::DemoArea, anchor, &mut minter());
    for b in &placed {
        sim.spawn_body(b.guid, b.spawn.clone()).unwrap();
    }
    let player = EntityGuid::new(ShardId(9), 1000);
    sim.spawn_ship(player, 1, space_sim::ShipState::at_rest(anchor));
    let at = |sim: &SimWorld, name: &str, k: usize| -> Vec<[f64; 3]> {
        placed
            .iter()
            .filter(|b| b.name == name)
            .map(|b| WorldPos::delta(anchor, sim.part_states_of(b.guid).unwrap()[k].pos))
            .collect()
    };
    let start_ramp = at(&sim, "ramp ball", 0);
    let start_pair = at(&sim, "spring pair a", 0);
    let mut pair_x = Vec::new();
    for _ in 0..300 {
        sim.step();
        pair_x.push(at(&sim, "spring pair a", 0)[0][0]);
    }
    let f = demo::FLOOR;
    for (i, p) in at(&sim, "stack crate", 0).iter().enumerate() {
        assert!(
            (p[0] - 12.0).abs() < 0.05 && (p[1] - (f + 0.3 + 0.6 * i as f64)).abs() < 0.03,
            "stack {i}: {p:?}"
        );
    }
    for p in at(&sim, "bowl ball", 0) {
        let d = ((p[0] - demo::BOWL_AT[0]).powi(2) + (p[2] - demo::BOWL_AT[2]).powi(2)).sqrt();
        assert!(
            d < 1.0 && p[1] < f + 0.6,
            "a bowl ball left the bottom: {p:?}"
        );
    }
    for (p, s) in at(&sim, "ramp ball", 0).iter().zip(&start_ramp) {
        assert!(
            p[2] - s[2] > 1.0 && p[1] < f + 0.3,
            "a ramp ball rolled off: {s:?} -> {p:?}"
        );
    }
    for p in at(&sim, "swinging pendulum", 0)
        .iter()
        .chain(&at(&sim, "bobbing pendulum", 0))
    {
        assert!(p[1] < f + 4.9 && p[1] > f + 2.3, "a bob hangs: {p:?}");
    }
    let chain_bottom = at(&sim, "chain", 5)[0];
    assert!(
        chain_bottom[1] < f + 3.0,
        "the chain hangs: {chain_bottom:?}"
    );
    let plank = placed.iter().find(|b| b.name == "seesaw plank").unwrap();
    let q = sim.state_of(plank.guid).unwrap().quat;
    let tilt = 2.0 * f64::from(q[3]).atan2(f64::from(q[0]));
    assert!(
        tilt.abs() > 0.2 && tilt.abs() < 0.32,
        "the seesaw tipped to its limit: {tilt}"
    );
    for (p, b) in at(&sim, "floating crate", 0)
        .iter()
        .zip(placed.iter().filter(|b| b.name == "floating crate"))
    {
        let s = WorldPos::delta(anchor, b.spawn.state.pos);
        assert!(
            (0..3).all(|k| (p[k] - s[k]).abs() < 1e-6),
            "a weightless crate moved"
        );
    }
    let (lo, hi) = pair_x
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
    assert!(
        hi - lo > 0.3 && start_pair[0][0] < hi,
        "the spring pair oscillates: {lo}..{hi}"
    );
    let me = WorldPos::delta(anchor, sim.state_of(player).unwrap().pos);
    assert!(
        me.iter().all(|v| v.abs() < 1e-9),
        "the spawn point is weightless: {me:?}"
    );
}

/// The planetoid over 20 s, far from the world origin: its crates stay put
/// on its surface, its three balls keep their orbits, a player dropped over
/// its top falls onto it, stands, and lifts off with the lift engines — and
/// the zero-g corner and the spawn point stay weightless to the bit (the
/// planetoid's pull is exactly zero beyond its reach).
#[test]
fn the_planetoid_holds_its_crates_and_moons_and_lets_a_player_land() {
    use space_protocol::PlayerInput;
    let anchor = WorldPos::from_metres([3_000_000_000, -7_000, 9_000_000_000_000]);
    let mut sim = SimWorld::new(ShardId(9));
    space_scene::install_environment(&mut sim, Scene::DemoArea, anchor).unwrap();
    let placed = space_scene::bodies(Scene::DemoArea, anchor, &mut minter());
    for b in &placed {
        sim.spawn_body(b.guid, b.spawn.clone()).unwrap();
    }
    let c = demo::PLANETOID_CENTER;
    let r = demo::PLANETOID_RADIUS;
    let from_centre = |p: WorldPos| {
        let d = WorldPos::delta(anchor, p);
        [d[0] - c[0], d[1] - c[1], d[2] - c[2]]
    };
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    // A player whose feet are a metre over the top.
    let player = EntityGuid::new(ShardId(9), 1000);
    let eye = |height: f64| {
        WorldPos::from_f64_metres_around(anchor, [c[0], c[1] + r + 1.7 + height, c[2]])
    };
    sim.spawn_ship(player, 1, space_sim::ShipState::at_rest(eye(1.0)));
    let of = |name: &str| -> Vec<&space_scene::SceneBody> {
        placed.iter().filter(|b| b.name == name).collect()
    };
    let crates = of("planetoid crate");
    let moons = of("moon");
    assert_eq!((crates.len(), moons.len()), (3, 3));
    let floating: Vec<(EntityGuid, space_sim::ShipState)> = of("floating crate")
        .iter()
        .map(|b| (b.guid, sim.state_of(b.guid).unwrap()))
        .collect();
    let mut swept = [0.0f64; 3];
    let mut last: Vec<[f64; 3]> = moons
        .iter()
        .map(|m| from_centre(sim.state_of(m.guid).unwrap().pos))
        .collect();
    for _ in 0..600 {
        sim.step();
        for (i, m) in moons.iter().enumerate() {
            let d = from_centre(sim.state_of(m.guid).unwrap().pos);
            let want = demo::PLANETOID_ORBITS[i];
            assert!(
                (len(d) - want).abs() < want * 0.03,
                "moon {i} left its orbit: {} m",
                len(d)
            );
            let cos = (0..3).map(|k| d[k] * last[i][k]).sum::<f64>() / (len(d) * len(last[i]));
            swept[i] += cos.clamp(-1.0, 1.0).acos();
            last[i] = d;
        }
    }
    for (i, s) in swept.iter().enumerate() {
        let period =
            std::f64::consts::TAU * (demo::PLANETOID_ORBITS[i].powi(3) / demo::PLANETOID_GM).sqrt();
        let expected = std::f64::consts::TAU * 20.0 / period;
        assert!(
            (s - expected).abs() / expected < 0.01,
            "moon {i} swept {s} rad, Kepler says {expected}"
        );
    }
    for b in &crates {
        let start = from_centre(b.spawn.state.pos);
        let now = from_centre(sim.state_of(b.guid).unwrap().pos);
        assert!(
            (len(now) - (r + 0.25)).abs() < 0.02,
            "a crate is {} m from the centre",
            len(now)
        );
        let moved = len([0, 1, 2].map(|k| now[k] - start[k]));
        assert!(moved < 0.05, "a resting crate slid {moved} m");
    }
    // The player stands on the top.
    let stood = sim.state_of(player).unwrap();
    let height = from_centre(stood.pos)[1] - r - 1.7;
    assert!(
        height.abs() < 0.05,
        "the player's feet are {height} m off the top"
    );
    assert!(len(stood.vel.map(f64::from)) < 0.05, "{:?}", stood.vel);
    // The pull is taken at a ship's origin, its eye point 1.7 m over the
    // ground: there the inverse square has fallen to GM / (R + 1.68)².
    let g = sim.gravity().at(stood.pos, sim.tick());
    let at_eye = demo::PLANETOID_GM / len(from_centre(stood.pos)).powi(2);
    assert!((len(g) - at_eye).abs() < 1e-9 && at_eye > 1.9, "{g:?}");
    // On the surface itself (where a crate sits) it is the 4 m/s².
    let surface = WorldPos::from_f64_metres_around(anchor, [c[0], c[1] + r, c[2]]);
    let g = sim.gravity().at(surface, sim.tick());
    assert!(
        (len(g) - demo::PLANETOID_SURFACE_GRAVITY).abs() < 1e-6,
        "{g:?}"
    );
    // Weightless where it was: the zero-g corner, unchanged to the bit.
    for (guid, start) in &floating {
        assert_eq!(
            sim.state_of(*guid).unwrap(),
            *start,
            "a floating crate moved"
        );
        let g = sim.gravity().at(start.pos, sim.tick());
        assert!(g.iter().all(|v| v.to_bits() == 0), "{g:?}");
    }
    let g = sim.gravity().at(anchor, sim.tick());
    assert!(g.iter().all(|v| v.to_bits() == 0), "the spawn point: {g:?}");
    // Lift-off: a second of "up".
    for _ in 0..30 {
        let tick = sim.tick() + 1;
        sim.apply_inputs(
            tick,
            &[PlayerInput {
                entity: player,
                thrust: [0.0, 1.0, 0.0],
                torque: [0.0; 3],
                tick,
                fast: false,
            }],
        );
        sim.step();
    }
    let up = from_centre(sim.state_of(player).unwrap().pos)[1] - r - 1.7;
    assert!(up > 2.0, "a second of lift raised the player only {up} m");
}

// ---------------------------------------------------------------- the sky

/// The demo area's solar system, as the physics has it: Earth fixed eight
/// Earth radii from the spawn point, the Moon at its distance and the Sun at
/// one AU at any scene time, the colliders where the gravity says the bodies
/// are, and every reach ending far from the demo area — the spawn point, the
/// zero-g corner and the gravity room feel exactly what they felt before.
#[test]
fn the_sky_moves_on_its_orbits_and_leaves_the_demo_area_alone() {
    use space_scene::solar::{self, EARTH_RADIUS, MOON_DISTANCE, MOON_RADIUS};
    let anchor = WorldPos::from_metres([3_000_000, 0, -2_000_000]);
    let env = space_scene::environment(Scene::DemoArea, anchor);
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let earth_c = solar::EARTH_CENTER;
    assert!(
        (len(earth_c) - 8.0 * EARTH_RADIUS).abs() < 1.0,
        "8 Earth radii out"
    );
    let body = |name: &str| {
        env.gravity
            .bodies
            .iter()
            .find(|b| b.name == name)
            .unwrap()
            .clone()
    };
    for t_us in [
        0,
        1_763_900_000_000_000,
        1_763_900_067_500_000,
        -5_000_000_000,
    ] {
        let at = |name: &str| WorldPos::delta(anchor, body(name).center_at(0, t_us).to_world_pos());
        let earth = at("Earth");
        assert!(
            (0..3).all(|k| (earth[k] - earth_c[k]).abs() < 1e-3),
            "Earth stands still"
        );
        let rel = |p: [f64; 3]| [p[0] - earth[0], p[1] - earth[1], p[2] - earth[2]];
        assert!((len(rel(at("Moon"))) - MOON_DISTANCE).abs() < 1.0);
        assert!((len(rel(at("Sun"))) - solar::AU).abs() < 1.0e3);
        let mars = WorldPos::delta(
            body("Sun").center_at(0, t_us).to_world_pos(),
            body("Mars").center_at(0, t_us).to_world_pos(),
        );
        assert!((len(mars) - 2.2794e11).abs() < 1.0e3);
        // The collider moves with the body, to the bit.
        let moon_static = env
            .statics
            .iter()
            .find(|s| {
                matches!(s.shape, space_sim::ShapeRef::Ball { radius }
                    if (f64::from(radius) - MOON_RADIUS).abs() < 1.0)
            })
            .unwrap();
        let p = moon_static.place_at(t_us);
        let c = body("Moon").center_at(0, t_us);
        assert_eq!((p.origin, p.offset), (c.origin, c.offset));
        // Nothing of it reaches the demo area.
        for b in &env.gravity.bodies {
            if b.name == "planetoid" {
                continue;
            }
            let d = len(b.center_at(0, t_us).delta_to(anchor));
            assert!(
                d > b.reach + 1.0e6,
                "{} reaches to {d} m of the spawn point",
                b.name
            );
            assert!(b.reach.is_finite() && b.reach > b.radius, "{}", b.name);
        }
        let mut field = env.gravity.clone();
        field.clock = space_protocol::SceneClock::new(0, t_us);
        let g = field.at(anchor, 0);
        assert!(
            g.iter().all(|v| v.to_bits() == 0),
            "the spawn point floats: {g:?}"
        );
        let room = WorldPos::from_f64_metres_around(anchor, demo::GRAVITY_ROOM_CENTER);
        assert_eq!(field.at(room, 0), [0.0, -9.81, 0.0]);
        let carry = field.carry(anchor, [0.0; 3], 0, t_us, t_us + 33_333);
        assert!(
            carry.iter().all(|v| v.to_bits() == 0),
            "nothing carries the demo area"
        );
    }
    // The reaches as documented.
    assert!((body("Earth").reach - 6.0 * EARTH_RADIUS).abs() < 1.0);
    assert!((body("Moon").reach - 10.0 * MOON_RADIUS).abs() < 1.0);
    assert!((body("Sun").reach - 10.0 * solar::SUN_RADIUS).abs() < 1.0);
    // On the Moon's surface: its 1.62 m/s².
    let t = 1_763_900_000_000_000;
    let moon = body("Moon").center_at(0, t);
    let surface = WorldPoint::new(
        moon.origin,
        [moon.offset[0], moon.offset[1] + MOON_RADIUS, moon.offset[2]],
    )
    .to_world_pos();
    let mut field = env.gravity.clone();
    field.clock = space_protocol::SceneClock::new(0, t);
    assert!((len(field.at(surface, 0)) - 1.62).abs() < 0.01);
    // The Moon goes round in about four and a half minutes, the year is an
    // hour.
    let month = solar::sky_body("Moon")
        .unwrap()
        .path
        .as_ref()
        .unwrap()
        .terms[0]
        .period_us;
    assert!((month as f64 / 1e6 - 269.3).abs() < 0.5, "{month}");
    let year = solar::period_us(solar::AU, solar::SUN_GM);
    assert!((year as f64 / 1e6 - 3600.0).abs() < 1.0, "{year}");
}

/// Fast travel from the demo area to the Moon in about a minute, then a
/// landing on it, through the same physics a shard runs: the ship steers at
/// the Moon in fast travel, is caught up and carried along inside its reach,
/// slows as the ground comes near, and stands on the Moon at the end.
#[test]
fn a_ship_fast_travels_to_the_moon_and_lands() {
    use space_protocol::PlayerInput;
    use space_scene::solar::MOON_RADIUS;
    let anchor = WorldPos::from_metres([3_000_000, 0, -2_000_000]);
    let mut sim = SimWorld::new(ShardId(9));
    sim.set_scene_clock(space_protocol::SceneClock::new(0, 1_763_900_000_000_000));
    space_scene::install_environment(&mut sim, Scene::DemoArea, anchor).unwrap();
    let ship = EntityGuid::new(ShardId(9), 1);
    sim.spawn_ship(ship, 1, space_sim::ShipState::at_rest(anchor));
    let moon = |sim: &SimWorld| {
        let b = sim
            .gravity()
            .bodies
            .iter()
            .find(|b| b.name == "Moon")
            .unwrap();
        b.center_at(sim.tick(), sim.scene_time())
    };
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut arrived = None;
    let mut top = 0.0f64;
    for tick in 1..=(120 * 30) {
        let s = sim.ship_state(ship).unwrap();
        let to = moon(&sim).delta_to(s.pos).map(|v| -v);
        let surface = len(to) - MOON_RADIUS;
        if surface < 3_000.0 {
            arrived = Some(tick);
            break;
        }
        let dir = to.map(|v| (v / len(to)) as f32);
        let t = sim.tick() + 1;
        sim.apply_inputs(
            t,
            &[PlayerInput {
                entity: ship,
                thrust: dir,
                torque: [0.0; 3],
                tick: t,
                fast: true,
            }],
        );
        sim.step();
        top = top.max(len(sim.ship_state(ship).unwrap().vel.map(f64::from)));
    }
    let arrived = f64::from(arrived.expect("reached the Moon")) / 30.0;
    assert!(arrived < 45.0, "{arrived} s");
    assert!(top > 1.0e7, "fast in between: {top} m/s");
    // Fast travel off (the ship drops to its normal top speed) and down in
    // normal flight, the way a pilot does it: push towards the ground while
    // slower than a twentieth of the height per second, brake otherwise.
    let mut settled = 0;
    for _ in 0..(150 * 30) {
        let s = sim.ship_state(ship).unwrap();
        let up = moon(&sim).delta_to(s.pos);
        let height = len(up) - MOON_RADIUS;
        let v = s.vel.map(f64::from);
        let speed = len(v);
        settled = if height < 2.5 && speed < 0.3 {
            settled + 1
        } else {
            0
        };
        if settled > 60 {
            break;
        }
        // On the ground: hands off (gravity holds it).
        let thrust = if height < 1.0 {
            [0.0; 3]
        } else if speed < (height / 20.0).max(1.0) {
            up.map(|c| (-c / len(up)) as f32)
        } else {
            let k = (speed / 1.5).min(1.0) / speed;
            v.map(|c| (-c * k) as f32)
        };
        let t = sim.tick() + 1;
        sim.apply_inputs(
            t,
            &[PlayerInput {
                entity: ship,
                thrust,
                torque: [0.0; 3],
                tick: t,
                fast: false,
            }],
        );
        sim.step();
    }
    assert!(settled > 60, "came to rest on the ground");
    let s = sim.ship_state(ship).unwrap();
    let height = len(moon(&sim).delta_to(s.pos)) - MOON_RADIUS;
    assert!(
        height > 0.0 && height < 3.0,
        "landed: {height} m over the ground"
    );
    assert!(
        len(s.vel.map(f64::from)) < 0.5,
        "at rest on it: {:?}",
        s.vel
    );
    let g = len(sim.gravity().at(s.pos, sim.tick()));
    assert!(g > 1.5, "the Moon's pull: {g}");
}
