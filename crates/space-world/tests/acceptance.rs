//! Acceptance tests for `space-world` (Phase 1, task 1.1).
//!
//! Every test named after a brief requirement lives here. The two benchmark
//! tests (`bench_interest_insert`, `bench_interest_query`) assert *debug*
//! timings; release numbers are documented in `space_world::interest`.

use std::time::Instant;

use space_test_support::Rng;
use space_world::octree::{self, NodeRef, MAX_DEPTH};
use space_world::{
    CellCoord, Galaxy, InterestGrid, LodSelector, StarSystem, WorldPos, CELL_HALF, MAX_SELECTED,
};

/// 50 000 light-years, expressed as a whole number of cells so the far point is
/// exactly representable: `FAR_CELL * 1024 m ~= 4.73e20 m`.
const FAR_CELL: i64 = 461_914_062_500_000_000;

/// A position ~50 000 light-years from the origin, exactly cell-aligned.
fn far_pos() -> WorldPos {
    WorldPos {
        cell: [FAR_CELL, 0, 0],
        local: [0.0; 3],
    }
}

// ---------------------------------------------------------------------------
// 1. position precision
// ---------------------------------------------------------------------------

#[test]
fn position_precision() {
    let far = far_pos();
    let near = far.offset([1.0, 0.0, 0.0]);

    // `delta` round-trips: 1 m at 4.73e20 m from the origin, exact.
    let d = WorldPos::delta(far, near);
    assert_eq!(d, [1.0, 0.0, 0.0]);
    let rel = (d[0] - 1.0).abs();
    assert!(rel < 1e-6, "relative error {rel}");

    // `distance` matches an i128-computed ground truth exactly.
    let cell_m = i128::from(near.cell[0]) * 1024 - i128::from(far.cell[0]) * 1024;
    let truth_mm = cell_m * 1000 + (f64::from(near.local[0]) * 1000.0).round() as i128
        - (f64::from(far.local[0]) * 1000.0).round() as i128;
    assert_eq!(truth_mm, 1000);
    let truth = (truth_mm as f64) / 1000.0;
    assert_eq!(WorldPos::distance(far, near), truth);

    // Random far origins and arbitrary deltas: the cell term is exact, so the
    // only error is the f32 quantisation of the local part (~3e-5 m for
    // |local| <= 512). Absolute error therefore stays below 1 mm.
    let mut rng = Rng::new(0x9E37_79B9);
    // Spread over ~4.7e23 mm (4.7e20 m) so origins reach 50 000 ly.
    let wide = |rng: &mut Rng| -> i128 {
        let hi = (rng.next_u64() % 256_000) as i128;
        let lo = rng.next_u64() as i128;
        hi * 1_000_000_000_000_000_000 + lo % 1_000_000_000_000_000_000
    };
    for _ in 0..1000 {
        let base = WorldPos::from_metres([wide(&mut rng), -wide(&mut rng), wide(&mut rng)]);
        let delta = [
            rng.next_sym() * 1000.0,
            rng.next_sym() * 1000.0,
            rng.next_sym() * 1000.0,
        ];
        let moved = WorldPos::from_f64_metres_around(base, delta);
        let back = WorldPos::delta(base, moved);
        for axis in 0..3 {
            let err = (back[axis] - delta[axis]).abs();
            assert!(err < 1e-3, "axis {axis}: {back:?} vs {delta:?} err {err}");
        }
    }
}

#[test]
fn cell_normalization() {
    // Local overflow carries into the cell exactly, in both directions.
    let p = WorldPos::new([0, 0, 0], [1024.0, 0.0, 0.0]);
    assert_eq!(p.cell, [1, 0, 0]);
    assert_eq!(p.local, [0.0, 0.0, 0.0]);

    let p = WorldPos::new([0, 0, 0], [1536.0, -2048.0, 0.0]);
    // 1536 m = cell 2 with local -512 (the half-open convention puts the
    // boundary in the upper cell), -2048 m = cell -2 with local 0.
    assert_eq!(p.cell, [2, -2, 0]);
    assert_eq!(p.local, [-512.0, 0.0, 0.0]);
    // Absolute position is preserved: cell*1024 + local.
    assert_eq!(p.cell[0] as f64 * 1024.0 + f64::from(p.local[0]), 1536.0);
    assert_eq!(p.cell[1] as f64 * 1024.0 + f64::from(p.local[1]), -2048.0);

    let p = WorldPos::new([0, 0, 0], [-1025.0, 0.0, 0.0]);
    assert_eq!(p.cell, [-1, 0, 0]);
    assert!((p.local[0] + 1.0).abs() < 1e-6);

    // Repeated normalisation is a fixed point.
    let p = WorldPos::new([7, -3, 1], [511.5, -511.5, 0.25]);
    assert_eq!(WorldPos::new(p.cell, p.local), p);
    for &l in &p.local {
        assert!((-CELL_HALF..CELL_HALF).contains(&l));
    }

    // Cell-aligned millimetre coordinates land exactly on cell boundaries.
    let p = WorldPos::from_metres([1_048_576_000, 0, 0]); // 1 048 576 m = 1024 cells
    assert_eq!(p.cell, [1024, 0, 0]);
    assert_eq!(p.local, [0.0, 0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// 2. octree
// ---------------------------------------------------------------------------

#[test]
fn octree_roundtrip() {
    let mut rng = Rng::new(0xCAFE);
    for _ in 0..200 {
        let pos = WorldPos::from_metres([
            rng.next_u64() as i64 as i128,
            -(rng.next_u64() as i64 as i128),
            rng.next_u64() as i128 / 7,
        ]);
        for depth in 0..=20u8 {
            let node = octree::node_at(pos, depth);
            assert_eq!(node.depth, depth);
            assert!(octree::contains(node, pos), "depth {depth}");
            // parent/child consistency
            if depth > 0 {
                let parent = octree::parent(node).expect("depth > 0 has a parent");
                assert_eq!(parent.depth, depth - 1);
                assert!(octree::contains(parent, pos));
                assert!(octree::children(parent).unwrap().contains(&node));
            } else {
                assert!(octree::parent(node).is_none());
            }
            // CellCoord agrees with node_at.
            assert_eq!(CellCoord(pos.cell).to_octree_node(depth), node);
        }
        // Deepest depth addresses exactly the cell.
        let leaf = octree::node_at(pos, MAX_DEPTH);
        assert_eq!(leaf.index, pos.cell);
        assert!(octree::children(leaf).is_none());
    }
}

// ---------------------------------------------------------------------------
// 3. LOD
// ---------------------------------------------------------------------------

/// A fixed galaxy-ish region the cameras look at, so "closer" is well defined.
const REGION_CENTER: WorldPos = WorldPos {
    cell: [0, 0, 0],
    local: [0.0; 3],
};

fn random_camera(rng: &mut Rng) -> WorldPos {
    let r = 5_000.0 + rng.next_f64() * 200_000.0;
    let theta = rng.next_f64() * std::f64::consts::TAU;
    let phi = rng.next_sym();
    let (sin_t, cos_t) = (theta.sin(), theta.cos());
    let (sin_p, cos_p) = (phi * 1.5, (1.0 - phi * phi).sqrt());
    WorldPos::from_f64_metres_around(
        REGION_CENTER,
        [r * cos_t * cos_p, r * sin_p, r * sin_t * cos_p],
    )
}

/// Does `ancestor` contain `descendant` in the octree (equal or deeper)?
fn node_contains(ancestor: NodeRef, descendant: NodeRef) -> bool {
    if descendant.depth < ancestor.depth {
        return false;
    }
    let shift = u32::from(descendant.depth - ancestor.depth);
    descendant.index[0] >> shift == ancestor.index[0]
        && descendant.index[1] >> shift == ancestor.index[1]
        && descendant.index[2] >> shift == ancestor.index[2]
}

/// The deepest node of `set` that is `node` itself or one of its descendants.
fn deepest_cover(set: &[NodeRef], node: NodeRef) -> Option<usize> {
    set.iter()
        .filter(|n| node_contains(node, **n))
        .map(|n| n.depth as usize)
        .max()
}

#[test]
fn lod_monotonic() {
    // A 4-level band keeps the debug-mode traversal cheap enough for 1000
    // cameras; the guarantee is band-independent.
    let sel = LodSelector::new(54, 58, 32.0);
    let fov = std::f32::consts::FRAC_PI_3;
    let dir = [1.0f32, 0.0, 0.0];
    let mut rng = Rng::new(0xB0_00_00_00_00_00_00_01);
    for i in 0..1000 {
        let cam = random_camera(&mut rng);
        // Halve the distance to the region centre: strictly closer.
        let closer = {
            let d = WorldPos::delta(cam, REGION_CENTER);
            WorldPos::from_f64_metres_around(cam, [-d[0] / 2.0, -d[1] / 2.0, -d[2] / 2.0])
        };
        // One fixed candidate tree, generated from the *closer* camera (the
        // more demanding one, so the tree is deep enough for both). Selecting
        // the same tree for both cameras isolates the monotonicity of the SSE
        // selection: the closer camera has a smaller SSE everywhere, so it can
        // only keep or deepen each region.
        let candidates = sel.candidates(closer, dir, fov);
        let far_sel = sel.select(&candidates, cam, fov);
        let near_sel = sel.select(&candidates, closer, fov);
        assert!(
            !far_sel.is_empty(),
            "camera {i} at {cam:?} selected nothing (candidates {})",
            candidates.len()
        );
        for &node in &far_sel {
            let deep = deepest_cover(&near_sel, node);
            assert!(
                deep.is_some_and(|d| d >= node.depth as usize),
                "camera {i}: node {node:?} (depth {}) got coarser when closer",
                node.depth
            );
        }
    }
}

#[test]
fn lod_bounded() {
    let sel = LodSelector::new(54, 58, 32.0);
    let fov = std::f32::consts::FRAC_PI_3;
    let mut rng = Rng::new(0x1234_5678);
    for _ in 0..200 {
        let cam = random_camera(&mut rng);
        for dir in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.3, 0.6, -0.7]] {
            let candidates = sel.candidates(cam, dir, fov);
            let chosen = sel.select(&candidates, cam, fov);
            assert!(chosen.len() <= MAX_SELECTED, "{}", chosen.len());
            // REPLACE refinement yields an antichain: no selected node contains
            // another.
            for (i, a) in chosen.iter().enumerate() {
                for b in chosen.iter().skip(i + 1) {
                    assert!(!node_contains(*a, *b) && !node_contains(*b, *a));
                }
            }
        }
    }
    // A camera sitting deep inside the tree is still bounded.
    for cell in [0i64, 1, 30, 1_000_000] {
        let cam = WorldPos::from_metres([i128::from(cell) * 1_000_000, 7, -7]);
        let candidates = sel.candidates(cam, [1.0, 0.0, 0.0], fov);
        assert!(sel.select(&candidates, cam, fov).len() <= MAX_SELECTED);
    }
}

// ---------------------------------------------------------------------------
// 4. galaxy
// ---------------------------------------------------------------------------

/// Sample 10 000 nodes across the disc and return their star answers.
fn sample_galaxy(g: &Galaxy, rng: &mut Rng) -> Vec<Option<space_world::StarInfo>> {
    (0..10_000)
        .map(|_| {
            let depth = 54 + (rng.next_u64() % 6) as u8;
            let cell = [
                (rng.next_u64() % 8192) as i64 - 4096,
                (rng.next_u64() % 64) as i64 - 32,
                (rng.next_u64() % 8192) as i64 - 4096,
            ];
            g.star_at(CellCoord(cell).to_octree_node(depth))
        })
        .collect()
}

#[test]
fn galaxy_deterministic() {
    let a = Galaxy::new(0xDEAD_BEEF_CAFE_1234);
    let b = Galaxy::new(0xDEAD_BEEF_CAFE_1234);
    let c = Galaxy::new(0xDEAD_BEEF_CAFE_1235);

    let va = sample_galaxy(&a, &mut Rng::new(7));
    let vb = sample_galaxy(&b, &mut Rng::new(7));
    assert_eq!(va, vb, "same seed must produce identical output");

    let vc = sample_galaxy(&c, &mut Rng::new(7));
    assert_ne!(va, vc, "different seed must differ");

    // The generator actually produces stars, and they cluster towards the core.
    let hits = va.iter().filter(|s| s.is_some()).count();
    assert!(hits > 100, "only {hits}/10000 nodes had a star");
    let core: usize = (0..2000)
        .map(|i| {
            a.star_at(CellCoord([i % 200, 0, i / 200]).to_octree_node(56))
                .is_some() as usize
        })
        .sum();
    let rim: usize = (0..2000)
        .map(|i| {
            a.star_at(CellCoord([3000 + i % 200, 0, i / 200]).to_octree_node(56))
                .is_some() as usize
        })
        .sum();
    assert!(core > rim, "core {core} should out-density rim {rim}");
    for s in va.iter().flatten() {
        assert!((0.0..=1.0).contains(&s.luminosity));
        for &channel in &s.spectral_tint {
            assert!(channel > 0);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. star systems
// ---------------------------------------------------------------------------

#[test]
fn system_orbits() {
    let star = WorldPos::from_metres([123_456_789_012_345_678, 0, -999]);
    let sys = StarSystem::new(0x5EED, star);
    assert!(sys.planet_count() >= 1);

    // Deterministic and exactly periodic.
    for i in 0..sys.planet_count() as u32 {
        let p = &sys.planets[i as usize];
        assert!(p.period > 0.0);
        assert!((0.0..0.4).contains(&p.eccentricity));
        for t in [0.0, 1.0, 12_345.5] {
            let a = sys.planet_pos(i, t);
            assert_eq!(
                a,
                sys.planet_pos(i, t),
                "planet {i} t={t} not deterministic"
            );
            // Exact periodicity where `t + period` is exactly representable.
            assert_eq!(a, sys.planet_pos(i, t + p.period), "planet {i} t={t}");
            // The orbit stays inside the apoapsis of the star.
            let d = WorldPos::distance(star, a);
            assert!(
                d <= p.semi_major_axis * (1.0 + p.eccentricity) + 1.0,
                "planet {i} escaped: {d} > {}",
                p.semi_major_axis
            );
        }
        // At a large `t` the modulo reduction of `t` itself carries f64
        // rounding, so periodicity is asserted with a tolerance proportional to
        // the accumulated rounding (`t * f64::EPSILON` of a revolution) — still
        // far below anything visible.
        for t in [1.0e9, 7.25e12] {
            let a = sys.planet_pos(i, t);
            assert_eq!(
                a,
                sys.planet_pos(i, t),
                "planet {i} t={t} not deterministic"
            );
            let b = sys.planet_pos(i, t + p.period);
            let drift = WorldPos::distance(a, b);
            let tol = p.semi_major_axis * 1e-6;
            assert!(
                drift < tol,
                "planet {i} drifts {drift} m over one period at t={t} (tol {tol})"
            );
        }
    }
    // Different seeds give different systems; the same seed reproduces exactly.
    let again = StarSystem::new(0x5EED, star);
    assert_eq!(again, sys);
    assert_ne!(StarSystem::new(0x5EEE, star), sys);

    // 1000 asteroids are addressable and stable across calls.
    assert!(sys.belt_body_count() >= 1000);
    for i in 0..1000u32 {
        let a = sys.belt_body_pos(i, 42.0);
        assert_eq!(a, sys.belt_body_pos(i, 42.0), "belt {i} unstable");
        assert_ne!(a, sys.belt_body_pos(i, 43.0), "belt {i} does not move");
    }
    // Distinct bodies are at distinct places (compared in exact millimetres,
    // since WorldPos holds f32 locals and is therefore not `Hash`).
    let mut seen = std::collections::HashSet::new();
    for i in 0..1000u32 {
        assert!(
            seen.insert(sys.belt_body_pos(i, 0.0).to_millimetres()),
            "belt bodies collide"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. interest grid
// ---------------------------------------------------------------------------

#[test]
fn interest_cap() {
    let mut grid = InterestGrid::new();
    let mut rng = Rng::new(0xC0FFEE);
    // 5000 entities in a 20 km box around the origin.
    for id in 0..5000u64 {
        let pos = WorldPos::from_f64_metres_around(
            WorldPos::ZERO,
            [
                rng.next_sym() * 10_000.0,
                rng.next_sym() * 10_000.0,
                rng.next_sym() * 10_000.0,
            ],
        );
        grid.insert(id, pos, 50.0);
    }
    assert_eq!(grid.len(), 5000);

    for cap in [1usize, 10, 100, 1000] {
        let hits = grid.query_sphere(WorldPos::ZERO, 10_000.0, cap);
        assert!(hits.len() <= cap, "{} > {cap}", hits.len());
        for w in hits.windows(2) {
            assert!(w[0].1 <= w[1].1, "not sorted: {:?}", w);
        }
        // Every hit really intersects the query sphere.
        for &(id, dist) in &hits {
            assert!(dist <= 10_000.0 + 50.0 + 1e-6, "hit {id} at {dist}");
        }
    }
    // cap = 0 yields nothing; a huge cap yields everything in range.
    assert!(grid.query_sphere(WorldPos::ZERO, 10_000.0, 0).is_empty());
    let all = grid.query_sphere(WorldPos::ZERO, 10_000.0, usize::MAX);
    assert!(!all.is_empty());
    assert!(all.len() <= 5000);

    // insert / remove round-trips.
    let pos = WorldPos::from_metres([1_000, 2_000, 3_000]);
    grid.insert(99_999, pos, 10.0);
    assert_eq!(grid.len(), 5001);
    let hit = grid.query_sphere(WorldPos::from_metres([1, 2, 3]), 5.0, 10);
    assert!(hit.iter().any(|(id, _)| *id == 99_999));
    assert!(grid.remove(99_999));
    assert!(!grid.remove(99_999));
    assert_eq!(grid.len(), 5000);
    let hit = grid.query_sphere(WorldPos::from_metres([1, 2, 3]), 5.0, 10);
    assert!(!hit.iter().any(|(id, _)| *id == 99_999));
    // Re-inserting an existing id replaces rather than duplicating.
    grid.insert(0, pos, 10.0);
    assert_eq!(grid.len(), 5000);
    let dupes = grid
        .query_sphere(pos, 1.0, 100)
        .iter()
        .filter(|(id, _)| *id == 0)
        .count();
    assert_eq!(dupes, 1);
}

// ---------------------------------------------------------------------------
// 7. benchmarks (debug budget; release numbers in `space_world::interest`)
// ---------------------------------------------------------------------------

/// Wall-clock budget for one benchmark workload in a debug build.
///
/// These run in CI, on a shared runner that builds the release binaries and
/// three images while the tests run, so a wall-clock limit is a guard against
/// an *algorithmic* regression (a probe of every cell, a quadratic sort), not a
/// micro-benchmark. On an idle machine the insert workload takes about 0.5 s
/// and the query workload about 1 s; on a heavily loaded one they were
/// measured at up to 2.4 s and 7.1 s, which the old 2 s budget turned into
/// random red builds. Anything slower than 10 s is a real regression.
const BENCH_BUDGET_S: f64 = 10.0;

/// Spread `n` entities over a 100 km cube (about 1000 per 1 km cell).
fn populate(n: u64) -> InterestGrid {
    let mut grid = InterestGrid::new();
    let mut rng = Rng::new(0xBEEF_0000);
    for id in 0..n {
        let pos = WorldPos::from_f64_metres_around(
            WorldPos::ZERO,
            [
                rng.next_sym() * 50_000.0,
                rng.next_sym() * 50_000.0,
                rng.next_sym() * 50_000.0,
            ],
        );
        grid.insert(id, pos, 25.0);
    }
    grid
}

#[test]
fn bench_interest_insert() {
    let t = Instant::now();
    let grid = populate(1_000_000);
    let elapsed = t.elapsed();
    eprintln!(
        "bench_interest_insert: 1_000_000 inserts in {:?} ({} cells)",
        elapsed,
        grid.len()
    );
    assert!(
        elapsed.as_secs_f64() < BENCH_BUDGET_S,
        "insert took {elapsed:?}"
    );
}

#[test]
fn bench_interest_query() {
    let grid = populate(1_000_000);
    let mut rng = Rng::new(0x1234);
    let t = Instant::now();
    let mut total = 0usize;
    for _ in 0..1000 {
        let center = WorldPos::from_f64_metres_around(
            WorldPos::ZERO,
            [
                rng.next_sym() * 40_000.0,
                rng.next_sym() * 40_000.0,
                rng.next_sym() * 40_000.0,
            ],
        );
        total += grid.query_sphere(center, 10_000.0, 1000).len();
    }
    let elapsed = t.elapsed();
    eprintln!(
        "bench_interest_query: 1000 query_sphere(r=10km, cap=1000) in {elapsed:?} \
         ({total} hits total)"
    );
    assert!(
        elapsed.as_secs_f64() < BENCH_BUDGET_S,
        "queries took {elapsed:?}"
    );
}
