//! Server-side spatial index for interest management.
//!
//! [`InterestGrid`] buckets entities by world cell ([`WorldPos::cell_key`]) in a
//! [`std::collections::HashMap`]. A sphere query walks only the cells that can
//! intersect the query sphere (expanded by the largest stored radius) — by
//! probing that cube of cells, or by filtering the occupied cells when there are
//! fewer of those — then sorts the hits by distance (ties by entity id) and
//! truncates to the caller's cap.
//!
//! # Performance
//!
//! Measured on the development machine (`x86_64`, `--release`, `lto = "thin"`):
//! 1 000 000 [`InterestGrid::insert`] calls in ~0.74 s and 1 000
//! [`InterestGrid::query_sphere`] calls (radius 10 km, cap 1000, ~1000 entities
//! per cell) in ~1.38 s. The debug-mode benchmark tests in
//! `tests/acceptance.rs` run the same workloads with a 10 s budget each: a
//! guard against algorithmic regressions that survives a busy CI runner.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::pos::WorldPos;

/// A small FxHash-style hasher for the grid's integer keys.
///
/// The standard library's SipHash is overkill for `[i64; 3]` cell keys and
/// costs measurable time in the interest hot loop; this mixer is a few
/// multiplies per key. It is *not* DoS-resistant, which is fine: keys are
/// server-generated cell indices, never attacker-controlled strings.
#[derive(Default, Clone, Copy)]
struct CellHasher(u64);

impl Hasher for CellHasher {
    #[inline]
    #[allow(clippy::chunks_exact_to_as_chunks)] // tail bytes are handled below
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_ne_bytes(chunk.try_into().unwrap_or([0; 8])));
        }
        let mut tail = 0u64;
        for &b in chunks.remainder() {
            tail = (tail << 8) | u64::from(b);
        }
        self.add(tail);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.add(i as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

impl CellHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
}

/// `HashMap` specialised to [`CellHasher`].
type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<CellHasher>>;

/// An entity in the grid.
#[derive(Clone, Copy, Debug)]
struct Entry {
    id: u64,
    pos: WorldPos,
    /// Interest radius in metres; the entity can be seen from this far away.
    radius: f32,
}

/// Spatial index of entities, keyed by world cell.
///
/// ```
/// use space_world::{InterestGrid, WorldPos};
///
/// let mut grid = InterestGrid::new();
/// // from_metres takes integer millimetres.
/// grid.insert(1, WorldPos::from_metres([0, 0, 0]), 100.0);
/// grid.insert(2, WorldPos::from_metres([50_000_000, 0, 0]), 100.0); // 50 km away
/// let hits = grid.query_sphere(WorldPos::ZERO, 1000.0, 10);
/// assert_eq!(hits.len(), 1);
/// assert_eq!(hits[0].0, 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct InterestGrid {
    /// cell key -> entities in that cell (a flat `Vec`, cheap to allocate).
    cells: FastMap<[i64; 3], Vec<Entry>>,
    /// entity -> cell key it is currently stored in.
    locations: FastMap<u64, [i64; 3]>,
    /// Largest radius of any stored entity (0 when empty); used to bound the
    /// cell scan of a query. Only ever grows — a removal may leave it larger
    /// than necessary, which costs scan time but never correctness.
    max_radius: f32,
}

impl InterestGrid {
    /// An empty grid.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored entities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    /// Whether the grid holds no entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    /// Insert `entity` at `pos` with interest `radius` (metres).
    ///
    /// Re-inserting an existing entity replaces its previous entry, so the grid
    /// always holds exactly one entry per entity id.
    ///
    /// A server refreshes every live entity every tick, and most of them have
    /// not left their cell (a cell is 1024 m): that case overwrites the entry in
    /// place — no map churn, and no free-then-reallocate of the cell's `Vec`
    /// when the entity is alone in it.
    pub fn insert(&mut self, entity: u64, pos: WorldPos, radius: f32) {
        let key = pos.cell_key();
        if radius > self.max_radius {
            self.max_radius = radius;
        }
        match self.locations.get(&entity).copied() {
            Some(old) if old == key => {
                let slot = self
                    .cells
                    .get_mut(&key)
                    .and_then(|cell| cell.iter_mut().find(|e| e.id == entity));
                if let Some(entry) = slot {
                    entry.pos = pos;
                    entry.radius = radius;
                    return;
                }
                // The two maps always agree, so this is not reached; falling
                // through re-creates the entry rather than trusting that.
            }
            Some(_) => {
                self.remove(entity);
            }
            None => {}
        }
        self.cells.entry(key).or_default().push(Entry {
            id: entity,
            pos,
            radius,
        });
        self.locations.insert(entity, key);
    }

    /// Remove `entity`; returns whether it was present.
    pub fn remove(&mut self, entity: u64) -> bool {
        let Some(key) = self.locations.remove(&entity) else {
            return false;
        };
        if let Some(cell) = self.cells.get_mut(&key) {
            // The order of a cell's entries is never observable (query results
            // are sorted by distance, then id), so the hole is filled with the
            // last entry instead of shifting the tail down.
            if let Some(i) = cell.iter().position(|e| e.id == entity) {
                cell.swap_remove(i);
            }
            if cell.is_empty() {
                self.cells.remove(&key);
            }
        }
        true
    }

    /// Entities whose interest sphere intersects the query sphere, nearest
    /// first, at most `cap` of them.
    ///
    /// Each hit is `(entity, distance)` where `distance` is the metres between
    /// the entity and `center` (0 when the entity sits at the centre). Results
    /// are sorted by ascending distance, ties by ascending entity id, so the
    /// result is a pure function of the stored set — it does not depend on
    /// insertion order or on the hash map's iteration order.
    ///
    /// # Cost
    ///
    /// The candidate cells are the cube of `2·reach + 1` cells per axis around
    /// the centre, where `reach = ceil((radius + largest stored radius) / 1024)`.
    /// That cube is 9 261 cells for a 10 km query and grows with the cube of the
    /// radius, while the grid usually holds far fewer *occupied* cells than that.
    /// The query therefore walks whichever is smaller: the cube (one map probe
    /// per cell) or the occupied cells (one cube-membership test per cell). Both
    /// walks visit exactly the same set of occupied cells, so the result is the
    /// same either way, and an enormous radius costs at most one pass over the
    /// grid instead of an unbounded loop.
    #[must_use]
    pub fn query_sphere(&self, center: WorldPos, radius: f32, cap: usize) -> Vec<(u64, f64)> {
        if cap == 0 || self.is_empty() {
            return Vec::new();
        }
        // Reach in metres: a stored entity is a hit when its own sphere meets
        // the query sphere, so cells are scanned up to radius + max_radius.
        let reach = f64::from(radius) + f64::from(self.max_radius);
        // Saturating cast: an infinite reach becomes `i64::MAX` cells, a NaN
        // one becomes 0 (and then matches nothing, see `collect_hits`).
        let reach_cells = (reach / crate::pos::CELL_SIZE).ceil() as i64;
        let mut hits: Vec<(u64, f64)> = Vec::new();
        // A negative reach (a negative radius larger than every stored one)
        // describes an empty cube.
        if let Ok(reach_cells) = u64::try_from(reach_cells) {
            let side = 2 * u128::from(reach_cells) + 1;
            let cube = side.saturating_mul(side).saturating_mul(side);
            if cube <= self.cells.len() as u128 {
                self.hits_by_cube(center, radius, reach, reach_cells, &mut hits);
            } else {
                self.hits_by_occupied_cells(center, radius, reach, reach_cells, &mut hits);
            }
        }
        // Partial selection: partition the `cap` nearest to the front, then
        // sort only those. Avoids a full sort of every match, which dominates
        // the query when the sphere holds many entities.
        if hits.len() > cap {
            hits.select_nth_unstable_by(cap, |a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            hits.truncate(cap);
        }
        hits.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        // Convert the squared distances of the surviving hits to metres.
        for hit in &mut hits {
            hit.1 = hit.1.sqrt();
        }
        hits
    }

    /// Collect the hits by probing every cell of the query cube.
    ///
    /// Only called when the cube is no larger than the number of occupied
    /// cells, so `reach_cells` is small here (the cube fits in a `usize`).
    fn hits_by_cube(
        &self,
        center: WorldPos,
        radius: f32,
        reach: f64,
        reach_cells: u64,
        hits: &mut Vec<(u64, f64)>,
    ) {
        let base = center.cell;
        let r = i64::try_from(reach_cells).unwrap_or(i64::MAX);
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if cell_out_of_reach([dx, dy, dz], reach) {
                        continue;
                    }
                    let key = [
                        base[0].wrapping_add(dx),
                        base[1].wrapping_add(dy),
                        base[2].wrapping_add(dz),
                    ];
                    if let Some(cell) = self.cells.get(&key) {
                        collect_hits(cell, center, radius, hits);
                    }
                }
            }
        }
    }

    /// Collect the hits by walking the occupied cells and keeping those inside
    /// the query cube.
    ///
    /// The map's iteration order is arbitrary, but it only decides the order
    /// the hits are *found* in; [`InterestGrid::query_sphere`] sorts them.
    fn hits_by_occupied_cells(
        &self,
        center: WorldPos,
        radius: f32,
        reach: f64,
        reach_cells: u64,
        hits: &mut Vec<(u64, f64)>,
    ) {
        let base = center.cell;
        for (key, cell) in &self.cells {
            // The offset the cube walk would have reached this cell with
            // (wrapping, exactly as it builds its keys).
            let delta = [
                key[0].wrapping_sub(base[0]),
                key[1].wrapping_sub(base[1]),
                key[2].wrapping_sub(base[2]),
            ];
            if delta.iter().any(|d| d.unsigned_abs() > reach_cells) {
                continue; // outside the cube
            }
            if cell_out_of_reach(delta, reach) {
                continue;
            }
            collect_hits(cell, center, radius, hits);
        }
    }
}

/// Can the cell at `delta` cells from the query centre's cell be skipped?
///
/// These are the corners of the cell cube that cannot reach the sphere: the
/// closest point of the cell at offset `(dx, dy, dz)` is at least
/// `max(|d| − 1, 0)` cells away per axis (the centre sits somewhere inside the
/// base cell), so the true distance is at least `1024 · sqrt(Σ far²)` metres.
///
/// Computed in `f64` so an offset of billions of cells cannot overflow; for the
/// small offsets of an ordinary query every term is an exact integer.
#[inline]
fn cell_out_of_reach(delta: [i64; 3], reach: f64) -> bool {
    let far = |d: i64| d.unsigned_abs().saturating_sub(1) as f64;
    let (fx, fy, fz) = (far(delta[0]), far(delta[1]), far(delta[2]));
    let min_dist = fx * fx + fy * fy + fz * fz;
    min_dist * crate::pos::CELL_SIZE * crate::pos::CELL_SIZE > reach * reach
}

/// Append the entries of one cell whose sphere meets the query sphere.
#[inline]
fn collect_hits(cell: &[Entry], center: WorldPos, radius: f32, hits: &mut Vec<(u64, f64)>) {
    for entry in cell {
        // Compare squared distances to avoid a `sqrt` per candidate; the
        // square root is taken only for the surviving hits.
        let dist_sq = fast_distance_sq(center, entry.pos);
        let allowed = f64::from(radius) + f64::from(entry.radius);
        if dist_sq <= allowed * allowed {
            hits.push((entry.id, dist_sq));
        }
    }
}

/// Squared distance between two nearby positions, using an `i64` cell delta.
///
/// Equivalent to [`WorldPos::distance_squared`] for entities within a query's
/// reach, but without the `i128` widening that dominates the query loop in
/// debug builds. The cell delta is taken **wrapping**, the same way the query
/// builds its cell keys, so a query that straddles the `i64` wrap of the cell
/// space measures the short way round instead of overflowing.
#[inline]
fn fast_distance_sq(a: WorldPos, b: WorldPos) -> f64 {
    let mut sum = 0.0f64;
    for axis in 0..3 {
        let cell_m = b.cell[axis].wrapping_sub(a.cell[axis]) as f64 * crate::pos::CELL_SIZE;
        let d = cell_m + (f64::from(b.local[axis]) - f64::from(a.local[axis]));
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::InterestGrid;
    use crate::hash::splitmix64;
    use crate::pos::WorldPos;

    /// A grid of `n` entities scattered over a `span_m` cube around the origin,
    /// each with a radius of 0..40 m.
    fn scattered(n: u64, span_m: f64, seed: u64) -> (InterestGrid, Vec<(u64, WorldPos, f32)>) {
        let mut grid = InterestGrid::new();
        let mut all = Vec::new();
        let unit = |h: u64| (h % 2_000_001) as f64 / 1_000_000.0 - 1.0;
        for id in 0..n {
            let h = |k: u64| splitmix64(seed ^ (id * 8 + k));
            let pos = WorldPos::from_f64_metres_around(
                WorldPos::ZERO,
                [
                    unit(h(0)) * span_m,
                    unit(h(1)) * span_m,
                    unit(h(2)) * span_m,
                ],
            );
            let radius = (h(3) % 40) as f32;
            grid.insert(id, pos, radius);
            all.push((id, pos, radius));
        }
        (grid, all)
    }

    /// Ground truth: every stored entity whose sphere meets the query sphere,
    /// by exhaustive `WorldPos::distance`, sorted like `query_sphere` sorts.
    fn brute_force(all: &[(u64, WorldPos, f32)], center: WorldPos, radius: f32) -> Vec<(u64, f64)> {
        let mut hits: Vec<(u64, f64)> = all
            .iter()
            .filter_map(|&(id, pos, r)| {
                let d = WorldPos::distance(center, pos);
                (d <= f64::from(radius) + f64::from(r)).then_some((id, d))
            })
            .collect();
        hits.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        hits
    }

    fn ids(hits: &[(u64, f64)]) -> Vec<u64> {
        hits.iter().map(|h| h.0).collect()
    }

    /// The two cell walks must find exactly the same hits: the choice between
    /// them is a cost decision, never a result decision.
    #[test]
    fn cube_walk_and_occupied_walk_agree() {
        let (grid, all) = scattered(3_000, 12_000.0, 0xA11CE);
        let centers = [
            WorldPos::ZERO,
            WorldPos::from_f64_metres_around(WorldPos::ZERO, [5_000.0, -3_000.0, 900.0]),
            WorldPos::from_f64_metres_around(WorldPos::ZERO, [-11_900.0, 11_900.0, 0.0]),
            WorldPos::from_f64_metres_around(WorldPos::ZERO, [40_000.0, 0.0, 0.0]),
        ];
        for center in centers {
            for radius in [0.0f32, 1.0, 600.0, 2_500.0, 9_000.0] {
                let reach = f64::from(radius) + f64::from(grid.max_radius);
                let reach_cells = (reach / crate::pos::CELL_SIZE).ceil() as u64;
                let mut by_cube = Vec::new();
                grid.hits_by_cube(center, radius, reach, reach_cells, &mut by_cube);
                let mut by_cells = Vec::new();
                grid.hits_by_occupied_cells(center, radius, reach, reach_cells, &mut by_cells);
                let order = |a: &(u64, f64), b: &(u64, f64)| a.0.cmp(&b.0);
                by_cube.sort_unstable_by(order);
                by_cells.sort_unstable_by(order);
                // Same entities, and the very same squared distances.
                assert_eq!(by_cube, by_cells, "radius {radius} at {center:?}");

                // And the public query agrees with an exhaustive search.
                let got = grid.query_sphere(center, radius, usize::MAX);
                let want = brute_force(&all, center, radius);
                assert_eq!(ids(&got), ids(&want), "radius {radius} at {center:?}");
                for (g, w) in got.iter().zip(&want) {
                    assert!((g.1 - w.1).abs() < 1e-6, "{g:?} vs {w:?}");
                }
            }
        }
    }

    /// A radius whose cell cube is astronomically large must cost one pass over
    /// the occupied cells, not a loop over the cube (10^12 m is a cube of
    /// ~7·10^27 cells: the old triple loop never returned).
    #[test]
    fn a_huge_radius_walks_the_occupied_cells() {
        let (grid, all) = scattered(500, 50_000.0, 0xB16);
        for radius in [1.0e12f32, f32::MAX, f32::INFINITY] {
            let hits = grid.query_sphere(WorldPos::ZERO, radius, usize::MAX);
            assert_eq!(hits.len(), all.len(), "radius {radius} sees everything");
            for w in hits.windows(2) {
                assert!(
                    w[0].1 < w[1].1 || (w[0].1 == w[1].1 && w[0].0 < w[1].0),
                    "sorted by distance, then id: {w:?}"
                );
            }
        }
        // One entity with a planet-sized interest radius widens every later
        // query's reach to ~10^5 cells per axis; a small query still answers,
        // and still only reports what its own sphere meets.
        let (mut grid, mut all) = scattered(500, 50_000.0, 0xB16);
        let far = WorldPos::from_f64_metres_around(WorldPos::ZERO, [3.0e8, 0.0, 0.0]);
        grid.insert(10_000, far, 1.0e8);
        all.push((10_000, far, 1.0e8));
        let got = grid.query_sphere(WorldPos::ZERO, 2_000.0, usize::MAX);
        assert_eq!(ids(&got), ids(&brute_force(&all, WorldPos::ZERO, 2_000.0)));
        assert!(!ids(&got).contains(&10_000), "3e8 m away, radius 1e8 m");
        // From 2.5e8 m out the gap to it is 5e7 m, inside its 1e8 m radius.
        let near_it = WorldPos::from_f64_metres_around(WorldPos::ZERO, [2.5e8, 0.0, 0.0]);
        assert_eq!(ids(&grid.query_sphere(near_it, 2_000.0, 10)), vec![10_000]);
    }

    /// A NaN or negative radius matches nothing (and does not loop or panic).
    #[test]
    fn degenerate_radii_match_nothing() {
        let (grid, _) = scattered(200, 3_000.0, 7);
        assert!(grid
            .query_sphere(WorldPos::ZERO, f32::NAN, usize::MAX)
            .is_empty());
        assert!(grid
            .query_sphere(WorldPos::ZERO, -1.0e9, usize::MAX)
            .is_empty());
        assert!(grid
            .query_sphere(WorldPos::ZERO, f32::NEG_INFINITY, usize::MAX)
            .is_empty());
    }

    /// Refreshing an entity that stays in its cell overwrites the entry in
    /// place; one that changes cell moves; neither duplicates it.
    #[test]
    fn reinsert_updates_in_place_and_moves_across_cells() {
        let at = |m: [f64; 3]| WorldPos::from_f64_metres_around(WorldPos::ZERO, m);
        // The cells the positions below fall in (x = 0, ~5 km and ~9 km).
        let (cell_0, cell_5, cell_9) = ([0i64, 0, 0], [5i64, 0, 0], [9i64, 0, 0]);
        let mut grid = InterestGrid::new();
        grid.insert(1, at([10.0, 0.0, 0.0]), 5.0);
        grid.insert(2, at([20.0, 0.0, 0.0]), 5.0);
        grid.insert(3, at([30.0, 0.0, 0.0]), 5.0);
        assert_eq!((grid.len(), grid.cells.len()), (3, 1));
        let allocation = grid.cells[&cell_0].as_ptr();

        // Same cell: new position and radius, same storage, same neighbours.
        grid.insert(2, at([200.0, 0.0, 0.0]), 9.0);
        assert_eq!((grid.len(), grid.cells.len()), (3, 1));
        assert_eq!(grid.cells[&cell_0].len(), 3);
        assert_eq!(
            grid.cells[&cell_0].as_ptr(),
            allocation,
            "a same-cell refresh must not reallocate the cell"
        );
        let hits = grid.query_sphere(at([0.0; 3]), 1_000.0, 10);
        assert_eq!(hits, vec![(1, 10.0), (3, 30.0), (2, 200.0)]);
        assert_eq!(grid.max_radius, 9.0);
        // The old position is gone, the new radius is in force.
        let none: Vec<u64> = Vec::new();
        assert_eq!(ids(&grid.query_sphere(at([20.0, 0.0, 0.0]), 0.0, 10)), none);
        assert_eq!(
            ids(&grid.query_sphere(at([208.0, 0.0, 0.0]), 0.0, 10)),
            vec![2]
        );

        // A lone entity refreshed in place keeps its cell alive (it used to be
        // removed and re-created, freeing and reallocating the `Vec`).
        grid.insert(7, at([5_000.0, 0.0, 0.0]), 1.0);
        let lone = grid.cells[&cell_5].as_ptr();
        grid.insert(7, at([5_001.0, 0.0, 0.0]), 1.0);
        assert_eq!(grid.cells[&cell_5].as_ptr(), lone);
        assert_eq!(grid.len(), 4);

        // Another cell: the entry moves, the emptied cell is dropped.
        grid.insert(7, at([9_000.0, 0.0, 0.0]), 1.0);
        assert_eq!(grid.len(), 4);
        assert!(!grid.cells.contains_key(&cell_5));
        assert_eq!(grid.cells[&cell_9].len(), 1);
        assert_eq!(
            ids(&grid.query_sphere(at([9_000.0, 0.0, 0.0]), 10.0, 10)),
            vec![7]
        );

        // Removal from the middle of a cell leaves the others intact.
        assert!(grid.remove(1));
        assert!(!grid.remove(1));
        let hits = grid.query_sphere(at([0.0; 3]), 1_000.0, 10);
        assert_eq!(ids(&hits), vec![3, 2]);
    }

    /// A query that straddles the `i64` wrap of the cell space finds its
    /// neighbour across the seam, at the short distance. (The cell keys were
    /// always built wrapping; the distance used a plain subtraction, which
    /// overflowed — a panic in a debug build.)
    #[test]
    fn a_query_across_the_cell_wrap_measures_the_short_way() {
        let mut grid = InterestGrid::new();
        // 12 m above the lower face of the lowest cell...
        let below = WorldPos::new([i64::MIN, 0, 0], [-500.0, 0.0, 0.0]);
        // ...and 12 m below the upper face of the highest: 24 m apart.
        let above = WorldPos::new([i64::MAX, 0, 0], [500.0, 0.0, 0.0]);
        grid.insert(1, below, 0.0);
        grid.insert(2, above, 0.0);
        assert_eq!(
            grid.query_sphere(above, 100.0, 10),
            vec![(2, 0.0), (1, 24.0)]
        );
        assert_eq!(
            grid.query_sphere(below, 100.0, 10),
            vec![(1, 0.0), (2, 24.0)]
        );
        assert_eq!(ids(&grid.query_sphere(below, 23.0, 10)), vec![1]);
    }
}
