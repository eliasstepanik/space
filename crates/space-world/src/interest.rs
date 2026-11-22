//! Server-side spatial index for interest management.
//!
//! [`InterestGrid`] buckets entities by world cell ([`WorldPos::cell_key`]) in a
//! [`std::collections::HashMap`]. A sphere query walks only the cells that can
//! intersect the query sphere (expanded by the largest stored radius), then
//! sorts the hits by distance and truncates to the caller's cap.
//!
//! # Performance
//!
//! Measured on the development machine (`x86_64`, `--release`, `lto = "thin"`):
//! 1 000 000 [`InterestGrid::insert`] calls in ~0.74 s and 1 000
//! [`InterestGrid::query_sphere`] calls (radius 10 km, cap 1000, ~1000 entities
//! per cell) in ~1.38 s. The debug-mode benchmark test in
//! `tests/acceptance.rs` asserts a 2 s budget for the same workload.

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
    pub fn insert(&mut self, entity: u64, pos: WorldPos, radius: f32) {
        self.remove(entity);
        let key = pos.cell_key();
        self.cells.entry(key).or_default().push(Entry {
            id: entity,
            pos,
            radius,
        });
        self.locations.insert(entity, key);
        if radius > self.max_radius {
            self.max_radius = radius;
        }
    }

    /// Remove `entity`; returns whether it was present.
    pub fn remove(&mut self, entity: u64) -> bool {
        let Some(key) = self.locations.remove(&entity) else {
            return false;
        };
        if let Some(cell) = self.cells.get_mut(&key) {
            cell.retain(|e| e.id != entity);
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
    /// are sorted by ascending distance; ties keep a stable, arbitrary order.
    #[must_use]
    pub fn query_sphere(&self, center: WorldPos, radius: f32, cap: usize) -> Vec<(u64, f64)> {
        if cap == 0 || self.is_empty() {
            return Vec::new();
        }
        // Reach in metres: a stored entity is a hit when its own sphere meets
        // the query sphere, so cells are scanned up to radius + max_radius.
        let reach = f64::from(radius) + f64::from(self.max_radius);
        let reach_cells = (reach / crate::pos::CELL_SIZE).ceil() as i64;
        let mut hits: Vec<(u64, f64)> = Vec::new();
        let base = center.cell;
        for dx in -reach_cells..=reach_cells {
            for dy in -reach_cells..=reach_cells {
                for dz in -reach_cells..=reach_cells {
                    // Skip the corners of the cell cube that cannot reach the
                    // sphere: the closest point of cell offset (dx, dy, dz) is
                    // at least `max(|d|-1, 0)` cells away per axis (the centre
                    // sits somewhere inside the base cell), so the true
                    // distance is at least `1024 * sqrt(sum far^2)` metres.
                    let far_x = (dx.abs() - 1).max(0);
                    let far_y = (dy.abs() - 1).max(0);
                    let far_z = (dz.abs() - 1).max(0);
                    let min_dist = (far_x * far_x + far_y * far_y + far_z * far_z) as f64;
                    if min_dist * crate::pos::CELL_SIZE * crate::pos::CELL_SIZE > reach * reach {
                        continue;
                    }
                    let key = [
                        base[0].wrapping_add(dx),
                        base[1].wrapping_add(dy),
                        base[2].wrapping_add(dz),
                    ];
                    let Some(cell) = self.cells.get(&key) else {
                        continue;
                    };
                    for entry in cell {
                        // Compare squared distances to avoid a `sqrt` per
                        // candidate; the square root is taken only for the
                        // surviving hits. Entities live within `reach` of the
                        // centre, so the cell delta fits comfortably in i64.
                        let dist_sq = fast_distance_sq(center, entry.pos);
                        let allowed = f64::from(radius) + f64::from(entry.radius);
                        if dist_sq <= allowed * allowed {
                            hits.push((entry.id, dist_sq));
                        }
                    }
                }
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
}

/// Squared distance between two nearby positions, using an `i64` cell delta.
///
/// Equivalent to [`WorldPos::distance_squared`] whenever the cell delta fits in
/// `i64` (always true for entities within a query's reach), but without the
/// `i128` widening that dominates the query loop in debug builds.
#[inline]
fn fast_distance_sq(a: WorldPos, b: WorldPos) -> f64 {
    let mut sum = 0.0f64;
    for axis in 0..3 {
        let cell_m = (b.cell[axis] - a.cell[axis]) as f64 * crate::pos::CELL_SIZE;
        let d = cell_m + (f64::from(b.local[axis]) - f64::from(a.local[axis]));
        sum += d * d;
    }
    sum
}
