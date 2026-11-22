//! World positions: `i64` cell index + `f32` offset inside the cell.
//!
//! The world is partitioned into cubic cells of [`CELL_SIZE`] = 1024 m. A
//! [`WorldPos`] stores the integer cell index plus a sub-cell offset in metres,
//! which keeps the *large* part of a coordinate exact (`i64`) and the *small*
//! part in `f32` where it has ~1e-4 m resolution. Distances are always derived
//! from the cell delta in `i128` metres plus a local correction, so precision
//! does not degrade with distance from the origin.
//!
//! There is no global origin: every conversion that needs a reference point
//! takes it as an argument ([`WorldPos::from_f64_metres_around`],
//! [`WorldPos::delta`], [`WorldPos::distance`]).

/// Edge length of one world cell, in metres.
pub const CELL_SIZE: f64 = 1024.0;

/// Half a cell edge, in metres: the half-open local range is
/// `[-CELL_HALF, CELL_HALF)`.
pub const CELL_HALF: f32 = 512.0;

/// Millimetres per metre (used by the exact integer conversions).
const MM_PER_M: i128 = 1000;

/// Metres per cell as an integer.
const CELL_M: i128 = 1024;

/// Floor division for `i128` (Rust's `/` truncates toward zero).
const fn floor_div(a: i128, b: i128) -> i128 {
    let q = a / b;
    if a % b != 0 && (a < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}

/// A position in the world: cell index + offset inside the cell.
///
/// `local` is normalised into the half-open interval `[-CELL_HALF, CELL_HALF)`
/// by every constructor, so the representation is canonical: two equal
/// positions always compare equal. (The brief's `|local| < CELL_HALF` holds for
/// every value except the single boundary `-CELL_HALF`, which is unavoidable
/// for a position that falls exactly on a cell boundary: it must belong to one
/// of the two neighbouring cells.)
///
/// ```
/// use space_world::{WorldPos, CELL_HALF};
///
/// // A local overflow of 600 m carries into the next cell.
/// let p = WorldPos::new([0, 0, 0], [600.0, 0.0, 0.0]);
/// assert_eq!(p.cell, [1, 0, 0]);
/// assert!((p.local[0] - (600.0 - 2.0 * f64::from(CELL_HALF) as f32)).abs() < 1e-6);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct WorldPos {
    /// Index of the cell containing the position.
    pub cell: [i64; 3],
    /// Offset from the centre of [`cell`](Self::cell), in metres, each
    /// component normalised into `[-CELL_HALF, CELL_HALF)`.
    pub local: [f32; 3],
}

impl WorldPos {
    /// The origin `(cell = [0,0,0], local = [0,0,0])`.
    pub const ZERO: Self = Self {
        cell: [0; 3],
        local: [0.0; 3],
    };

    /// Build a position from a cell index and a local offset, normalising any
    /// local overflow (positive or negative) into the cell index.
    ///
    /// Cell indices wrap on `i64` overflow (the addressable space is ~9.5e21 m
    /// per axis, far beyond anything the game addresses).
    #[must_use]
    pub fn new(cell: [i64; 3], local: [f32; 3]) -> Self {
        let mut out_cell = cell;
        let mut out_local = [0.0f32; 3];
        for axis in 0..3 {
            let (c, l) = normalise(cell[axis], f64::from(local[axis]));
            out_cell[axis] = c;
            out_local[axis] = l;
        }
        Self {
            cell: out_cell,
            local: out_local,
        }
    }

    /// Build a position from exact integer **millimetre** coordinates measured
    /// from the origin.
    ///
    /// Millimetres (not whole metres) are chosen so that sub-metre positions
    /// survive the round trip: `to_millimetres` is the exact inverse for every
    /// position whose `local` has millimetre resolution.
    #[must_use]
    pub fn from_metres(mm: [i128; 3]) -> Self {
        let mut cell = [0i64; 3];
        let mut local = [0.0f32; 3];
        for axis in 0..3 {
            let mut c = floor_div(mm[axis], CELL_M * MM_PER_M);
            let rem_mm = mm[axis] - c * CELL_M * MM_PER_M; // 0 ..< 1_048_576
            let mut l = rem_mm as f64 / 1000.0_f64;
            if l >= f64::from(CELL_HALF) {
                l -= CELL_SIZE;
                c += 1;
            }
            cell[axis] = c as i64;
            local[axis] = l as f32;
        }
        Self { cell, local }
    }

    /// Exact integer millimetre coordinates of this position, measured from the
    /// origin.
    ///
    /// The cell term is exact; the local term is rounded to the nearest
    /// millimetre.
    #[must_use]
    pub fn to_millimetres(self) -> [i128; 3] {
        let mut out = [0i128; 3];
        for axis in 0..3 {
            let cell_mm = i128::from(self.cell[axis]) * CELL_M * MM_PER_M;
            let local_mm = (f64::from(self.local[axis]) * 1000.0_f64).round();
            out[axis] = cell_mm + local_mm as i128;
        }
        out
    }

    /// `origin + delta`, with `delta` in metres as `f64`.
    ///
    /// The arithmetic is done in the local frame of `origin`, so the precision
    /// of `delta` is preserved no matter how far `origin` is from the world
    /// origin (no catastrophic cancellation against a huge absolute
    /// coordinate).
    #[must_use]
    pub fn from_f64_metres_around(origin: WorldPos, delta: [f64; 3]) -> Self {
        let mut cell = origin.cell;
        let mut local = origin.local;
        for axis in 0..3 {
            let (c, l) = normalise(
                origin.cell[axis],
                f64::from(origin.local[axis]) + delta[axis],
            );
            cell[axis] = c;
            local[axis] = l;
        }
        Self { cell, local }
    }

    /// `self + delta`, with `delta` in metres.
    #[must_use]
    pub fn offset(self, delta: [f32; 3]) -> Self {
        Self::from_f64_metres_around(
            self,
            [
                f64::from(delta[0]),
                f64::from(delta[1]),
                f64::from(delta[2]),
            ],
        )
    }

    /// The cell index, as a map key.
    #[must_use]
    pub fn cell_key(self) -> [i64; 3] {
        self.cell
    }

    /// Signed per-axis difference `b - a` in metres.
    ///
    /// Each axis is `(cell delta) * CELL_SIZE + (local delta)`. The cell term is
    /// exact in `f64` for `|cell delta| <= 2^53`, so nearby positions keep full
    /// `f32` precision even at galactic distances.
    #[must_use]
    pub fn delta(a: WorldPos, b: WorldPos) -> [f64; 3] {
        let mut out = [0.0f64; 3];
        for axis in 0..3 {
            out[axis] = axis_delta_m(a.cell[axis], b.cell[axis], a.local[axis], b.local[axis]);
        }
        out
    }

    /// Euclidean distance between `a` and `b` in metres.
    #[must_use]
    pub fn distance(a: WorldPos, b: WorldPos) -> f64 {
        let d = Self::delta(a, b);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    /// Squared distance in metres\^2 (avoids the square root for comparisons).
    #[must_use]
    pub fn distance_squared(a: WorldPos, b: WorldPos) -> f64 {
        let d = Self::delta(a, b);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }
}

/// Per-axis difference in metres, exact for the cell term.
fn axis_delta_m(a_cell: i64, b_cell: i64, a_local: f32, b_local: f32) -> f64 {
    // i128 keeps the cell delta exact even if the cells are at opposite ends of
    // the i64 space.
    let cell_delta = i128::from(b_cell) - i128::from(a_cell);
    // Exact in f64 while |cell_delta| <= 2^53 (i.e. while the cell delta in
    // metres fits an f64 mantissa); beyond that the cell term is still the
    // correctly rounded value of the exact integer product.
    let cell_m = (cell_delta as f64) * CELL_SIZE;
    cell_m + (f64::from(b_local) - f64::from(a_local))
}

/// Normalise an absolute metre coordinate (given as cell + value) into the
/// canonical `cell`/`local` pair.
fn normalise(cell: i64, value_m: f64) -> (i64, f32) {
    // Number of whole cells to carry; `floor` so the result lands in
    // [-CELL_HALF, CELL_HALF).
    let carry = ((value_m + f64::from(CELL_HALF)) / CELL_SIZE).floor();
    let carry_i = carry as i64;
    let mut local = value_m - carry * CELL_SIZE;
    // Guard the exact boundary and any f64 rounding at huge magnitudes.
    if local >= f64::from(CELL_HALF) {
        local -= CELL_SIZE;
        return (cell.wrapping_add(carry_i).wrapping_add(1), local as f32);
    }
    if local < -f64::from(CELL_HALF) {
        local += CELL_SIZE;
        return (cell.wrapping_add(carry_i).wrapping_sub(1), local as f32);
    }
    (cell.wrapping_add(carry_i), local as f32)
}

#[cfg(test)]
mod tests {
    use super::{WorldPos, CELL_HALF, CELL_SIZE};

    /// 50 000 light-years in metres (rounded to whole millimetres).
    const FAR_MM: i128 = 473_000_000_000_000_000_000_000;

    #[test]
    fn new_normalises_local_overflow() {
        let cell_size = CELL_SIZE as f32;
        let p = WorldPos::new([0, 0, 0], [cell_size, 0.0, 0.0]);
        assert_eq!(p.cell, [1, 0, 0]);
        assert_eq!(p.local, [0.0, 0.0, 0.0]);

        let p = WorldPos::new([0, 0, 0], [-cell_size - 1.0, 0.0, 0.0]);
        assert_eq!(p.cell, [-1, 0, 0]);
        assert!((p.local[0] + 1.0).abs() < 1e-6);

        // Exactly -CELL_HALF stays put (half-open interval).
        let p = WorldPos::new([5, 0, 0], [-CELL_HALF, 0.0, 0.0]);
        assert_eq!(p.cell, [5, 0, 0]);
        assert_eq!(p.local[0], -CELL_HALF);

        // Exactly +CELL_HALF carries into the next cell.
        let p = WorldPos::new([5, 0, 0], [CELL_HALF, 0.0, 0.0]);
        assert_eq!(p.cell, [6, 0, 0]);
        assert_eq!(p.local[0], -CELL_HALF);
    }

    #[test]
    fn normalisation_keeps_local_in_range() {
        for raw in [-1e12f32, -12_345.5, -1.0, 0.0, 1.0, 777.25, 1e9] {
            for axis in 0..3 {
                let mut local = [0.0f32; 3];
                local[axis] = raw;
                let p = WorldPos::new([3, -7, 11], local);
                assert!(
                    p.local[axis] >= -CELL_HALF && p.local[axis] < CELL_HALF,
                    "raw {raw} axis {axis} -> {:?}",
                    p.local[axis]
                );
            }
        }
    }

    #[test]
    fn millimetre_round_trip_is_exact() {
        for mm in [
            0i128,
            1,
            -1,
            511_999,
            512_000,
            -512_000,
            1_048_576,
            -1_048_576,
            FAR_MM,
            FAR_MM + 999,
            FAR_MM - 1,
            -FAR_MM,
        ] {
            let p = WorldPos::from_metres([mm, -mm, mm]);
            let back = p.to_millimetres();
            assert_eq!(back[0], mm, "axis 0 for {mm}");
            assert_eq!(back[1], -mm, "axis 1 for {mm}");
            assert_eq!(back[2], mm, "axis 2 for {mm}");
        }
    }

    #[test]
    fn offset_preserves_precision_far_from_origin() {
        let far = WorldPos::from_metres([FAR_MM, 0, 0]);
        let near = far.offset([1.0, 0.0, 0.0]);
        assert_eq!(WorldPos::delta(far, near), [1.0, 0.0, 0.0]);
        assert_eq!(WorldPos::distance(far, near), 1.0);
    }

    #[test]
    fn cell_key_matches_cell() {
        let p = WorldPos::from_metres([FAR_MM, 0, 0]);
        assert_eq!(p.cell_key(), p.cell);
    }
}
