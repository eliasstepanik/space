//! Deterministic, seeded galaxy generation.
//!
//! [`Galaxy::star_at`] answers one question per octree node: does this node
//! contain a star, and if so where? The answer is a pure function of
//! `(seed, node)`, computed from integer hashing only, so the same seed yields
//! bit-identical output on every platform and in every process.
//!
//! The density model is deliberately simple: a smooth radial falloff from the
//! galactic centre (the world origin) multiplied by a spiral-arm modulation of
//! the azimuthal angle. Both are evaluated with integer arithmetic on the node
//! centre, then compared against a hashed threshold.

use crate::hash::{mix, splitmix64};
use crate::octree::NodeRef;
use crate::pos::WorldPos;

/// A star generated inside an octree node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarInfo {
    /// Position of the star relative to the centre of the node it belongs to,
    /// in metres, each component within half the node size.
    pub pos_in_node: [f32; 3],
    /// Relative luminosity in `(0, 1]`.
    pub luminosity: f32,
    /// Approximate spectral class as an RGB tint (0..=255 per channel).
    pub spectral_tint: [u8; 3],
}

/// A deterministic galaxy: a seed plus the density model.
///
/// ```
/// use space_world::{Galaxy, Octree, WorldPos};
///
/// let galaxy = Galaxy::new(0xC0FFEE);
/// let node = space_world::octree::node_at(WorldPos::ZERO, 56);
/// let _ = galaxy.star_at(node); // Option<StarInfo>, same every run
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Galaxy {
    seed: u64,
}

/// Radial scale of the galactic disc, in cells (1 cell = 1024 m). The whole
/// disc spans a few thousand cells, i.e. a few light-seconds at this scale —
/// the world scale is a compact "sandbox galaxy", not 1:1 with the Milky Way.
const DISC_CELLS: i64 = 4096;

/// Number of spiral arms.
const ARMS: i64 = 2;

/// Pitch of the spiral, in turns per `DISC_CELLS` of radius.
const ARM_TURNS: i64 = 3;

impl Galaxy {
    /// The galaxy generated from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The seed this galaxy was created with.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The star inside `node`, if any.
    ///
    /// A star appears iff a hash of `(seed, node)` falls below the local star
    /// density. Nodes coarser than the disc scale are almost always empty, so
    /// callers should query a depth where a node is around a cell or two across.
    #[must_use]
    pub fn star_at(&self, node: NodeRef) -> Option<StarInfo> {
        let center = node.center();
        let density = self.density_permille(center)?;
        let h = mix(
            self.seed,
            &[
                u64::from(node.depth),
                node.index[0] as u64,
                node.index[1] as u64,
                node.index[2] as u64,
            ],
        );
        // `density` is in 0..=1000 (per mille); compare against a 0..=999 draw.
        if h % 1000 >= density {
            return None;
        }
        Some(self.star_details(node, h))
    }

    /// Star density around `pos`, in per mille (0..=1000). `None` when the
    /// location is outside the disc entirely.
    fn density_permille(&self, pos: WorldPos) -> Option<u64> {
        let x = pos.cell[0];
        let z = pos.cell[2];
        // Integer radius, scaled by 1024 to keep precision after the division.
        let r2 = (x as i128) * (x as i128) + (z as i128) * (z as i128);
        let r = isqrt(r2);
        if r > i128::from(DISC_CELLS) * 2 {
            return None;
        }
        // Radial falloff: 1000 at the core, ~120 at the disc edge.
        let rr = r.min(i128::from(DISC_CELLS) * 2);
        let radial = 1000i128 - (rr * 440) / (i128::from(DISC_CELLS) * 2);
        // Spiral modulation: the azimuthal distance from the nearest arm.
        let arm = arm_distance_permille(x, z, r);
        let density = (radial * (300 + arm)) / 1300;
        Some(density.clamp(0, 1000) as u64)
    }

    /// Deterministic star attributes for a node that passed the density test.
    fn star_details(&self, node: NodeRef, h: u64) -> StarInfo {
        let half = NodeRef::size_m(node.depth) * 0.5;
        let mut pos_in_node = [0.0f32; 3];
        for (axis, slot) in pos_in_node.iter_mut().enumerate() {
            let u = splitmix64(h ^ (axis as u64).wrapping_mul(0x517C_C1B7_2722_0A95));
            // Uniform in (-half, half), derived from an integer draw.
            let t = (u % 2_000_001) as f64 / 1_000_000.0 - 1.0;
            *slot = (t * half) as f32;
        }
        let lum_raw = splitmix64(h ^ 0x2545_F491_4F6C_DD1D) % 900;
        let luminosity = (lum_raw as f32 + 100.0) / 1000.0;
        let tint_idx = splitmix64(h ^ 0x62A9_F9D5_734E_FD51) % 7;
        let spectral_tint = SPECTRAL_TINTS[tint_idx as usize];
        StarInfo {
            pos_in_node,
            luminosity,
            spectral_tint,
        }
    }
}

/// Seven coarse spectral tints (O, B, A, F, G, K, M), hottest to coolest.
const SPECTRAL_TINTS: [[u8; 3]; 7] = [
    [155, 176, 255],
    [170, 191, 255],
    [202, 215, 255],
    [248, 247, 255],
    [255, 244, 214],
    [255, 202, 143],
    [255, 170, 128],
];

/// Distance from the azimuth `(x, z)` at radius `r` to the nearest spiral arm,
/// in per mille (1000 = on an arm, 0 = midway between arms).
///
/// All integer: the azimuth is quantised to 1/65536 of a turn via an integer
/// atan2 approximation, and the arms wind by `ARM_TURNS` turns across the disc
/// radius.
fn arm_distance_permille(x: i64, z: i64, r: i128) -> i128 {
    if r == 0 {
        return 1000;
    }
    let turns = i128::from(atan2_turns_i64(z, x));
    // Azimuthal spacing between neighbouring arms, in 1/65536 turns.
    let spacing: i128 = i128::from(65536 / ARMS);
    // How far the arms have wound at this radius.
    let wind = (r * (ARM_TURNS as i128 * 65536) / i128::from(DISC_CELLS)).rem_euclid(spacing);
    let d = (turns - wind).rem_euclid(spacing);
    let nearest = d.min(spacing - d);
    // nearest in [0, spacing/2] -> 1000..0 per mille.
    let half = spacing / 2;
    ((half - nearest) * 1000 / half).clamp(0, 1000)
}

/// Integer `atan2(y, x)` returned in units of 1/65536 of a full turn,
/// in `[0, 65536)`.
///
/// Reduces to the first octant and uses the classic integer approximation
/// `atan(t) ~= t / (1 + 0.28086 t^2)`, accurate to ~0.001 rad — plenty for a
/// density modulation, monotone, and exactly reproducible on every platform.
fn atan2_turns_i64(y: i64, x: i64) -> i64 {
    let (y, x) = (i128::from(y), i128::from(x));
    // One eighth of a turn (pi/4) in 1/65536-turn units.
    let octant: i128 = 65536 / 8;
    let quarter = 2 * octant;
    if x == 0 && y == 0 {
        return 0;
    }
    let ax = x.abs();
    let ay = y.abs();
    let swapped = ay > ax;
    let (num, den) = if swapped { (ax, ay) } else { (ay, ax) };
    // t = num / den in [0, 1], scaled by 2^20. `den` is > 0 here.
    let t = (num << 20) / den;
    let t2 = (t * t) >> 20;
    // 0.28086 * 2^20 = 294_512
    let denom = (1i128 << 20) + ((t2 * 294_512) >> 20);
    // atan(t) scaled by 2^20, in [0, pi/4].
    let atan = (t << 20) / denom;
    // atan is scaled by 2^20 and lies in [0, pi/4]; convert to 1/65536-turn
    // units where pi/4 == octant (8192). pi/4 * 2^20 = 823_401.
    let mut a = (atan * octant) / 823_401;
    if swapped {
        a = 2 * octant - a;
    }
    if x < 0 {
        a = 2 * quarter - a;
    }
    if y < 0 {
        a = 4 * quarter - a;
    }
    (a.rem_euclid(4 * quarter)) as i64
}

/// Integer square root (floor) for `i128`, Newton's method.
pub(crate) fn isqrt(n: i128) -> i128 {
    if n <= 0 {
        return 0;
    }
    let mut x = 1i128 << ((127 - n.leading_zeros()) / 2 + 1);
    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

#[cfg(test)]
mod tests {
    use super::{atan2_turns_i64, isqrt, Galaxy};
    use crate::octree::NodeRef;

    #[test]
    fn isqrt_matches_u64_for_small_values() {
        for n in 0..10_000i128 {
            let expected = (n as f64).sqrt() as i128;
            let got = isqrt(n);
            assert!(
                got * got <= n && (got + 1) * (got + 1) > n,
                "n={n} got={got}"
            );
            let _ = expected;
        }
    }

    #[test]
    fn atan2_turns_matches_quadrants() {
        // Units are 1/65536 of a full turn: quarter = 16384, half = 32768.
        assert_eq!(atan2_turns_i64(0, 1), 0);
        assert_eq!(atan2_turns_i64(1, 0), 16384); // +y = quarter turn
        assert_eq!(atan2_turns_i64(0, -1), 32768); // half turn
        assert_eq!(atan2_turns_i64(-1, 0), 49152); // three quarters
                                                   // Diagonals: 1/8 turn = 8192, within the integer atan approximation's
                                                   // error (~60 units of 1/65536 turn).
        assert!((atan2_turns_i64(1, 1) - 8192).abs() <= 100);
        assert!((atan2_turns_i64(-1, -1) - (32768 + 8192)).abs() <= 100);
        // Monotone in the first quadrant.
        let mut prev = -1i64;
        for x in 1..200i64 {
            let a = atan2_turns_i64(x, 200);
            assert!(a >= prev, "x={x}");
            prev = a;
        }
    }

    #[test]
    fn star_at_is_a_pure_function_of_seed_and_node() {
        let g = Galaxy::new(42);
        for depth in 50..=60u8 {
            let node = NodeRef::new(depth, [12, -34, 56]);
            assert_eq!(g.star_at(node), g.star_at(node));
        }
    }

    #[test]
    fn density_falls_with_radius() {
        use crate::cell::CellCoord;
        let g = Galaxy::new(1);
        // Depth 56 nodes are a 256th of a cell across, so address them by the
        // cell their centre falls in.
        let core = CellCoord([0, 0, 0]).to_octree_node(56);
        let edge = CellCoord([3000, 0, 0]).to_octree_node(56);
        let far = CellCoord([9000, 0, 0]).to_octree_node(56);
        assert!(
            g.density_permille(core.center()).unwrap() > g.density_permille(edge.center()).unwrap()
        );
        assert!(g.density_permille(far.center()).is_none());
    }
}
