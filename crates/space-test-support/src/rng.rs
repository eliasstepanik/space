//! A seeded SplitMix64 generator: integer arithmetic only, so a seed gives the
//! same sequence on every platform.

/// The SplitMix64 increment (the golden ratio, scaled to 64 bits).
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// A deterministic pseudo-random generator (SplitMix64).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng(u64);

impl Rng {
    /// A generator that starts from `seed`.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GAMMA);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n` (by remainder). Panics if `n` is 0.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// A float in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A float in `[-1, 1)`.
    pub fn next_sym(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }

    /// A float in `[-1, 1]` with 1e-6 resolution, derived from integers only.
    pub fn next_signed(&mut self) -> f32 {
        ((self.next_u64() % 2_000_001) as f64 / 1_000_000.0 - 1.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn seed_zero_gives_the_published_splitmix64_vectors() {
        let mut rng = Rng::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn one_seed_gives_one_sequence() {
        let mut a = Rng::new(0x5EA1_2345_6789_ABCD);
        let mut b = Rng::new(0x5EA1_2345_6789_ABCD);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn floats_stay_in_their_ranges() {
        let mut rng = Rng::new(42);
        for _ in 0..10_000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f), "next_f64 {f}");
            let s = rng.next_sym();
            assert!((-1.0..1.0).contains(&s), "next_sym {s}");
            let g = rng.next_signed();
            assert!((-1.0..=1.0).contains(&g), "next_signed {g}");
        }
    }
}
