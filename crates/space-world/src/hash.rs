//! Integer hashing used by every deterministic generator in this crate.
//!
//! Only integer arithmetic is used so that generators are bit-identical on
//! every platform and evaluation order (no floating-point RNG state).

/// The SplitMix64 mixer (Steele / Lea / Flood, "Fast Splittable Pseudorandom
/// Number Generators", 2018).
///
/// Given a counter-like input it returns a well-mixed 64-bit value. Pure and
/// branch-free, therefore deterministic across platforms.
#[inline]
#[must_use]
pub const fn splitmix64(x: u64) -> u64 {
    let z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Combine a seed with a sequence of `u64` parts into one well-mixed hash.
///
/// Order-sensitive and avalanche-complete enough for procedural generation:
/// changing any single bit of any part changes every output bit with
/// probability 1/2.
#[inline]
#[must_use]
pub const fn mix(seed: u64, parts: &[u64]) -> u64 {
    let mut h = seed;
    let mut i = 0usize;
    while i < parts.len() {
        h = splitmix64(h ^ parts[i]);
        i += 1;
    }
    h
}

/// Uniformly map a hash to a value in `[0, range)` without floating point.
#[inline]
#[must_use]
pub const fn hash_bounded(hash: u64, range: u64) -> u64 {
    if range == 0 {
        0
    } else {
        hash % range
    }
}

/// Uniformly map a hash to a value in `[0, 1]` with 1e-6 resolution.
///
/// The result is a `f64` derived from an integer division, so it is exactly
/// reproducible on every platform.
#[inline]
#[must_use]
pub const fn hash_unit(hash: u64) -> f64 {
    // 0 ..= 1_000_000 / 1_000_000
    (hash % 1_000_001) as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::{hash_bounded, hash_unit, mix, splitmix64};

    #[test]
    fn splitmix64_matches_reference_vector() {
        // SplitMix64 reference: first output of the stream seeded with 0.
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(1), 0x910A_2DEC_8902_5CC1);
    }

    #[test]
    fn mix_is_order_sensitive_and_deterministic() {
        let a = mix(7, &[1, 2, 3]);
        assert_eq!(a, mix(7, &[1, 2, 3]));
        assert_ne!(a, mix(7, &[3, 2, 1]));
        assert_ne!(a, mix(8, &[1, 2, 3]));
        assert_ne!(a, mix(7, &[1, 2]));
    }

    #[test]
    fn helpers_stay_in_range() {
        assert_eq!(hash_bounded(5, 0), 0);
        for i in 0..1000u64 {
            let h = splitmix64(i);
            assert!(hash_bounded(h, 37) < 37);
            let u = hash_unit(h);
            assert!((0.0..=1.0).contains(&u));
        }
    }
}
