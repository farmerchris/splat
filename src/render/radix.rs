//! LSD radix sort for the CPU rasterizer's per-tile depth keys.

/// Below this length a comparison sort beats the radix sort's fixed costs.
const SMALL: usize = 192;

/// Sort `keys` by their upper 32 bits, **stably** (keys with equal upper
/// halves keep their relative order). `tmp` is scratch space of at least
/// `keys.len()` elements.
///
/// Small inputs fall back to a comparison sort of the full 64-bit key, which
/// gives the same result whenever the lower halves are already ascending —
/// the only way the rasterizer calls it (lower half = insertion position).
pub fn sort_hi32(keys: &mut [u64], tmp: &mut [u64]) {
    let n = keys.len();
    if n <= SMALL {
        debug_assert!(keys.windows(2).all(|w| w[0] as u32 <= w[1] as u32));
        keys.sort_unstable();
        return;
    }
    let tmp = &mut tmp[..n];
    let mut hist = [[0u32; 256]; 4];
    for &k in keys.iter() {
        let h = (k >> 32) as u32;
        hist[0][(h & 0xff) as usize] += 1;
        hist[1][(h >> 8 & 0xff) as usize] += 1;
        hist[2][(h >> 16 & 0xff) as usize] += 1;
        hist[3][(h >> 24) as usize] += 1;
    }
    let (mut src, mut dst): (&mut [u64], &mut [u64]) = (keys, tmp);
    let mut in_tmp = false;
    for (d, h) in hist.iter_mut().enumerate() {
        // A digit shared by every key doesn't change the order: skip it.
        let shift = 32 + 8 * d;
        if h[((src[0] >> shift) & 0xff) as usize] as usize == n {
            continue;
        }
        let mut sum = 0u32;
        for c in h.iter_mut() {
            let v = *c;
            *c = sum;
            sum += v;
        }
        for &k in src.iter() {
            let b = ((k >> shift) & 0xff) as usize;
            // SAFETY: the prefix sums of a histogram of `src` index within
            // `dst` (same length) and each slot is written once.
            unsafe { *dst.get_unchecked_mut(h[b] as usize) = k };
            h[b] += 1;
        }
        std::mem::swap(&mut src, &mut dst);
        in_tmp = !in_tmp;
    }
    if in_tmp {
        // `src` is the scratch buffer holding the result, `dst` is `keys`.
        dst.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn check(hi: impl Fn(&mut u64) -> u32, n: usize) {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ n as u64;
        // Lower half = insertion index, as the rasterizer produces them.
        let keys: Vec<u64> = (0..n).map(|i| ((hi(&mut seed) as u64) << 32) | i as u64).collect();
        let mut want = keys.clone();
        want.sort_by_key(|k| k >> 32); // stable
        let mut got = keys.clone();
        let mut tmp = vec![0; n];
        sort_hi32(&mut got, &mut tmp);
        assert_eq!(got, want, "n = {n}");
    }

    #[test]
    fn matches_stable_sort() {
        for n in [0, 1, 2, 7, SMALL, SMALL + 1, 1000, 5000, 70_000] {
            check(|s| rng(s) as u32, n);
            // Many duplicates (ties must keep insertion order).
            check(|s| (rng(s) % 5) as u32, n);
            // Positive float depths in a narrow range: high digits constant.
            check(|s| (1.0 + (rng(s) % 1000) as f32 * 1e-3).to_bits(), n);
            // All equal.
            check(|_| 42, n);
        }
    }

    #[test]
    fn float_bits_order() {
        // Positive finite floats order like their bit patterns.
        let mut seed = 7u64;
        let n = 3000;
        let depths: Vec<f32> = (0..n).map(|_| (rng(&mut seed) % 100_000) as f32 * 0.37 + 0.01).collect();
        let mut keys: Vec<u64> = depths.iter().enumerate().map(|(i, d)| ((d.to_bits() as u64) << 32) | i as u64).collect();
        let mut tmp = vec![0; n];
        sort_hi32(&mut keys, &mut tmp);
        assert!(keys.windows(2).all(|w| depths[w[0] as u32 as usize] <= depths[w[1] as u32 as usize]));
    }
}
