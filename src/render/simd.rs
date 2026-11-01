//! Minimal 4-lane f32 vector used by the CPU rasterizer's inner loop.
//!
//! NEON on aarch64, plain arrays elsewhere (which LLVM vectorizes on most
//! targets). Only the handful of operations the blender needs are provided.

#![allow(dead_code)]

/// Lanes of [`F4`].
pub const LANES: usize = 4;

#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::aarch64::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct F4(pub float32x4_t);
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct M4(pub uint32x4_t);

    // SAFETY (whole module): NEON is a baseline feature of aarch64, so every
    // intrinsic below is always available.
    impl F4 {
        #[inline(always)]
        pub fn splat(v: f32) -> F4 {
            unsafe { F4(vdupq_n_f32(v)) }
        }
        #[inline(always)]
        pub fn new(v: [f32; 4]) -> F4 {
            unsafe { F4(vld1q_f32(v.as_ptr())) }
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            let mut o = [0f32; 4];
            unsafe { vst1q_f32(o.as_mut_ptr(), self.0) };
            o
        }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            unsafe { F4(vaddq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            unsafe { F4(vsubq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            unsafe { F4(vmulq_f32(self.0, o.0)) }
        }
        /// `self + a * b` (fused).
        #[inline(always)]
        pub fn mul_add(self, a: F4, b: F4) -> F4 {
            unsafe { F4(vfmaq_f32(self.0, a.0, b.0)) }
        }
        #[inline(always)]
        pub fn div(self, o: F4) -> F4 {
            unsafe { F4(vdivq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn sqrt(self) -> F4 {
            unsafe { F4(vsqrtq_f32(self.0)) }
        }
        #[inline(always)]
        pub fn abs(self) -> F4 {
            unsafe { F4(vabsq_f32(self.0)) }
        }
        #[inline(always)]
        pub fn neg(self) -> F4 {
            unsafe { F4(vnegq_f32(self.0)) }
        }
        #[inline(always)]
        pub fn gt(self, o: F4) -> M4 {
            unsafe { M4(vcgtq_f32(self.0, o.0)) }
        }
        /// Load 4 consecutive values.
        #[inline(always)]
        pub fn load(s: &[f32]) -> F4 {
            assert!(s.len() >= 4);
            unsafe { F4(vld1q_f32(s.as_ptr())) }
        }
        #[inline(always)]
        pub fn min(self, o: F4) -> F4 {
            unsafe { F4(vminq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: F4) -> F4 {
            unsafe { F4(vmaxq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn round(self) -> F4 {
            unsafe { F4(vrndnq_f32(self.0)) }
        }
        #[inline(always)]
        pub fn floor(self) -> F4 {
            unsafe { F4(vrndmq_f32(self.0)) }
        }
        #[inline(always)]
        pub fn ceil(self) -> F4 {
            unsafe { F4(vrndpq_f32(self.0)) }
        }
        /// Saturating conversion like `as u32` (negative and NaN -> 0).
        #[inline(always)]
        pub fn to_u32(self) -> [u32; 4] {
            let mut o = [0u32; 4];
            unsafe { vst1q_u32(o.as_mut_ptr(), vcvtq_u32_f32(self.0)) };
            o
        }
        #[inline(always)]
        pub fn le(self, o: F4) -> M4 {
            unsafe { M4(vcleq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn lt(self, o: F4) -> M4 {
            unsafe { M4(vcltq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn ge(self, o: F4) -> M4 {
            unsafe { M4(vcgeq_f32(self.0, o.0)) }
        }
        /// Lanes of `self` where `m` is set, else 0.
        #[inline(always)]
        pub fn and_mask(self, m: M4) -> F4 {
            unsafe { F4(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(self.0), m.0))) }
        }
        /// `2^n` for integral-valued lanes `n` in [-126, 127].
        #[inline(always)]
        pub fn exp2_int(self) -> F4 {
            unsafe {
                let n = vcvtq_s32_f32(self.0);
                let bits = vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127)));
                F4(vreinterpretq_f32_s32(bits))
            }
        }
    }

    impl M4 {
        /// Lane `i` set iff bit `i` of `b` is set.
        #[inline(always)]
        pub fn from_bits(b: u32) -> M4 {
            unsafe { M4(vtstq_u32(vdupq_n_u32(b), vld1q_u32([1u32, 2, 4, 8].as_ptr()))) }
        }
        #[inline(always)]
        pub fn and(self, o: M4) -> M4 {
            unsafe { M4(vandq_u32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn or(self, o: M4) -> M4 {
            unsafe { M4(vorrq_u32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn not(self) -> M4 {
            unsafe { M4(vmvnq_u32(self.0)) }
        }
        /// Bit `i` set iff lane `i` is set.
        #[inline(always)]
        pub fn bits(self) -> u32 {
            unsafe {
                let w = vld1q_u32([1u32, 2, 4, 8].as_ptr());
                vaddvq_u32(vandq_u32(self.0, w))
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct F4(pub [f32; 4]);
    #[derive(Clone, Copy)]
    pub struct M4(pub [bool; 4]);

    #[inline(always)]
    fn map(a: [f32; 4], f: impl Fn(f32) -> f32) -> [f32; 4] {
        [f(a[0]), f(a[1]), f(a[2]), f(a[3])]
    }
    #[inline(always)]
    fn zip(a: [f32; 4], b: [f32; 4], f: impl Fn(f32, f32) -> f32) -> [f32; 4] {
        [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
    }
    #[inline(always)]
    fn cmp(a: [f32; 4], b: [f32; 4], f: impl Fn(f32, f32) -> bool) -> [bool; 4] {
        [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
    }

    impl F4 {
        #[inline(always)]
        pub fn splat(v: f32) -> F4 {
            F4([v; 4])
        }
        #[inline(always)]
        pub fn new(v: [f32; 4]) -> F4 {
            F4(v)
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            self.0
        }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            F4(zip(self.0, o.0, |a, b| a + b))
        }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            F4(zip(self.0, o.0, |a, b| a - b))
        }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            F4(zip(self.0, o.0, |a, b| a * b))
        }
        #[inline(always)]
        pub fn mul_add(self, a: F4, b: F4) -> F4 {
            self.add(a.mul(b))
        }
        #[inline(always)]
        pub fn div(self, o: F4) -> F4 {
            F4(zip(self.0, o.0, |a, b| a / b))
        }
        #[inline(always)]
        pub fn sqrt(self) -> F4 {
            F4(map(self.0, f32::sqrt))
        }
        #[inline(always)]
        pub fn abs(self) -> F4 {
            F4(map(self.0, f32::abs))
        }
        #[inline(always)]
        pub fn neg(self) -> F4 {
            F4(map(self.0, |a| -a))
        }
        #[inline(always)]
        pub fn gt(self, o: F4) -> M4 {
            M4(cmp(self.0, o.0, |a, b| a > b))
        }
        #[inline(always)]
        pub fn load(s: &[f32]) -> F4 {
            F4([s[0], s[1], s[2], s[3]])
        }
        #[inline(always)]
        pub fn min(self, o: F4) -> F4 {
            F4(zip(self.0, o.0, f32::min))
        }
        #[inline(always)]
        pub fn max(self, o: F4) -> F4 {
            F4(zip(self.0, o.0, f32::max))
        }
        #[inline(always)]
        pub fn round(self) -> F4 {
            F4(map(self.0, f32::round_ties_even))
        }
        #[inline(always)]
        pub fn floor(self) -> F4 {
            F4(map(self.0, f32::floor))
        }
        #[inline(always)]
        pub fn ceil(self) -> F4 {
            F4(map(self.0, f32::ceil))
        }
        #[inline(always)]
        pub fn to_u32(self) -> [u32; 4] {
            self.0.map(|v| v as u32)
        }
        #[inline(always)]
        pub fn le(self, o: F4) -> M4 {
            M4(cmp(self.0, o.0, |a, b| a <= b))
        }
        #[inline(always)]
        pub fn lt(self, o: F4) -> M4 {
            M4(cmp(self.0, o.0, |a, b| a < b))
        }
        #[inline(always)]
        pub fn ge(self, o: F4) -> M4 {
            M4(cmp(self.0, o.0, |a, b| a >= b))
        }
        #[inline(always)]
        pub fn and_mask(self, m: M4) -> F4 {
            F4([0, 1, 2, 3].map(|i| if m.0[i] { self.0[i] } else { 0.0 }))
        }
        #[inline(always)]
        pub fn exp2_int(self) -> F4 {
            F4(map(self.0, |n| f32::from_bits(((n as i32 + 127) as u32) << 23)))
        }
    }

    impl M4 {
        #[inline(always)]
        pub fn from_bits(b: u32) -> M4 {
            M4([0, 1, 2, 3].map(|i| b >> i & 1 != 0))
        }
        #[inline(always)]
        pub fn and(self, o: M4) -> M4 {
            M4([0, 1, 2, 3].map(|i| self.0[i] && o.0[i]))
        }
        #[inline(always)]
        pub fn or(self, o: M4) -> M4 {
            M4([0, 1, 2, 3].map(|i| self.0[i] || o.0[i]))
        }
        #[inline(always)]
        pub fn not(self) -> M4 {
            M4(self.0.map(|b| !b))
        }
        #[inline(always)]
        pub fn bits(self) -> u32 {
            (0..4).map(|i| (self.0[i] as u32) << i).sum()
        }
    }
}

pub use imp::{F4, M4};

impl F4 {
    /// `e^x` for `x <= 0` (inputs below -87 are clamped), within ~2 ulp of
    /// `f32::exp`. Cephes-style range reduction plus a degree-6 polynomial.
    #[inline(always)]
    pub fn exp_neg(self) -> F4 {
        const LOG2E: f32 = std::f32::consts::LOG2_E;
        const C1: f32 = 0.693_359_4;
        const C2: f32 = -2.121_944_4e-4;
        let x = self.max(F4::splat(-87.0));
        let n = x.mul(F4::splat(LOG2E)).round();
        let f = x.sub(n.mul(F4::splat(C1))).sub(n.mul(F4::splat(C2)));
        let mut p = F4::splat(1.987_569_1e-4);
        p = F4::splat(1.398_199_9e-3).mul_add(p, f);
        p = F4::splat(8.333_452e-3).mul_add(p, f);
        p = F4::splat(4.166_579_6e-2).mul_add(p, f);
        p = F4::splat(1.666_666_5e-1).mul_add(p, f);
        p = F4::splat(0.5).mul_add(p, f);
        let f2 = f.mul(f);
        let r = f.add(F4::splat(1.0)).mul_add(p, f2);
        r.mul(n.exp2_int())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std() {
        let mut worst = 0f32;
        let mut x = -90.0f32;
        while x <= 0.0 {
            let v = F4::new([x, x * 0.5, x * 0.1, x * 0.01]).exp_neg().to_array();
            for (i, s) in [1.0, 0.5, 0.1, 0.01].iter().enumerate() {
                let xi = x * s;
                let want = xi.max(-87.0).exp();
                let rel = ((v[i] - want) / want).abs();
                worst = worst.max(rel);
            }
            x += 0.0137;
        }
        assert!(worst < 5e-7, "max rel err {worst}");
        assert_eq!(F4::splat(0.0).exp_neg().to_array(), [1.0; 4]);
    }

    #[test]
    fn masks() {
        let a = F4::new([1.0, 2.0, 3.0, 4.0]);
        let m = a.ge(F4::splat(2.5));
        assert_eq!(m.bits(), 0b1100);
        assert_eq!(a.and_mask(m).to_array(), [0.0, 0.0, 3.0, 4.0]);
        assert_eq!(M4::from_bits(0b1010).bits(), 0b1010);
        let v = F4::new([-1.5, 2.25, f32::NAN, 1e20]);
        assert_eq!(v.to_u32(), [0, 2, 0, u32::MAX]);
        assert_eq!(v.floor().to_array()[..2], [-2.0, 2.0]);
        assert_eq!(v.ceil().to_array()[..2], [-1.0, 3.0]);
        assert_eq!(a.le(F4::splat(2.0)).and(a.lt(F4::splat(9.0))).bits(), 0b0011);
    }
}
