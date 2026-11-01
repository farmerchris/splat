//! Real spherical harmonics evaluation matching the reference 3DGS implementation.

use glam::Vec3;

pub const SH_C0: f32 = 0.282_094_8;
const SH_C1: f32 = 0.488_602_5;
const SH_C2: [f32; 5] = [
    1.092_548_5,
    -1.092_548_5,
    0.315_391_57,
    -1.092_548_5,
    0.546_274_24,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_33,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

/// Number of SH coefficients per color channel for a given degree.
pub const fn coeffs_for_degree(degree: u32) -> usize {
    ((degree + 1) * (degree + 1)) as usize
}

/// Infer the SH degree from the number of `f_rest_*` properties (3 channels).
pub fn degree_from_rest_count(rest: usize) -> Option<u32> {
    (0..=3).find(|&d| 3 * (coeffs_for_degree(d) - 1) == rest)
}

/// Evaluate the view-dependent part (degrees >= 1) of the SH color.
///
/// `rest` holds `coeffs_for_degree(degree) - 1` RGB coefficients, `dir` is the
/// normalized direction from the camera to the gaussian.
#[inline]
pub fn eval_rest(degree: u32, rest: &[Vec3], dir: Vec3) -> Vec3 {
    if degree == 0 || rest.is_empty() {
        return Vec3::ZERO;
    }
    let (x, y, z) = (dir.x, dir.y, dir.z);
    let mut c = SH_C1 * (-y * rest[0] + z * rest[1] - x * rest[2]);
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, yz, xz) = (x * y, y * z, x * z);
        c += SH_C2[0] * xy * rest[3]
            + SH_C2[1] * yz * rest[4]
            + SH_C2[2] * (2.0 * zz - xx - yy) * rest[5]
            + SH_C2[3] * xz * rest[6]
            + SH_C2[4] * (xx - yy) * rest[7];
        if degree >= 3 {
            c += SH_C3[0] * y * (3.0 * xx - yy) * rest[8]
                + SH_C3[1] * xy * z * rest[9]
                + SH_C3[2] * y * (4.0 * zz - xx - yy) * rest[10]
                + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * rest[11]
                + SH_C3[4] * x * (4.0 * zz - xx - yy) * rest[12]
                + SH_C3[5] * z * (xx - yy) * rest[13]
                + SH_C3[6] * x * (xx - 3.0 * yy) * rest[14];
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_inference() {
        assert_eq!(degree_from_rest_count(0), Some(0));
        assert_eq!(degree_from_rest_count(9), Some(1));
        assert_eq!(degree_from_rest_count(24), Some(2));
        assert_eq!(degree_from_rest_count(45), Some(3));
        assert_eq!(degree_from_rest_count(10), None);
    }

    #[test]
    fn degree_zero_is_view_independent() {
        assert_eq!(eval_rest(0, &[], Vec3::X), Vec3::ZERO);
    }

    #[test]
    fn degree_one_flips_with_direction() {
        let rest = [Vec3::ZERO, Vec3::ONE, Vec3::ZERO];
        let a = eval_rest(1, &rest, Vec3::Z);
        let b = eval_rest(1, &rest, -Vec3::Z);
        assert!((a + b).length() < 1e-6);
        assert!(a.x > 0.0);
    }
}
