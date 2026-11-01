//! In-memory gaussian scene representation (structure-of-arrays).

use std::path::Path;

use anyhow::{bail, Context, Result};
use glam::{Mat3, Quat, Vec3};

/// A collection of 3D gaussians with *activated* parameters
/// (linear scales, normalized rotations, opacities in `[0, 1]`).
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub positions: Vec<Vec3>,
    /// Per-axis standard deviations (already `exp`-activated).
    pub scales: Vec<Vec3>,
    /// Unit quaternions.
    pub rotations: Vec<Quat>,
    /// Opacity in `[0, 1]` (already sigmoid-activated).
    pub opacities: Vec<f32>,
    /// Base (view-independent) color: `SH_C0 * f_dc + 0.5`, roughly in `[0, 1]`.
    pub colors: Vec<Vec3>,
    /// Spherical harmonics degree of `sh_rest` (0..=3).
    pub sh_degree: u32,
    /// Higher-order SH coefficients, `coeffs_for_degree(sh_degree) - 1` per gaussian.
    pub sh_rest: Vec<Vec3>,
    /// Upper triangle of the 3D covariance (xx, xy, xz, yy, yz, zz), derived from
    /// scales and rotations by [`Scene::finalize`].
    pub cov3d: Vec<[f32; 6]>,
}

/// Robust scene extent used to place the initial camera.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub center: Vec3,
    pub radius: f32,
}

impl Scene {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Number of SH "rest" coefficients stored per gaussian.
    pub fn rest_per_splat(&self) -> usize {
        crate::sh::coeffs_for_degree(self.sh_degree) - 1
    }

    /// Append a single gaussian (degree-0 color only).
    pub fn push(&mut self, pos: Vec3, scale: Vec3, rot: Quat, opacity: f32, color: Vec3) {
        debug_assert_eq!(self.sh_degree, 0);
        self.positions.push(pos);
        self.scales.push(scale);
        self.rotations.push(rot);
        self.opacities.push(opacity);
        self.colors.push(color);
    }

    /// Validate array lengths, sanitize values and compute derived data.
    pub fn finalize(&mut self) -> Result<()> {
        let n = self.positions.len();
        if self.scales.len() != n
            || self.rotations.len() != n
            || self.opacities.len() != n
            || self.colors.len() != n
            || self.sh_rest.len() != n * self.rest_per_splat()
        {
            bail!("inconsistent scene arrays");
        }
        for q in &mut self.rotations {
            let len = q.length();
            *q = if len.is_finite() && len > 1e-12 {
                *q / len
            } else {
                Quat::IDENTITY
            };
        }
        for o in &mut self.opacities {
            *o = if o.is_finite() {
                o.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        self.cov3d = self
            .scales
            .iter()
            .zip(&self.rotations)
            .map(|(s, q)| covariance(*s, *q))
            .collect();
        Ok(())
    }

    /// Load a scene from disk; format is chosen by extension (`.ply` or `.splat`).
    pub fn load(path: &Path) -> Result<Scene> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let mut scene = match ext.as_str() {
            "splat" => crate::splat_format::parse(&data)?,
            "ply" => crate::ply::parse(&data)?,
            _ if data.starts_with(b"ply") => crate::ply::parse(&data)?,
            _ => bail!(
                "unknown file type (expected .ply or .splat): {}",
                path.display()
            ),
        };
        scene.finalize()?;
        if scene.is_empty() {
            bail!("{} contains no gaussians", path.display());
        }
        Ok(scene)
    }

    /// Robust center/radius estimate: the per-axis median as center and a
    /// percentile-based radius, so far-away floaters and backgrounds in real
    /// captures don't push the camera out to infinity.
    pub fn bounds(&self) -> Bounds {
        if self.positions.is_empty() {
            return Bounds {
                center: Vec3::ZERO,
                radius: 1.0,
            };
        }
        // Subsample big scenes; the estimate doesn't need every point.
        let step = (self.positions.len() / 200_000).max(1);
        let pts: Vec<Vec3> = self
            .positions
            .iter()
            .step_by(step)
            .copied()
            .filter(|p| p.is_finite())
            .collect();
        if pts.is_empty() {
            return Bounds {
                center: Vec3::ZERO,
                radius: 1.0,
            };
        }
        let median = |mut v: Vec<f32>| {
            let mid = v.len() / 2;
            *v.select_nth_unstable_by(mid, f32::total_cmp).1
        };
        let center = Vec3::new(
            median(pts.iter().map(|p| p.x).collect()),
            median(pts.iter().map(|p| p.y).collect()),
            median(pts.iter().map(|p| p.z).collect()),
        );
        let mut d: Vec<f32> = pts.iter().map(|p| p.distance(center)).collect();
        let mut pct = |q: f32| {
            let idx = ((d.len() as f32 * q) as usize).min(d.len() - 1);
            *d.select_nth_unstable_by(idx, f32::total_cmp).1
        };
        // Compact objects have p90 ≈ 1.5 × p50; unbounded captures have long
        // tails (sky, background), where 2 × p50 frames the subject better.
        let radius = pct(0.9).min(2.0 * pct(0.5));
        Bounds {
            center,
            radius: if radius > 1e-6 { radius } else { 1.0 },
        }
    }
}

/// 3D covariance `R S Sᵀ Rᵀ` as its upper triangle.
pub fn covariance(scale: Vec3, rot: Quat) -> [f32; 6] {
    let m = Mat3::from_quat(rot) * Mat3::from_diagonal(scale);
    let s = m * m.transpose();
    [
        s.x_axis.x, s.y_axis.x, s.z_axis.x, // xx, xy, xz
        s.y_axis.y, s.z_axis.y, // yy, yz
        s.z_axis.z, // zz
    ]
}

/// Logistic sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse sigmoid, clamped so the result stays finite.
#[inline]
pub fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_axis_aligned() {
        let c = covariance(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        assert_eq!(c, [1.0, 0.0, 0.0, 4.0, 0.0, 9.0]);
    }

    #[test]
    fn covariance_rotated_swaps_axes() {
        let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let c = covariance(Vec3::new(1.0, 2.0, 3.0), q);
        assert!((c[0] - 4.0).abs() < 1e-5);
        assert!((c[3] - 1.0).abs() < 1e-5);
        assert!(c[1].abs() < 1e-5);
    }

    #[test]
    fn sigmoid_logit_roundtrip() {
        for p in [0.01, 0.3, 0.5, 0.9, 0.999] {
            assert!((sigmoid(logit(p)) - p).abs() < 1e-5);
        }
    }

    #[test]
    fn bounds_ignore_outliers() {
        let mut s = Scene::default();
        for i in 0..100 {
            let t = i as f32 / 100.0;
            s.push(
                Vec3::new(t, 0.0, 0.0),
                Vec3::ONE,
                Quat::IDENTITY,
                1.0,
                Vec3::ONE,
            );
        }
        s.push(Vec3::splat(1e6), Vec3::ONE, Quat::IDENTITY, 1.0, Vec3::ONE);
        let b = s.bounds();
        assert!(b.radius < 1.0, "radius {}", b.radius);
        assert!((b.center.x - 0.5).abs() < 0.05);
    }

    #[test]
    fn finalize_rejects_mismatched_arrays() {
        let mut s = Scene::default();
        s.positions.push(Vec3::ZERO);
        assert!(s.finalize().is_err());
    }
}
