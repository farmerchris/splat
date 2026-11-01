//! Reader/writer for the compact `.splat` format popularized by
//! antimatter15/splat: 32 bytes per gaussian:
//!
//! | bytes | content                                   |
//! |-------|-------------------------------------------|
//! | 0..12 | position, 3 × f32 LE                       |
//! | 12..24| scale (linear), 3 × f32 LE                 |
//! | 24..28| RGBA, 4 × u8 (A = opacity)                 |
//! | 28..32| rotation (w, x, y, z), 4 × u8, `(q*128)+128` |

use anyhow::{bail, Result};
use glam::{Quat, Vec3};

use crate::scene::Scene;

pub const RECORD_SIZE: usize = 32;

pub fn parse(data: &[u8]) -> Result<Scene> {
    if data.len() % RECORD_SIZE != 0 {
        bail!(
            ".splat file size {} is not a multiple of {RECORD_SIZE} bytes",
            data.len()
        );
    }
    let n = data.len() / RECORD_SIZE;
    let mut s = Scene::default();
    s.positions.reserve(n);
    s.scales.reserve(n);
    s.rotations.reserve(n);
    s.opacities.reserve(n);
    s.colors.reserve(n);
    let f = |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    for r in data.chunks_exact(RECORD_SIZE) {
        s.positions.push(Vec3::new(f(r, 0), f(r, 1), f(r, 2)));
        s.scales.push(Vec3::new(f(r, 3), f(r, 4), f(r, 5)).abs());
        s.colors
            .push(Vec3::new(r[24] as f32, r[25] as f32, r[26] as f32) / 255.0);
        s.opacities.push(r[27] as f32 / 255.0);
        let q = |b: u8| (b as f32 - 128.0) / 128.0;
        s.rotations
            .push(Quat::from_xyzw(q(r[29]), q(r[30]), q(r[31]), q(r[28])));
    }
    Ok(s)
}

pub fn write(scene: &Scene) -> Vec<u8> {
    let mut out = Vec::with_capacity(scene.len() * RECORD_SIZE);
    let u8c = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    let qc = |v: f32| (v * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8;
    for i in 0..scene.len() {
        let p = scene.positions[i];
        let s = scene.scales[i];
        for v in [p.x, p.y, p.z, s.x, s.y, s.z] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let c = scene.colors[i];
        out.extend_from_slice(&[u8c(c.x), u8c(c.y), u8c(c.z), u8c(scene.opacities[i])]);
        let q = scene.rotations[i].normalize();
        out.extend_from_slice(&[qc(q.w), qc(q.x), qc(q.y), qc(q.z)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut s = Scene::default();
        s.push(
            Vec3::new(1.0, -2.0, 3.5),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::from_rotation_x(0.7),
            0.5,
            Vec3::new(1.0, 0.5, 0.0),
        );
        let bytes = write(&s);
        assert_eq!(bytes.len(), 32);
        let r = parse(&bytes).unwrap();
        assert_eq!(r.positions[0], s.positions[0]);
        assert_eq!(r.scales[0], s.scales[0]);
        assert!((r.opacities[0] - 0.5).abs() < 0.01);
        assert!((r.colors[0] - s.colors[0]).length() < 0.01);
        assert!(r.rotations[0].normalize().angle_between(s.rotations[0]) < 0.05);
    }

    #[test]
    fn bad_size() {
        assert!(parse(&[0u8; 33]).is_err());
    }
}
