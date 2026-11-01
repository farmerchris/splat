//! Procedural example scenes.
//!
//! Scenes are modelled Y-up and then rotated 180° about X so they follow the
//! Y-down convention of COLMAP-based captures (the viewer's default `--up -y`).

use std::f32::consts::{PI, TAU};

use anyhow::{bail, Result};
use glam::{Quat, Vec3};

use crate::scene::Scene;

pub const SCENES: &[(&str, &str)] = &[
    ("cube", "RGB color cube lattice (small, good smoke test)"),
    ("knot", "rainbow (3,5) torus knot tube"),
    ("galaxy", "spiral galaxy with a warm core and dust lanes"),
    ("planet", "banded planet with a ring system and starfield"),
    ("shballs", "spheres with view-dependent (degree-3 SH) color"),
];

/// SplitMix64: tiny deterministic PRNG so scenes are reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn f(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u = self.f().max(1e-7);
        let v = self.f();
        (-2.0 * u.ln()).sqrt() * (TAU * v).cos()
    }
    pub fn unit(&mut self) -> Vec3 {
        Vec3::new(self.normal(), self.normal(), self.normal()).normalize_or(Vec3::X)
    }
    pub fn quat(&mut self) -> Quat {
        Quat::from_xyzw(self.normal(), self.normal(), self.normal(), self.normal()).normalize()
    }
}

fn hsv(h: f32, s: f32, v: f32) -> Vec3 {
    let h = h.rem_euclid(1.0) * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Vec3::new(r, g, b) + Vec3::splat(v - c)
}

/// A flat disc-like gaussian lying in the plane with normal `n`.
fn disc(n: Vec3, radius: f32, thickness: f32, spin: f32) -> (Vec3, Quat) {
    let q = Quat::from_rotation_arc(Vec3::Z, n.normalize_or(Vec3::Z)) * Quat::from_rotation_z(spin);
    (Vec3::new(radius, radius, thickness), q)
}

pub fn generate(name: &str) -> Result<Scene> {
    let mut s = match name {
        "cube" => cube(),
        "knot" => knot(),
        "galaxy" => galaxy(),
        "planet" => planet(),
        "shballs" => shballs(),
        _ => bail!(
            "unknown scene '{name}'; available: {}",
            SCENES.iter().map(|s| s.0).collect::<Vec<_>>().join(", ")
        ),
    };
    to_y_down(&mut s);
    s.finalize()?;
    Ok(s)
}

/// Rotate 180° about X (y → -y, z → -z). SH coefficients are stored in the
/// rotated frame by the generators that use them, so only geometry changes.
fn to_y_down(s: &mut Scene) {
    let r = Quat::from_rotation_x(PI);
    for p in &mut s.positions {
        *p = r * *p;
    }
    for q in &mut s.rotations {
        *q = (r * *q).normalize();
    }
}

fn cube() -> Scene {
    let mut s = Scene::default();
    let n = 12;
    let step = 2.0 / (n - 1) as f32;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let t = Vec3::new(i as f32, j as f32, k as f32) / (n - 1) as f32;
                let p = t * 2.0 - Vec3::ONE;
                s.push(p, Vec3::splat(step * 0.28), Quat::IDENTITY, 0.95, t);
            }
        }
    }
    // A white floor grid so the cube has context.
    for i in -20..=20 {
        for j in -20..=20 {
            let p = Vec3::new(i as f32 * 0.12, -1.4, j as f32 * 0.12);
            let shade = if (i + j) % 2 == 0 { 0.85 } else { 0.35 };
            let (sc, q) = disc(Vec3::Y, 0.07, 0.005, 0.0);
            s.push(p, sc, q, 0.9, Vec3::splat(shade));
        }
    }
    s
}

fn knot() -> Scene {
    let mut s = Scene::default();
    let mut rng = Rng::new(7);
    let (p, q) = (3.0f32, 5.0f32);
    let curve = |t: f32| {
        let r = 1.0 + 0.4 * (q * t).cos();
        Vec3::new(r * (p * t).cos(), 0.4 * (q * t).sin(), r * (p * t).sin())
    };
    let segments = 3000;
    let ring = 24;
    let tube = 0.14;
    for i in 0..segments {
        let t = i as f32 / segments as f32 * TAU;
        let c = curve(t);
        let tangent = (curve(t + 1e-3) - curve(t - 1e-3)).normalize();
        let side = tangent.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = side.cross(tangent);
        for j in 0..ring {
            let a = j as f32 / ring as f32 * TAU + rng.range(-0.1, 0.1);
            let n = side * a.cos() + up * a.sin();
            let pos = c + n * tube;
            let (sc, rot) = disc(n, 0.03, 0.006, rng.range(0.0, TAU));
            let shade = 0.55 + 0.45 * n.dot(Vec3::new(0.3, 0.8, 0.5).normalize()).max(0.0);
            s.push(pos, sc, rot, 0.9, hsv(t / TAU, 0.85, 1.0) * shade);
        }
    }
    s
}

fn galaxy() -> Scene {
    let mut s = Scene::default();
    let mut rng = Rng::new(42);
    let arms = 4;
    // Stars in log-spiral arms.
    for _ in 0..90_000 {
        let arm = (rng.next_u64() % arms) as f32;
        let r = rng.f().powf(0.6) * 3.0 + 0.05;
        let theta = arm / arms as f32 * TAU + r.ln() * 2.2 + rng.normal() * 0.25 / (0.3 + r);
        let h = rng.normal() * 0.06 * (1.0 + 0.5 / (r + 0.2));
        let pos = Vec3::new(r * theta.cos(), h, r * theta.sin());
        let heat = (-r * 0.9).exp();
        let color =
            Vec3::new(0.55, 0.65, 1.0).lerp(Vec3::new(1.0, 0.85, 0.55), heat) * rng.range(0.6, 1.2);
        let size = rng.range(0.006, 0.018);
        s.push(
            pos,
            Vec3::splat(size),
            Quat::IDENTITY,
            rng.range(0.5, 1.0),
            color,
        );
    }
    // Glowing bulge.
    for _ in 0..8_000 {
        let pos = rng.unit() * rng.normal().abs() * 0.35 * Vec3::new(1.0, 0.5, 1.0);
        let color = Vec3::new(1.0, rng.range(0.75, 0.9), rng.range(0.45, 0.6));
        s.push(
            pos,
            Vec3::splat(rng.range(0.02, 0.06)),
            Quat::IDENTITY,
            0.35,
            color,
        );
    }
    // Dark dust lanes trailing the arms: wide, flat, semi-transparent.
    for _ in 0..6_000 {
        let arm = (rng.next_u64() % arms) as f32;
        let r = rng.range(0.4, 2.6);
        let theta = arm / arms as f32 * TAU + r.ln() * 2.2 - 0.25;
        let pos = Vec3::new(r * theta.cos(), rng.normal() * 0.01, r * theta.sin());
        let (sc, q) = disc(Vec3::Y, rng.range(0.04, 0.09), 0.008, rng.range(0.0, TAU));
        s.push(pos, sc, q, 0.25, Vec3::new(0.12, 0.07, 0.05));
    }
    s
}

fn planet() -> Scene {
    let mut s = Scene::default();
    let mut rng = Rng::new(3);
    let radius = 1.0;
    // Fibonacci sphere surface.
    let n = 40_000;
    let golden = PI * (3.0 - 5f32.sqrt());
    let light = Vec3::new(-0.6, 0.4, 0.7).normalize();
    for i in 0..n {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
        let r = (1.0 - y * y).sqrt();
        let th = golden * i as f32;
        let nrm = Vec3::new(r * th.cos(), y, r * th.sin());
        let lat = y.asin();
        let band = (lat * 9.0 + (th * 0.7).sin() * 0.3).sin() * 0.5 + 0.5;
        let base = Vec3::new(0.85, 0.6, 0.35).lerp(Vec3::new(0.95, 0.9, 0.75), band);
        let storm = (nrm - Vec3::new(0.6, -0.35, 0.72).normalize()).length() < 0.18;
        let base = if storm {
            Vec3::new(0.8, 0.3, 0.2)
        } else {
            base
        };
        let shade = 0.15 + 0.85 * nrm.dot(light).max(0.0);
        let (sc, q) = disc(nrm, 0.028, 0.004, rng.range(0.0, TAU));
        s.push(nrm * radius, sc, q, 1.0, base * shade);
    }
    // Ring system, tilted.
    let tilt = Quat::from_rotation_z(0.35);
    for _ in 0..25_000 {
        let r = rng.range(1.4, 2.3);
        if (1.85..1.92).contains(&r) {
            continue; // Cassini-like gap
        }
        let a = rng.range(0.0, TAU);
        let pos = tilt * Vec3::new(r * a.cos(), rng.normal() * 0.004, r * a.sin());
        let c = 0.55 + 0.35 * ((r * 23.0).sin() * 0.5 + 0.5);
        let shadowed = pos.dot(light) < 0.0 && (pos - light * pos.dot(light)).length() < radius;
        let lum = if shadowed { 0.15 } else { 1.0 };
        let (sc, q) = disc(tilt * Vec3::Y, 0.02, 0.003, a);
        s.push(pos, sc, q, 0.7, Vec3::new(c, c * 0.92, c * 0.8) * lum);
    }
    // Background stars on a large shell.
    for _ in 0..3_000 {
        let pos = rng.unit() * rng.range(12.0, 15.0);
        let b = rng.range(0.6, 1.0);
        s.push(pos, Vec3::splat(0.03), Quat::IDENTITY, 1.0, Vec3::splat(b));
    }
    s
}

fn shballs() -> Scene {
    let mut s = Scene::default();
    s.sh_degree = 3;
    let per = crate::sh::coeffs_for_degree(3) - 1;
    let mut rng = Rng::new(11);
    let centers = [
        (Vec3::new(-1.2, 0.0, 0.0), 0),
        (Vec3::new(0.0, 0.0, 0.0), 1),
        (Vec3::new(1.2, 0.0, 0.0), 2),
    ];
    for (c, kind) in centers {
        for _ in 0..12_000 {
            let n = rng.unit();
            let (sc, q) = disc(n, 0.035, 0.006, rng.range(0.0, TAU));
            s.push(c + n * 0.5, sc, q, 0.95, Vec3::splat(0.5));
            let mut rest = vec![Vec3::ZERO; per];
            match kind {
                // Degree 1: red when viewed from one side, cyan from the other.
                0 => rest[2] = Vec3::new(-0.9, 0.6, 0.6),
                // Degree 2: color bands that depend on elevation.
                1 => {
                    rest[5] = Vec3::new(0.9, -0.3, -0.6);
                    rest[7] = Vec3::new(-0.4, 0.8, -0.2);
                }
                // Degree 3: a fast-changing rainbow sheen.
                _ => {
                    rest[0] = Vec3::new(0.2, -0.2, 0.0);
                    rest[9] = Vec3::new(0.0, 1.2, -1.2);
                    rest[13] = Vec3::new(1.2, -0.6, 0.0);
                    rest[14] = Vec3::new(-0.6, 0.0, 1.0);
                }
            }
            s.sh_rest.extend(rest);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_scenes_generate() {
        for (name, _) in SCENES {
            let s = generate(name).unwrap();
            assert!(s.len() > 1000, "{name} has {} splats", s.len());
            assert!(s.positions.iter().all(|p| p.is_finite()));
            assert_eq!(s.cov3d.len(), s.len());
        }
    }

    #[test]
    fn deterministic() {
        let a = generate("galaxy").unwrap();
        let b = generate("galaxy").unwrap();
        assert_eq!(a.positions, b.positions);
    }

    #[test]
    fn unknown_scene() {
        assert!(generate("nope").is_err());
    }
}
