//! Headless benchmarking and snapshot helpers.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec3;

use crate::camera::{OrbitCamera, UpAxis};
use crate::render::{self, Backend, Frame, RenderStats};
use crate::scene::Scene;

#[derive(Clone, Debug)]
pub struct BenchResult {
    pub backend: String,
    pub frames: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub min_ms: f64,
    pub avg: RenderStats,
    /// PSNR of the first frame against the first backend's first frame.
    pub psnr_vs_ref: Option<f64>,
    pub init_ms: f64,
}

pub struct BenchConfig {
    pub width: u32,
    pub height: u32,
    pub frames: usize,
    pub warmup: usize,
    pub up: UpAxis,
}

/// Render `frames` views orbiting the scene with each backend.
pub fn run(scene: Arc<Scene>, backends: &[Backend], cfg: &BenchConfig) -> Vec<Result<BenchResult>> {
    let bounds = scene.bounds();
    let mut reference: Option<Frame> = None;
    let mut out = Vec::new();
    for &b in backends {
        let r = (|| -> Result<BenchResult> {
            let t0 = Instant::now();
            let mut renderer = render::create(b, scene.clone())?;
            let init_ms = t0.elapsed().as_secs_f64() * 1e3;
            let mut cam = OrbitCamera::new(bounds, cfg.up);
            let mut frame = Frame::default();
            let bg = Vec3::splat(0.0);

            // First frame at the home view, used for the correctness comparison.
            renderer.render(&cam.view(), cfg.width, cfg.height, bg, &mut frame);
            let psnr = match &reference {
                Some(r) => Some(psnr(r, &frame)),
                None => {
                    reference = Some(frame.clone());
                    None
                }
            };
            for _ in 0..cfg.warmup {
                cam.rotate(0.05, 0.0);
                renderer.render(&cam.view(), cfg.width, cfg.height, bg, &mut frame);
            }
            let mut times = Vec::with_capacity(cfg.frames);
            let mut sum = RenderStats::default();
            for i in 0..cfg.frames {
                cam.rotate(std::f32::consts::TAU / cfg.frames as f32, 0.0);
                cam.pitch = 0.4 * (i as f32 * 0.1).sin();
                let t = Instant::now();
                let s = renderer.render(&cam.view(), cfg.width, cfg.height, bg, &mut frame);
                times.push(t.elapsed().as_secs_f64() * 1e3);
                sum.visible += s.visible;
                sum.overlaps += s.overlaps;
                sum.project_ms += s.project_ms;
                sum.sort_ms += s.sort_ms;
                sum.raster_ms += s.raster_ms;
                sum.total_ms += s.total_ms;
            }
            let n = cfg.frames.max(1);
            let nf = n as f64;
            let avg = RenderStats {
                visible: sum.visible / n,
                overlaps: sum.overlaps / n,
                project_ms: sum.project_ms / nf,
                sort_ms: sum.sort_ms / nf,
                raster_ms: sum.raster_ms / nf,
                total_ms: sum.total_ms / nf,
            };
            times.sort_by(f64::total_cmp);
            let pct = |p: f64| {
                times
                    .get(((times.len() as f64 - 1.0) * p).round() as usize)
                    .copied()
                    .unwrap_or(0.0)
            };
            Ok(BenchResult {
                backend: renderer.name().to_string(),
                frames: cfg.frames,
                mean_ms: times.iter().sum::<f64>() / nf,
                p50_ms: pct(0.5),
                p95_ms: pct(0.95),
                min_ms: pct(0.0),
                avg,
                psnr_vs_ref: psnr,
                init_ms,
            })
        })();
        out.push(r);
    }
    out
}

pub fn print_table(results: &[Result<BenchResult>], backends: &[Backend]) {
    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8} {:>7} {:>8} {:>8} {:>8} {:>9} {:>10} {:>8}",
        "backend",
        "mean",
        "p50",
        "p95",
        "min",
        "fps",
        "project",
        "sort",
        "raster",
        "visible",
        "overlaps",
        "psnr"
    );
    for (r, b) in results.iter().zip(backends) {
        match r {
            Ok(r) => println!(
                "{:<10} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>7.1} {:>8.2} {:>8.2} {:>8.2} {:>9} {:>10} {:>8}",
                r.backend,
                r.mean_ms,
                r.p50_ms,
                r.p95_ms,
                r.min_ms,
                1e3 / r.mean_ms.max(1e-9),
                r.avg.project_ms,
                r.avg.sort_ms,
                r.avg.raster_ms,
                r.avg.visible,
                r.avg.overlaps,
                r.psnr_vs_ref.map(|p| if p.is_infinite() { "inf".into() } else { format!("{p:.1}") })
                    .unwrap_or_else(|| "ref".into()),
            ),
            Err(e) => println!("{:<10} unavailable: {e:#}", format!("{b:?}").to_lowercase()),
        }
    }
}

/// Peak signal-to-noise ratio in dB between two equally sized frames.
pub fn psnr(a: &Frame, b: &Frame) -> f64 {
    if a.width != b.width || a.height != b.height || a.pixels.is_empty() {
        return 0.0;
    }
    let mut se = 0f64;
    for (p, q) in a.pixels.iter().zip(&b.pixels) {
        for c in 0..3 {
            let d = p[c] as f64 - q[c] as f64;
            se += d * d;
        }
    }
    let mse = se / (a.pixels.len() * 3) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// Write an uncompressed (stored-deflate) PNG. Dependency-free and plenty for
/// debug snapshots.
pub fn write_png(path: &Path, frame: &Frame) -> Result<()> {
    let mut raw = Vec::with_capacity((frame.width as usize * 3 + 1) * frame.height as usize);
    for y in 0..frame.height {
        raw.push(0); // filter: none
        for x in 0..frame.width {
            raw.extend_from_slice(&frame.get(x, y));
        }
    }
    let mut z = vec![0x78, 0x01];
    let mut chunks = raw.chunks(65_535).peekable();
    if chunks.peek().is_none() {
        z.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(c) = chunks.next() {
        z.push(if chunks.peek().is_none() { 1 } else { 0 });
        let len = c.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(c);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + x as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut out = Vec::new();
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut chunk = |ty: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut c = Vec::with_capacity(4 + data.len());
        c.extend_from_slice(ty);
        c.extend_from_slice(data);
        out.extend_from_slice(&c);
        out.extend_from_slice(&crc32(&c).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&frame.width.to_be_bytes());
    ihdr.extend_from_slice(&frame.height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    let mut f =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    f.write_all(&out)?;
    Ok(())
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn psnr_identical_is_infinite() {
        let f = Frame::new(4, 4);
        assert!(psnr(&f, &f).is_infinite());
    }

    #[test]
    fn png_has_signature() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.png");
        write_png(&p, &Frame::new(3, 2)).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert!(bytes.starts_with(b"\x89PNG"));
    }
}
