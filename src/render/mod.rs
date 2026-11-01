//! Rasterizer backends.
//!
//! Every backend implements [`Renderer`]: given a camera [`View`] and an output
//! size it produces an 8-bit RGB [`Frame`]. Backends share the projection math
//! in this module (EWA splatting as in Kerbl et al. 2023, "3D Gaussian
//! Splatting for Real-Time Radiance Field Rendering").

use std::sync::Arc;

use anyhow::Result;
use glam::{Mat3, Vec3};

use crate::camera::View;
use crate::scene::Scene;

pub mod cpu;
pub mod naive;

/// Low-pass filter added to the projected covariance (in pixels²) so that
/// sub-pixel gaussians still cover at least ~one pixel.
pub const DILATION: f32 = 0.3;
/// Alpha below which a gaussian's contribution is ignored.
pub const ALPHA_MIN: f32 = 1.0 / 255.0;
/// Maximum alpha of a single gaussian (avoids fully opaque layers).
pub const ALPHA_MAX: f32 = 0.99;
/// Transmittance below which a pixel is considered saturated.
pub const T_MIN: f32 = 1e-4;

/// 8-bit RGB image, row-major.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[u8; 3]>,
}

impl Frame {
    pub fn new(width: u32, height: u32) -> Self {
        Frame {
            width,
            height,
            pixels: vec![[0; 3]; (width * height) as usize],
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.pixels.resize((width * height) as usize, [0; 3]);
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> [u8; 3] {
        self.pixels[(y * self.width + x) as usize]
    }
}

/// Timing and workload counters for one rendered frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {
    /// Gaussians that survived culling.
    pub visible: usize,
    /// Gaussian/tile pairs (tiled backends) or rasterized gaussians.
    pub overlaps: usize,
    pub project_ms: f64,
    pub sort_ms: f64,
    pub raster_ms: f64,
    pub total_ms: f64,
}

/// A rasterizer backend bound to one scene.
pub trait Renderer: Send {
    fn name(&self) -> &'static str;
    /// Render the scene into `frame`, resizing it to `width`×`height`.
    fn render(
        &mut self,
        view: &View,
        width: u32,
        height: u32,
        background: Vec3,
        frame: &mut Frame,
    ) -> RenderStats;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// Multi-threaded tile-based CPU rasterizer (rayon).
    Cpu,
    /// Single-threaded reference rasterizer: global depth sort, per-splat blending.
    CpuNaive,
    /// GPU compute rasterizer (wgpu: Metal / Vulkan / DX12). Requires the `gpu` feature.
    Gpu,
}

impl Backend {
    pub fn all() -> &'static [Backend] {
        &[Backend::Cpu, Backend::CpuNaive, Backend::Gpu]
    }
}

pub fn create(backend: Backend, scene: Arc<Scene>) -> Result<Box<dyn Renderer>> {
    Ok(match backend {
        Backend::Cpu => Box::new(cpu::CpuTiled::new(scene)),
        Backend::CpuNaive => Box::new(naive::CpuNaive::new(scene)),
        Backend::Gpu => create_gpu(scene)?,
    })
}

#[cfg(feature = "gpu")]
fn create_gpu(scene: Arc<Scene>) -> Result<Box<dyn Renderer>> {
    Ok(Box::new(gpu::GpuRenderer::new(scene)?))
}

#[cfg(not(feature = "gpu"))]
fn create_gpu(_scene: Arc<Scene>) -> Result<Box<dyn Renderer>> {
    anyhow::bail!("this build has no GPU backend; rebuild with `--features gpu`")
}

#[cfg(feature = "gpu")]
pub mod gpu;

/// A gaussian projected to screen space.
#[derive(Clone, Copy, Debug, Default)]
pub struct Splat2D {
    /// Pixel-space center.
    pub x: f32,
    pub y: f32,
    /// Inverse 2D covariance (a, b, c) for `a dx² + 2b dx dy + c dy²`.
    pub conic: [f32; 3],
    pub color: [f32; 3],
    pub opacity: f32,
    /// Camera-space depth (> 0).
    pub depth: f32,
    /// Half-extent of the bounding box in pixels.
    pub ext_x: f32,
    pub ext_y: f32,
}

/// Per-frame constants for [`project`].
#[derive(Clone, Copy, Debug)]
pub struct ProjectCtx {
    pub rot: Mat3,
    pub eye: Vec3,
    pub focal: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: f32,
    pub height: f32,
    pub near: f32,
    pub lim_x: f32,
    pub lim_y: f32,
}

impl ProjectCtx {
    pub fn new(view: &View, width: u32, height: u32) -> Self {
        let focal = view.focal(height);
        let (w, h) = (width as f32, height as f32);
        ProjectCtx {
            rot: view.rotation(),
            eye: view.eye,
            focal,
            cx: w * 0.5,
            cy: h * 0.5,
            width: w,
            height: h,
            near: view.near,
            // Clamp the Jacobian's evaluation point like the reference code so
            // gaussians far outside the frustum don't blow up.
            lim_x: 1.3 * (w * 0.5 / focal),
            lim_y: 1.3 * (h * 0.5 / focal),
        }
    }
}

/// Project gaussian `i` of `scene`. Returns `None` when it is culled.
#[inline]
pub fn project(scene: &Scene, i: usize, ctx: &ProjectCtx) -> Option<Splat2D> {
    let opacity = scene.opacities[i];
    if opacity < ALPHA_MIN {
        return None;
    }
    let p = scene.positions[i];
    let t = ctx.rot * (p - ctx.eye);
    if !(t.z > ctx.near) {
        return None;
    }
    let inv_z = 1.0 / t.z;
    let x = ctx.focal * t.x * inv_z + ctx.cx;
    let y = ctx.focal * t.y * inv_z + ctx.cy;

    // Jacobian of the perspective projection at the (clamped) mean.
    let tx = (t.x * inv_z).clamp(-ctx.lim_x, ctx.lim_x);
    let ty = (t.y * inv_z).clamp(-ctx.lim_y, ctx.lim_y);
    let f = ctx.focal * inv_z;
    let j0 = Vec3::new(f, 0.0, -f * tx);
    let j1 = Vec3::new(0.0, f, -f * ty);
    // Rows of T = J * R.
    let rt = ctx.rot.transpose();
    let t0 = rt * j0;
    let t1 = rt * j1;
    let c = &scene.cov3d[i];
    let sigma = |v: Vec3| {
        Vec3::new(
            c[0] * v.x + c[1] * v.y + c[2] * v.z,
            c[1] * v.x + c[3] * v.y + c[4] * v.z,
            c[2] * v.x + c[4] * v.y + c[5] * v.z,
        )
    };
    let s0 = sigma(t0);
    let a = t0.dot(s0) + DILATION;
    let b = t1.dot(s0);
    let cc = t1.dot(sigma(t1)) + DILATION;
    let det = a * cc - b * b;
    if !(det > 0.0) {
        return None;
    }
    let inv_det = 1.0 / det;

    // Extent where opacity * exp(-r²/2) drops below ALPHA_MIN.
    let k = (2.0 * (opacity / ALPHA_MIN).ln()).sqrt();
    let ext_x = k * a.sqrt();
    let ext_y = k * cc.sqrt();
    if x + ext_x <= 0.0 || x - ext_x >= ctx.width || y + ext_y <= 0.0 || y - ext_y >= ctx.height {
        return None;
    }

    let mut color = scene.colors[i];
    if scene.sh_degree > 0 {
        let per = scene.rest_per_splat();
        let dir = (p - ctx.eye).normalize_or_zero();
        color += crate::sh::eval_rest(scene.sh_degree, &scene.sh_rest[i * per..(i + 1) * per], dir);
    }
    let color = color.max(Vec3::ZERO);
    if !(x.is_finite() && y.is_finite() && ext_x.is_finite() && ext_y.is_finite()) {
        return None;
    }

    Some(Splat2D {
        x,
        y,
        conic: [cc * inv_det, -b * inv_det, a * inv_det],
        color: color.to_array(),
        opacity,
        depth: t.z,
        ext_x,
        ext_y,
    })
}

/// Gaussian falloff exponent at pixel offset (dx, dy).
#[inline(always)]
pub fn power(conic: &[f32; 3], dx: f32, dy: f32) -> f32 {
    -0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) - conic[1] * dx * dy
}

/// Convert an accumulated linear color to 8-bit.
#[inline(always)]
pub fn to_u8(c: [f32; 3]) -> [u8; 3] {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    [q(c[0]), q(c[1]), q(c[2])]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{OrbitCamera, UpAxis};
    use crate::scene::Bounds;
    use glam::Quat;

    fn one_splat(pos: Vec3, opacity: f32) -> Scene {
        let mut s = Scene::default();
        s.push(
            pos,
            Vec3::splat(0.1),
            Quat::IDENTITY,
            opacity,
            Vec3::new(1.0, 0.0, 0.0),
        );
        s.finalize().unwrap();
        s
    }

    fn view() -> View {
        let mut c = OrbitCamera::new(
            Bounds {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            UpAxis::NegY,
        );
        c.pitch = 0.0;
        c.view()
    }

    #[test]
    fn centered_splat_projects_to_center() {
        let s = one_splat(Vec3::ZERO, 0.9);
        let ctx = ProjectCtx::new(&view(), 100, 80);
        let p = project(&s, 0, &ctx).unwrap();
        assert!((p.x - 50.0).abs() < 1e-3 && (p.y - 40.0).abs() < 1e-3);
        assert!(p.ext_x > 0.0 && p.ext_y > 0.0);
        // Isotropic gaussian → isotropic conic.
        assert!((p.conic[0] - p.conic[2]).abs() < 1e-4);
        assert!(p.conic[1].abs() < 1e-4);
    }

    #[test]
    fn culls_behind_camera_and_transparent() {
        let v = view();
        let ctx = ProjectCtx::new(&v, 100, 80);
        let behind = one_splat(v.eye - v.forward * 1.0, 1.0);
        assert!(project(&behind, 0, &ctx).is_none());
        let clear = one_splat(Vec3::ZERO, 0.001);
        assert!(project(&clear, 0, &ctx).is_none());
    }

    #[test]
    fn culls_offscreen() {
        let v = view();
        let ctx = ProjectCtx::new(&v, 100, 80);
        let off = one_splat(v.right * 100.0, 1.0);
        assert!(project(&off, 0, &ctx).is_none());
    }
}
