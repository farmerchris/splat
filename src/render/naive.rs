//! Straightforward single-threaded rasterizer, used as a correctness
//! reference and as the performance baseline in benchmarks.

use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;

use super::{power, project, to_u8, Frame, ProjectCtx, RenderStats, Renderer, Splat2D};
use super::{ALPHA_MAX, ALPHA_MIN, T_MIN};
use crate::camera::View;
use crate::scene::Scene;

pub struct CpuNaive {
    scene: Arc<Scene>,
}

impl CpuNaive {
    pub fn new(scene: Arc<Scene>) -> Self {
        CpuNaive { scene }
    }
}

impl Renderer for CpuNaive {
    fn name(&self) -> &'static str {
        "cpu-naive"
    }

    fn render(
        &mut self,
        view: &View,
        width: u32,
        height: u32,
        background: Vec3,
        frame: &mut Frame,
    ) -> RenderStats {
        let start = Instant::now();
        frame.resize(width, height);
        let mut stats = RenderStats::default();
        let ctx = ProjectCtx::new(view, width, height);
        let scene = &*self.scene;
        let mut splats: Vec<Splat2D> = (0..scene.len())
            .filter_map(|i| project(scene, i, &ctx))
            .collect();
        stats.visible = splats.len();
        let t_proj = Instant::now();
        splats.sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
        let t_sort = Instant::now();

        let n = (width * height) as usize;
        let mut color = vec![[0f32; 3]; n];
        let mut trans = vec![1f32; n];
        for s in &splats {
            let x0 = (s.x - s.ext_x).floor().max(0.0) as u32;
            let y0 = (s.y - s.ext_y).floor().max(0.0) as u32;
            let x1 = ((s.x + s.ext_x).ceil().max(0.0) as u32).min(width);
            let y1 = ((s.y + s.ext_y).ceil().max(0.0) as u32).min(height);
            for py in y0..y1 {
                let dy = py as f32 + 0.5 - s.y;
                for px in x0..x1 {
                    let i = (py * width + px) as usize;
                    let t = trans[i];
                    if t < T_MIN {
                        continue;
                    }
                    let pw = power(&s.conic, px as f32 + 0.5 - s.x, dy);
                    if pw > 0.0 {
                        continue;
                    }
                    let alpha = (s.opacity * pw.exp()).min(ALPHA_MAX);
                    if alpha < ALPHA_MIN {
                        continue;
                    }
                    let w = alpha * t;
                    for c in 0..3 {
                        color[i][c] += s.color[c] * w;
                    }
                    trans[i] = t * (1.0 - alpha);
                }
            }
        }
        stats.overlaps = splats.len();
        let bg = background.to_array();
        for (i, px) in frame.pixels.iter_mut().enumerate() {
            let t = trans[i];
            let c = color[i];
            *px = to_u8([c[0] + t * bg[0], c[1] + t * bg[1], c[2] + t * bg[2]]);
        }
        let end = Instant::now();
        stats.project_ms = (t_proj - start).as_secs_f64() * 1e3;
        stats.sort_ms = (t_sort - t_proj).as_secs_f64() * 1e3;
        stats.raster_ms = (end - t_sort).as_secs_f64() * 1e3;
        stats.total_ms = (end - start).as_secs_f64() * 1e3;
        stats
    }
}
