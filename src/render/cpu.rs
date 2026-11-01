//! Multi-threaded tile-based CPU rasterizer.
//!
//! Mirrors the structure of the CUDA reference rasterizer:
//! 1. project all gaussians in parallel,
//! 2. emit one `(tile, depth)` key per gaussian/tile overlap and sort them,
//! 3. blend each tile front-to-back independently with early termination.

use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use super::{power, project, to_u8, Frame, ProjectCtx, RenderStats, Renderer, Splat2D};
use super::{ALPHA_MAX, ALPHA_MIN, T_MIN};
use crate::camera::View;
use crate::scene::Scene;

pub const TILE: u32 = 16;
const TILE_PIXELS: usize = (TILE * TILE) as usize;

pub struct CpuTiled {
    scene: Arc<Scene>,
    splats: Vec<Splat2D>,
    pairs: Vec<(u64, u32)>,
    ranges: Vec<(u32, u32)>,
}

impl CpuTiled {
    pub fn new(scene: Arc<Scene>) -> Self {
        CpuTiled {
            scene,
            splats: Vec::new(),
            pairs: Vec::new(),
            ranges: Vec::new(),
        }
    }
}

/// Inclusive-exclusive tile rectangle covered by a splat's bounding box.
#[inline]
fn tile_rect(s: &Splat2D, tiles_x: u32, tiles_y: u32) -> (u32, u32, u32, u32) {
    let t = TILE as f32;
    let x0 = ((s.x - s.ext_x) / t).floor().max(0.0) as u32;
    let y0 = ((s.y - s.ext_y) / t).floor().max(0.0) as u32;
    let x1 = (((s.x + s.ext_x) / t).ceil().max(0.0) as u32).min(tiles_x);
    let y1 = (((s.y + s.ext_y) / t).ceil().max(0.0) as u32).min(tiles_y);
    (x0.min(x1), y0.min(y1), x1, y1)
}

impl Renderer for CpuTiled {
    fn name(&self) -> &'static str {
        "cpu"
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
        if width == 0 || height == 0 {
            return stats;
        }
        let scene = &*self.scene;
        let ctx = ProjectCtx::new(view, width, height);

        self.splats.clear();
        self.splats.par_extend(
            (0..scene.len())
                .into_par_iter()
                .filter_map(|i| project(scene, i, &ctx)),
        );
        stats.visible = self.splats.len();
        let t_proj = Instant::now();

        let tiles_x = width.div_ceil(TILE);
        let tiles_y = height.div_ceil(TILE);
        let num_tiles = (tiles_x * tiles_y) as usize;
        self.pairs.clear();
        self.pairs
            .par_extend(self.splats.par_iter().enumerate().flat_map_iter(|(i, s)| {
                let (x0, y0, x1, y1) = tile_rect(s, tiles_x, tiles_y);
                let depth = s.depth.to_bits() as u64;
                (y0..y1).flat_map(move |ty| {
                    (x0..x1).map(move |tx| ((((ty * tiles_x + tx) as u64) << 32) | depth, i as u32))
                })
            }));
        self.pairs.par_sort_unstable_by_key(|p| p.0);
        stats.overlaps = self.pairs.len();

        self.ranges.clear();
        self.ranges.resize(num_tiles, (0, 0));
        let mut i = 0;
        while i < self.pairs.len() {
            let tile = (self.pairs[i].0 >> 32) as usize;
            let begin = i;
            while i < self.pairs.len() && (self.pairs[i].0 >> 32) as usize == tile {
                i += 1;
            }
            self.ranges[tile] = (begin as u32, i as u32);
        }
        let t_sort = Instant::now();

        let splats = &self.splats;
        let pairs = &self.pairs;
        let ranges = &self.ranges;
        let bg = background.to_array();
        let tiles: Vec<[[u8; 3]; TILE_PIXELS]> = (0..num_tiles)
            .into_par_iter()
            .map(|t| {
                let tx = t as u32 % tiles_x;
                let ty = t as u32 / tiles_x;
                let (b, e) = ranges[t];
                raster_tile(
                    splats,
                    pairs[b as usize..e as usize].iter().map(|p| p.1),
                    tx * TILE,
                    ty * TILE,
                    width,
                    height,
                    bg,
                )
            })
            .collect();

        // Scatter tiles into the frame, one parallel job per tile row.
        let w = width as usize;
        frame
            .pixels
            .par_chunks_mut(w * TILE as usize)
            .enumerate()
            .for_each(|(ty, rows)| {
                let rows_here = rows.len() / w;
                for tx in 0..tiles_x as usize {
                    let tile = &tiles[ty * tiles_x as usize + tx];
                    let x0 = tx * TILE as usize;
                    let cols = (w - x0).min(TILE as usize);
                    for ly in 0..rows_here {
                        rows[ly * w + x0..ly * w + x0 + cols]
                            .copy_from_slice(&tile[ly * TILE as usize..ly * TILE as usize + cols]);
                    }
                }
            });
        let end = Instant::now();

        stats.project_ms = (t_proj - start).as_secs_f64() * 1e3;
        stats.sort_ms = (t_sort - t_proj).as_secs_f64() * 1e3;
        stats.raster_ms = (end - t_sort).as_secs_f64() * 1e3;
        stats.total_ms = (end - start).as_secs_f64() * 1e3;
        stats
    }
}

/// Blend the depth-sorted splats overlapping one tile.
fn raster_tile(
    splats: &[Splat2D],
    order: impl Iterator<Item = u32>,
    ox: u32,
    oy: u32,
    width: u32,
    height: u32,
    bg: [f32; 3],
) -> [[u8; 3]; TILE_PIXELS] {
    let tw = TILE.min(width - ox) as usize;
    let th = TILE.min(height - oy) as usize;
    let npix = tw * th;
    let mut color = [[0f32; 3]; TILE_PIXELS];
    let mut trans = [1f32; TILE_PIXELS];
    let mut done = 0usize;
    let (fx, fy) = (ox as f32, oy as f32);

    for si in order {
        let s = &splats[si as usize];
        // Clip the splat's bounding box to this tile (local pixel coords).
        let lx0 = (s.x - s.ext_x - fx).floor().max(0.0) as usize;
        let ly0 = (s.y - s.ext_y - fy).floor().max(0.0) as usize;
        let lx1 = ((s.x + s.ext_x - fx).ceil().max(0.0) as usize).min(tw);
        let ly1 = ((s.y + s.ext_y - fy).ceil().max(0.0) as usize).min(th);
        for ly in ly0..ly1 {
            let dy = fy + ly as f32 + 0.5 - s.y;
            let row = ly * TILE as usize;
            for lx in lx0..lx1 {
                let li = row + lx;
                let t = trans[li];
                if t < T_MIN {
                    continue;
                }
                let dx = fx + lx as f32 + 0.5 - s.x;
                let pw = power(&s.conic, dx, dy);
                if pw > 0.0 {
                    continue;
                }
                let alpha = (s.opacity * pw.exp()).min(ALPHA_MAX);
                if alpha < ALPHA_MIN {
                    continue;
                }
                let wgt = alpha * t;
                let c = &mut color[li];
                c[0] += s.color[0] * wgt;
                c[1] += s.color[1] * wgt;
                c[2] += s.color[2] * wgt;
                let nt = t * (1.0 - alpha);
                trans[li] = nt;
                if nt < T_MIN {
                    done += 1;
                }
            }
        }
        if done >= npix {
            break;
        }
    }

    let mut out = [[0u8; 3]; TILE_PIXELS];
    for i in 0..TILE_PIXELS {
        let t = trans[i];
        let c = color[i];
        out[i] = to_u8([c[0] + t * bg[0], c[1] + t * bg[1], c[2] + t * bg[2]]);
    }
    out
}
