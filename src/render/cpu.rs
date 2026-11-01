//! Multi-threaded tile-based CPU rasterizer.
//!
//! Mirrors the structure of the CUDA reference rasterizer, tuned for CPUs:
//! 1. project all gaussians in parallel, 4 at a time with SIMD (bit-identical
//!    to [`super::project`]), in a few index-range jobs per thread with reused
//!    buffers, emitting one `(tile, depth|index)` pair per gaussian/tile
//!    overlap and counting overlaps per tile,
//! 2. bucket the pairs by tile with a counting sort (prefix sums of the
//!    per-job histograms, then a parallel scatter; no global sort),
//! 3. per tile, in parallel and largest tile first: radix-sort the bucket by
//!    depth (ties broken by original scene index, like the reference), then
//!    blend front-to-back in 4-lane SIMD groups, skipping saturated pixels via
//!    per-row bitmasks, and write straight into the output frame.
//!
//! The scene is copied once, reordered along a Morton curve, so the splats a
//! tile reads in depth order are mostly close together in memory.
//!
//! In [`RenderStats`], `sort_ms` covers the tile bucketing; the per-tile depth
//! sorts run fused with blending and are counted in `raster_ms`.

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat3, Vec3};
use rayon::prelude::*;

use super::{to_u8, Frame, ProjectCtx, RenderStats, Renderer, Splat2D, DILATION};
use super::radix;
use super::simd::{F4, LANES, M4};
use super::{ALPHA_MAX, ALPHA_MIN, T_MIN};
use crate::camera::View;
use crate::scene::Scene;

pub const TILE: u32 = 16;
const TILE_PIXELS: usize = (TILE * TILE) as usize;
const _: () = assert!(TILE <= 32 && TILE.is_multiple_of(4), "row masks are u32, groups are 4 lanes");

/// Minimum gaussians per projection job, and projection jobs per thread.
const MIN_GROUP: usize = 1024;
const GROUPS_PER_THREAD: usize = 8;

/// Per-splat data needed by the rasterizer.
#[derive(Clone, Copy, Debug, Default)]
struct RSplat {
    x: f32,
    y: f32,
    /// Conic (a, b, c) of the inverse 2D covariance.
    conic: [f32; 3],
    color: [f32; 3],
    opacity: f32,
    /// Pixel bounding box `[x0, y0, x1)`×`[y0, y1)` clamped to the frame,
    /// computed exactly like the reference renderer's.
    bbox: [u16; 4],
}

/// Per-projection-job scratch, reused across frames.
#[derive(Default)]
struct Group {
    /// Sort keys (depth bits << 32 | reordered index) of this group's visible
    /// splats, in index order.
    keys: Vec<u64>,
    /// Overlap pairs: (tile index, index into `keys`).
    tiles: Vec<(u32, u32)>,
    /// Overlaps per tile (later: write cursor per tile).
    counts: Vec<u32>,
}

pub struct CpuTiled {
    prep: Prep,
    groups: Vec<Group>,
    /// Per-overlap sort keys (depth bits << 32 | reordered index), bucketed
    /// by tile; `tile_start` holds the bucket boundaries.
    keys: Vec<u64>,
    /// Projected splats by reordered index (only visible ones are valid).
    splats: Vec<RSplat>,
    /// Radix sort scratch, same layout as `keys`.
    sort_tmp: Vec<u64>,
    tile_start: Vec<u32>,
}

impl CpuTiled {
    pub fn new(scene: Arc<Scene>) -> Self {
        CpuTiled {
            prep: Prep::new(&scene),
            groups: Vec::new(),
            keys: Vec::new(),
            splats: Vec::new(),
            sort_tmp: Vec::new(),
            tile_start: Vec::new(),
        }
    }
}

/// Clamped pixel bounding box of a projected splat (as in the reference).
#[inline]
fn pixel_bbox(s: &Splat2D, width: u32, height: u32) -> [u16; 4] {
    let x0 = (s.x - s.ext_x).floor().max(0.0) as u32;
    let y0 = (s.y - s.ext_y).floor().max(0.0) as u32;
    let x1 = ((s.x + s.ext_x).ceil().max(0.0) as u32).min(width);
    let y1 = ((s.y + s.ext_y).ceil().max(0.0) as u32).min(height);
    [x0.min(x1) as u16, y0.min(y1) as u16, x1 as u16, y1 as u16]
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
        let prep = &self.prep;
        let scene = &prep.scene;
        let n = scene.len();
        let ctx = ProjectCtx::new(view, width, height);
        let tiles_x = width.div_ceil(TILE);
        let tiles_y = height.div_ceil(TILE);
        let num_tiles = (tiles_x * tiles_y) as usize;

        // 1. Project + emit overlap pairs, per index group.
        // A few jobs per thread balances load; more would only add per-job
        // tile histograms to merge.
        let jobs = n.div_ceil(MIN_GROUP).min(rayon::current_num_threads() * GROUPS_PER_THREAD).max(1);
        let group = n.div_ceil(jobs).max(1);
        self.groups.resize_with(n.div_ceil(group), Group::default);
        let grid = Grid { tiles_x, width, height };
        if self.splats.len() != n {
            self.splats = vec![RSplat::default(); n];
        }
        self.groups
            .par_iter_mut()
            .zip(self.splats.par_chunks_mut(group))
            .enumerate()
            .for_each(|(gi, (g, out))| project_group(g, gi * group, out, scene, prep, &ctx, grid, num_tiles));
        stats.visible = self.groups.iter().map(|g| g.keys.len()).sum();
        let t_proj = Instant::now();

        // 2. Counting sort by tile: tile starts, then per-group write cursors.
        self.tile_start.clear();
        self.tile_start.resize(num_tiles + 1, 0);
        let mut total = 0u32;
        for t in 0..num_tiles {
            self.tile_start[t] = total;
            for g in self.groups.iter_mut() {
                let c = g.counts[t];
                g.counts[t] = total;
                total += c;
            }
        }
        self.tile_start[num_tiles] = total;
        let total = total as usize;
        stats.overlaps = total;
        // Scatter the keys into per-tile buckets. Buckets fill in index order,
        // which the (stable) radix sort relies on.
        self.keys.clear();
        self.keys.reserve(total);
        {
            let keys = SyncPtr(self.keys.as_mut_ptr());
            let keys = &keys;
            self.groups.par_iter_mut().for_each(|g| {
                for &(t, r) in &g.tiles {
                    let c = &mut g.counts[t as usize];
                    // SAFETY: cursors of different (group, tile) pairs index
                    // disjoint ranges of the `total`-sized buffer.
                    unsafe { keys.0.add(*c as usize).write(g.keys[r as usize]) };
                    *c += 1;
                }
            });
            // SAFETY: every slot in 0..total was written exactly once above.
            unsafe { self.keys.set_len(total) };
        }
        if self.sort_tmp.len() < total {
            self.sort_tmp.resize(total, 0);
        }
        let mut buckets: Vec<Bucket> = Vec::with_capacity(num_tiles);
        let (mut rest, mut rest_tmp) = (&mut self.keys[..], &mut self.sort_tmp[..]);
        for t in 0..num_tiles {
            let (b, e) = (self.tile_start[t] as usize, self.tile_start[t + 1] as usize);
            let (keys, tail) = rest.split_at_mut(e - b);
            rest = tail;
            let (tmp, tail) = rest_tmp.split_at_mut(e - b);
            rest_tmp = tail;
            buckets.push(Bucket { tile: t as u32, keys, tmp });
        }
        // Schedule the most expensive tiles first so they don't straggle.
        buckets.sort_unstable_by_key(|b| std::cmp::Reverse(b.keys.len()));
        let t_sort = Instant::now();

        // 3. Sort each tile's bucket by depth and rasterize it directly into
        // the frame (fused, so the keys are still in cache).
        let bg = background.to_array();
        let out = SyncPtr(frame.pixels.as_mut_ptr());
        let out = &out;
        let splats = &self.splats;
        buckets.into_par_iter().with_max_len(1).for_each(|b| {
            radix::sort_hi32(b.keys, b.tmp);
            fix_ties(b.keys, &prep.orig);
            let tx = b.tile % tiles_x;
            let ty = b.tile / tiles_x;
            raster_tile(splats, b.keys, tx * TILE, ty * TILE, width, height, bg, out);
        });
        let end = Instant::now();

        stats.project_ms = (t_proj - start).as_secs_f64() * 1e3;
        stats.sort_ms = (t_sort - t_proj).as_secs_f64() * 1e3;
        stats.raster_ms = (end - t_sort).as_secs_f64() * 1e3;
        stats.total_ms = (end - start).as_secs_f64() * 1e3;
        stats
    }
}

#[derive(Clone, Copy)]
struct Grid {
    tiles_x: u32,
    width: u32,
    height: u32,
}

/// Project (reordered) gaussians `base..base + out.len()` into `out` and `g`:
/// sort keys and one `(tile, key)` pair per overlapped tile, counted per tile.
#[allow(clippy::too_many_arguments)]
fn project_group(
    g: &mut Group,
    base: usize,
    out: &mut [RSplat],
    scene: &Scene,
    prep: &Prep,
    ctx: &ProjectCtx,
    grid: Grid,
    num_tiles: usize,
) {
    let Grid { tiles_x, width, height } = grid;
    let rt = ctx.rot.transpose();
    g.keys.clear();
    g.tiles.clear();
    g.counts.clear();
    g.counts.resize(num_tiles, 0);
    let mut emit = |i: usize, slot: &mut RSplat, s: Splat2D, bbox: [u16; 4]| {
        let rec = g.keys.len() as u32;
        *slot = RSplat {
            x: s.x,
            y: s.y,
            conic: s.conic,
            color: s.color,
            opacity: s.opacity,
            bbox,
        };
        g.keys.push(((s.depth.to_bits() as u64) << 32) | i as u64);
        if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
            return;
        }
        // Tiles containing the bounding box's pixels.
        let (x0, y0) = (bbox[0] as u32 / TILE, bbox[1] as u32 / TILE);
        let (x1, y1) = ((bbox[2] as u32).div_ceil(TILE), (bbox[3] as u32).div_ceil(TILE));
        // (Exact ellipse/tile intersection culls only a few percent of these
        // pairs on real scenes and costs more than it saves.)
        for ty in y0..y1 {
            for tx in x0..x1 {
                let t = ty * tiles_x + tx;
                g.tiles.push((t, rec));
                g.counts[t as usize] += 1;
            }
        }
    };
    let mut chunks = out.chunks_exact_mut(LANES);
    let mut i0 = base;
    for chunk in &mut chunks {
        let (mut mask, p) = project4(prep, i0, ctx, &rt);
        if mask != 0 {
            let (x, y, depth) = (p.x.to_array(), p.y.to_array(), p.depth.to_array());
            let (ext_x, ext_y) = (p.ext_x.to_array(), p.ext_y.to_array());
            let conic = p.conic.map(F4::to_array);
            // Pixel bounding boxes, exactly as `pixel_bbox` computes them.
            let (w, h) = (F4::splat(width as f32), F4::splat(height as f32));
            let bx0 = p.x.sub(p.ext_x).floor().to_u32();
            let by0 = p.y.sub(p.ext_y).floor().to_u32();
            let bx1 = p.x.add(p.ext_x).ceil().min(w).to_u32();
            let by1 = p.y.add(p.ext_y).ceil().min(h).to_u32();
            while mask != 0 {
                let l = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let i = i0 + l;
                let s = Splat2D {
                    x: x[l],
                    y: y[l],
                    conic: [conic[0][l], conic[1][l], conic[2][l]],
                    color: splat_color(scene, i, ctx),
                    opacity: scene.opacities[i],
                    depth: depth[l],
                    ext_x: ext_x[l],
                    ext_y: ext_y[l],
                };
                let bbox = [bx0[l].min(bx1[l]) as u16, by0[l].min(by1[l]) as u16, bx1[l] as u16, by1[l] as u16];
                debug_assert_eq!(bbox, pixel_bbox(&s, width, height));
                emit(i, &mut chunk[l], s, bbox);
            }
        }
        i0 += LANES;
    }
    for (j, slot) in chunks.into_remainder().iter_mut().enumerate() {
        let i = i0 + j;
        if let Some(s) = project_fast(scene, prep, i, ctx, &rt) {
            emit(i, slot, s, pixel_bbox(&s, width, height));
        }
    }
}

/// Per-scene data precomputed once so per-frame work is cheaper.
struct Prep {
    /// The scene's gaussians reordered along a 3D Morton curve, so splats that
    /// land in the same tile are mostly close together in memory (the
    /// rasterizer reads them in depth order, i.e. otherwise randomly).
    scene: Scene,
    /// Original scene index of each reordered gaussian (for depth ties).
    orig: Vec<u32>,
    /// `sqrt(2 ln(opacity / ALPHA_MIN))`: the bounding-box radius in standard
    /// deviations (0 for culled-by-opacity gaussians).
    k: Vec<f32>,
    /// Structure-of-arrays copies of positions and covariances for the
    /// 4-wide projection.
    pos: [Vec<f32>; 3],
    cov: [Vec<f32>; 6],
}

/// Spread the low 10 bits of `v` to every third bit.
fn spread3(v: u32) -> u32 {
    let mut x = v & 0x3ff;
    x = (x | (x << 16)) & 0x0300_00ff;
    x = (x | (x << 8)) & 0x0300_f00f;
    x = (x | (x << 4)) & 0x030c_30c3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

impl Prep {
    fn new(src: &Scene) -> Self {
        let n = src.len();
        // Quantize to a 1024³ grid over the robust bounds (outliers clamp to
        // the border cells) and sort by Morton code.
        let b = src.bounds();
        let lo = b.center - Vec3::splat(4.0 * b.radius);
        let scale = 1024.0 / (8.0 * b.radius);
        let mut order: Vec<u64> = src
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let q = ((*p - lo) * scale).clamp(Vec3::ZERO, Vec3::splat(1023.0));
                // (`as` maps NaN to 0.)
                let code = spread3(q.x as u32) | spread3(q.y as u32) << 1 | spread3(q.z as u32) << 2;
                (code as u64) << 32 | i as u64
            })
            .collect();
        order.par_sort_unstable();
        let orig: Vec<u32> = order.iter().map(|&o| o as u32).collect();
        let per = src.rest_per_splat();
        let scene = Scene {
            positions: orig.iter().map(|&i| src.positions[i as usize]).collect(),
            opacities: orig.iter().map(|&i| src.opacities[i as usize]).collect(),
            colors: orig.iter().map(|&i| src.colors[i as usize]).collect(),
            cov3d: orig.iter().map(|&i| src.cov3d[i as usize]).collect(),
            sh_degree: src.sh_degree,
            sh_rest: orig
                .iter()
                .flat_map(|&i| src.sh_rest[i as usize * per..(i as usize + 1) * per].iter().copied())
                .collect(),
            // Not needed for rendering (covariances are precomputed).
            scales: Vec::new(),
            rotations: Vec::new(),
        };
        let k = scene
            .opacities
            .iter()
            .map(|&o| if o < ALPHA_MIN { 0.0 } else { (2.0 * (o / ALPHA_MIN).ln()).sqrt() })
            .collect();
        debug_assert_eq!(scene.len(), n);
        let pos = [0, 1, 2].map(|c| scene.positions.iter().map(|p| p[c]).collect());
        let cov = [0, 1, 2, 3, 4, 5].map(|c| scene.cov3d.iter().map(|v| v[c]).collect());
        Prep { scene, orig, k, pos, cov }
    }
}

/// Within runs of equal depth in depth-sorted `keys`, order by original scene
/// index (the reference renderer's tie-break). Runs are rare, so this is
/// essentially one sequential scan.
fn fix_ties(keys: &mut [u64], orig: &[u32]) {
    let mut i = 0;
    while i + 1 < keys.len() {
        let d = keys[i] >> 32;
        if keys[i + 1] >> 32 != d {
            i += 1;
            continue;
        }
        let mut j = i + 2;
        while j < keys.len() && keys[j] >> 32 == d {
            j += 1;
        }
        keys[i..j].sort_unstable_by_key(|&k| orig[k as u32 as usize]);
        i = j;
    }
}

/// Exactly [`super::project`] (same operations, bit-identical results), but
/// with per-scene and per-frame constants hoisted out and fully inlined.
#[inline(always)]
fn project_fast(scene: &Scene, prep: &Prep, i: usize, ctx: &ProjectCtx, rt: &Mat3) -> Option<Splat2D> {
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
    let tx = (t.x * inv_z).clamp(-ctx.lim_x, ctx.lim_x);
    let ty = (t.y * inv_z).clamp(-ctx.lim_y, ctx.lim_y);
    let f = ctx.focal * inv_z;
    let j0 = Vec3::new(f, 0.0, -f * tx);
    let j1 = Vec3::new(0.0, f, -f * ty);
    let t0 = *rt * j0;
    let t1 = *rt * j1;
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
    let k = prep.k[i];
    let ext_x = k * a.sqrt();
    let ext_y = k * cc.sqrt();
    if x + ext_x <= 0.0 || x - ext_x >= ctx.width || y + ext_y <= 0.0 || y - ext_y >= ctx.height {
        return None;
    }
    let color = splat_color(scene, i, ctx);
    if !(x.is_finite() && y.is_finite() && ext_x.is_finite() && ext_y.is_finite()) {
        return None;
    }
    Some(Splat2D {
        x,
        y,
        conic: [cc * inv_det, -b * inv_det, a * inv_det],
        color,
        opacity,
        depth: t.z,
        ext_x,
        ext_y,
    })
}

/// View-dependent color of gaussian `i`, as in [`super::project`].
#[inline(always)]
fn splat_color(scene: &Scene, i: usize, ctx: &ProjectCtx) -> [f32; 3] {
    let mut color = scene.colors[i];
    if scene.sh_degree > 0 {
        let per = scene.rest_per_splat();
        let dir = (scene.positions[i] - ctx.eye).normalize_or_zero();
        color += crate::sh::eval_rest(scene.sh_degree, &scene.sh_rest[i * per..(i + 1) * per], dir);
    }
    color.max(Vec3::ZERO).to_array()
}

/// Geometry of 4 consecutive projected gaussians (see [`project4`]).
struct Proj4 {
    x: F4,
    y: F4,
    conic: [F4; 3],
    depth: F4,
    ext_x: F4,
    ext_y: F4,
}

/// [`project_fast`] for gaussians `i..i + 4` at once, minus the color: the
/// same IEEE operations in the same order (no fused multiply-adds), so the
/// results are bit-identical. Returns the mask of lanes that survive culling.
#[inline(always)]
fn project4(prep: &Prep, i: usize, ctx: &ProjectCtx, rt: &Mat3) -> (u32, Proj4) {
    let s = F4::splat;
    let ld = |v: &[f32]| F4::load(&v[i..i + 4]);
    let opacity = ld(&prep.scene.opacities);
    let (dx, dy, dz) = (
        ld(&prep.pos[0]).sub(s(ctx.eye.x)),
        ld(&prep.pos[1]).sub(s(ctx.eye.y)),
        ld(&prep.pos[2]).sub(s(ctx.eye.z)),
    );
    // Mat3 * Vec3 as glam computes it: x_axis * v.x + y_axis * v.y + z_axis * v.z.
    let mat = |m: &Mat3, vx: F4, vy: F4, vz: F4| {
        let row = |c: usize| {
            s(m.x_axis[c]).mul(vx).add(s(m.y_axis[c]).mul(vy)).add(s(m.z_axis[c]).mul(vz))
        };
        (row(0), row(1), row(2))
    };
    let (tx_, ty_, tz) = mat(&ctx.rot, dx, dy, dz);
    let mut ok = opacity.lt(s(ALPHA_MIN)).not().and(tz.gt(s(ctx.near)));
    let inv_z = s(1.0).div(tz);
    let x = s(ctx.focal).mul(tx_).mul(inv_z).add(s(ctx.cx));
    let y = s(ctx.focal).mul(ty_).mul(inv_z).add(s(ctx.cy));
    let tx = tx_.mul(inv_z).max(s(-ctx.lim_x)).min(s(ctx.lim_x));
    let ty = ty_.mul(inv_z).max(s(-ctx.lim_y)).min(s(ctx.lim_y));
    let f = s(ctx.focal).mul(inv_z);
    let zero = s(0.0);
    let (t0x, t0y, t0z) = mat(rt, f, zero, f.neg().mul(tx));
    let (t1x, t1y, t1z) = mat(rt, zero, f, f.neg().mul(ty));
    let c: [F4; 6] = std::array::from_fn(|k| ld(&prep.cov[k]));
    let sigma = |vx: F4, vy: F4, vz: F4| {
        (
            c[0].mul(vx).add(c[1].mul(vy)).add(c[2].mul(vz)),
            c[1].mul(vx).add(c[3].mul(vy)).add(c[4].mul(vz)),
            c[2].mul(vx).add(c[4].mul(vy)).add(c[5].mul(vz)),
        )
    };
    let dot = |ax: F4, ay: F4, az: F4, (bx, by, bz): (F4, F4, F4)| ax.mul(bx).add(ay.mul(by)).add(az.mul(bz));
    let s0 = sigma(t0x, t0y, t0z);
    let a = dot(t0x, t0y, t0z, s0).add(s(DILATION));
    let b = dot(t1x, t1y, t1z, s0);
    let cc = dot(t1x, t1y, t1z, sigma(t1x, t1y, t1z)).add(s(DILATION));
    let det = a.mul(cc).sub(b.mul(b));
    ok = ok.and(det.gt(zero));
    let inv_det = s(1.0).div(det);
    let k = ld(&prep.k);
    let ext_x = k.mul(a.sqrt());
    let ext_y = k.mul(cc.sqrt());
    let off = x.add(ext_x).le(zero)
        .or(x.sub(ext_x).ge(s(ctx.width)))
        .or(y.add(ext_y).le(zero))
        .or(y.sub(ext_y).ge(s(ctx.height)));
    let inf = s(f32::INFINITY);
    let finite = x.abs().lt(inf).and(y.abs().lt(inf)).and(ext_x.abs().lt(inf)).and(ext_y.abs().lt(inf));
    ok = ok.and(off.not()).and(finite);
    let p = Proj4 {
        x,
        y,
        conic: [cc.mul(inv_det), b.neg().mul(inv_det), a.mul(inv_det)],
        depth: tz,
        ext_x,
        ext_y,
    };
    (ok.bits(), p)
}

/// How many keys ahead the rasterizer prefetches splat data.
const PREFETCH: usize = 16;

/// Hint the CPU to pull `p` into L1. Splat data is gathered in depth order,
/// i.e. randomly, so without this the rasterizer is bound by memory latency.
#[inline(always)]
fn prefetch<T>(p: &T) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: prefetch is a hint and never faults.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p as *const T, options(nostack, preserves_flags, readonly));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as above.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const T as *const i8);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}

/// One tile's overlaps: sort keys (+ sort scratch) and splat records.
struct Bucket<'a> {
    tile: u32,
    keys: &'a mut [u64],
    tmp: &'a mut [u64],
}

/// Raw pointer shared by parallel jobs that write disjoint elements.
struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}

/// 4-lane groups per tile row.
const ROW_GROUPS: usize = TILE as usize / LANES;
const TILE_GROUPS: usize = TILE_PIXELS / LANES;

/// Blend the depth-sorted splats overlapping one tile and write the result.
///
/// Pixels are processed in 4-lane groups along rows. A per-row bitmask of
/// unsaturated pixels lets whole groups/rows/tiles be skipped once opaque.
#[allow(clippy::too_many_arguments)]
fn raster_tile(
    splats: &[RSplat],
    keys: &[u64],
    ox: u32,
    oy: u32,
    width: u32,
    height: u32,
    bg: [f32; 3],
    out: &SyncPtr<[u8; 3]>,
) {
    let tw = TILE.min(width - ox) as usize;
    let th = TILE.min(height - oy) as usize;
    // Structure-of-arrays tile state: transmittance and accumulated color.
    let mut tr = [F4::splat(1.0); TILE_GROUPS];
    let mut cr = [F4::splat(0.0); TILE_GROUPS];
    let mut cg = [F4::splat(0.0); TILE_GROUPS];
    let mut cb = [F4::splat(0.0); TILE_GROUPS];
    // Bit `x` of `live[y]` is set while pixel (x, y) is not saturated.
    let full_row = ((1u64 << tw) - 1) as u32;
    let mut live = [0u32; TILE as usize];
    live[..th].fill(full_row);
    let mut live_rows = th;
    let (fx, fy) = (ox as f32, oy as f32);
    let lane = F4::new([0.0, 1.0, 2.0, 3.0]);
    let (zero, one) = (F4::splat(0.0), F4::splat(1.0));
    let (amin, amax, tmin) = (F4::splat(ALPHA_MIN), F4::splat(ALPHA_MAX), F4::splat(T_MIN));
    let mut dxs = [zero; ROW_GROUPS];
    let mut qx = [zero; ROW_GROUPS];

    for (ki, &key) in keys.iter().enumerate() {
        if let Some(&ahead) = keys.get(ki + PREFETCH) {
            prefetch(&splats[ahead as u32 as usize]);
        }
        let s = &splats[key as u32 as usize];
        // Clip the splat's bounding box to this tile (local pixel coords).
        let [bx0, by0, bx1, by1] = s.bbox.map(|v| v as usize);
        let lx0 = bx0.saturating_sub(ox as usize);
        let lx1 = bx1.saturating_sub(ox as usize).min(tw);
        let ly0 = by0.saturating_sub(oy as usize);
        let ly1 = by1.saturating_sub(oy as usize).min(th);
        if lx0 >= lx1 {
            continue;
        }
        let bbox_bits = (((1u64 << lx1) - 1) as u32) & !((1u32 << lx0) - 1);
        let g0 = lx0 / LANES;
        let g1 = lx1.div_ceil(LANES);
        let [a, b, c] = s.conic;
        // power = -0.5 a dx² + (-0.5 c dy²) + (-b dy) dx; the dx-only part is
        // shared by all rows.
        let cx0 = fx + 0.5 - s.x;
        for g in g0..g1 {
            let dx = lane.add(F4::splat((g * LANES) as f32 + cx0));
            dxs[g] = dx;
            qx[g] = dx.mul(dx).mul(F4::splat(-0.5 * a));
        }
        let (op, sr, sg, sb) = (F4::splat(s.opacity), F4::splat(s.color[0]), F4::splat(s.color[1]), F4::splat(s.color[2]));
        for ly in ly0..ly1 {
            let lm = live[ly] & bbox_bits;
            if lm == 0 {
                continue;
            }
            let dy = fy + ly as f32 + 0.5 - s.y;
            let qy = F4::splat(-0.5 * c * dy * dy);
            let bdy = F4::splat(-b * dy);
            let mut sat = 0u32;
            for g in g0..g1 {
                let gm = (lm >> (g * LANES)) & 0xF;
                if gm == 0 {
                    continue;
                }
                let i = ly * ROW_GROUPS + g;
                let pw = qx[g].add(qy).mul_add(bdy, dxs[g]);
                let alpha = pw.exp_neg().mul(op).min(amax);
                let m = M4::from_bits(gm).and(pw.le(zero)).and(alpha.ge(amin));
                let alpha = alpha.and_mask(m);
                let t = tr[i];
                let w = alpha.mul(t);
                cr[i] = cr[i].mul_add(sr, w);
                cg[i] = cg[i].mul_add(sg, w);
                cb[i] = cb[i].mul_add(sb, w);
                let nt = t.mul(one.sub(alpha));
                tr[i] = nt;
                sat |= nt.lt(tmin).and(m).bits() << (g * LANES);
            }
            if sat != 0 {
                live[ly] &= !sat;
                if live[ly] == 0 {
                    live_rows -= 1;
                }
            }
        }
        if live_rows == 0 {
            break;
        }
    }

    for ly in 0..th {
        // SAFETY: each tile writes only its own disjoint pixel rectangle of a
        // frame with `width * height` pixels.
        let row = unsafe {
            std::slice::from_raw_parts_mut(out.0.add((oy as usize + ly) * width as usize + ox as usize), tw)
        };
        for (g, px) in row.chunks_mut(LANES).enumerate() {
            let i = ly * ROW_GROUPS + g;
            let (t, r, gg, bb) = (tr[i].to_array(), cr[i].to_array(), cg[i].to_array(), cb[i].to_array());
            for (l, p) in px.iter_mut().enumerate() {
                *p = to_u8([r[l] + t[l] * bg[0], gg[l] + t[l] * bg[1], bb[l] + t[l] * bg[2]]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{OrbitCamera, UpAxis};

    fn views(scene: &Scene) -> Vec<View> {
        let mut cam = OrbitCamera::new(scene.bounds(), UpAxis::NegY);
        let mut v = Vec::new();
        for i in 0..6 {
            cam.rotate(1.1, 0.0);
            cam.pitch = 0.5 * (i as f32).sin();
            cam.zoom(if i % 2 == 0 { 1.7 } else { 0.6 });
            v.push(cam.view());
        }
        v
    }

    #[test]
    fn fix_ties_orders_equal_depths_by_original_index() {
        // Reordered index -> original index.
        let orig = [5u32, 3, 9, 0, 7, 1];
        let k = |d: u64, i: u64| d << 32 | i;
        let mut keys = vec![k(1, 4), k(2, 0), k(2, 1), k(2, 2), k(3, 5), k(4, 3), k(4, 4)];
        fix_ties(&mut keys, &orig);
        assert_eq!(keys, vec![k(1, 4), k(2, 1), k(2, 0), k(2, 2), k(3, 5), k(4, 3), k(4, 4)]);
    }

    #[test]
    fn prep_reorders_consistently() {
        let src = crate::gen::generate("shballs").unwrap();
        let prep = Prep::new(&src);
        let per = src.rest_per_splat();
        let mut seen = vec![false; src.len()];
        for (j, &i) in prep.orig.iter().enumerate() {
            let i = i as usize;
            assert!(!std::mem::replace(&mut seen[i], true));
            assert_eq!(prep.scene.positions[j], src.positions[i]);
            assert_eq!(prep.scene.cov3d[j], src.cov3d[i]);
            assert_eq!(prep.scene.sh_rest[j * per..(j + 1) * per], src.sh_rest[i * per..(i + 1) * per]);
        }
    }

    #[test]
    fn project_fast_is_bit_identical() {
        for name in ["galaxy", "shballs", "cube", "knot"] {
            let prep = Prep::new(&crate::gen::generate(name).unwrap());
            let scene = &prep.scene;
            for view in views(&scene) {
                for (w, h) in [(320, 180), (37, 23)] {
                    let ctx = ProjectCtx::new(&view, w, h);
                    let rt = ctx.rot.transpose();
                    for i in 0..scene.len() {
                        let a = super::super::project(scene, i, &ctx);
                        let b = project_fast(scene, &prep, i, &ctx, &rt);
                        // The 4-wide path must agree lane by lane too.
                        let base = i & !(LANES - 1);
                        if base + LANES <= scene.len() {
                            let (mask, p) = project4(&prep, base, &ctx, &rt);
                            let l = i - base;
                            assert_eq!(mask >> l & 1 != 0, a.is_some(), "{name} splat {i}: 4-wide culling differs");
                            if let Some(a) = a {
                                let got = [p.x, p.y, p.conic[0], p.conic[1], p.conic[2], p.depth, p.ext_x, p.ext_y]
                                    .map(|v| v.to_array()[l].to_bits());
                                let want = [a.x, a.y, a.conic[0], a.conic[1], a.conic[2], a.depth, a.ext_x, a.ext_y]
                                    .map(f32::to_bits);
                                assert_eq!(got, want, "{name} splat {i}: 4-wide values differ");
                            }
                        }
                        match (a, b) {
                            (None, None) => {}
                            (Some(a), Some(b)) => {
                                let bits = |s: Splat2D| {
                                    [s.x, s.y, s.conic[0], s.conic[1], s.conic[2], s.color[0], s.color[1], s.color[2], s.opacity, s.depth, s.ext_x, s.ext_y]
                                        .map(f32::to_bits)
                                };
                                assert_eq!(bits(a), bits(b), "{name} splat {i}");
                            }
                            _ => panic!("{name} splat {i}: culling differs"),
                        }
                    }
                }
            }
        }
    }
}
