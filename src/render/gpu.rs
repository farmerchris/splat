//! Headless GPU compute rasterizer (wgpu: Metal / Vulkan / DX12).
//!
//! The scene is uploaded once. Every frame is a single command buffer:
//!
//! 1. `project`: EWA projection of every gaussian (same math as
//!    [`super::project`]); visible splats are compacted with an atomic counter
//!    and get a 24-bit depth key.
//! 2. Radix sort of the visible splats by depth (3 stable 8-bit passes).
//! 3. `count` + prefix scan + `emit`: expand each splat, *in depth order*,
//!    into one `(tile, splat)` pair per covered 8×8 tile.
//! 4. Stable radix sort of the pairs by tile id (1–2 passes for terminal
//!    sizes); since the input was depth ordered, each tile's run is sorted
//!    front to back. This is cheaper than sorting `(tile, depth)` keys: the
//!    multi-pass depth sort runs over the visible splats rather than over the
//!    (2× larger) pair list.
//! 5. `ranges`: per-tile `[begin, end)` runs.
//! 6. `raster`: one 8×8 workgroup per (tile, chunk of ≤ CHUNK splats), splats
//!    staged through shared memory, early termination once every pixel is
//!    saturated; `resolve` composites the chunks of each tile.
//! 7. Copy the packed RGBA8 image plus counters/timestamps into a staging
//!    buffer and map it.
//!
//! All element counts stay on the GPU (kernels read them from a counter
//! buffer), so there is exactly one CPU↔GPU round trip per frame. The pair
//! buffers have a fixed capacity; if a frame overflows it they are grown and
//! the frame is rendered again.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use super::{Frame, ProjectCtx, RenderStats, Renderer};
use super::{ALPHA_MAX, ALPHA_MIN, DILATION, T_MIN};
use crate::camera::View;
use crate::scene::Scene;

const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");
const PROJECT_WGSL: &str = include_str!("shaders/project.wgsl");
const TILES_WGSL: &str = include_str!("shaders/tiles.wgsl");
const SCAN_WGSL: &str = include_str!("shaders/scan.wgsl");
const SORT_WGSL: &str = include_str!("shaders/sort.wgsl");
const RASTER_WGSL: &str = include_str!("shaders/raster.wgsl");

/// Raster tile edge in pixels. Smaller than the CPU's 16: GPU threads evaluate
/// every pixel of a tile for every splat in its list, so small tiles waste
/// less work on small splats and give more workgroups to balance load.
const TILE: u32 = 8;
/// Max splats one raster workgroup blends; longer tile lists are split and
/// composited (see raster.wgsl).
const CHUNK: u32 = 4096;
/// Threads per workgroup of the 1D kernels.
const WG: u32 = 256;
/// Elements per workgroup of the prefix scan (must match scan.wgsl).
const SCAN_BLOCK: u32 = 1024;
/// Keys per thread in a radix sort pass (a block is 256 * SORT_CHUNKS keys).
const SORT_CHUNKS: u32 = 16;
/// 8-bit radix passes of the depth sort. Depth keys are the f32 bit pattern
/// relative to the near plane with the low bits dropped (see project.wgsl):
/// 3 passes keep 2^-20 relative depth precision over 16 octaves. (2 passes,
/// i.e. ~0.8% precision, measurably changes the blend order: 33 dB on plush.)
const DEPTH_PASSES: u32 = 3;
/// Max workgroups per dispatch dimension.
const MAX_GROUPS: u32 = 65_535;

/// Per-frame uniforms (layout must match `Params` in common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    right: [f32; 4],
    down: [f32; 4],
    fwd: [f32; 4],
    eye: [f32; 4],
    cam: [f32; 4],
    size: [f32; 4],
    bg: [f32; 4],
    n: u32,
    sh_degree: u32,
    tiles_x: u32,
    tiles_y: u32,
    cap: u32,
    sh_stride: u32,
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortParams {
    shift: u32,
    slot: u32,
    cap: u32,
    max_blocks: u32,
}

/// Binding kinds used to build explicit bind group layouts.
#[derive(Clone, Copy)]
enum B {
    Uniform,
    Read,
    Write,
}

fn layout(device: &wgpu::Device, label: &str, kinds: &[B]) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = kinds
        .iter()
        .enumerate()
        .map(|(i, k)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match k {
                    B::Uniform => wgpu::BufferBindingType::Uniform,
                    B::Read => wgpu::BufferBindingType::Storage { read_only: true },
                    B::Write => wgpu::BufferBindingType::Storage { read_only: false },
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pl),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bufs: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn uniform<T: Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Split a workgroup count over x/y so it fits the per-dimension limit.
fn groups_2d(groups: u32) -> (u32, u32) {
    let groups = groups.max(1);
    if groups <= MAX_GROUPS {
        (groups, 1)
    } else {
        let y = groups.div_ceil(MAX_GROUPS);
        (groups.div_ceil(y), y)
    }
}

fn dispatch(
    pass: &mut wgpu::ComputePass<'_>,
    pipe: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    groups: u32,
) {
    let (x, y) = groups_2d(groups);
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

/// Pipelines that don't depend on buffer sizes.
struct Kernels {
    frame_layout_project: wgpu::BindGroupLayout,
    project: wgpu::ComputePipeline,
    tiles_layout: wgpu::BindGroupLayout,
    count: wgpu::ComputePipeline,
    emit: wgpu::ComputePipeline,
    ranges: wgpu::ComputePipeline,
    raster_layout: wgpu::BindGroupLayout,
    raster_chunks: wgpu::ComputePipeline,
    raster: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    scan_layout: wgpu::BindGroupLayout,
    scan_blocks: wgpu::ComputePipeline,
    scan_add: wgpu::ComputePipeline,
    sort_layout: wgpu::BindGroupLayout,
    sort_hist: wgpu::ComputePipeline,
    sort_scatter: wgpu::ComputePipeline,
}

impl Kernels {
    fn new(device: &wgpu::Device) -> Self {
        let header = format!(
            "const ALPHA_MIN: f32 = {:?};\nconst ALPHA_MAX: f32 = {:?};\nconst T_MIN: f32 = {:?};\n\
             const DILATION: f32 = {:?};\nconst DEPTH_SHIFT: u32 = {}u;\nconst TILE: u32 = {}u;\nconst CHUNK: u32 = {}u;\n{}\n",
            ALPHA_MIN, ALPHA_MAX, T_MIN, DILATION, 32 - 8 * DEPTH_PASSES, TILE, CHUNK, COMMON_WGSL
        );
        let module = |label: &str, body: &str, with_header: bool| {
            let src = if with_header {
                format!("{header}{body}")
            } else {
                body.to_string()
            };
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let m_project = module("project", PROJECT_WGSL, true);
        let m_tiles = module("tiles", TILES_WGSL, true);
        let m_raster = module("raster", RASTER_WGSL, true);
        let m_scan = module("scan", SCAN_WGSL, false);
        let m_sort = module(
            "sort",
            &format!("const CHUNKS: u32 = {}u;\n{SORT_WGSL}", SORT_CHUNKS),
            false,
        );

        use B::*;
        let frame_layout_project = layout(
            device,
            "project",
            &[Uniform, Read, Read, Write, Write, Write, Write, Write],
        );
        let tiles_layout = layout(
            device,
            "tiles",
            &[
                Uniform, Write, Read, Read, Write, Write, Write, Write, Write,
            ],
        );
        let raster_layout = layout(
            device,
            "raster",
            &[Uniform, Read, Read, Read, Write, Write, Write],
        );
        let scan_layout = layout(device, "scan", &[Uniform, Write, Write]);
        let sort_layout = layout(
            device,
            "sort",
            &[Uniform, Read, Read, Read, Write, Write, Write],
        );
        Kernels {
            project: pipeline(device, &m_project, "main", &frame_layout_project),
            count: pipeline(device, &m_tiles, "count", &tiles_layout),
            emit: pipeline(device, &m_tiles, "emit", &tiles_layout),
            ranges: pipeline(device, &m_tiles, "ranges_main", &tiles_layout),
            raster_chunks: pipeline(device, &m_raster, "chunks", &raster_layout),
            raster: pipeline(device, &m_raster, "main", &raster_layout),
            resolve: pipeline(device, &m_raster, "resolve", &raster_layout),
            scan_blocks: pipeline(device, &m_scan, "scan_blocks", &scan_layout),
            scan_add: pipeline(device, &m_scan, "add", &scan_layout),
            sort_hist: pipeline(device, &m_sort, "hist_main", &sort_layout),
            sort_scatter: pipeline(device, &m_sort, "scatter", &sort_layout),
            frame_layout_project,
            tiles_layout,
            raster_layout,
            scan_layout,
            sort_layout,
        }
    }
}

/// Exclusive prefix scan of a fixed-length u32 buffer (recursive block sums).
struct Scan {
    /// (workgroups, bind group) per level, finest first.
    levels: Vec<(u32, wgpu::BindGroup)>,
}

impl Scan {
    fn new(device: &wgpu::Device, k: &Kernels, data: &wgpu::Buffer, len: u32) -> Self {
        let mut levels = Vec::new();
        let mut cur = data.clone();
        let mut len = len.max(1);
        loop {
            let groups = len.div_ceil(SCAN_BLOCK);
            let sums = storage(device, "scan sums", groups as u64 * 4);
            let params = uniform(device, "scan params", &[len, 0, 0, 0]);
            levels.push((
                groups,
                bind(device, &k.scan_layout, &[&params, &cur, &sums]),
            ));
            if groups == 1 {
                break;
            }
            cur = sums;
            len = groups;
        }
        Scan { levels }
    }

    fn encode(&self, pass: &mut wgpu::ComputePass<'_>, k: &Kernels) {
        for (groups, bg) in &self.levels {
            dispatch(pass, &k.scan_blocks, bg, *groups);
        }
        for (groups, bg) in self.levels.iter().rev().skip(1) {
            dispatch(pass, &k.scan_add, bg, *groups);
        }
    }
}

/// Multi-pass LSD radix sort of key/value buffer pairs (ping-pong).
struct RadixSort {
    max_blocks: u32,
    /// One bind group per pass; pass `p` reads buffer `p % 2`, writes `1 - p % 2`.
    passes: Vec<wgpu::BindGroup>,
    scan: Scan,
}

impl RadixSort {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        k: &Kernels,
        counters: &wgpu::Buffer,
        slot: u32,
        cap: u32,
        keys: [&wgpu::Buffer; 2],
        vals: [&wgpu::Buffer; 2],
        passes: u32,
    ) -> Self {
        let max_blocks = cap.div_ceil(256 * SORT_CHUNKS).max(1);
        let hist = storage(device, "sort hist", 256 * max_blocks as u64 * 4);
        let scan = Scan::new(device, k, &hist, 256 * max_blocks);
        let passes = (0..passes)
            .map(|p| {
                let (i, o) = ((p % 2) as usize, (1 - p % 2) as usize);
                let params = uniform(
                    device,
                    "sort params",
                    &SortParams {
                        shift: 8 * p,
                        slot,
                        cap,
                        max_blocks,
                    },
                );
                bind(
                    device,
                    &k.sort_layout,
                    &[&params, counters, keys[i], vals[i], keys[o], vals[o], &hist],
                )
            })
            .collect();
        RadixSort {
            max_blocks,
            passes,
            scan,
        }
    }

    fn encode(&self, pass: &mut wgpu::ComputePass<'_>, k: &Kernels, passes: usize) {
        for bg in &self.passes[..passes] {
            dispatch(pass, &k.sort_hist, bg, self.max_blocks);
            self.scan.encode(pass, k);
            dispatch(pass, &k.sort_scatter, bg, self.max_blocks);
        }
    }
}

/// Buffers and bind groups that depend on the output size and pair capacity.
struct Sized {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    cap: u32,
    tile_passes: usize,
    ranges: wgpu::Buffer,
    out: wgpu::Buffer,
    /// Indirect dispatch args of the raster pass.
    raster_args: wgpu::Buffer,
    staging: wgpu::Buffer,
    tile_sort: RadixSort,
    emit_bg: wgpu::BindGroup,
    ranges_bg: wgpu::BindGroup,
    raster_bg: wgpu::BindGroup,
}

/// Timed compute passes per frame: project, depth sort, binning, tile sort, raster.
const STAGES: usize = 5;
/// Bytes after the pixels in the staging buffer: 4 counters + 2 timestamps per stage.
const STAGING_TAIL: u64 = 16 + 16 * STAGES as u64;

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    k: Kernels,
    n: u32,
    sh_degree: u32,
    sh_stride: u32,
    params: wgpu::Buffer,
    counters: wgpu::Buffer,
    /// Per visible splat: x, y, ext_x, ext_y (binning).
    geom: wgpu::Buffer,
    /// Per visible splat: packed raster record (`RSplat`, 32 bytes).
    rsplats: wgpu::Buffer,
    counts: wgpu::Buffer,
    /// Packed tile rect per depth-sorted splat.
    rects: wgpu::Buffer,
    project_bg: wgpu::BindGroup,
    depth_sort: RadixSort,
    count_scan: Scan,
    /// Depth-sorted visible splat indices (result of `depth_sort`).
    order_vals: wgpu::Buffer,
    timestamps: Option<(wgpu::QuerySet, wgpu::Buffer)>,
    timestamp_period: f32,
    sized: Option<Sized>,
    debug: bool,
    pixels: Vec<u32>,
    /// GPU busy time of the previous frame (timestamps), used to pace waiting.
    last_gpu_ms: f64,
    /// Initial pair capacity floor (small in tests to exercise growth).
    min_cap: u32,
}

impl GpuRenderer {
    pub fn new(scene: Arc<Scene>) -> Result<Self> {
        let t0 = Instant::now();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| anyhow!("no GPU adapter available: {e}"))?;

        let n = u32::try_from(scene.len()).context("scene too large for the GPU backend")?;
        let sh_stride = (scene.rest_per_splat() * 3) as u32;
        let gauss_bytes = scene.len() as u64 * 64;
        let sh_bytes = scene.sh_rest.len() as u64 * 12;
        let need = gauss_bytes.max(sh_bytes);

        let alimits = adapter.limits();
        if (alimits.max_storage_buffer_binding_size as u64) < need || alimits.max_buffer_size < need
        {
            anyhow::bail!(
                "scene needs {} MB GPU buffers, adapter allows {} MB",
                need >> 20,
                alimits.max_storage_buffer_binding_size >> 20
            );
        }
        let want_ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let features = if want_ts {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: alimits.max_storage_buffer_binding_size,
            max_buffer_size: alimits.max_buffer_size,
            max_storage_buffers_per_shader_stage: 8,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("splat-tui"),
            required_features: features,
            required_limits: limits,
            ..Default::default()
        }))
        .context("creating GPU device")?;

        let k = Kernels::new(&device);

        // Scene upload: 16 floats per gaussian (pos, opacity, cov3d, color).
        let mut gauss = vec![0f32; scene.len() * 16];
        for i in 0..scene.len() {
            let p = scene.positions[i];
            let c = &scene.cov3d[i];
            let col = scene.colors[i];
            gauss[i * 16..i * 16 + 13].copy_from_slice(&[
                p.x,
                p.y,
                p.z,
                scene.opacities[i],
                c[0],
                c[1],
                c[2],
                c[3],
                c[4],
                c[5],
                col.x,
                col.y,
                col.z,
            ]);
        }
        let init = |label: &str, contents: &[u8]| {
            if contents.is_empty() {
                storage(&device, label, 16)
            } else {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
        };
        let gauss_buf = init("gaussians", bytemuck::cast_slice(&gauss));
        drop(gauss);
        let sh: Vec<f32> = scene.sh_rest.iter().flat_map(|v| v.to_array()).collect();
        let sh_buf = init("sh", bytemuck::cast_slice(&sh));
        drop(sh);

        let nn = scene.len() as u64;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counters = storage(&device, "counters", 16);
        let geom = storage(&device, "splat geometry", nn * 16);
        let rsplats = storage(&device, "raster splats", nn * 32);
        let dkeys = [
            storage(&device, "depth keys 0", nn * 4),
            storage(&device, "depth keys 1", nn * 4),
        ];
        let dvals = [
            storage(&device, "depth vals 0", nn * 4),
            storage(&device, "depth vals 1", nn * 4),
        ];
        let counts = storage(&device, "tile counts", nn * 4);
        let rects = storage(&device, "tile rects", nn * 8);

        let project_bg = bind(
            &device,
            &k.frame_layout_project,
            &[
                &params, &gauss_buf, &sh_buf, &counters, &geom, &dkeys[0], &dvals[0], &rsplats,
            ],
        );
        let depth_sort = RadixSort::new(
            &device,
            &k,
            &counters,
            0,
            n.max(1),
            [&dkeys[0], &dkeys[1]],
            [&dvals[0], &dvals[1]],
            DEPTH_PASSES,
        );
        let count_scan = Scan::new(&device, &k, &counts, n.max(1));

        let timestamps = want_ts.then(|| {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * STAGES as u32,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp resolve"),
                size: 16 * STAGES as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            (qs, resolve)
        });
        let timestamp_period = queue.get_timestamp_period();
        let debug = std::env::var_os("SPLAT_GPU_DEBUG").is_some();
        if debug {
            let info = adapter.get_info();
            eprintln!(
                "gpu: {} ({:?}), {} gaussians uploaded, init {:.0} ms, timestamps: {}",
                info.name,
                info.backend,
                n,
                t0.elapsed().as_secs_f64() * 1e3,
                want_ts
            );
        }

        Ok(GpuRenderer {
            device,
            queue,
            k,
            n,
            sh_degree: scene.sh_degree,
            sh_stride,
            params,
            counters,
            geom,
            rsplats,
            counts,
            rects,
            project_bg,
            depth_sort,
            count_scan,
            order_vals: dvals[DEPTH_PASSES as usize % 2].clone(),
            timestamps,
            timestamp_period,
            sized: None,
            debug,
            pixels: Vec::new(),
            last_gpu_ms: 0.0,
            // Start with room for ~2 pairs per gaussian; grow on overflow.
            min_cap: n.saturating_mul(2).max(1 << 18),
        })
    }

    /// (Re)create everything that depends on the output size or pair capacity.
    fn ensure_sized(&mut self, width: u32, height: u32, min_cap: u32) {
        if let Some(s) = &self.sized {
            if s.width == width && s.height == height && s.cap >= min_cap {
                return;
            }
        }
        let cap = match &self.sized {
            Some(s) => s.cap.max(min_cap),
            None => min_cap,
        };
        let device = &self.device;
        let k = &self.k;
        let tiles_x = width.div_ceil(TILE);
        let tiles_y = height.div_ceil(TILE);
        let num_tiles = tiles_x * tiles_y;
        let tile_bits = 32 - (num_tiles.max(2) - 1).leading_zeros();
        let tile_passes = tile_bits.div_ceil(8).max(1) as usize;

        let c = cap as u64 * 4;
        let tkeys = [
            storage(device, "tile keys 0", c),
            storage(device, "tile keys 1", c),
        ];
        let tvals = [
            storage(device, "tile vals 0", c),
            storage(device, "tile vals 1", c),
        ];
        let tile_sort = RadixSort::new(
            device,
            k,
            &self.counters,
            1,
            cap,
            [&tkeys[0], &tkeys[1]],
            [&tvals[0], &tvals[1]],
            tile_passes as u32,
        );
        let sorted = tile_passes % 2;
        let ranges = storage(device, "tile ranges", num_tiles as u64 * 8);
        let px_bytes = width as u64 * height as u64 * 4;
        let out = storage(device, "image", px_bytes);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: px_bytes + STAGING_TAIL,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tiles_bg = |keys: &wgpu::Buffer, vals: &wgpu::Buffer| {
            bind(
                device,
                &k.tiles_layout,
                &[
                    &self.params,
                    &self.counters,
                    &self.geom,
                    &self.order_vals,
                    &self.counts,
                    keys,
                    vals,
                    &ranges,
                    &self.rects,
                ],
            )
        };
        let emit_bg = tiles_bg(&tkeys[0], &tvals[0]);
        let ranges_bg = tiles_bg(&tkeys[sorted], &tvals[sorted]);
        let slots = num_tiles as u64 + (cap / CHUNK) as u64 + 1;
        let partial = storage(device, "raster partials", slots * (TILE * TILE) as u64 * 16);
        let raster_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster args"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raster_bg = bind(
            device,
            &k.raster_layout,
            &[
                &self.params,
                &self.rsplats,
                &tvals[sorted],
                &ranges,
                &out,
                &partial,
                &raster_args,
            ],
        );
        self.sized = Some(Sized {
            width,
            height,
            tiles_x,
            tiles_y,
            cap,
            tile_passes,
            ranges,
            out,
            raster_args,
            staging,
            tile_sort,
            emit_bg,
            ranges_bg,
            raster_bg,
        });
    }

    /// Encode, submit and read back one frame. Returns (visible, pairs, gpu stage ms).
    fn run_frame(
        &mut self,
        view: &View,
        width: u32,
        height: u32,
        bg: Vec3,
    ) -> (u32, u32, Option<[f64; STAGES]>) {
        let s = self.sized.as_ref().expect("sized");
        let k = &self.k;
        let ctx = ProjectCtx::new(view, width, height);
        let params = Params {
            right: view.right.extend(0.0).to_array(),
            down: view.down.extend(0.0).to_array(),
            fwd: view.forward.extend(0.0).to_array(),
            eye: ctx.eye.extend(0.0).to_array(),
            cam: [ctx.focal, ctx.cx, ctx.cy, ctx.near],
            size: [ctx.width, ctx.height, ctx.lim_x, ctx.lim_y],
            bg: bg.extend(0.0).to_array(),
            n: self.n,
            sh_degree: self.sh_degree,
            tiles_x: s.tiles_x,
            tiles_y: s.tiles_y,
            cap: s.cap,
            sh_stride: self.sh_stride,
            pad: [0; 2],
        };
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &s.raster_args,
            0,
            bytemuck::cast_slice(&[s.tiles_x, s.tiles_y, 0, 0]),
        );

        let t0 = Instant::now();
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.counters, 0, None);
        enc.clear_buffer(&s.ranges, 0, None);
        let elem_groups = self.n.div_ceil(WG);
        let timestamps = self.timestamps.as_ref();
        let stage = |enc: &mut wgpu::CommandEncoder,
                     i: u32,
                     label: &str,
                     f: &mut dyn FnMut(&mut wgpu::ComputePass<'_>)| {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: timestamps.map(|(qs, _)| wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(2 * i),
                    end_of_pass_write_index: Some(2 * i + 1),
                }),
            });
            f(&mut pass);
        };
        stage(&mut enc, 0, "project", &mut |pass| {
            dispatch(pass, &k.project, &self.project_bg, elem_groups)
        });
        stage(&mut enc, 1, "depth sort", &mut |pass| {
            self.depth_sort
                .encode(pass, k, self.depth_sort.passes.len())
        });
        stage(&mut enc, 2, "binning", &mut |pass| {
            dispatch(pass, &k.count, &s.emit_bg, elem_groups);
            self.count_scan.encode(pass, k);
            dispatch(pass, &k.emit, &s.emit_bg, elem_groups);
        });
        stage(&mut enc, 3, "tile sort", &mut |pass| {
            s.tile_sort.encode(pass, k, s.tile_passes);
            dispatch(pass, &k.ranges, &s.ranges_bg, s.cap.div_ceil(WG));
        });
        stage(&mut enc, 4, "raster", &mut |pass| {
            dispatch(
                pass,
                &k.raster_chunks,
                &s.raster_bg,
                (s.tiles_x * s.tiles_y).div_ceil(WG),
            );
            pass.set_pipeline(&k.raster);
            pass.dispatch_workgroups_indirect(&s.raster_args, 0);
            pass.set_pipeline(&k.resolve);
            pass.dispatch_workgroups(s.tiles_x, s.tiles_y, 1);
        });
        let px_bytes = width as u64 * height as u64 * 4;
        enc.copy_buffer_to_buffer(&s.out, 0, &s.staging, 0, px_bytes);
        enc.copy_buffer_to_buffer(&self.counters, 0, &s.staging, px_bytes, 16);
        if let Some((qs, resolve)) = &self.timestamps {
            enc.resolve_query_set(qs, 0..2 * STAGES as u32, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, &s.staging, px_bytes + 16, 16 * STAGES as u64);
        }
        self.queue.submit([enc.finish()]);
        let t1 = Instant::now();

        let slice = s.staging.slice(..);
        let mapped = Arc::new(AtomicBool::new(false));
        let flag = mapped.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            r.expect("mapping staging buffer");
            flag.store(true, Ordering::Release);
        });
        // wgpu's blocking wait on Metal polls the command buffer status with
        // 1 ms sleeps, which costs ~1 ms per frame here. Instead sleep through
        // part of the expected GPU time (the previous frame's GPU-timestamped
        // busy time, which is unaffected by how long we slept) and then spin
        // on non-blocking polls.
        let sleep_ms = 0.8 * self.last_gpu_ms - 0.5;
        if sleep_ms > 0.5 {
            std::thread::sleep(std::time::Duration::from_secs_f64(sleep_ms * 1e-3));
        }
        while !mapped.load(Ordering::Acquire) {
            self.device.poll(wgpu::PollType::Poll).expect("polling GPU");
            if !mapped.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        let t2 = Instant::now();
        let (vis, pairs, gpu_ms);
        {
            let data = slice.get_mapped_range();
            let px: &[u32] = bytemuck::cast_slice(&data[..px_bytes as usize]);
            self.pixels.clear();
            self.pixels.extend_from_slice(px);
            let tail: &[u32] =
                bytemuck::cast_slice(&data[px_bytes as usize..px_bytes as usize + 16]);
            vis = tail[0];
            pairs = tail[1];
            gpu_ms = self.timestamps.as_ref().map(|_| {
                let raw = &data[px_bytes as usize + 16..px_bytes as usize + STAGING_TAIL as usize];
                let t: Vec<u64> = raw
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let ms = |a: u64, b: u64| {
                    b.saturating_sub(a) as f64 * self.timestamp_period as f64 * 1e-6
                };
                std::array::from_fn::<f64, STAGES, _>(|i| ms(t[2 * i], t[2 * i + 1]))
            });
        }
        s.staging.unmap();
        self.last_gpu_ms = gpu_ms.map_or(0.0, |t| t.iter().sum());
        if self.debug {
            eprintln!(
                "gpu: encode+submit {:.2} ms, wait {:.2} ms, readback {:.2} ms, \
                 stages [project, depth sort, binning, tile sort, raster] {:.2?}, vis {vis}, pairs {pairs}",
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3,
                t2.elapsed().as_secs_f64() * 1e3,
                gpu_ms
            );
        }
        (vis, pairs, gpu_ms)
    }
}

impl Renderer for GpuRenderer {
    fn name(&self) -> &'static str {
        "gpu"
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
        if self.n == 0 {
            let c = super::to_u8(background.to_array());
            frame.pixels.fill(c);
            stats.total_ms = start.elapsed().as_secs_f64() * 1e3;
            return stats;
        }
        self.ensure_sized(width, height, self.min_cap);
        let (mut vis, mut pairs, mut gpu_ms) = self.run_frame(view, width, height, background);
        let cap = self.sized.as_ref().map_or(0, |s| s.cap);
        if pairs > cap {
            let new_cap = pairs.saturating_add(pairs / 4);
            self.ensure_sized(width, height, new_cap);
            (vis, pairs, gpu_ms) = self.run_frame(view, width, height, background);
        }
        for (dst, &p) in frame.pixels.iter_mut().zip(&self.pixels) {
            *dst = [p as u8, (p >> 8) as u8, (p >> 16) as u8];
        }
        stats.visible = vis as usize;
        stats.overlaps = pairs as usize;
        if let Some([project, depth_sort, binning, tile_sort, raster]) = gpu_ms {
            stats.project_ms = project;
            stats.sort_ms = depth_sort + binning + tile_sort;
            stats.raster_ms = raster;
        }
        stats.total_ms = start.elapsed().as_secs_f64() * 1e3;
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::psnr;
    use crate::camera::{OrbitCamera, UpAxis};
    use crate::render::cpu::CpuTiled;

    fn compare(scene_name: &str, width: u32, height: u32, min_psnr: f64, tiny_cap: bool) {
        let scene = Arc::new(crate::gen::generate(scene_name).unwrap());
        let mut gpu = match GpuRenderer::new(scene.clone()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e:#}");
                return;
            }
        };
        if tiny_cap {
            gpu.min_cap = 64;
        }
        let mut cpu = CpuTiled::new(scene.clone());
        let mut cam = OrbitCamera::new(scene.bounds(), UpAxis::NegY);
        let bg = Vec3::new(0.1, 0.2, 0.3);
        for step in 0..3 {
            let view = cam.view();
            let (mut a, mut b) = (Frame::default(), Frame::default());
            let sa = cpu.render(&view, width, height, bg, &mut a);
            let sb = gpu.render(&view, width, height, bg, &mut b);
            let p = psnr(&a, &b);
            assert!(
                p >= min_psnr,
                "{scene_name} step {step}: psnr {p:.1} dB (cpu visible {}, gpu visible {})",
                sa.visible,
                sb.visible
            );
            let tol = sa.visible / 1000 + 1;
            assert!(
                sa.visible.abs_diff(sb.visible) <= tol,
                "{scene_name} step {step}: visible {} vs {}",
                sa.visible,
                sb.visible
            );
            // GPU tiles are 8x8 vs the CPU's 16x16: every CPU (splat, tile) pair
            // covers at least one GPU pair.
            assert!(
                sb.overlaps >= sa.overlaps,
                "pairs {} < {}",
                sb.overlaps,
                sa.overlaps
            );
            cam.rotate(0.7, 0.2);
            cam.zoom(0.8);
        }
        if tiny_cap {
            assert!(
                gpu.sized.as_ref().unwrap().cap > 64,
                "pair buffers never grew"
            );
        }
    }

    #[test]
    fn gpu_matches_cpu_galaxy() {
        compare("galaxy", 160, 90, 40.0, false);
    }

    #[test]
    fn gpu_matches_cpu_sh_degree3() {
        compare("shballs", 96, 64, 40.0, false);
    }

    #[test]
    fn gpu_handles_odd_sizes() {
        // Partial tiles; 333x201 needs 2 tile-sort passes (> 256 tiles).
        compare("knot", 37, 23, 40.0, false);
        compare("planet", 333, 201, 40.0, false);
    }

    #[test]
    fn gpu_grows_pair_buffers_on_overflow() {
        compare("galaxy", 200, 120, 40.0, true);
    }
}
