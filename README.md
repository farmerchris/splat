# splat-tui

View 3D Gaussian Splatting scenes in your terminal. Supports standard 3DGS
`.ply` files (including spherical harmonics up to degree 3) and the compact
`.splat` format. Rotate with the mouse or arrow keys.

Each terminal cell shows two pixels stacked vertically, using `▀` with a
24-bit foreground and background color. Rendering is supersampled 2× by
default for anti-aliasing.

## Quick start

```sh
cargo build --release
./target/release/splat-tui gen:galaxy                  # built-in procedural scene
./target/release/splat-tui gen -o scenes               # write example scenes to ./scenes
./target/release/splat-tui scenes/planet.ply
./target/release/splat-tui path/to/point_cloud.ply --up +y   # if the model shows upside down

# optional GPU backend (wgpu: Metal / Vulkan / DX12)
cargo build --release --features gpu
./target/release/splat-tui scene.ply --backend gpu
```

### Example scenes

* **Procedural** (`splat-tui gen --list`). Use them directly as `gen:<name>`,
  or write them to disk with `splat-tui gen [name] -o dir --format ply|splat`.
  * `cube`: RGB lattice (3k splats)
  * `knot`: torus knot (72k)
  * `galaxy`: spiral galaxy (104k)
  * `planet`: ringed planet with starfield (66k)
  * `shballs`: degree-3 SH, so colors change with view angle (36k)
* **Real captures.** Tested with these downloads:

  ```sh
  mkdir -p scenes/downloaded && cd scenes/downloaded
  for f in nike plush train; do
    curl -LO https://huggingface.co/cakewalk/splat-data/resolve/main/$f.splat
  done
  curl -L -o luigi.ply  https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply   # use --up +y
  curl -L -o bonsai.ply https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/point_cloud/iteration_7000/point_cloud.ply  # 287 MB, SH degree 3
  ```

  Unbounded room and outdoor captures (`train`, `bonsai`) only look right from
  roughly where the training cameras were. Seen from outside, the background
  and floaters get in the way, so zoom in and rotate to find the subject.

## Controls

| Input | Action |
|---|---|
| left-drag | orbit |
| right-drag / shift-drag | pan |
| scroll wheel | zoom |
| ← → ↑ ↓ | orbit by 5° |
| shift + arrows, `w a s d` | pan |
| `+` `-` / PgUp PgDn | zoom |
| `[` `]` | widen / narrow field of view |
| `r` | reset camera |
| `u` | cycle up axis (+Y, -Y, +Z, -Z) |
| space | toggle auto-rotate |
| `b` | cycle background color |
| `x` | cycle supersampling 1×–4× |
| `m` | toggle truecolor / 256-color |
| `n` | switch render backend live |
| `?` | help overlay |
| `q` / Esc | quit |

The status bar shows visible/total splats, frame time, fps, camera angles,
backend, and a timing breakdown (project / sort / raster / terminal draw).

## CLI

```
splat-tui [SCENE] [--backend cpu|cpu-naive|gpu] [--up -y|+y|+z|-z] [--supersample N] [--color truecolor|ansi256]
splat-tui gen [NAME|all] [-o DIR] [--format ply|splat] [--list]
splat-tui bench SCENE [--backend cpu,gpu,...] [--size WxH] [--frames N]
splat-tui render SCENE -o out.png [--size WxH] [--yaw DEG] [--pitch DEG] [--zoom F] [--backend ...]
```

`bench` renders an orbit headlessly with each backend. It reports mean, p50,
p95, min and per-stage timings, plus PSNR against the first backend, so the
backends are checked against each other.

## Architecture

| Module | Purpose |
|---|---|
| `scene.rs` | Scene storage as structure-of-arrays (activated params plus precomputed 3D covariance); robust bounds |
| `ply.rs` | PLY reader for ascii and binary LE/BE, all scalar types, 3DGS layout, SH degree 1–3; 3DGS PLY writer |
| `splat_format.rs` | `.splat` reader and writer (32 bytes per splat) |
| `camera.rs` | Orbit camera with selectable up axis; camera space is x right, y down, z forward (COLMAP) |
| `render/mod.rs` | `Renderer` trait and the shared EWA projection (Kerbl et al. 2023) |
| `render/cpu.rs`, `radix.rs`, `simd.rs` | Optimized multi-threaded tile rasterizer (default) |
| `render/naive.rs` | Single-threaded reference rasterizer |
| `render/gpu.rs`, `shaders/*.wgsl` | wgpu compute pipeline (feature `gpu`) |
| `term.rs` | Half-block widget, box-filter downsampling, 256-color fallback |
| `app.rs` | ratatui/crossterm TUI with mouse capture, synchronized output and a buffered writer |
| `gen.rs` | Procedural scenes |
| `bench.rs` | Benchmark harness, PSNR, dependency-free PNG writer |

Tests: 44 unit tests (51 with `--features gpu`) and 83 integration tests in
`tests/`, covering loaders and malformed input, backend agreement at odd sizes
and poses, camera, terminal mapping, and the CLI.

## Rendering backends: options evaluated and final decision

All numbers are from an Apple M1 (8 cores) and are milliseconds per frame,
measured with `splat-tui bench` (40-frame orbit). The PSNR column compares
each backend's image against the `cpu` backend.

### Options considered

| Option | Verdict |
|---|---|
| **CPU tiled, rayon** (`cpu`) | **Chosen as default.** Fastest at terminal resolutions, exact, no GPU dependency, works over SSH and in containers. |
| CPU single-threaded reference (`cpu-naive`) | Kept as the correctness reference and baseline. 6–10× slower. |
| **wgpu 27 compute** (`gpu`, Metal here) | Built and kept as an optional feature. Matches the CPU at ≈65 dB PSNR. Wins by 10–30% at 640×360 and above; loses or ties at terminal sizes. |
| Direct Metal (`metal` crate) | Rejected: macOS-only and unsafe. Its main advantage over wgpu (avoiding wgpu's 1 ms Metal wait sleep) was solved inside the wgpu backend instead. |
| Vulkan via MoltenVK (ash, vulkano) | Rejected: needs an SDK and translates to the same Metal driver, so it adds setup with no speed gain. |
| OpenCL / CUDA | OpenCL is deprecated on macOS (1.2); CUDA is unavailable on Apple hardware. |
| Hybrid (GPU project + CPU sort) | Measured and rejected: the whole GPU sort stage takes about 7 ms, while the CPU sort alone took 11–21 ms. |
| 2D vector libraries (tiny-skia, vello) | Not applicable: they draw paths, not depth-sorted anisotropic gaussians with front-to-back alpha blending. |
| Terminal output: ratatui + crossterm, half blocks | Chosen. Wrapping stdout in a 1 MB `BufWriter` and enabling synchronized updates cut terminal draw time in tmux from ~246 ms to ~4 ms per frame. Braille or ASCII output would lose color resolution; sixel and kitty graphics are not widely supported. |

### Final head-to-head (after optimization)

| Scene | Size | cpu | gpu | cpu-naive | gpu PSNR |
|---|---|---|---|---|---|
| train.splat (1.03M) | 240×120 | **17.2** | 18.9 | 132.2 | 65.2 |
| train.splat | 640×360 | 25.8 | **21.7** | 246.2 | 64.9 |
| bonsai.ply (1.16M, SH3) | 240×120 | 26.4 | **26.3** | 178.1 | 64.0 |
| bonsai.ply | 640×360 | 33.7 | **29.3** | 272.3 | 64.7 |
| plush.splat (281k) | 240×120 | **5.5** | 6.4 | 41.3 | 68.1 |
| plush.splat | 640×360 | 10.0 | **7.2** | 86.2 | 67.7 |
| gen:galaxy (104k) | 240×120 | **2.1** | 3.2 | 13.9 | 72.0 |
| gen:galaxy | 640×360 | 4.1 | **3.7** | 31.7 | 70.3 |
| gen:shballs (36k, SH3) | 240×120 | **1.6** | 2.5 | 9.5 | 81.5 |
| gen:shballs | 640×360 | 4.4 | **3.5** | 38.5 | 80.7 |

The optimized CPU rasterizer runs 2.4–3.5× faster than its first version. Train at
320×180 went from 57.8 to 16.6 ms; galaxy from 10.5 to 2.6 ms.

### Why CPU is the default

A 200×50 terminal at 2× supersampling is 400×200 pixels. At that size the GPU
pipeline's fixed cost (about 13 dispatches, plus waiting for submission and
readback, about 2–3 ms) is as large as the whole CPU frame, so the CPU wins or
ties at terminal sizes. The GPU pulls ahead at larger outputs (640×360+), by
10–30%, which matters for `render` and huge terminals but not for a typical
viewer window. The CPU path also gives bit-exact output against the reference, avoids
pulling in about 55 extra crates, and works anywhere a terminal does (SSH, no GPU).
The GPU backend remains available with `--features gpu` / `--backend gpu`, and
you can switch live with `n`.

### Optimizations that mattered

**CPU** (`render/cpu.rs`):
* Counting-sort pairs into per-tile buckets, then radix-sort each tile by
  depth (with an index tie-break), instead of one global sort. Sort went from
  12.8 to 0.9 ms on train.
* Per-row bitmasks of unsaturated pixels. Before this, 89% of rasterizer pixel
  visits hit pixels that were already opaque.
* Blending 4 pixels at a time with SIMD (NEON, or SSE on x86) and a polynomial
  `exp`.
* Software prefetch of splat records, and the scene reordered along a Morton
  curve (Z-order) at load, so a tile's splats sit close together in memory.
* Projecting 4 splats at a time with SIMD (bit-identical to the scalar
  version); adaptive job sizes; the per-tile sort fused into the raster job;
  largest tiles scheduled first.
* Tried and reverted: exact ellipse/tile culling (removed only 1–4% of pairs,
  and binning cost more), 8×8 and 32×32 tiles, an 11-bit radix, gather passes.

**GPU** (`render/gpu.rs`):
* 8×8 tiles: raster went from 32 to about 15 ms.
* Splitting long tile lists into 4096-splat chunks composited afterwards:
  raster went to about 6 ms. This was the biggest win.
* Waiting by GPU timestamps instead of wgpu-hal's 1 ms sleep loop:
  galaxy went from 6.3 to 3.3 ms.
* A 24-bit depth key with 3 radix passes. The 16-bit key was rejected
  (33 dB on plush).
* Splat records packed to 32 bytes (color stored as f16), plus `exp2`.
* Deterministic stream compaction (flag, scan, scatter) so equal-depth splats
  keep file order and repeated frames are bit-identical. This was also slightly
  faster than the atomic-append version it replaced.
* If a frame's per-tile buffers would exceed the device's buffer limits
  (only near the 16384×16384 maximum), that frame is rendered on the CPU instead.

## License

MIT
