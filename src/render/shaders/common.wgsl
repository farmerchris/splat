// Shared declarations for the per-frame kernels. The Rust side prepends the
// constants ALPHA_MIN, ALPHA_MAX, T_MIN, DILATION and TILE (generated from
// `render::*` so the GPU and CPU paths cannot drift apart).

struct Params {
    right: vec4<f32>,
    down: vec4<f32>,
    fwd: vec4<f32>,
    eye: vec4<f32>,
    // focal, cx, cy, near
    cam: vec4<f32>,
    // width, height, lim_x, lim_y
    size: vec4<f32>,
    // background rgb
    bg: vec4<f32>,
    n: u32,
    sh_degree: u32,
    tiles_x: u32,
    tiles_y: u32,
    // capacity of the tile key/value buffers
    cap: u32,
    // floats of SH "rest" data per gaussian
    sh_stride: u32,
    pad0: u32,
    pad1: u32,
}

// A projected gaussian as the rasterizer needs it (cf. `render::Splat2D`),
// packed into 32 bytes so each random gather touches a single cache sector.
// The bounding box half-extents are stored separately (`geom`: x, y, ext_x, ext_y).
struct RSplat {
    // center x, y and the conic pre-scaled for exp2:
    // (A, B, C) = log2(e) * (-a/2, -b, -c/2), so that
    // exp(power) = exp2(A dx^2 + B dx dy + C dy^2).
    v0: vec4<f32>,
    // bitcast(C), bitcast(opacity), pack2x16float(r, g), pack2x16float(b, 0)
    v1: vec4<u32>,
}

const LOG2_E: f32 = 1.4426950408889634;

fn is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn linear_id(wid: vec3<u32>, nwg: vec3<u32>, lid: u32) -> u32 {
    return (wid.y * nwg.x + wid.x) * 256u + lid;
}

// Tile rectangle [x0, x1) x [y0, y1) covered by a splat (same as `cpu::tile_rect`).
fn tile_rect(s: vec4<f32>, tiles_x: u32, tiles_y: u32) -> vec4<u32> {
    let t = f32(TILE);
    let fx = f32(tiles_x);
    let fy = f32(tiles_y);
    let x0 = u32(clamp(floor((s.x - s.z) / t), 0.0, fx));
    let y0 = u32(clamp(floor((s.y - s.w) / t), 0.0, fy));
    let x1 = u32(clamp(ceil((s.x + s.z) / t), 0.0, fx));
    let y1 = u32(clamp(ceil((s.y + s.w) / t), 0.0, fy));
    return vec4<u32>(min(x0, x1), min(y0, y1), x1, y1);
}
