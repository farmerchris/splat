// Per-gaussian EWA projection (mirrors `render::project`).
//
// Every gaussian i writes a visibility flag (0/1) to `flags[i]` and its depth
// key (or NO_KEY) to `keys[i]`; visible ones also write `geom[i]` and
// `rsplats[i]`. The host then runs an exclusive scan over the flags and
// `compact.wgsl` packs the visible (key, i) pairs in scene-index order. Unlike
// an atomic counter this is deterministic, and since the depth sort is stable
// the final order is (depth key, original index), like the CPU backends.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> gauss: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sh: array<f32>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> geom: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> rsplats: array<RSplat>;

const SH_C1: f32 = 0.4886025;
const SH_C2_0: f32 = 1.0925485;
const SH_C2_1: f32 = -1.0925485;
const SH_C2_2: f32 = 0.31539157;
const SH_C2_3: f32 = -1.0925485;
const SH_C2_4: f32 = 0.54627424;
const SH_C3_0: f32 = -0.5900436;
const SH_C3_1: f32 = 2.8906114;
const SH_C3_2: f32 = -0.4570458;
const SH_C3_3: f32 = 0.37317633;
const SH_C3_4: f32 = -0.4570458;
const SH_C3_5: f32 = 1.4453057;
const SH_C3_6: f32 = -0.5900436;

fn shc(base: u32, k: u32) -> vec3<f32> {
    let o = base + 3u * k;
    return vec3<f32>(sh[o], sh[o + 1u], sh[o + 2u]);
}

// View-dependent SH color (mirrors `sh::eval_rest`).
fn eval_sh(i: u32, dir: vec3<f32>) -> vec3<f32> {
    let base = i * P.sh_stride;
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    var c = SH_C1 * (-y * shc(base, 0u) + z * shc(base, 1u) - x * shc(base, 2u));
    if (P.sh_degree >= 2u) {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let yz = y * z;
        let xz = x * z;
        c += SH_C2_0 * xy * shc(base, 3u)
            + SH_C2_1 * yz * shc(base, 4u)
            + SH_C2_2 * (2.0 * zz - xx - yy) * shc(base, 5u)
            + SH_C2_3 * xz * shc(base, 6u)
            + SH_C2_4 * (xx - yy) * shc(base, 7u);
        if (P.sh_degree >= 3u) {
            c += SH_C3_0 * y * (3.0 * xx - yy) * shc(base, 8u)
                + SH_C3_1 * xy * z * shc(base, 9u)
                + SH_C3_2 * y * (4.0 * zz - xx - yy) * shc(base, 10u)
                + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shc(base, 11u)
                + SH_C3_4 * x * (4.0 * zz - xx - yy) * shc(base, 12u)
                + SH_C3_5 * z * (xx - yy) * shc(base, 13u)
                + SH_C3_6 * x * (xx - 3.0 * yy) * shc(base, 14u);
        }
    }
    return c;
}

// Max color channel before packing to f16 (whose max finite value is 65504).
const COLOR_MAX: f32 = 16.0;

fn clamp_color(c: f32) -> f32 {
    // Bit test rather than a comparison, which fast math may fold away.
    if (!is_finite(c)) {
        return 0.0;
    }
    return clamp(c, 0.0, COLOR_MAX);
}

fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let r = 1.0 / length(v);
    if (is_finite(r) && r > 0.0) {
        return v * r;
    }
    return vec3<f32>(0.0);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = linear_id(wid, nwg, lid);
    if (i >= P.n) {
        return;
    }
    let dk = project_one(i);
    keys[i] = dk;
    flags[i] = select(1u, 0u, dk == NO_KEY);
}

// Projects gaussian i, writing geom[i] / rsplats[i] if it is visible.
// Returns its depth key, or NO_KEY if it is culled.
fn project_one(i: u32) -> u32 {
    let g0 = gauss[4u * i];
    let opacity = g0.w;
    if (opacity < ALPHA_MIN) {
        return NO_KEY;
    }
    let d = g0.xyz - P.eye.xyz;
    let t = vec3<f32>(dot(P.right.xyz, d), dot(P.down.xyz, d), dot(P.fwd.xyz, d));
    if (!(t.z > P.cam.w)) {
        return NO_KEY;
    }
    let focal = P.cam.x;
    let inv_z = 1.0 / t.z;
    let x = focal * t.x * inv_z + P.cam.y;
    let y = focal * t.y * inv_z + P.cam.z;

    // Jacobian of the perspective projection at the (clamped) mean.
    let tx = clamp(t.x * inv_z, -P.size.z, P.size.z);
    let ty = clamp(t.y * inv_z, -P.size.w, P.size.w);
    let f = focal * inv_z;
    // Rows of T = J * R.
    let t0 = f * P.right.xyz - (f * tx) * P.fwd.xyz;
    let t1 = f * P.down.xyz - (f * ty) * P.fwd.xyz;

    let g1 = gauss[4u * i + 1u];
    let g2 = gauss[4u * i + 2u];
    let g3 = gauss[4u * i + 3u];
    // Covariance upper triangle: xx xy xz yy yz zz.
    let c0 = g1.x;
    let c1 = g1.y;
    let c2 = g1.z;
    let c3 = g1.w;
    let c4 = g2.x;
    let c5 = g2.y;
    let s0 = vec3<f32>(
        c0 * t0.x + c1 * t0.y + c2 * t0.z,
        c1 * t0.x + c3 * t0.y + c4 * t0.z,
        c2 * t0.x + c4 * t0.y + c5 * t0.z,
    );
    let s1 = vec3<f32>(
        c0 * t1.x + c1 * t1.y + c2 * t1.z,
        c1 * t1.x + c3 * t1.y + c4 * t1.z,
        c2 * t1.x + c4 * t1.y + c5 * t1.z,
    );
    let a = dot(t0, s0) + DILATION;
    let b = dot(t1, s0);
    let cc = dot(t1, s1) + DILATION;
    let det = a * cc - b * b;
    if (!(det > 0.0)) {
        return NO_KEY;
    }
    let inv_det = 1.0 / det;

    // Extent where opacity * exp(-r^2/2) drops below ALPHA_MIN.
    let k = sqrt(2.0 * log(opacity / ALPHA_MIN));
    let ext_x = k * sqrt(a);
    let ext_y = k * sqrt(cc);
    if (x + ext_x <= 0.0 || x - ext_x >= P.size.x || y + ext_y <= 0.0 || y - ext_y >= P.size.y) {
        return NO_KEY;
    }
    if (!(is_finite(x) && is_finite(y) && is_finite(ext_x) && is_finite(ext_y))) {
        return NO_KEY;
    }

    var color = vec3<f32>(g2.z, g2.w, g3.x);
    if (P.sh_degree > 0u) {
        color += eval_sh(i, normalize_or_zero(d));
    }
    // Clamp (and drop NaN from degenerate SH) so pack2x16float can't produce
    // inf, which would turn into NaN when blended with zero weight.
    color = vec3<f32>(clamp_color(color.x), clamp_color(color.y), clamp_color(color.z));

    geom[i] = vec4<f32>(x, y, ext_x, ext_y);
    // Conic (cc, -b, a) / det, pre-scaled for exp2 (see `RSplat`).
    let s = LOG2_E * inv_det;
    rsplats[i] = RSplat(
        vec4<f32>(x, y, -0.5 * s * cc, s * b),
        vec4<u32>(
            bitcast<u32>(-0.5 * s * a),
            bitcast<u32>(opacity),
            pack2x16float(color.xy),
            pack2x16float(vec2<f32>(color.z, 0.0)),
        ),
    );
    // Positive floats order like their bit patterns, and every visible depth
    // is > near, so the offset from near's bit pattern is a monotonic key.
    // Dropping DEPTH_SHIFT low bits keeps 23 - DEPTH_SHIFT mantissa bits,
    // i.e. 2^-(23 - DEPTH_SHIFT) relative precision, and lets the sort use
    // fewer radix passes. Keys within one bucket keep scene-index order.
    let dk = (bitcast<u32>(t.z) - bitcast<u32>(P.cam.w)) >> DEPTH_SHIFT;
    return min(dk, 0xffffffffu >> DEPTH_SHIFT);
}
