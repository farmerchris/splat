// Per-tile front-to-back blending (mirrors `cpu::raster_tile`).
//
// A tile's depth-sorted splat list can be extremely long (a few tiles of a
// real capture touch >100k splats), and one workgroup walking it serially
// becomes the critical path of the whole frame. So each tile's list is split
// into chunks of CHUNK splats; every (tile, chunk) gets its own TILE x TILE
// workgroup that blends its chunk into a partial (color, transmittance), and
// `resolve` composites the chunks front to back:
//     C = C0 + T0 C1 + T0 T1 C2 + ...,   T = T0 T1 T2 ...
// Within a chunk splats are staged through shared memory in batches of
// TILE^2, and the workgroup stops once all its pixels are saturated.

const BATCH: u32 = TILE * TILE;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> rsplats: array<RSplat>;
@group(0) @binding(2) var<storage, read> tile_vals: array<u32>;
@group(0) @binding(3) var<storage, read> ranges: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_px: array<u32>;
// Per (tile, chunk) slot: TILE^2 partial (rgb, transmittance) values.
@group(0) @binding(5) var<storage, read_write> partial: array<vec4<f32>>;
// Indirect dispatch args for `main`: (tiles_x, tiles_y, max chunks per tile).
@group(0) @binding(6) var<storage, read_write> args: array<atomic<u32>>;

// (x, y, A, B) and (C, opacity, r, g); blue separately.
var<workgroup> s_a: array<vec4<f32>, BATCH>;
var<workgroup> s_b: array<vec4<f32>, BATCH>;
var<workgroup> s_blue: array<f32, BATCH>;
var<workgroup> done_count: atomic<u32>;
var<workgroup> u_shared: vec3<u32>;

fn quantize(v: f32) -> u32 {
    return u32(clamp(v, 0.0, 1.0) * 255.0 + 0.5);
}

fn tile_range(tile: u32) -> vec2<u32> {
    let b = ranges[2u * tile];
    return vec2<u32>(b, max(ranges[2u * tile + 1u], b));
}

// Slot of chunk 0 of a tile in `partial`. Unique because tile runs are laid
// out in tile order: slot(t + 1) >= slot(t) + chunks(t).
fn first_slot(tile: u32, begin: u32) -> u32 {
    return tile + begin / CHUNK;
}

// Max chunk count over all tiles -> z dimension of the raster dispatch.
@compute @workgroup_size(256)
fn chunks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile = gid.x;
    if (tile >= P.tiles_x * P.tiles_y) {
        return;
    }
    let r = tile_range(tile);
    atomicMax(&args[2], (r.y - r.x + CHUNK - 1u) / CHUNK);
}

@compute @workgroup_size(TILE, TILE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid2: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let tile = wid.y * P.tiles_x + wid.x;
    let chunk = wid.z;
    let px = wid.x * TILE + lid2.x;
    let py = wid.y * TILE + lid2.y;
    let inside = px < u32(P.size.x) && py < u32(P.size.y);
    let fx = f32(px) + 0.5;
    let fy = f32(py) + 0.5;

    if (lid == 0u) {
        atomicStore(&done_count, 0u);
        let r = tile_range(tile);
        u_shared = vec3<u32>(r, 0u);
    }
    let range = workgroupUniformLoad(&u_shared);
    let tile_begin = range.x;
    let begin = tile_begin + chunk * CHUNK;
    if (begin >= range.y) {
        return;
    }
    let end = min(range.y, begin + CHUNK);

    var done = !inside;
    if (done) {
        atomicAdd(&done_count, 1u);
    }
    var trans = 1.0;
    var color = vec3<f32>(0.0);
    let batches = (end - begin + BATCH - 1u) / BATCH;
    // Software pipelining: the dependent random-access gather for batch
    // b + 1 is issued before blending batch b.
    var next: RSplat;
    if (begin + lid < end) {
        next = rsplats[tile_vals[begin + lid]];
    }
    for (var batch = 0u; batch < batches; batch++) {
        workgroupBarrier();
        if (lid == 0u) {
            u_shared.z = atomicLoad(&done_count);
        }
        let all_done = workgroupUniformLoad(&u_shared).z;
        if (all_done >= BATCH) {
            break;
        }
        let first = begin + batch * BATCH;
        let rg = unpack2x16float(next.v1.z);
        s_a[lid] = next.v0;
        s_b[lid] = vec4<f32>(bitcast<f32>(next.v1.x), bitcast<f32>(next.v1.y), rg.x, rg.y);
        s_blue[lid] = unpack2x16float(next.v1.w).x;
        let k = first + BATCH + lid;
        if (k < end) {
            next = rsplats[tile_vals[k]];
        }
        workgroupBarrier();
        if (!done) {
            let cnt = min(BATCH, end - first);
            for (var j = 0u; j < cnt; j++) {
                let sa = s_a[j];
                let sb = s_b[j];
                let dx = fx - sa.x;
                let dy = fy - sa.y;
                let pw = sa.z * dx * dx + sb.x * dy * dy + sa.w * dx * dy;
                if (pw > 0.0) {
                    continue;
                }
                let alpha = min(sb.y * exp2(pw), ALPHA_MAX);
                if (alpha < ALPHA_MIN) {
                    continue;
                }
                color += vec3<f32>(sb.z, sb.w, s_blue[j]) * (alpha * trans);
                trans = trans * (1.0 - alpha);
                if (trans < T_MIN) {
                    done = true;
                    atomicAdd(&done_count, 1u);
                    break;
                }
            }
        }
    }
    partial[(first_slot(tile, tile_begin) + chunk) * BATCH + lid] = vec4<f32>(color, trans);
}

@compute @workgroup_size(TILE, TILE)
fn resolve(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid2: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let px = wid.x * TILE + lid2.x;
    let py = wid.y * TILE + lid2.y;
    if (px >= u32(P.size.x) || py >= u32(P.size.y)) {
        return;
    }
    let tile = wid.y * P.tiles_x + wid.x;
    let r = tile_range(tile);
    let n = (r.y - r.x + CHUNK - 1u) / CHUNK;
    let slot = first_slot(tile, r.x);
    var color = vec3<f32>(0.0);
    var trans = 1.0;
    for (var c = 0u; c < n; c++) {
        if (trans < T_MIN) {
            break;
        }
        let p = partial[(slot + c) * BATCH + lid];
        color += trans * p.xyz;
        trans *= p.w;
    }
    let c = color + trans * P.bg.xyz;
    out_px[py * u32(P.size.x) + px] = quantize(c.x) | (quantize(c.y) << 8u) | (quantize(c.z) << 16u);
}
