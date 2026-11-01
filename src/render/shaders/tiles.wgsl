// Tile binning kernels. Each entry point uses its own bind group layout, so
// bindings are declared per kernel with distinct names.

// ---- count: tiles touched by each splat, in depth order ------------------
// Entries past the visible count are zeroed so the following scan can run
// over a fixed length.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> geom: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
// count: per-splat tile counts (written); emit: their exclusive scan (read).
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> tile_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> tile_vals: array<u32>;
@group(0) @binding(7) var<storage, read_write> ranges: array<u32>;
// Tile rect of each splat in depth order, packed as (x0 | y0 << 16, x1 | y1 << 16),
// so `emit` reads it sequentially instead of gathering splats again.
@group(0) @binding(8) var<storage, read_write> rects: array<vec2<u32>>;

@compute @workgroup_size(256)
fn count(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let j = linear_id(wid, nwg, lid);
    if (j >= P.n) {
        return;
    }
    var cnt = 0u;
    if (j < atomicLoad(&counters[0])) {
        let r = tile_rect(geom[order[j]], P.tiles_x, P.tiles_y);
        cnt = (r.z - r.x) * (r.w - r.y);
        rects[j] = vec2<u32>(r.x | (r.y << 16u), r.z | (r.w << 16u));
    }
    counts[j] = cnt;
}

// ---- emit: one (tile, splat) pair per covered tile ------------------------
// Splats are visited in depth order, so a *stable* sort by tile id yields
// per-tile lists that are already sorted front to back.

@compute @workgroup_size(256)
fn emit(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let j = linear_id(wid, nwg, lid);
    let vis = atomicLoad(&counters[0]);
    if (j >= vis) {
        return;
    }
    let si = order[j];
    let pr = rects[j];
    let r = vec4<u32>(pr.x & 0xffffu, pr.x >> 16u, pr.y & 0xffffu, pr.y >> 16u);
    var o = counts[j];
    if (j == vis - 1u) {
        atomicStore(&counters[1], o + (r.z - r.x) * (r.w - r.y));
    }
    for (var ty = r.y; ty < r.w; ty++) {
        for (var tx = r.x; tx < r.z; tx++) {
            if (o < P.cap) {
                tile_keys[o] = ty * P.tiles_x + tx;
                tile_vals[o] = si;
            }
            o++;
        }
    }
}

// ---- ranges: [begin, end) of each tile's run in the sorted pair list ------

@compute @workgroup_size(256)
fn ranges_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let k = linear_id(wid, nwg, lid);
    let n = min(atomicLoad(&counters[1]), P.cap);
    if (k >= n) {
        return;
    }
    let t = tile_keys[k];
    if (k == 0u || tile_keys[k - 1u] != t) {
        ranges[2u * t] = k;
    }
    if (k + 1u == n || tile_keys[k + 1u] != t) {
        ranges[2u * t + 1u] = k + 1u;
    }
}
