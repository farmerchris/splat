// Deterministic stream compaction of the visible gaussians (see project.wgsl).
// `offsets` is the exclusive scan of the visibility flags, so visible
// gaussians land in scene-index order and the stable depth sort then breaks
// depth-key ties by original index.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> counters: array<atomic<u32>>;
// Per gaussian: depth key or NO_KEY (written by `project`).
@group(0) @binding(2) var<storage, read> keys_in: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals_out: array<u32>;

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
    let k = keys_in[i];
    let o = offsets[i];
    let visible = k != NO_KEY;
    if (visible) {
        keys_out[o] = k;
        vals_out[o] = i;
    }
    if (i + 1u == P.n) {
        atomicStore(&counters[0], o + select(0u, 1u, visible));
    }
}
