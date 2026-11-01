// Multi-level exclusive prefix sum over u32. Each workgroup scans a block of
// 1024 elements and writes the block total to `sums`; the host scans `sums`
// recursively and then adds the scanned block offsets back with `add`.

struct ScanParams {
    len: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> S: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;

const PER_THREAD: u32 = 4u;
const BLOCK: u32 = 1024u;

var<workgroup> tmp: array<u32, 256>;

fn block_id(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.y * nwg.x + wid.x;
}

@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let b = block_id(wid, nwg);
    if (b * BLOCK >= max(S.len, 1u)) {
        return;
    }
    let base = b * BLOCK + lid * PER_THREAD;
    var v: array<u32, 4>;
    var total = 0u;
    for (var k = 0u; k < PER_THREAD; k++) {
        let idx = base + k;
        var x = 0u;
        if (idx < S.len) {
            x = data[idx];
        }
        v[k] = total;
        total += x;
    }
    // Hillis-Steele inclusive scan of the per-thread totals.
    tmp[lid] = total;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off <<= 1u) {
        var add = 0u;
        if (lid >= off) {
            add = tmp[lid - off];
        }
        workgroupBarrier();
        tmp[lid] += add;
        workgroupBarrier();
    }
    let excl = tmp[lid] - total;
    for (var k = 0u; k < PER_THREAD; k++) {
        let idx = base + k;
        if (idx < S.len) {
            data[idx] = v[k] + excl;
        }
    }
    if (lid == 255u) {
        sums[b] = tmp[255];
    }
}

@compute @workgroup_size(256)
fn add(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let b = block_id(wid, nwg);
    let s = sums[b];
    let base = b * BLOCK + lid * PER_THREAD;
    for (var k = 0u; k < PER_THREAD; k++) {
        let idx = base + k;
        if (idx < S.len) {
            data[idx] += s;
        }
    }
}
