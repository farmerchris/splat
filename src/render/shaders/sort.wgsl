// Stable LSD radix sort pass (8-bit digits) over u32 key/value pairs.
//
// Per pass: `hist` counts digits per block of BLOCK elements into a
// digit-major table (hist[digit * max_blocks + block]); the host runs an
// exclusive scan over that table, which turns it into each (digit, block)'s
// global output offset; `scatter` then writes every element to
// offset + its stable rank among equal digits inside the block.
//
// The element count is read from `counters[slot]` on the GPU, so no host
// round trip is needed between building the keys and sorting them.

struct SortParams {
    shift: u32,
    slot: u32,
    cap: u32,
    max_blocks: u32,
}

@group(0) @binding(0) var<uniform> R: SortParams;
@group(0) @binding(1) var<storage, read> counters: array<u32>;
@group(0) @binding(2) var<storage, read> keys_in: array<u32>;
@group(0) @binding(3) var<storage, read> vals_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> hist: array<u32>;

// CHUNKS (elements per thread) is prepended by the host.
const BLOCK: u32 = 256u * CHUNKS;

var<workgroup> wh: array<atomic<u32>, 256>;
var<workgroup> n_shared: u32;
// One 256-bit thread mask per digit (8 words each).
var<workgroup> masks: array<atomic<u32>, 2048>;
var<workgroup> base: array<u32, 256>;

fn block_id(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.y * nwg.x + wid.x;
}

@compute @workgroup_size(256)
fn hist_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let b = block_id(wid, nwg);
    if (b >= R.max_blocks) {
        return;
    }
    atomicStore(&wh[lid], 0u);
    if (lid == 0u) {
        n_shared = min(counters[R.slot], R.cap);
    }
    let n = workgroupUniformLoad(&n_shared);
    // Each thread walks CHUNKS consecutive keys and run-length encodes equal
    // digits, which keeps shared-atomic contention low when most keys share
    // a digit (e.g. the exponent byte of depths).
    let start = b * BLOCK + lid * CHUNKS;
    var cur = 0xffffffffu;
    var run = 0u;
    for (var k = 0u; k < CHUNKS; k++) {
        let idx = start + k;
        if (idx < n) {
            let d = (keys_in[idx] >> R.shift) & 255u;
            if (d == cur) {
                run++;
            } else {
                if (run > 0u) {
                    atomicAdd(&wh[cur], run);
                }
                cur = d;
                run = 1u;
            }
        }
    }
    if (run > 0u) {
        atomicAdd(&wh[cur], run);
    }
    workgroupBarrier();
    hist[lid * R.max_blocks + b] = atomicLoad(&wh[lid]);
}

@compute @workgroup_size(256)
fn scatter(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let b = block_id(wid, nwg);
    if (b >= R.max_blocks) {
        return;
    }
    if (lid == 0u) {
        n_shared = min(counters[R.slot], R.cap);
    }
    let n = workgroupUniformLoad(&n_shared);
    if (b * BLOCK >= n) {
        return;
    }
    base[lid] = hist[lid * R.max_blocks + b];
    for (var w = 0u; w < 8u; w++) {
        atomicStore(&masks[lid * 8u + w], 0u);
    }
    workgroupBarrier();

    let word = lid >> 5u;
    let bit = 1u << (lid & 31u);
    let chunks = min(CHUNKS, (n - b * BLOCK + 255u) / 256u);
    for (var c = 0u; c < chunks; c++) {
        let idx = b * BLOCK + c * 256u + lid;
        let valid = idx < n;
        var key = 0u;
        var val = 0u;
        var d = 0u;
        if (valid) {
            key = keys_in[idx];
            val = vals_in[idx];
            d = (key >> R.shift) & 255u;
            atomicOr(&masks[d * 8u + word], bit);
        }
        workgroupBarrier();
        var rank = 0u;
        var total = 0u;
        if (valid) {
            for (var w = 0u; w < 8u; w++) {
                let m = atomicLoad(&masks[d * 8u + w]);
                let pc = countOneBits(m);
                if (w < word) {
                    rank += pc;
                } else if (w == word) {
                    rank += countOneBits(m & (bit - 1u));
                }
                total += pc;
            }
            let pos = base[d] + rank;
            keys_out[pos] = key;
            vals_out[pos] = val;
        }
        workgroupBarrier();
        // The first thread of each digit advances its offset and clears its mask.
        if (valid && rank == 0u) {
            base[d] += total;
            for (var w = 0u; w < 8u; w++) {
                atomicStore(&masks[d * 8u + w], 0u);
            }
        }
        workgroupBarrier();
    }
}
