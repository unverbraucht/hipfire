#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ═══ HFQ6-G256 × Q8_1 dp4a MMQ kernel body for gfx906 ═══
//
// Port of gemm_hfq4g256_residual_mmq_gfx906_body.cuh to HFQ6 weights.
// Phase B.2 session 1 (plan: docs/plans/gfx906-mq6-mmq-port-phase-b2.md).
//
// Differences from HFQ4 body:
//   1. HFQ6 group size = 200 B (vs HFQ4's 136 B). Per-window weight bytes:
//      HFQ4 = 64, HFQ6 = 96.
//   2. HFQ6 stores unsigned q ∈ [0, 63] directly as int8 (fits since
//      63 < 127). HFQ4 stores nibbles as (n - 8) for signed int8.
//      → x_dm carries the shift compensation for HFQ4 only.
//
// LANDMINE 1 (plan §3.1.1):
//   HFQ4: x_dm[i] = make_float2(sc, zp + 8.0f * sc)  [zp_eff compensates -8 shift]
//   HFQ6: x_dm[i] = make_float2(sc, zp)              [no shift, zp passes through]
//
// LANDMINE 2 (plan §3.1.2):
//   The 0.25f factor in gemm_hfq6g256_residual_wave64_dp4a.hip:116 is
//   per-lane share (each lane covers 1/4 of a sub-block). The MMQ body
//   covers a FULL sub-block per thread (vdr=8 ints over 32 K-elements),
//   so the accumulation formula is `zp_eff * sum_x` with NO factor.
//
// ═══ KERNEL INVARIANTS — caller MUST satisfy ═══
//
//   1. K must be a multiple of 256 (= group_size). Body assumes
//      `groups_per_row = K / 256`; not asserted at runtime (caller's job).
//
//   2. M alignment depends on the kernel variant:
//        `_x{N}`        (need_check=true)  → ANY M ≥ 1 is safe.
//                                            OOB rows clamp to M-1 weights
//                                            in shared memory and accumulate
//                                            spurious sums, but write_back
//                                            skips them so observable Y is
//                                            correct.
//        `_full_add_x{N}` (need_check=false, add=1) → CALLER MUST GUARANTEE
//                                                     `M % 128 == 0`.
//        `_full_set_x{N}` (need_check=false, add=0) → CALLER MUST GUARANTEE
//                                                     `M % 128 == 0`.
//      The `_full_*` variants drop the `(row >= M ? skip)` check from
//      writeback for ~5 % perf — they will write GARBAGE to OOB rows
//      otherwise. The dispatcher at `dispatch.rs:7468` enforces
//      `is_full = m % 128 == 0 && batch_size % mmq_x == 0` before naming
//      a `_full_*` kernel; do not bypass that gate.
//
//   3. N (batch_size) alignment depends on the kernel variant:
//        `_x{N}`        → ANY N ≥ 1 is safe (need_check=true skips OOB cols).
//        `_full_*_x{N}` → CALLER MUST GUARANTEE `batch_size % mmq_x == 0`.
//
//   4. Body's row-clamp at lines `(row0 + i < M) ? (row0 + i) : (M - 1)`
//      means non-`_full_*` variants over-fetch row M-1 weights into the
//      LDS slots that correspond to OOB rows. This costs ~0 % perf (cache-
//      hot reads) but breaks if M==0 (would index row -1 underflowing
//      to 2^64-1). Caller must also guarantee M ≥ 1.
//
// Topology (wave-native gfx906) — UNCHANGED from HFQ4:
//   block dim (64, 4, 1) = 256 threads = 4 wave64s
//   mmq_y = 128
//   mmq_x = templated {8, 16, 24, 32, 40, 48, 56, 64}
//
// Option C+pad (Window Streaming) preserved:
//   HFQ6 group = 256 K-elements = 2 Q8_1 blocks = 2 windows.
//   Each window covers 128 K-elements (= one Q8_1 block, = 4 sub-blocks
//   of 32 K-elements each). Window weight bytes: 96 (HFQ6) vs 64 (HFQ4).
//
// Per-window pipeline (= 4 syncs/group total, identical to HFQ4):
//   1. Load 128-K of x_qs (32 data ints + pad per row) + Q8_1 block of tile_y
//   2. __syncthreads
//   3. 4 sub-blocks computed back-to-back (no syncs between)
//   4. __syncthreads

#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32 // 32 K-elements per streaming sub-iter
#define QK8_1 32
#define QI8_1 8

// Reuse HFQ4's stride choices. HFQ4 PMC validated per
// docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md §5.
// HFQ6 PMC sweep (Phase B.2 follow-up, 2026-05-08) found b32 path
// underperforms at mmq_x=32 (MemUnitBusy 13.8% vs 28.8% at mmq_x=16)
// — LDS issue-rate starvation. Lowering the b128 cliff to mmq_x>=32
// activates b128 (= 2 ds_read_b128 per inner ALU iter vs 8 ds_read_b32)
// for mmq_x ∈ {32, 40, 48, 56, 64}. mmq_x=16/24 stay on b32 because
// the unpack overhead at small mmq_x exceeds the issue-rate win
// (HFQ4 cliff at 64 was empirically tuned for that family; HFQ6's
// heavier unpack benefits from b128 sooner).
template <int mmq_x>
constexpr int x_stride_for() { return mmq_x >= 32 ? 40 : 33; }

#define Y_STRIDE 36

// LDS layout invariant — KEEP IN SYNC WITH dispatch.rs:
//   [x_qs:   i32    × MMQ_Y * x_stride       ]
//   [x_dm:   float2 × MMQ_Y                  ]
//   [tile_y: i32    × mmq_x * Y_STRIDE       ]
// Same shape as HFQ4 (decoded ints in LDS — only the source byte count
// differs).
struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// ─── Tile loaders ─────────────────────────────────────────────────────────

// Load 128 K-elements of X (HFQ6, unpacked) for one window.
// window ∈ [0, 1]. Each window loads 96 bytes of weight per row
// = 16 (int_a, int_b) pairs = 32 dst ints per row in x_qs.
template <int x_stride>
static __device__ __forceinline__ void load_hfq6_tile_streaming(
    const char* __restrict__ A,
    int* __restrict__ x_qs,
    float2* __restrict__ x_dm,
    int row0, int kg, int window, int M, int groups_per_row
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255

    // Header (sc, zp) once per group — only window 0 loads it.
    if (window == 0) {
        if (tid < 128) {
            const int i = tid;
            const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kg) * 200;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            // ── LANDMINE 1 GUARD ──
            // HFQ6 stores q unsigned in [0, 63], packed directly as int8 —
            // no -8 shift like HFQ4. So zp passes through; do NOT add 8*sc.
            x_dm[i] = make_float2(sc, zp);
        }
    }

    // X-qs window load: 128 rows × 16 pairs/row = 2048 pairs/window.
    // Distributed across 256 threads → 8 pairs/thread.
    // Pair p in row r occupies 6 bytes at offset (8 + window*96 + 6*p)
    // within the group, decoding to (int_a, int_b) — 2 dst ints written
    // at x_qs[r * x_stride + 2*p + {0,1}].
    //
    // Per-tid: 8 pairs covering half a row (16 pairs per row, 2 tids per row).
    //   tid t → row = t/2, pair_in_row ∈ [8*(t%2), 8*(t%2) + 8)
    //
    // This task-id mapping mirrors HFQ4's structure (8 tasks/tid, half a
    // row per tid) so warp-level access patterns are similar. Bytes are
    // loaded individually via uint8_t reads — same approach as the
    // wave64_dp4a HFQ6 kernel. A vectorized load (uint+ushort, or 3-uint
    // gather across pair boundaries) is a S2 optimization candidate.
    #pragma unroll
    for (int loop = 0; loop < 8; ++loop) {
        const int task_id = tid * 8 + loop; // 0..2047
        const int i = task_id / 16;         // row, 0..127
        const int p = task_id % 16;         // pair index in row, 0..15

        const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kg) * 200;

        // Offset: 8 (header) + window * 96 + 6 * p.
        const unsigned char* dp = (const unsigned char*)(gp + 8 + window * 96 + p * 6);
        const unsigned char b0 = dp[0];
        const unsigned char b1 = dp[1];
        const unsigned char b2 = dp[2];
        const unsigned char b3 = dp[3];
        const unsigned char b4 = dp[4];
        const unsigned char b5 = dp[5];

        // 6-bit unpack: 8 weights from 6 bytes.
        // Identical algebra to gemm_hfq6g256_residual_wave64_dp4a.hip:78-92.
        const unsigned int q0 = b0 & 63;
        const unsigned int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        const unsigned int q2 = (b1 >> 4) | ((b2 & 3)  << 4);
        const unsigned int q3 = b2 >> 2;
        const unsigned int q4 = b3 & 63;
        const unsigned int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        const unsigned int q6 = (b4 >> 4) | ((b5 & 3)  << 4);
        const unsigned int q7 = b5 >> 2;

        // Pack 4 weights per int (each ∈ [0, 63] fits int8 unsigned).
        // No -8 shift here — HFQ6 stores q directly as int8.
        const int int_a = (int)((q0 & 0xFF)
                              | ((q1 & 0xFF) << 8)
                              | ((q2 & 0xFF) << 16)
                              | ((q3 & 0xFF) << 24));
        const int int_b = (int)((q4 & 0xFF)
                              | ((q5 & 0xFF) << 8)
                              | ((q6 & 0xFF) << 16)
                              | ((q7 & 0xFF) << 24));

        x_qs[i * x_stride + 2 * p + 0] = int_a;
        x_qs[i * x_stride + 2 * p + 1] = int_b;
    }
}

// Load one Q8_1 block (128 elements) for all mmq_x columns.
// REUSED FROM HFQ4 BODY UNCHANGED — Q8_1 activation layout is dtype-agnostic.
template <int mmq_x>
static __device__ __forceinline__ void load_q8_1_tile_coalesced(
    const block_q8_1_mmq* __restrict__ Xq,
    int* __restrict__ tile_y,
    int col0, int kb, int N
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255
    const int total_ints = mmq_x * Y_STRIDE;

    // Each thread loads total_ints / 256 ints.
    #pragma unroll
    for (int u = tid; u < total_ints; u += 256) {
        const int j = u / Y_STRIDE;
        const int slot = u % Y_STRIDE;
        const bool valid = (col0 + j) < N;
        const int col = valid ? (col0 + j) : (N - 1);

        const int* src = (const int*)(Xq + (long long)kb * N + col);
        tile_y[u] = valid ? src[slot] : 0;
    }
}

// ─── dp4a compute ─────────────────────────────────────────────────────────

// Compute one 32-K sub-block's contribution into `sum`.
// REUSED FROM HFQ4 BODY UNCHANGED — operates on decoded int8 in LDS,
// which has identical layout for HFQ4 and HFQ6 after the unpack.
//
// LANDMINE 2 GUARD:
//   The accumulation formula at the end of the inner loop is
//   `sum[idx] += scale_w * d_x * sumi + zp_eff * sum_x` (NO 0.25f).
//   The 0.25f factor in gemm_hfq6g256_residual_wave64_dp4a.hip:116 is
//   the per-lane share for that kernel's lane-mapping — each lane there
//   covers 8 of 32 sub-block elements. The MMQ body covers the FULL
//   sub-block per thread via the inner vdr=8 loop, so no factor is
//   needed.
template <int mmq_x>
static __device__ __forceinline__ void vec_dot_dp4a_streaming(
    const int* __restrict__ x_qs,
    const float2* __restrict__ x_dm,
    const int* __restrict__ tile_y,
    float* __restrict__ sum,
    int sub_block
) {
    constexpr int vdr = 8; // 8 ints cover 32 K-elements.
    constexpr int x_stride = x_stride_for<mmq_x>();
    const int kx_start = sub_block * 8;
    const int ky_start = 4 + sub_block * 8;

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;

        const half2* y_ds_col = (const half2*)(tile_y + j * Y_STRIDE);
        const half2 ds_j = y_ds_col[sub_block];

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {
            const int i = i0 + threadIdx.x;

            int sumi = 0;
            // HFQ6 b128 cliff at mmq_x>=32 (lowered from HFQ4's >=64 per
            // PMC sweep — see x_stride_for<>() comment above).
            if constexpr (mmq_x >= 32) {
                // b128 path: issue 8 ints per operand as 2× int4 (b128) reads.
                const int4 x_v0 = *(const int4*)&x_qs[i * x_stride + kx_start + 0];
                const int4 x_v1 = *(const int4*)&x_qs[i * x_stride + kx_start + 4];
                const int4 y_v0 = *(const int4*)&tile_y[j * Y_STRIDE + ky_start + 0];
                const int4 y_v1 = *(const int4*)&tile_y[j * Y_STRIDE + ky_start + 4];
                sumi = __builtin_amdgcn_sdot4(x_v0.x, y_v0.x, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v0.y, y_v0.y, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v0.z, y_v0.z, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v0.w, y_v0.w, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v1.x, y_v1.x, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v1.y, y_v1.y, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v1.z, y_v1.z, sumi, false);
                sumi = __builtin_amdgcn_sdot4(x_v1.w, y_v1.w, sumi, false);
            } else {
                // Scalar (b32) path for small mmq_x.
                #pragma unroll
                for (int v = 0; v < vdr; ++v) {
                    const int x_int = x_qs[i * x_stride + kx_start + v];
                    const int y_int = tile_y[j * Y_STRIDE + ky_start + v];
                    sumi = __builtin_amdgcn_sdot4(x_int, y_int, sumi, false);
                }
            }

            const float2 dm_i = x_dm[i];
            const float2 dsf = __half22float2(ds_j);
            const float scale_w = dm_i.x;
            const float zp_eff  = dm_i.y;
            const float d_x     = dsf.x;
            const float sum_x   = dsf.y;

            const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
            // ── LANDMINE 2 GUARD ──
            // Full sub-block per thread; no 0.25f correction.
            sum[idx] += scale_w * d_x * (float)sumi + zp_eff * sum_x;
        }
    }
}

// ─── Write-back ───────────────────────────────────────────────────────────
// REUSED FROM HFQ4 BODY UNCHANGED — operates on accumulated float sums.

template <int mmq_x, bool need_check>
static __device__ __forceinline__ void write_back_residual_templated(
    float* __restrict__ Y,
    const float* __restrict__ sum,
    int row0, int col0, int M, int N, int add
) {
    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col = col0 + j;
        if (need_check && col >= N) continue;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WAVE_SIZE) {
            const int i = i0 + threadIdx.x;
            const int row = row0 + i;
            if (need_check && row >= M) continue;

            const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
            const long long out_idx = (long long)col * M + row;
            if (add) {
                Y[out_idx] += sum[idx];
            } else {
                Y[out_idx]  = sum[idx];
            }
        }
    }
}

// ─── Shared kernel body ───────────────────────────────────────────────────

template <int mmq_x, bool need_check, int add_mode>
static __device__ __forceinline__ void mmq_body_templated(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add_param
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * mmq_x;
    if (need_check && (row0 >= M || col0 >= N)) return;

    const int groups_per_row = K / 256;
    const int add = (add_mode == -1) ? add_param : add_mode;

    // LDS layout (must match dispatch.rs shared_mem calc):
    //   x_qs   : MMQ_Y * x_stride ints
    //   x_dm   : MMQ_Y float2
    //   tile_y : mmq_x * Y_STRIDE ints
    constexpr int x_stride = x_stride_for<mmq_x>();
    extern __shared__ int smem[];
    int*    x_qs   = smem;
    float2* x_dm   = (float2*)(x_qs + MMQ_Y * x_stride);
    int*    tile_y = (int*)(x_dm + MMQ_Y);

    float sum[(mmq_x / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE)] = {0.0f};

    for (int kg = 0; kg < groups_per_row; ++kg) {
        // Window Streaming: 2 windows × 4 sub-blocks each, 4 syncs/group.
        for (int window = 0; window < 2; ++window) {
            load_q8_1_tile_coalesced<mmq_x>(Xq, tile_y, col0, 2*kg + window, N);
            load_hfq6_tile_streaming<x_stride>(A, x_qs, x_dm, row0, kg, window, M, groups_per_row);
            __syncthreads();
            #pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                vec_dot_dp4a_streaming<mmq_x>(x_qs, x_dm, tile_y, sum, sub);
            }
            __syncthreads();
        }
    }

    write_back_residual_templated<mmq_x, need_check>(Y, sum, row0, col0, M, N, add);
}
