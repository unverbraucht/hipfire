#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ═══ HFP4-G32 × Q8_1 dp4a MMQ kernel body for gfx906 ═══
//
// Sister of gemm_hfq4g256_residual_mmq_gfx906_body.cuh. SAME outer
// topology (block (64,4,1) = 4 wave64s, MMQ_Y=128, mmq_x ∈ {8..64},
// 2-window streaming with 4 syncs/group). The dp4a inner loop is
// BYTE-IDENTICAL — only the X tile loader differs:
//
//   HFQ4: per-256-K group = [sc:fp32 | zp:fp32 | nibbles:128 B] = 136 B
//         decode: int8 = (nibble - 8)                  symmetric INT4
//         scale:  fp32 per 256-K  (= per 4 sub-blocks)
//
//   HFP4: per-256-K group = [(block_e:u8 | nibbles:16 B) × 8]  = 136 B
//         decode: int8 = kvalues_hfp4[nibble]           E2M1 LUT, signed
//         scale:  row_scale_a (fp16) × 2^(block_e - 127)     per 32-K
//                 (per sub-block, 8 sub-blocks per 256-K group)
//
// Both formats are 136 B per 256 K-elements; only the byte layout inside
// the group + the dequant of (nibble, scale) → (int8_qs, fp32_scale)
// differs. The Q8_1 activation tile loader and the dp4a accumulation
// loop are unchanged from HFQ4.
//
// The LUT is the **2× E2M1** lookup table in INT8:
//     kvalues_hfp4 = 2 × {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
//                  = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
// The ×2 factor folds into the per-block scale at load time as `sc *
// 0.5f` so the dp4a output represents the correctly-scaled inner product.
// This matches llama.cpp's MXFP4 pattern (`ggml/src/ggml-cuda/mmq.cuh`)
// where `kvalues_mxfp4 = 2 × E2M1_LUT` and the ×0.5 fold lives at
// `x_df[i] = ggml_cuda_e8m0_to_fp32(bxi->e) * 0.5f`.
//
// LDS budget: HFQ4 stores ONE (sc, zp+8sc) per row per 256-K (8 B
// resident). HFP4 needs 4 per-sub-block scales (= 4 floats = 16 B
// resident) per row per WINDOW (one window = 128 K = 4 sub-blocks),
// reloaded each window. Net delta vs HFQ4 at MMQ_Y=128:
//   HFQ4 x_dm = 128 × 8  = 1,024 B
//   HFP4 x_ds = 128 × 16 = 2,048 B   (+1,024 B)
// At mmq_x=64 (max): 30,720 + 1,024 = 31,744 B ≤ 32 KiB → 2 WGs/CU OK.

#define MMQ_Y 128
#define MMQ_NWARPS 4
#define WAVE_SIZE 64
#define MMQ_TILE_NE_K 32 // 32 K-elements per streaming sub-iter (= 1 HFP4 block)
#define QK8_1 32
#define QI8_1 8

// Per-mmq_x X_STRIDE — same b128 / bank-conflict tradeoff as the HFQ4
// body. Cliff lowered from HFQ4's mmq_x≥64 to mmq_x≥32 because HFP4
// carries a heavier unpack (16-entry signed-INT8 LUT lookup per nibble
// vs HFQ4's `(n - 8)` subtract), so the LDS issue-rate win from b128
// loads activates sooner. This mirrors the HFQ6 PMC sweep
// (feat/gfx906-mq6-phase-a-dp4a, 2026-05-08) which observed
// MemUnitBusy starvation at b32 + mmq_x=32 and lowered the same cliff.
// First-PR safety: keep mmq_x ∈ {8, 16, 24} on b32. PMC sweep TODO post-bringup.
template <int mmq_x>
constexpr int x_stride_for() { return mmq_x >= 32 ? 40 : 33; }

#define Y_STRIDE 36

// LDS layout — KEEP IN SYNC WITH dispatch.rs (HFP4 variant):
//   [x_qs:   i32   × MMQ_Y * x_stride       ]  = 128 * x_stride * 4
//   [x_ds:   float × MMQ_Y * 4              ]  = 128 * 16   =  2,048 B
//   [tile_y: i32   × mmq_x * Y_STRIDE       ]  = mmq_x * 144 B
// At mmq_x=64 (stride 40): 20,480 + 2,048 + 9,216 = 31,744 B per WG.
// At mmq_x=8  (stride 33): 16,896 + 2,048 + 1,152 = 20,096 B per WG.
// Budget: ≤ 32 KiB/WG so 2 WGs/CU fit in 64 KiB cap. ✅

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// 2× OCP E2M1 lattice as signed INT8 — matches llama.cpp's
// kvalues_mxfp4. The ×2 folds into the scale (× 0.5f at load time).
//
// PERF CRITICAL: the LUT must live in LDS, not .rodata or
// __constant__. The compiler can't promote a per-lane indexed lookup
// to scalar K$, so both `__device__ static const` and `__constant__`
// emit `global_load_ubyte` per nibble (~195 globals/kernel observed
// at mmq_x=32 — see ISA inspection 2026-05-11). LDS holds the table
// in 16 B/WG and each lookup is one ds_read_u8. Initialized once per
// WG by lanes 0..15 in `mmq_body_templated` before the kg loop.

// ─── Tile loaders ─────────────────────────────────────────────────────────

// Load 128 K-elements of HFP4 X (LUT-decoded as INT8 + per-sub-block
// fp32 scale) for one window. window ∈ [0, 1]. Each window covers
// 4 sub-blocks (= 4 HFP4 blocks) × 32 K-elements.
//
// HFP4 byte layout per row:
//   [row header: 16 B]
//   [block 0: 1 B block_e + 16 B nibbles] × (K/32) blocks
// row_off = row * (16 + K/256 * 136) = row * (16 + groups_per_row * 136)
//
// Within a 256-K group (g): blocks 0..3 = window 0; blocks 4..7 = window 1.
template <int x_stride>
static __device__ __forceinline__ void load_hfp4_tile_streaming(
    const char* __restrict__ A,
    int* __restrict__ x_qs,
    float* __restrict__ x_ds,           // MMQ_Y × 4 (one float per row per sub-block)
    const int8_t* __restrict__ lut,     // 16 B in LDS — see body header
    int row0, int kg, int window, int M, int groups_per_row, int row_stride_bytes
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255

    // Scale collection: 4 sub-blocks × 128 rows = 512 (row, sub) pairs.
    // Distribute across 256 threads → 2 pairs/thread.
    //
    // Row scale: read FP16 from header byte 0..1 ONCE per row per
    // (kg=0, window=0) and broadcast — but we don't have a "first call"
    // hook here. Instead we re-read it each window; the bandwidth cost
    // is negligible (128 × 2 B = 256 B from HBM per window, fully
    // coalesced from the row-header stride).
    #pragma unroll
    for (int u = tid; u < MMQ_Y * 4; u += 256) {
        const int i = u >> 2;            // row 0..127
        const int sub = u & 3;           // sub-block 0..3 within window
        const int row = (row0 + i < M) ? (row0 + i) : (M - 1);

        const char* row_base = A + (long long)row * row_stride_bytes;
        const unsigned short row_scale_bits = *(const unsigned short*)(row_base);
        const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
        const float row_scale = (float)row_scale_h;

        // Block index within this row: kg * 8 + window * 4 + sub.
        const int block_idx = kg * 8 + window * 4 + sub;
        const char* gp = row_base + 16 + (long long)block_idx * 17;
        const unsigned char block_e = *(const unsigned char*)gp;

        // UE8M0 dequant via FP32 bit construction (avoid host-only
        // ldexpf on ROCm ≥ 7.2.2). 2^(e-127) = float with biased
        // exponent = e, mantissa = 0.
        const float blk = __builtin_bit_cast(float, ((unsigned int)block_e) << 23);

        // Fold the ×2 LUT factor (kvalues_hfp4 = 2 × E2M1) and the
        // row scale into a single per-sub-block fp32:
        //     scale_eff = row_scale * 2^(e-127) * 0.5
        x_ds[i * 4 + sub] = row_scale * blk * 0.5f;
    }

    // X-qs window load: 128 rows × 16 uints/row = 2048 uint reads,
    // distributed across 256 threads → 8 uints/thread.
    //
    // Each uint encodes 8 nibbles. We LUT-decode each nibble to an
    // INT8 value via kvalues_hfp4_mmq and pack 4 INT8s per int. The
    // result lands in x_qs at the same stride/layout HFQ4 uses, so
    // the dp4a compute path is byte-identical.
    //
    // Source byte offset per (row, chunk):
    //   gp = row_base + 16 + window_first_block * 17
    //      = row_base + 16 + (kg * 8 + window * 4) * 17
    //   Within the 4 blocks of this window, byte = 1 (block_e) + offset
    //   into 16 B nibbles. The 16 chunks per row map to:
    //     chunk c ∈ [0, 16) → block (c / 4), nibble byte (c % 4) * 4
    //   so each chunk reads 4 bytes of nibbles from one block,
    //   producing 8 nibbles → 2 ints in x_qs.
    #pragma unroll
    for (int loop = 0; loop < 8; ++loop) {
        const int task_id = tid * 8 + loop; // 0..2047
        const int i = task_id / 16;         // row, 0..127
        const int chunk = task_id % 16;     // which uint in row, 0..15

        const int row = (row0 + i < M) ? (row0 + i) : (M - 1);
        const int block_in_window = chunk >> 2; // 0..3
        const int byte_in_block = (chunk & 3) << 2; // 0/4/8/12

        const int block_idx = kg * 8 + window * 4 + block_in_window;
        const char* gp = A + (long long)row * row_stride_bytes
                       + 16
                       + (long long)block_idx * 17
                       + 1                       // skip block_e byte
                       + byte_in_block;
        const unsigned int qs0 = *(const unsigned int*)gp;

        // LUT-decode 8 nibbles → 8 INT8 codes → 2 packed ints.
        const int v0 = lut[(qs0 >>  0) & 0xFu];
        const int v1 = lut[(qs0 >>  4) & 0xFu];
        const int v2 = lut[(qs0 >>  8) & 0xFu];
        const int v3 = lut[(qs0 >> 12) & 0xFu];
        const int v4 = lut[(qs0 >> 16) & 0xFu];
        const int v5 = lut[(qs0 >> 20) & 0xFu];
        const int v6 = lut[(qs0 >> 24) & 0xFu];
        const int v7 = lut[(qs0 >> 28) & 0xFu];

        const int int_a = (int)((v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) | ((v3 & 0xFF) << 24));
        const int int_b = (int)((v4 & 0xFF) | ((v5 & 0xFF) << 8) | ((v6 & 0xFF) << 16) | ((v7 & 0xFF) << 24));

        x_qs[i * x_stride + 2 * chunk + 0] = int_a;
        x_qs[i * x_stride + 2 * chunk + 1] = int_b;
    }
}

// Q8_1 activation tile loader — VERBATIM from HFQ4 body. Activations
// are format-agnostic (Q8_1 doesn't care what the weight format is).
template <int mmq_x>
static __device__ __forceinline__ void load_q8_1_tile_coalesced(
    const block_q8_1_mmq* __restrict__ Xq,
    int* __restrict__ tile_y,
    int col0, int kb, int N
) {
    const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x; // 0..255
    const int total_ints = mmq_x * Y_STRIDE;

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
//
// HFP4 diff from HFQ4: per-row scale lives in x_ds[row * 4 + sub] (one
// fp32 per sub-block) rather than x_dm[row] (one fp32 pair per 256-K).
// Zero point is gone — HFP4 is signed-symmetric (kvalues_hfp4 includes
// negatives), so the (zp + 8*sc) bias term that HFQ4 carried for its
// (n - 8) unpack is no longer needed.
//
// dp4a inner loop and y-side scaling (d_x, sum_x) are UNCHANGED.
template <int mmq_x>
static __device__ __forceinline__ void vec_dot_dp4a_streaming(
    const int* __restrict__ x_qs,
    const float* __restrict__ x_ds,
    const int* __restrict__ tile_y,
    float* __restrict__ sum,
    int sub_block
) {
    constexpr int vdr = 8; // 8 ints cover 32 K-elements
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
            // b128 cliff at mmq_x≥32 (lower than HFQ4's ≥64 — see
            // x_stride_for<>() comment for the rationale).
            if constexpr (mmq_x >= 32) {
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
                #pragma unroll
                for (int v = 0; v < vdr; ++v) {
                    const int x_int = x_qs[i * x_stride + kx_start + v];
                    const int y_int = tile_y[j * Y_STRIDE + ky_start + v];
                    sumi = __builtin_amdgcn_sdot4(x_int, y_int, sumi, false);
                }
            }

            const float scale_w = x_ds[i * 4 + sub_block];
            const float2 dsf = __half22float2(ds_j);
            const float d_x   = dsf.x;
            // HFP4 is signed-symmetric — no zp term. The Q8_1 sum_x
            // factor (dsf.y) drops out entirely because the implicit
            // zero from a signed-symmetric weight ⟨0, x⟩ = 0.

            const int idx = (j0 / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE) + (i0 / WAVE_SIZE);
            sum[idx] += scale_w * d_x * (float)sumi;
        }
    }
}

// ─── Write-back ─── (verbatim from HFQ4 body)

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
    // HFP4 row stride includes the 16-B header and (K/32) × 17 B
    // blocks. K/32 = 8 * groups_per_row → blocks-bytes = 136 *
    // groups_per_row. Total = 16 + 136 * groups_per_row.
    const int row_stride_bytes = 16 + groups_per_row * 136;
    const int add = (add_mode == -1) ? add_param : add_mode;

    constexpr int x_stride = x_stride_for<mmq_x>();
    extern __shared__ int smem[];
    // LUT: 16 B at the start of LDS (4 ints). x_qs / x_ds / tile_y
    // follow. Adding 4 ints (16 B) to the LDS budget is negligible
    // against the 32 KiB cap (still leaves 2 WGs/CU at mmq_x=64).
    int8_t* lut = (int8_t*)smem;
    int*    x_qs   = smem + 4;
    float*  x_ds   = (float*)(x_qs + MMQ_Y * x_stride);
    int*    tile_y = (int*)(x_ds + MMQ_Y * 4);

    // Init LDS LUT (one-time per WG; reused across all kg windows).
    {
        const int tid = threadIdx.y * WAVE_SIZE + threadIdx.x;
        if (tid < 16) {
            static constexpr int8_t init[16] = {
                0,  1,  2,  3,  4,  6,  8,  12,
                0, -1, -2, -3, -4, -6, -8, -12,
            };
            lut[tid] = init[tid];
        }
        __syncthreads();
    }

    float sum[(mmq_x / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE)] = {0.0f};

    for (int kg = 0; kg < groups_per_row; ++kg) {
        for (int window = 0; window < 2; ++window) {
            load_q8_1_tile_coalesced<mmq_x>(Xq, tile_y, col0, 2*kg + window, N);
            load_hfp4_tile_streaming<x_stride>(
                A, x_qs, x_ds, lut, row0, kg, window, M, groups_per_row, row_stride_bytes
            );
            __syncthreads();
            #pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                vec_dot_dp4a_streaming<mmq_x>(x_qs, x_ds, tile_y, sum, sub);
            }
            __syncthreads();
        }
    }

    write_back_residual_templated<mmq_x, need_check>(Y, sum, row0, col0, M, N, add);
}
