#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Single-output wave64 dp4a HFP4G32 GEMV for gfx906.
// Peer of fused_qkv/fused_gate_up/fused_qkvza_hfp4g32_wave64_dp4a; the
// fused kernels add 2/3/4-way output routing on top of the same lane
// mapping. This version is the one-output baseline used by lm_head and
// any other single-row GEMV on the decode path.
//
// Replaces the wave32 F32 fallback `gemv_hfp4g32.hip` on gfx906. The
// fallback wastes half the wave (gfx906 is wave64-native) and does
// scalar F32 dequant in the inner loop; this kernel uses wave64 lanes
// fully and runs the inner loop as packed-INT8 dp4a.
//
// Activations are pre-quantized to Q8_1 (block_q8_1_mmq layout, see
// kernels/src/gemm_hfq4g256_residual_mmq.hip). The caller in
// dispatch.rs handles the prequant via `ensure_q8_1_mmq_x`.
//
// Lane mapping inside one 256-K HFP4 group (32 lanes per warp):
//   K range          : [8*lane, 8*lane+7]
//   HFP4 block idx   : lane / 4    (8 blocks per group, 4 lanes/block)
//   nibble byte off  : (lane % 4) * 4
//   Q8_1 block (A/B) : lane / 16   (HFP4 group = 2 Q8_1 blocks)
//   Q8_1 sub-block   : (lane / 4) % 4
//   Q8_1 qs offset   : 8 * (lane % 16)
//
// Math: same trick as the fused kernels — kvalues_hfp4 = 2 × E2M1
// stored as signed INT8, ×0.5 folded into the per-block scale, no zp
// term (signed-symmetric).

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// 2× OCP E2M1 LUT — must live in LDS on gfx906.
//
// PERF CRITICAL: `__device__ static const` and `__constant__` *both*
// emit `global_load_ubyte` per nibble in the dp4a inner loop, because
// the compiler can't promote a per-lane indexed lookup to scalar K$.
// LDS is the only fast option. Cost: 16 B/WG of LDS for the table +
// one ds_read_u8 per nibble (~30× cheaper than the global path).
// Reference: this exactly matches the LUT trick in `gemv_hfp4g32.hip`
// (the scalar fallback), where the table lives in __shared__ memory.

extern "C" __global__ void gemv_hfp4g32_wave64_dp4a(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,   // K/128 Q8_1 blocks
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ int8_t kvalues_hfp4_gemv[16];
    if (tid < 16) {
        static constexpr int8_t init[16] = {
            0,  1,  2,  3,  4,  6,  8,  12,
            0, -1, -2, -3, -4, -6, -8, -12,
        };
        kvalues_hfp4_gemv[tid] = init[tid];
    }
    __syncthreads();

    // 2 rows per wave64 (each 32-lane warp handles one output row).
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const int row_stride_bytes = 16 + groups_per_row * 136;
    const char* row_base = A + (long long)row * row_stride_bytes;

    // Hoist FP16 row scale once per row.
    const unsigned short row_scale_bits = *(const unsigned short*)(row_base);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale = (float)row_scale_h;

    const char* row_ptr = row_base + 16;

    // Lane decomposition (matches the fused kernels).
    const int hfp4_block_in_group = lane >> 2;       // 0..7
    const int nib_off             = (lane & 3) << 2; // 0/4/8/12
    const int sub_in_block        = (lane >> 2) & 3;
    const int block_within_group  = lane >> 4;
    const int qs_byte_off         = 8 * (lane & 15);

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* hp = row_ptr + (long long)g * 136 + hfp4_block_in_group * 17;

        const unsigned char block_e = *(const unsigned char*)hp;
        const float blk = __builtin_bit_cast(float, ((unsigned int)block_e) << 23);
        const float sc  = row_scale * blk * 0.5f;

        const unsigned int pk = *(const unsigned int*)(hp + 1 + nib_off);

        const int v0 = kvalues_hfp4_gemv[(pk >>  0) & 0xFu];
        const int v1 = kvalues_hfp4_gemv[(pk >>  4) & 0xFu];
        const int v2 = kvalues_hfp4_gemv[(pk >>  8) & 0xFu];
        const int v3 = kvalues_hfp4_gemv[(pk >> 12) & 0xFu];
        const int v4 = kvalues_hfp4_gemv[(pk >> 16) & 0xFu];
        const int v5 = kvalues_hfp4_gemv[(pk >> 20) & 0xFu];
        const int v6 = kvalues_hfp4_gemv[(pk >> 24) & 0xFu];
        const int v7 = kvalues_hfp4_gemv[(pk >> 28) & 0xFu];

        const int int_a = (int)((v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) | ((v3 & 0xFF) << 24));
        const int int_b = (int)((v4 & 0xFF) | ((v5 & 0xFF) << 8) | ((v6 & 0xFF) << 16) | ((v7 & 0xFF) << 24));

        const int kblock = 2 * g + block_within_group;
        const block_q8_1_mmq* xb = Xq + kblock;

        const int* xqs = (const int*)(xb->qs + qs_byte_off);
        const int xq_a = xqs[0];
        const int xq_b = xqs[1];

        int sumi = 0;
        sumi = __builtin_amdgcn_sdot4(int_a, xq_a, sumi, false);
        sumi = __builtin_amdgcn_sdot4(int_b, xq_b, sumi, false);

        const half2 ds = xb->ds4[sub_in_block];
        const float2 dsf = __half22float2(ds);
        const float d_x = dsf.x;

        acc += sc * d_x * (float)sumi;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] = acc;
}
