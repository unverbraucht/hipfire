#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Wave64+dp4a indexed MoE down + scaled-residual GEMV for gfx906.
// Issue #207 Gap 1 — dp4a port of
// `gemv_hfq4g256_moe_down_indexed_wave64.hip` (scalar FP wave64).
//
// `rot_batch` is [K_TOP × K] — one rotated activation row per top-K
// expert. The wave64 reference reads `x = rot_batch + krank * K` for
// each (row, krank) block. The dp4a port pre-quantizes all K_TOP rows
// into a single Q8_1 scratch via `ensure_q8_1_mmq_x(rot_batch,
// k_top, K)`, producing kblock-major layout [K/128 × K_TOP]. Each
// block reads `Xq[kblock * K_TOP + krank]`.
//
// Math + write semantic preserved: atomicAdd `topk_weights[krank] *
// acc` into `x_residual[row]`. The atomic across K_TOP experts is
// non-deterministic in summation order — same as the wave32 base.
//
// Topology: block=[64,1,1], grid=((M+1)/2, K_TOP, 1).

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64_dp4a(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [k_top]
    const float* __restrict__ topk_weights,              // [k_top]
    const block_q8_1_mmq* __restrict__ Xq,               // kblock-major: [K/128 × K_TOP]
    float*       __restrict__ x_residual,                // [M]
    int M, int K, int K_TOP
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;
    const int krank = blockIdx.y;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = lane * 4;

    const int sub_in_block       = (lane / 4) % 4;
    const int block_within_group = lane / 16;
    const int qs_byte_off        = 8 * (lane % 16);

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 136;

        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const float zp_eff = zp + 8.0f * sc;

        const unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        const unsigned int n0 = (pk >>  0) & 0xFu;
        const unsigned int n1 = (pk >>  4) & 0xFu;
        const unsigned int n2 = (pk >>  8) & 0xFu;
        const unsigned int n3 = (pk >> 12) & 0xFu;
        const unsigned int n4 = (pk >> 16) & 0xFu;
        const unsigned int n5 = (pk >> 20) & 0xFu;
        const unsigned int n6 = (pk >> 24) & 0xFu;
        const unsigned int n7 = (pk >> 28) & 0xFu;
        const int int_a = (int)(((n0 - 8) & 0xFF)
                              | (((n1 - 8) & 0xFF) << 8)
                              | (((n2 - 8) & 0xFF) << 16)
                              | (((n3 - 8) & 0xFF) << 24));
        const int int_b = (int)(((n4 - 8) & 0xFF)
                              | (((n5 - 8) & 0xFF) << 8)
                              | (((n6 - 8) & 0xFF) << 16)
                              | (((n7 - 8) & 0xFF) << 24));

        const int kblock = 2 * g + block_within_group;

        // kblock-major Xq lookup with batch dim = K_TOP, this block's token = krank.
        const block_q8_1_mmq* xb = Xq + (long long)kblock * K_TOP + krank;
        const int* xqs = (const int*)(xb->qs + qs_byte_off);
        const int xq_a = xqs[0];
        const int xq_b = xqs[1];

        int sumi = 0;
        sumi = __builtin_amdgcn_sdot4(int_a, xq_a, sumi, false);
        sumi = __builtin_amdgcn_sdot4(int_b, xq_b, sumi, false);

        const half2 ds = xb->ds4[sub_in_block];
        const float2 dsf = __half22float2(ds);
        const float d_x   = dsf.x;
        const float sum_x = dsf.y;

        acc += sc * d_x * (float)sumi + zp_eff * sum_x * 0.25f;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) {
        const float scale = topk_weights[krank];
        atomicAdd(&x_residual[row], scale * acc);
    }
}
