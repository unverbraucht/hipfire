#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Wave64+dp4a indexed MoE gate_up GEMV for gfx906 (Vega 20 / MI50/60).
// Issue #207 Gap 1: the existing wave64 indexed kernel
// (`gemv_hfq4g256_moe_gate_up_indexed_wave64.hip`) uses scalar FP inner
// loops. This kernel ports the dp4a inner loop from the non-MoE HFQ4
// dp4a family (gemm_hfq4g256_wave64_dp4a.hip / fused_qkv_*_dp4a.hip),
// using pre-quantized Q8_1 activations.
//
// Activation pre-quantize via `ensure_q8_1_mmq_x` — same Q8_1 scratch
// the dense MMQ / dp4a paths use. The MoE preamble already pays the
// Q8_1 quantize cost once for fused_qkvza on the FA-MoE path, so
// in principle the scratch can be reused; in this initial port we
// quantize fresh for clarity. Future optimization: share the Q8_1
// scratch across MoE preamble + indexed kernels.
//
// Math identity (signed HFQ4) — see gemm_hfq4g256_wave64_dp4a.hip:
//   acc += sc * d_x * sumi_dp4a + zp_eff * sum_x * 0.25f
// where zp_eff = zp + 8 * sc folds the unsigned-bias correction
// (HFQ4 nibbles ∈ [0,15] shifted to signed [−8,7] via (n−8) & 0xFF).
//
// Topology mirrors the wave64 indexed kernel: block=[64,1,1], grid =
// ((M+1)/2, K_TOP, 1), 2 rows per block via warp_id.
//
// Output write semantic preserved: first MI rows → y_gate[krank * mi
// + row]; second MI rows → y_up[krank * mi + (row - mi)]. SET write
// (not residual; the wave64 reference uses `=` not `+=` since this is
// the first writer of those outputs).

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void gemv_hfq4g256_moe_gate_up_k8_indexed_wave64_dp4a(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [k_top]
    const block_q8_1_mmq* __restrict__ Xq,               // kblock-major: [K/128]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;       // 0 or 1
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

    // Per-lane mapping inside one HFQ4-G256 group (256 K-elements):
    //   sub-block index s(l) = l/4 (0..7)
    //   Q8_1 block index within group b(l) = l/16 (0=A, 1=B)
    //   ds4 slot within block i(l) = (l/4) % 4
    //   qs byte offset within block: 8 * (lane % 16)
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

        // Single-token (batch_size=1) Xq lookup: kblock-major reduces to Xq[kblock].
        const block_q8_1_mmq* xb = Xq + (long long)kblock;
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

    // Per-warp 32-lane reduction. See gemv_hfq4g256_residual_wave64.hip
    // header for the wave64 cross-lane-read caveat (DO NOT add width=32).
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
