#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// dp4a-port of fused_qkvza_hfq4g256_wave64, retargeted to HFP4G32.
// 4-output routing (Q/K/V/Z/Alpha) for the DeltaNet LinearAttention
// preamble. Sister of fused_qkv/fused_gate_up HFP4 wave64 dp4a kernels;
// see fused_gate_up_hfp4g32_wave64_dp4a.hip for the lane-mapping
// derivation.

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// PERF CRITICAL on gfx906: `__constant__` not `__device__ static
// const`. See gemv_hfp4g32_wave64_dp4a.hip header for the diagnostic.
__constant__ int8_t kvalues_hfp4_qkvza[16] = {
    0,  1,  2,  3,  4,  6,  8,  12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

extern "C" __global__ void fused_qkvza_hfp4g32_wave64_dp4a(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const block_q8_1_mmq* __restrict__ Xq,
    float*       __restrict__ y_qkv,
    float*       __restrict__ y_z,
    float*       __restrict__ y_beta,
    float*       __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int gid = blockIdx.x * 2 + warp_id;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv; y = y_qkv; row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z; y = y_z; row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta; y = y_beta; row = gid - qkv_m - z_m;
    } else {
        A = A_alpha; y = y_alpha; row = gid - qkv_m - z_m - beta_m;
    }

    const int groups_per_row = K / 256;
    const int row_stride_bytes = 16 + groups_per_row * 136;
    const char* row_base = A + (long long)row * row_stride_bytes;

    const unsigned short row_scale_bits = *(const unsigned short*)(row_base);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale = (float)row_scale_h;

    const char* row_ptr = row_base + 16;

    const int hfp4_block_in_group = lane >> 2;
    const int nib_off             = (lane & 3) << 2;
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

        const int v0 = kvalues_hfp4_qkvza[(pk >>  0) & 0xFu];
        const int v1 = kvalues_hfp4_qkvza[(pk >>  4) & 0xFu];
        const int v2 = kvalues_hfp4_qkvza[(pk >>  8) & 0xFu];
        const int v3 = kvalues_hfp4_qkvza[(pk >> 12) & 0xFu];
        const int v4 = kvalues_hfp4_qkvza[(pk >> 16) & 0xFu];
        const int v5 = kvalues_hfp4_qkvza[(pk >> 20) & 0xFu];
        const int v6 = kvalues_hfp4_qkvza[(pk >> 24) & 0xFu];
        const int v7 = kvalues_hfp4_qkvza[(pk >> 28) & 0xFu];

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
