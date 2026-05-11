#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// dp4a-port of HFQ4G256's fused_gate_up wave64 kernel, retargeted to
// HFP4G32 weights. Activations are pre-quantized to Q8_1 (block_q8_1_mmq
// layout); X-side decode is byte-identical to the HFQ4 variant.
//
// HFP4 byte layout per row:
//   [16-B header: f16 row_scale_a | f16 row_scale_b | ...]
//   [block 0..K/32-1: 1 B block_e (UE8M0) | 16 B nibbles] × K/32
// Row stride: 16 + (K/32) * 17 = 16 + groups_per_row * 136 (since
// each 256-K "group" contains 8 HFP4 blocks → 8 × 17 = 136 B).
//
// Lane mapping inside one 256-K HFP4 group (32 lanes per warp):
//   K range          : [8*lane, 8*lane+7]
//   HFP4 block idx   : lane / 4    (8 blocks per group, 4 lanes/block)
//   nibble byte off  : (lane % 4) * 4
//   Q8_1 block (A/B) : lane / 16   (HFP4 group = 2 Q8_1 blocks)
//   Q8_1 sub-block   : (lane / 4) % 4
//   Q8_1 qs offset   : 8 * (lane % 16)
//
// Per-block weight value = row_scale * 2^(block_e - 127) * E2M1_LUT[nibble].
// The kvalues_hfp4 LUT is the 2× E2M1 lattice as INT8 codes, so the
// dp4a output already includes the ×2; we fold ×0.5 into the per-block
// scale at unpack time. Same trick as llama.cpp's MXFP4 path.
//
// HFP4 has no zp term (signed-symmetric), so the ⟨zp_eff, sum_x⟩
// correction from the HFQ4 kernel drops entirely.

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// 2× OCP E2M1 as signed INT8. ×0.5 fold lives in the per-block scale
// at unpack time so dp4a output is correctly scaled.
//
// PERF CRITICAL on gfx906: `__device__ static const` emits
// global_load_ubyte per nibble; `__constant__` caches in scalar K$.
// See gemv_hfp4g32_wave64_dp4a.hip header for the diagnostic.
__constant__ int8_t kvalues_hfp4_wave64[16] = {
    0,  1,  2,  3,  4,  6,  8,  12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

extern "C" __global__ void fused_gate_up_hfp4g32_wave64_dp4a(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const block_q8_1_mmq* __restrict__ Xq,   // K/128 blocks
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int gid = blockIdx.x * 2 + warp_id;
    const int total_m = gate_m + up_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int groups_per_row = K / 256;
    const int row_stride_bytes = 16 + groups_per_row * 136;
    const char* row_base = A + (long long)local_row * row_stride_bytes;

    // Hoist row scale: FP16 in header bytes 0..1.
    const unsigned short row_scale_bits = *(const unsigned short*)(row_base);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale = (float)row_scale_h;

    const char* row_ptr = row_base + 16;             // first HFP4 block

    // Lane → (HFP4 block, nibble byte, Q8_1 sub-block) decomposition.
    const int hfp4_block_in_group = lane >> 2;       // 0..7
    const int nib_off             = (lane & 3) << 2; // 0/4/8/12

    const int sub_in_block        = (lane >> 2) & 3; // ds4[i] index 0..3
    const int block_within_group  = lane >> 4;       // Q8_1 block A or B
    const int qs_byte_off         = 8 * (lane & 15); // qs[] byte offset

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        // This lane's HFP4 block within the 256-K group.
        const char* hp = row_ptr + (long long)g * 136 + hfp4_block_in_group * 17;

        const unsigned char block_e = *(const unsigned char*)hp;
        // UE8M0 dequant via FP32 bit construction: 2^(e-127) = float
        // with biased exponent = e, mantissa = 0. Fold row_scale and
        // the ×0.5 LUT factor (kvalues_hfp4 = 2 × E2M1).
        const float blk = __builtin_bit_cast(float, ((unsigned int)block_e) << 23);
        const float sc  = row_scale * blk * 0.5f;

        // Load this lane's 8 nibbles (1 uint = 4 bytes).
        const unsigned int pk = *(const unsigned int*)(hp + 1 + nib_off);

        // LUT-decode 8 nibbles → 8 signed INT8 → 2 packed ints.
        const int v0 = kvalues_hfp4_wave64[(pk >>  0) & 0xFu];
        const int v1 = kvalues_hfp4_wave64[(pk >>  4) & 0xFu];
        const int v2 = kvalues_hfp4_wave64[(pk >>  8) & 0xFu];
        const int v3 = kvalues_hfp4_wave64[(pk >> 12) & 0xFu];
        const int v4 = kvalues_hfp4_wave64[(pk >> 16) & 0xFu];
        const int v5 = kvalues_hfp4_wave64[(pk >> 20) & 0xFu];
        const int v6 = kvalues_hfp4_wave64[(pk >> 24) & 0xFu];
        const int v7 = kvalues_hfp4_wave64[(pk >> 28) & 0xFu];

        const int int_a = (int)((v0 & 0xFF)
                              | ((v1 & 0xFF) << 8)
                              | ((v2 & 0xFF) << 16)
                              | ((v3 & 0xFF) << 24));
        const int int_b = (int)((v4 & 0xFF)
                              | ((v5 & 0xFF) << 8)
                              | ((v6 & 0xFF) << 16)
                              | ((v7 & 0xFF) << 24));

        // Pick the right Q8_1 block (A or B) for this lane.
        const int kblock = 2 * g + block_within_group;
        const block_q8_1_mmq* xb = Xq + kblock;

        const int* xqs = (const int*)(xb->qs + qs_byte_off);
        const int xq_a = xqs[0];
        const int xq_b = xqs[1];

        int sumi = 0;
        sumi = __builtin_amdgcn_sdot4(int_a, xq_a, sumi, false);
        sumi = __builtin_amdgcn_sdot4(int_b, xq_b, sumi, false);

        // Per-sub-block d_x from Q8_1. HFP4 has no zp, so sum_x drops.
        const half2 ds = xb->ds4[sub_in_block];
        const float2 dsf = __half22float2(ds);
        const float d_x = dsf.x;

        acc += sc * d_x * (float)sumi;
    }

    // Cross-lane reduction within the 32-lane warp.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (lane == 0) y[local_row] = acc;
}
