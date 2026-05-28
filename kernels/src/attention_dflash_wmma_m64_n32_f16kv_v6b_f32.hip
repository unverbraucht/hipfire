// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// v6b: split head_dim into two 64-dim passes with n_tile=64.
// Each pass has s_acc[4] (n_tile/16=4, matching array size exactly).
//
// vs v5:  halved n_tile (64 vs 128), doubled K-tile count (305 vs 153),
//         halved Q_frags/O_frags/d_chunks (4 vs 8).
// vs v6:  n_tile=64 (was 128), n_chunks=4 (was 8), so s_acc[4] is correct.
//
// LDS budget:
//   V_lds: 32 × 64 × 2 =  4,096 B   (only 64 d-dims per pass)
//   S_lds: 64 × 66 × 2 =  8,448 B   (stride 66: 64 active + 2 pad)
//   scalars:           =    768 B
//   Total:               13,312 B = 13 KB → 5 WGs/CU (5×13=65 < 64 KB)
//   Actually 65 KB > 64 KB → 4 WGs/CU (4×13=52 < 64 KB).
//
// VGPR target: Q[4]+O[4]+s_acc[4] = 64+32+32 = 128 VGPR (bare minimum
//  for Q+O+s_acc coexistence), leaving ~128 VGPR for Q_other + b_reg +
//  temps + control flow. Realistic ~160–200 VGPR → 2–3 waves/SIMD.
//
// Softmax covers 64 K-columns per K-tile (stride 66, reads 2×32=64 elements).

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

constexpr int S_LDS_STRIDE  = 66;   // 64 active cols + 2 pad
constexpr int V_tile        = 32;
constexpr int d_half_chunks = 4;    // d_chunks per pass (64/16)
constexpr int n_chunks      = 4;    // K-tile chunks (64/16)
constexpr int n_tile        = 64;   // K-tile width
constexpr int m_tile        = 64;   // query rows
constexpr int v_chunks_per_ktile = 2;  // n_tile/V_tile = 64/32

extern "C" __global__ void __launch_bounds__(128, 4)
attention_dflash_wmma_m64_n32_f16kv_v6b_f32(
    const float* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale
) {
    const int head = blockIdx.x;
    const int qt = blockIdx.y;
    const int q_start = qt * m_tile;
    const int tid = threadIdx.x;

    if (head >= n_heads || q_start >= B) return;

    const int rep = n_heads / n_kv_heads;
    const int kv_head = head / rep;
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int my_row_base = wave * 16;

    if (head_dim != 128) return;

    extern __shared__ float sdata[];
    _Float16* V_lds = (_Float16*)sdata;
    _Float16* S_lds = (_Float16*)(V_lds + V_tile * 64);  // V_tile × 64 dims
    float*    m_lds = (float*)(S_lds + m_tile * S_LDS_STRIDE);
    float*    l_lds = m_lds + m_tile;
    float*    alpha_lds = l_lds + m_tile;

    // ================================================================
    // Two passes: d_half=0 (dims 0-63), d_half=1 (dims 64-127)
    // ================================================================
    for (int d_half = 0; d_half < 2; ++d_half) {
        const int d_base = d_half * 64;

        // Barrier: ensure all threads finished previous d_half's
        // finalize (which reads l_lds) before we re-init.
        if (d_half > 0) __syncthreads();

        // Initialize O_frags (output accumulator, stays in regs across all K-tiles)
        float8_t O_frags[d_half_chunks];
        #pragma unroll
        for (int dc = 0; dc < d_half_chunks; ++dc) {
            O_frags[dc] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
        if (lane < 16) {
            m_lds[my_row_base + lane] = -INFINITY;
            l_lds[my_row_base + lane] = 0.0f;
        }
        __syncthreads();

        // Loop over K-tiles of width 64
        for (int kt_start = 0; kt_start < L; kt_start += n_tile) {

            // ============================================================
            // Phase A: Compute full QK^T across both d-halves (64 K-cols)
            // ============================================================
            float8_t s_acc[n_chunks];  // 4 slots for 4 K-chunks ✓
            #pragma unroll
            for (int c = 0; c < n_chunks; ++c) {
                s_acc[c] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            }

            // Current d_half contribution: load Q into registers locally,
            // so the compiler can alias with Q_other later (VGPR reuse).
            {
                half16_t Q_cur[d_half_chunks];
                #pragma unroll
                for (int dc = 0; dc < d_half_chunks; ++dc) {
                    int d0 = d_base + dc * 16;
                    int gq = q_start + my_row_base + half;
                    const float* q_row = (gq < B)
                        ? (q + gq * q_stride + head * head_dim)
                        : nullptr;
                    if (q_row) {
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) Q_cur[dc][j] = (_Float16)q_row[d0 + j];
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) Q_cur[dc][j] = (_Float16)0.0f;
                    }
                }

                #pragma unroll
                for (int kc = 0; kc < n_chunks; ++kc) {
                    int my_k = kt_start + kc * 16 + half;
                    bool k_valid = my_k < L;
                    const _Float16* my_k_row = k_valid
                        ? (k + my_k * kv_stride + kv_head * head_dim + d_base)
                        : nullptr;
                    #pragma unroll 2
                    for (int dc = 0; dc < d_half_chunks; ++dc) {
                        int d0 = dc * 16;
                        half16_t b_reg;
                        if (k_valid) {
                            #pragma unroll
                            for (int j = 0; j < 16; ++j) b_reg[j] = my_k_row[d0 + j];
                        } else {
                            #pragma unroll
                            for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)0.0f;
                        }
                        s_acc[kc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Q_cur[dc], b_reg, s_acc[kc]);
                    }
                }
            }

            // Other d_half contribution: load Q from global memory.
            // Q_cur is dead here — compiler can alias Q_cur and Q_oth VGPRs.
            {
                const int other_base = (d_half == 0) ? 64 : 0;
                half16_t Q_oth[d_half_chunks];
                #pragma unroll
                for (int dc = 0; dc < d_half_chunks; ++dc) {
                    int d0 = other_base + dc * 16;
                    int gq = q_start + my_row_base + half;
                    const float* q_row = (gq < B)
                        ? (q + gq * q_stride + head * head_dim)
                        : nullptr;
                    if (q_row) {
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) Q_oth[dc][j] = (_Float16)q_row[d0 + j];
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) Q_oth[dc][j] = (_Float16)0.0f;
                    }
                }

                #pragma unroll
                for (int kc = 0; kc < n_chunks; ++kc) {
                    int my_k = kt_start + kc * 16 + half;
                    bool k_valid = my_k < L;
                    const _Float16* my_k_row = k_valid
                        ? (k + my_k * kv_stride + kv_head * head_dim + other_base)
                        : nullptr;
                    #pragma unroll 2
                    for (int dc = 0; dc < d_half_chunks; ++dc) {
                        int d0 = dc * 16;
                        half16_t b_reg;
                        if (k_valid) {
                            #pragma unroll
                            for (int j = 0; j < 16; ++j) b_reg[j] = my_k_row[d0 + j];
                        } else {
                            #pragma unroll
                            for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)0.0f;
                        }
                        s_acc[kc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Q_oth[dc], b_reg, s_acc[kc]);
                    }
                }
            }
            // Q_cur and Q_oth are both dead here.

            // Write S to LDS (f16, stride 66)
            #pragma unroll
            for (int c = 0; c < n_chunks; ++c) {
                int s_col = c * 16 + half;
                bool col_valid = (kt_start + s_col) < L;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int s_row = my_row_base + 2 * j + half_id;
                    float s_val_f32 = s_acc[c][j] * scale;
                    if (!col_valid) s_val_f32 = -INFINITY;
                    S_lds[s_row * S_LDS_STRIDE + s_col] = (_Float16)s_val_f32;
                }
            }
            // s_acc is dead here — compiler can alias with Phase C regs.

            // ============================================================
            // Phase B: Cooperative wave-32 softmax (64 columns)
            // ============================================================
            #pragma unroll 1
            for (int r_local = 0; r_local < 16; ++r_local) {
                int row = my_row_base + r_local;
                _Float16* s_row = S_lds + row * S_LDS_STRIDE;

                // Scan 64 elements (2 × 32) instead of 128 (4 × 32) as in v5.
                float local_max = -INFINITY;
                #pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    int idx = lane + kk * 32;
                    local_max = fmaxf(local_max, (float)s_row[idx]);
                }
                local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
                local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
                local_max = fmaxf(local_max, __shfl_xor(local_max, 4));
                local_max = fmaxf(local_max, __shfl_xor(local_max, 8));
                local_max = fmaxf(local_max, __shfl_xor(local_max, 16));
                float tm = local_max;

                float m_old = m_lds[row];
                float m_new = fmaxf(m_old, tm);
                float alpha = expf(m_old - m_new);

                float local_sum = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    int idx = lane + kk * 32;
                    float e = expf((float)s_row[idx] - m_new);
                    s_row[idx] = (_Float16)e;
                    local_sum += e;
                }
                local_sum += __shfl_xor(local_sum, 1);
                local_sum += __shfl_xor(local_sum, 2);
                local_sum += __shfl_xor(local_sum, 4);
                local_sum += __shfl_xor(local_sum, 8);
                local_sum += __shfl_xor(local_sum, 16);

                if (lane == 0) {
                    l_lds[row] = alpha * l_lds[row] + local_sum;
                    m_lds[row] = m_new;
                    alpha_lds[row] = alpha;
                }
            }
            __syncthreads();

            // ============================================================
            // Phase C: Alpha-fold + SV accumulation (64 d-dims, 2 v-chunks)
            // ============================================================
            #pragma unroll
            for (int dc = 0; dc < d_half_chunks; ++dc) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int o_row = my_row_base + 2 * j + half_id;
                    float alpha = alpha_lds[o_row];
                    O_frags[dc][j] *= alpha;
                }
            }

            // Process V in 2 chunks of 32 keys each (n_tile=64 / V_tile=32)
            #pragma unroll
            for (int v_chunk = 0; v_chunk < v_chunks_per_ktile; ++v_chunk) {
                int v_start = v_chunk * V_tile;

                // Stage V into V_lds: only this d_half's 64 dims.
                for (int kr_local = 0; kr_local < 8; ++kr_local) {
                    int kr = wave * 8 + kr_local;
                    int gk = kt_start + v_start + kr;
                    bool valid = gk < L;
                    const _Float16* v_row = valid
                        ? (v + gk * kv_stride + kv_head * head_dim + d_base)
                        : nullptr;
                    for (int dd = 0; dd < 64; dd += 32) {
                        int d = dd + lane;
                        _Float16 vv = v_row ? v_row[d] : (_Float16)0.0f;
                        V_lds[kr * 64 + d] = vv;
                    }
                }
                __syncthreads();

                // SV accumulation: 2 S-chunks per v_chunk (32 K-cols)
                #pragma unroll
                for (int c = 0; c < n_chunks; ++c) {
                    int s_col_base = c * 16;
                    if (s_col_base >= v_start && s_col_base < v_start + V_tile) {
                        int s_col_local = s_col_base - v_start;

                        half16_t a_reg_sm;
                        {
                            const _Float16* sm_row = S_lds + (my_row_base + half) * S_LDS_STRIDE + s_col_base;
                            #pragma unroll
                            for (int j = 0; j < 16; ++j) a_reg_sm[j] = sm_row[j];
                        }

                        #pragma unroll 2
                        for (int dc = 0; dc < d_half_chunks; ++dc) {
                            int d0 = dc * 16;
                            half16_t b_reg;
                            int my_d = d0 + half;
                            #pragma unroll
                            for (int j = 0; j < 16; ++j) {
                                b_reg[j] = V_lds[(s_col_local + j) * 64 + my_d];
                            }
                            O_frags[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                a_reg_sm, b_reg, O_frags[dc]);
                        }
                    }
                }
                __syncthreads();
            } // end v_chunk
        } // end K-tile loop

        // Finalize: write output for this d_half
        #pragma unroll
        for (int dc = 0; dc < d_half_chunks; ++dc) {
            int o_col = d_base + dc * 16 + half;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + half_id;
                int gq = q_start + o_row;
                if (gq < B) {
                    float l = l_lds[o_row];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    out[gq * q_stride + head * head_dim + o_col] = O_frags[dc][j] * inv_l;
                }
            }
        }
    } // end d_half loop
}
