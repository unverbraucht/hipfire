// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Persistent GEMM kernel: each workgroup loops over multiple output tiles.
// This amortizes launch overhead and improves L2 cache reuse.
//
// Grid: [ceil(M/16), TILES_PER_GROUP] where TILES_PER_GROUP = min(ceil(N/16)/TARGET_GROUPS, ...)
// Block: [32] (single wave)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define NB8 8

// Each workgroup handles a column of M-tiles across multiple N-tile groups
__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16_wmma_persistent(
    const _Float16* __restrict__ W,   // [M, K] row-major F16
    const float*    __restrict__ X,   // [N, K] row-major F32
    float*          __restrict__ Y,   // [N, M] row-major F32 (transposed)
    int M, int K, int N,
    int n_tile_groups   // total number of N-tile groups (ceil(N/128))
) {
    const int m_tile = blockIdx.x;  // which M-tile (0..ceil(M/16)-1)
    const int n_tile_start = blockIdx.y;  // starting N-tile group for this workgroup
    const int n_tile_stride = gridDim.y;  // stride across N-tile groups
    const int tid = threadIdx.x;
    
    if (m_tile * 16 >= M) return;
    
    const int my_a_row = m_tile * 16 + (tid & 15);
    const bool a_in_bounds = (my_a_row < M);
    
    // Loop over assigned N-tile groups
    for (int n_tile = n_tile_start; n_tile < n_tile_groups; n_tile += n_tile_stride) {
        const int col_start = n_tile * (16 * NB8);
        if (col_start >= N) break;
        
        float8_t acc[NB8];
        #pragma unroll
        for (int nb = 0; nb < NB8; nb++)
            acc[nb] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        
        // K-loop
        for (int k0 = 0; k0 < K; k0 += 16) {
            // Load W tile once
            half16_t a_reg;
            if (a_in_bounds) {
                const _Float16* src = W + (long long)my_a_row * K + k0;
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    a_reg[j] = (k0 + j < K) ? src[j] : (_Float16)0.0f;
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) a_reg[j] = (_Float16)0.0f;
            }
            
            // Process 8 N-subtiles
            #pragma unroll
            for (int nb = 0; nb < NB8; nb++) {
                const int my_b_row = col_start + nb * 16 + (tid & 15);
                half16_t b_reg;
                if (my_b_row < N) {
                    const float* src = X + (long long)my_b_row * K + k0;
                    #pragma unroll
                    for (int j = 0; j < 16; j++)
                        b_reg[j] = (k0 + j < K) ? (_Float16)src[j] : (_Float16)0.0f;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; j++) b_reg[j] = (_Float16)0.0f;
                }
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[nb]);
            }
        }
        
        // Write output
        #pragma unroll
        for (int nb = 0; nb < NB8; nb++) {
            const int out_col = col_start + nb * 16 + (tid & 15);
            if (out_col < N) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int out_row = m_tile * 16 + 2 * j + (tid >> 4);
                    if (out_row < M) {
                        Y[(long long)out_col * M + out_row] = acc[nb][j];
                    }
                }
            }
        }
    }
}
