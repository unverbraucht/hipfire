// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Optimized MB32 GEMM: 2×16 tiles in M-dimension, 128-col N-tile
// Reduces block count in Y by 2×, improving A-matrix reuse across blocks.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define MB 2
#define NB 8
#define TILE_M (16 * MB)
#define TILE_N (16 * NB)

__launch_bounds__(64, 2)  // 2 warps per block, 2 blocks per CU
extern "C" __global__ void gemm_f16_wmma_mb32_opt(
    const _Float16* __restrict__ W,  // [M, K] row-major F16
    const float*    __restrict__ X,  // [K, N] column-major F32 (input)
    float*          __restrict__ Y,  // [M, N] row-major F32 (output)
    int M, int K, int N
) {
    __shared__ float lds_output[TILE_N * TILE_M];

    const int row_start = blockIdx.x * TILE_M;    // M-tile origin (32 rows)
    const int col_start = blockIdx.y * TILE_N;    // N-tile origin (128 cols)
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;  // 0 or 1
    const int lane = tid & 31;

    if (row_start >= M || col_start >= N) return;

    // Each warp handles one 16×16 M-tile
    const int my_m_tile = row_start + warp_id * 16;
    const int my_a_row = my_m_tile + (lane & 15);
    const bool a_in_bounds = (my_m_tile < M) && (my_a_row < M);

    // Accumulators: each warp has NB=8 float8_t accumulators
    float8_t acc[NB];
    #pragma unroll
    for (int nb = 0; nb < NB; nb++)
        acc[nb] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < K; k0 += 16) {
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

        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            const int my_b_row = col_start + nb * 16 + (lane & 15);
            half16_t b_reg;
            if (my_b_row < N) {
                const float* src = X + (long long)k0 * N + my_b_row;  // column-major
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    b_reg[j] = (k0 + j < K) ? (_Float16)src[(long long)j * N] : (_Float16)0.0f;
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) b_reg[j] = (_Float16)0.0f;
            }
            acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[nb]);
        }
    }

    // Write to LDS: each warp writes its 16×128 tile
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const int out_col = col_start + nb * 16 + (lane & 15);
        if (out_col < N) {
            const int lds_col = nb * 16 + (lane & 15);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = my_m_tile + 2 * j + (lane >> 4);
                if (out_row < M) {
                    const int lds_row = warp_id * 16 + 2 * j + (lane >> 4);
                    lds_output[lds_col * TILE_M + lds_row] = acc[nb][j];
                }
            }
        }
    }
    __syncthreads();

    // Coalesced write-back: M-major (row-major Y)
    const int num_cols = min(N - col_start, TILE_N);
    for (int i = tid; i < TILE_M * num_cols; i += 64) {
        const int out_row = i / num_cols;
        const int out_col = i % num_cols;
        const int global_row = row_start + out_row;
        const int global_col = col_start + out_col;
        if (global_row < M && global_col < N) {
            Y[(long long)global_row * N + global_col] = lds_output[out_col * TILE_M + out_row];
        }
    }
}
