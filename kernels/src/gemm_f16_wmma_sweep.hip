// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Parameterized GEMM kernel for sweeping optimal configurations.
// Compile-time parameters control tiling, blocking, and occupancy.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Compile-time parameters (set via -D flags)
#ifndef NB
#define NB 8
#endif

#ifndef M_TILE
#define M_TILE 16
#endif

#ifndef K_UNROLL
#define K_UNROLL 16
#endif

#ifndef WAVES_PER_SIMD
#define WAVES_PER_SIMD 8
#endif

#ifndef PERSISTENT_GROUPS
#define PERSISTENT_GROUPS 768
#endif

typedef _Float16 __attribute__((ext_vector_type(K_UNROLL))) halfv_t;
typedef float    __attribute__((ext_vector_type(K_UNROLL))) floatv_t;

// Standard (non-persistent) variant
__launch_bounds__(32, WAVES_PER_SIMD)
extern "C" __global__ void gemm_f16_wmma_sweep(
    const _Float16* __restrict__ A,   // [M, K] row-major F16
    const float*    __restrict__ X,   // [N, K] row-major F32
    float*          __restrict__ Y,   // [N, M] row-major F32
    int M, int K, int N
) {
    const int m_block = blockIdx.x;
    const int n_block = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int m_start = m_block * M_TILE;
    const int n_start = n_block * NB;
    
    // Early exit
    if (m_start >= M || n_start >= N) return;
    
    // Accumulators
    float acc[NB] = {0};
    
    // K-loop with configurable unroll
    #pragma unroll K_UNROLL
    for (int k = 0; k < K; k += K_UNROLL) {
        halfv_t a_row;
        floatv_t x_row;
        
        // Load A row (broadcast to all threads)
        if (m_start + tid < M) {
            for (int i = 0; i < K_UNROLL; i++) {
                a_row[i] = (k + i < K) ? A[(m_start + tid) * K + k + i] : (_Float16)0.0f;
            }
        } else {
            for (int i = 0; i < K_UNROLL; i++) {
                a_row[i] = (_Float16)0.0f;
            }
        }
        
        // Process N-tile (NB rows)
        for (int nb = 0; nb < NB; nb++) {
            const int n_idx = n_start + nb;
            if (n_idx < N) {
                // Load X row and convert to F16
                for (int i = 0; i < K_UNROLL; i++) {
                    x_row[i] = (k + i < K) ? X[n_idx * K + k + i] : 0.0f;
                }
                
                // WMMA-style multiply-accumulate
                #pragma unroll
                for (int i = 0; i < K_UNROLL; i++) {
                    acc[nb] += (float)a_row[i] * x_row[i];
                }
            }
        }
    }
    
    // Write output
    if (m_start + tid < M) {
        for (int nb = 0; nb < NB; nb++) {
            const int n_idx = n_start + nb;
            if (n_idx < N) {
                Y[n_idx * M + m_start + tid] = acc[nb];
            }
        }
    }
}

// Persistent variant
__launch_bounds__(32, WAVES_PER_SIMD)
extern "C" __global__ void gemm_f16_wmma_sweep_persistent(
    const _Float16* __restrict__ A,
    const float*    __restrict__ X,
    float*          __restrict__ Y,
    int M, int K, int N,
    int total_n_blocks
) {
    const int m_block = blockIdx.x;
    const int n_start_block = blockIdx.y;
    const int n_stride = gridDim.y;
    const int tid = threadIdx.x;
    
    const int m_start = m_block * M_TILE;
    
    if (m_start >= M) return;
    
    // Loop over assigned N-tile groups
    for (int n_block = n_start_block; n_block < total_n_blocks; n_block += n_stride) {
        const int n_start = n_block * NB;
        
        float acc[NB] = {0};
        
        #pragma unroll K_UNROLL
        for (int k = 0; k < K; k += K_UNROLL) {
            halfv_t a_row;
            floatv_t x_row;
            
            if (m_start + tid < M) {
                for (int i = 0; i < K_UNROLL; i++) {
                    a_row[i] = (k + i < K) ? A[(m_start + tid) * K + k + i] : (_Float16)0.0f;
                }
            } else {
                for (int i = 0; i < K_UNROLL; i++) {
                    a_row[i] = (_Float16)0.0f;
                }
            }
            
            for (int nb = 0; nb < NB; nb++) {
                const int n_idx = n_start + nb;
                if (n_idx < N) {
                    for (int i = 0; i < K_UNROLL; i++) {
                        x_row[i] = (k + i < K) ? X[n_idx * K + k + i] : 0.0f;
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < K_UNROLL; i++) {
                        acc[nb] += (float)a_row[i] * x_row[i];
                    }
                }
            }
        }
        
        if (m_start + tid < M) {
            for (int nb = 0; nb < NB; nb++) {
                const int n_idx = n_start + nb;
                if (n_idx < N) {
                    Y[n_idx * M + m_start + tid] = acc[nb];
                }
            }
        }
    }
}
