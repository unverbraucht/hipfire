// Self-contained benchmark for parameter sweep
// Compile with: hipcc -O3 --offload-arch=gfx1100 -DNB=8 -DM_TILE=16 ... bench_sweep.cpp

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifndef NB
#define NB 8
#endif
#ifndef M_TILE  
#define M_TILE 16
#endif
#ifndef K_UNROLL
#define K_UNROLL 16
#endif
#ifndef PERSISTENT_GROUPS
#define PERSISTENT_GROUPS 768
#endif
#ifndef WAVES_PER_SIMD
#define WAVES_PER_SIMD 8
#endif

typedef _Float16 __attribute__((ext_vector_type(K_UNROLL/2))) half8_t;
typedef float __attribute__((ext_vector_type(K_UNROLL))) float16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

template<int WARP_SIZE, typename T>
__device__ __forceinline__ T warp_reduce(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__launch_bounds__(32, WAVES_PER_SIMD)
__global__ void gemm_f16_wmma_sweep_persistent(
    const _Float16* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int N,
    int total_n_blocks
) {
    int m_block = blockIdx.x;
    int n_start = blockIdx.y;
    int n_stride = gridDim.y;
    int tid = threadIdx.x;
    
    int m_row = m_block * M_TILE + tid;
    if (m_row >= M) return;
    
    float acc[NB] = {0};
    
    for (int n_block = n_start; n_block < total_n_blocks; n_block += n_stride) {
        int n_base = n_block * NB;
        
        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            int n_col = n_base + nb;
            if (n_col >= N) continue;
            
            float sum = 0.0f;
            
            #pragma unroll K_UNROLL/16
            for (int k = 0; k < K; k += K_UNROLL) {
                #pragma unroll
                for (int ki = 0; ki < K_UNROLL; ki++) {
                    if (k + ki < K) {
                        sum += __half2float(A[m_row * K + k + ki]) * X[n_col * K + k + ki];
                    }
                }
            }
            
            acc[nb] += sum;
        }
    }
    
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        int n_col = n_start * NB + nb;
        if (n_col < N) {
            Y[m_row * N + n_col] = acc[nb];
        }
    }
}

#define CHECK(call) do { \
    auto _e = (call); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error: %s\n", hipGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <M> <K> <N>\n", argv[0]);
        return 1;
    }
    
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    const double peak_gbs = 716.8;
    const double peak_tflops = 122.8;
    
    size_t A_bytes = M * K * sizeof(_Float16);
    size_t X_bytes = N * K * sizeof(float);
    size_t Y_bytes = M * N * sizeof(float);
    
    _Float16 *d_A;
    float *d_X, *d_Y;
    CHECK(hipMalloc(&d_A, A_bytes));
    CHECK(hipMalloc(&d_X, X_bytes));
    CHECK(hipMalloc(&d_Y, Y_bytes));
    
    int m_tiles = (M + M_TILE - 1) / M_TILE;
    int n_blocks = (N + NB - 1) / NB;
    dim3 grid(m_tiles, n_blocks);
    dim3 block(32);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        hipLaunchKernelGGL(gemm_f16_wmma_sweep_persistent, grid, block, 0, nullptr, d_A, d_X, d_Y, M, K, N, n_blocks);
    }
    CHECK(hipDeviceSynchronize());
    
    // Benchmark
    const int iters = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        hipLaunchKernelGGL(gemm_f16_wmma_sweep_persistent, grid, block, 0, nullptr, d_A, d_X, d_Y, M, K, N, n_blocks);
    }
    CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    
    double flops = 2.0 * M * K * N;
    double tflops = flops / time_us / 1e6;
    double tflops_pct = (tflops / peak_tflops) * 100.0;
    
    printf("%.2f %.3f %.1f\n", time_us, tflops, tflops_pct);
    
    hipFree(d_A);
    hipFree(d_X);
    hipFree(d_Y);
    
    return 0;
}
