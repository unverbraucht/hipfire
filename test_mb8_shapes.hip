#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

// Declare MB4 and MB8 kernels
extern "C" __global__ void gemm_f16_mb4(const _Float16* __restrict__ A, const _Float16* __restrict__ B, 
                                         float* __restrict__ C, int M, int N, int K);
extern "C" __global__ void gemm_f16_mb8(const _Float16* __restrict__ A, const _Float16* __restrict__ B, 
                                         float* __restrict__ C, int M, int N, int K);

#define CHECK(cmd) \
  do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
      printf("HIP error: %s\n", hipGetErrorString(err)); \
      exit(1); \
    } \
  } while(0)

void bench_gemm(const _Float16* d_A, const _Float16* d_B, float* d_C,
                int M, int N, int K, const char* name, bool use_mb8) {
    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));
    
    dim3 block(32);
    dim3 grid;
    
    if (use_mb8) {
        grid = dim3((N + 127)/128, (M + 15)/16);
    } else {
        grid = dim3((N + 63)/64, (M + 15)/16);
    }
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        if (use_mb8) {
            hipLaunchKernelGGL(gemm_f16_mb8, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
        } else {
            hipLaunchKernelGGL(gemm_f16_mb4, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
        }
    }
    CHECK(hipDeviceSynchronize());
    
    // Benchmark
    CHECK(hipEventRecord(start));
    int iters = 20;
    for (int i = 0; i < iters; i++) {
        if (use_mb8) {
            hipLaunchKernelGGL(gemm_f16_mb8, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
        } else {
            hipLaunchKernelGGL(gemm_f16_mb4, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
        }
    }
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    
    float ms;
    CHECK(hipEventElapsedTime(&ms, start, stop));
    printf("%-10s: %.1f ms (MB%d)\n", name, ms/iters, use_mb8?8:4);
    
    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));
}

int main() {
    printf("Testing MB8 on QKV and FC2 shapes\n");
    printf("==================================\n\n");
    
    // QKV shape: M=4608, K=1536, N=19520
    int M_qkv = 4608, K_qkv = 1536, N_qkv = 19520;
    printf("QKV: M=%d, K=%d, N=%d\n", M_qkv, K_qkv, N_qkv);
    
    _Float16 *d_A_qkv, *d_B_qkv;
    float *d_C_qkv;
    CHECK(hipMalloc(&d_A_qkv, M_qkv * K_qkv * sizeof(_Float16)));
    CHECK(hipMalloc(&d_B_qkv, K_qkv * N_qkv * sizeof(_Float16)));
    CHECK(hipMalloc(&d_C_qkv, M_qkv * N_qkv * sizeof(float)));
    
    bench_gemm(d_A_qkv, d_B_qkv, d_C_qkv, M_qkv, N_qkv, K_qkv, "QKV", false);
    bench_gemm(d_A_qkv, d_B_qkv, d_C_qkv, M_qkv, N_qkv, K_qkv, "QKV", true);
    
    hipFree(d_A_qkv); hipFree(d_B_qkv); hipFree(d_C_qkv);
    printf("\n");
    
    // FC2 shape: M=1536, K=4224, N=19520
    int M_fc2 = 1536, K_fc2 = 4224, N_fc2 = 19520;
    printf("FC2: M=%d, K=%d, N=%d\n", M_fc2, K_fc2, N_fc2);
    
    _Float16 *d_A_fc2, *d_B_fc2;
    float *d_C_fc2;
    CHECK(hipMalloc(&d_A_fc2, M_fc2 * K_fc2 * sizeof(_Float16)));
    CHECK(hipMalloc(&d_B_fc2, K_fc2 * N_fc2 * sizeof(_Float16)));
    CHECK(hipMalloc(&d_C_fc2, M_fc2 * N_fc2 * sizeof(float)));
    
    bench_gemm(d_A_fc2, d_B_fc2, d_C_fc2, M_fc2, N_fc2, K_fc2, "FC2", false);
    bench_gemm(d_A_fc2, d_B_fc2, d_C_fc2, M_fc2, N_fc2, K_fc2, "FC2", true);
    
    hipFree(d_A_fc2); hipFree(d_B_fc2); hipFree(d_C_fc2);
    
    return 0;
}
