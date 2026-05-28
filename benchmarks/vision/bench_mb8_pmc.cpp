#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <chrono>
#include <vector>

#define HIP_CHECK(expr) \
    do { \
        hipError_t err = expr; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Declare the kernel
extern "C" __global__ void gemm_f16_wmma_mb8(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K);

// QKV shape from dots-ocr
constexpr int M = 4608;
constexpr int N = 1536;
constexpr int K = 1536;

int main() {
    printf("MB8 GEMM PMC Benchmark (QKV: %d x %d x %d)\n", M, N, K);
    
    size_t A_size = M * K * sizeof(_Float16);
    size_t B_size = K * N * sizeof(_Float16);
    size_t C_size = M * N * sizeof(float);
    
    printf("Allocating %.1f MB (A: %.1f MB, B: %.1f MB, C: %.1f MB)\n",
           (A_size + B_size + C_size) / 1e6,
           A_size / 1e6, B_size / 1e6, C_size / 1e6);
    
    _Float16 *d_A, *d_B;
    float *d_C;
    HIP_CHECK(hipMalloc(&d_A, A_size));
    HIP_CHECK(hipMalloc(&d_B, B_size));
    HIP_CHECK(hipMalloc(&d_C, C_size));
    
    // Initialize with random data
    std::vector<_Float16> h_A(M * K);
    std::vector<_Float16> h_B(K * N);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(0.01f * (i % 100));
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(0.01f * (i % 100));
    
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), B_size, hipMemcpyHostToDevice));
    
    // Launch config: MB8 uses 16x128 tiles
    dim3 grid(M, (N + 127) / 128);
    dim3 block(128);  // 8 warps
    
    printf("Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);
    printf("Total blocks: %d\n", grid.x * grid.y);
    
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++) {
        gemm_f16_wmma_mb8<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    printf("Running benchmark (100 iterations)...\n");
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < 100; i++) {
        gemm_f16_wmma_mb8<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    ms /= 100.0f;
    
    double tflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e12;
    double bw = (A_size + B_size + C_size) / (ms / 1000.0) / 1e9;
    
    printf("\nResults:\n");
    printf("  Time: %.2f ms\n", ms);
    printf("  TFLOPS: %.2f (%.1f%% of 82.6 peak)\n", tflops, tflops / 82.6 * 100);
    printf("  Bandwidth: %.1f GB/s (%.1f%% of 960 peak)\n", bw, bw / 960 * 100);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    return 0;
}
