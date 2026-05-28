#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define HIP_CHECK(hip_call) { \
    hipError_t ret = hip_call; \
    if (ret != hipSuccess) { \
        printf("HIP error %d at %s:%d\n", ret, __FILE__, __LINE__); \
        exit(ret); \
    } \
}

// Declare external kernels
extern "C" __global__ void gemm_f16_wmma_mb4(
    const _Float16* __restrict__ W,
    const float*    __restrict__ X,
    float*          __restrict__ Y,
    int M, int K, int N
);

extern "C" __global__ void gemm_f16_wmma_mb8(
    const _Float16* __restrict__ W,
    const float*    __restrict__ X,
    float*          __restrict__ Y,
    int M, int K, int N
);

int main() {
    // Vision encoder fc13 shape
    int M = 4224;  // output features
    int K = 1536;  // input features
    int N = 19520; // batch size (patches)
    
    printf("Testing MB8 kernel: %dx%dx%d (fc13 shape)\n", M, K, N);
    
    // Allocate device memory
    _Float16 *d_W;
    float *d_X, *d_Y_mb4, *d_Y_mb8;
    
    HIP_CHECK(hipMalloc(&d_W, M * K * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_X, N * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Y_mb4, N * M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Y_mb8, N * M * sizeof(float)));
    
    // Initialize with random data
    _Float16 *h_W = (_Float16*)malloc(M * K * sizeof(_Float16));
    float *h_X = (float*)malloc(N * K * sizeof(float));
    
    for (int i = 0; i < M * K; i++) {
        float val = ((i * 1237) % 1000) * 1e-3f - 0.5f;
        h_W[i] = (_Float16)val;
    }
    for (int i = 0; i < N * K; i++) {
        h_X[i] = ((i * 4219) % 1000) * 1e-3f - 0.5f;
    }
    
    HIP_CHECK(hipMemcpy(d_W, h_W, M * K * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, h_X, N * K * sizeof(float), hipMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        dim3 grid_mb4((M + 15) / 16, (N + 63) / 64, 1);
        hipLaunchKernelGGL(gemm_f16_wmma_mb4, grid_mb4, dim3(32), 0, 0, d_W, d_X, d_Y_mb4, M, K, N);
        
        dim3 grid_mb8((M + 15) / 16, (N + 127) / 128, 1);
        hipLaunchKernelGGL(gemm_f16_wmma_mb8, grid_mb8, dim3(32), 0, 0, d_W, d_X, d_Y_mb8, M, K, N);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // MB4 timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    int iters = 20;
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; i++) {
        dim3 grid_mb4((M + 15) / 16, (N + 63) / 64, 1);
        hipLaunchKernelGGL(gemm_f16_wmma_mb4, grid_mb4, dim3(32), 0, 0, d_W, d_X, d_Y_mb4, M, K, N);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float mb4_ms;
    HIP_CHECK(hipEventElapsedTime(&mb4_ms, start, stop));
    mb4_ms /= iters;
    
    // MB8 timing
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; i++) {
        dim3 grid_mb8((M + 15) / 16, (N + 127) / 128, 1);
        hipLaunchKernelGGL(gemm_f16_wmma_mb8, grid_mb8, dim3(32), 0, 0, d_W, d_X, d_Y_mb8, M, K, N);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float mb8_ms;
    HIP_CHECK(hipEventElapsedTime(&mb8_ms, start, stop));
    mb8_ms /= iters;
    
    // Verify correctness
    float *h_Y_mb4 = (float*)malloc(N * M * sizeof(float));
    float *h_Y_mb8 = (float*)malloc(N * M * sizeof(float));
    HIP_CHECK(hipMemcpy(h_Y_mb4, d_Y_mb4, N * M * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_Y_mb8, d_Y_mb8, N * M * sizeof(float), hipMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    double sum_sq = 0.0;
    for (int i = 0; i < N * M; i++) {
        float diff = fabsf(h_Y_mb4[i] - h_Y_mb8[i]);
        if (diff > max_diff) max_diff = diff;
        sum_sq += (double)diff * (double)diff;
    }
    float rms_diff = sqrtf(sum_sq / (N * M));
    
    printf("MB4: %.2f ms\n", mb4_ms);
    printf("MB8: %.2f ms\n", mb8_ms);
    printf("Speedup: %.2fx\n", mb4_ms / mb8_ms);
    printf("Max diff: %.2e\n", max_diff);
    printf("RMS diff: %.2e\n", rms_diff);
    printf("Correctness: %s\n", (max_diff < 1e-2f && rms_diff < 1e-3f) ? "PASS" : "FAIL");
    
    // Cleanup
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_W);
    hipFree(d_X);
    hipFree(d_Y_mb4);
    hipFree(d_Y_mb8);
    free(h_W);
    free(h_X);
    free(h_Y_mb4);
    free(h_Y_mb8);
    
    return 0;
}
