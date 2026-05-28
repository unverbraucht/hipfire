#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Forward declarations
extern "C" __global__ void gemm_f16_wmma_mb4(
    const _Float16* __restrict__ a,
    const _Float16* __restrict__ b,
    float* __restrict__ c,
    int m, int k, int n);

extern "C" __global__ void gemm_f16_wmma_mb8(
    const _Float16* __restrict__ a,
    const _Float16* __restrict__ b,
    float* __restrict__ c,
    int m, int k, int n);

extern "C" void launch_mb4(const void* a, const void* b, void* c, int m, int k, int n) {
    dim3 block(32);
    dim3 grid((m + 15) / 16, (n + 63) / 64);
    gemm_f16_wmma_mb4<<<grid, block>>>((const _Float16*)a, (const _Float16*)b, (float*)c, m, k, n);
}

extern "C" void launch_mb8(const void* a, const void* b, void* c, int m, int k, int n) {
    dim3 block(32);
    dim3 grid((m + 15) / 16, (n + 127) / 128);
    gemm_f16_wmma_mb8<<<grid, block, 8192>>>((const _Float16*)a, (const _Float16*)b, (float*)c, m, k, n);
}
