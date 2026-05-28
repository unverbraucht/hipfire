#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Forward declarations - match actual kernel signatures
extern "C" __global__ void gemm_f16_wmma_mb4(
    const _Float16* __restrict__ W,   // F16 weights
    const float*    __restrict__ X,   // F32 input
    float*          __restrict__ Y,   // F32 output
    int m, int k, int n);

extern "C" __global__ void gemm_f16_wmma_mb8(
    const _Float16* __restrict__ W,   // F16 weights
    const float*    __restrict__ X,   // F32 input
    float*          __restrict__ Y,   // F32 output
    int m, int k, int n);

extern "C" void launch_mb4(const void* w, const void* x, void* y, int m, int k, int n) {
    dim3 block(32);
    dim3 grid((m + 15) / 16, (n + 63) / 64);
    gemm_f16_wmma_mb4<<<grid, block>>>((const _Float16*)w, (const float*)x, (float*)y, m, k, n);
}

extern "C" void launch_mb8(const void* w, const void* x, void* y, int m, int k, int n) {
    dim3 block(32);
    dim3 grid((m + 15) / 16, (n + 127) / 128);
    gemm_f16_wmma_mb8<<<grid, block, 8192>>>((const _Float16*)w, (const float*)x, (float*)y, m, k, n);
}
