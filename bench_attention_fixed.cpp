#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <chrono>

extern "C" __global__ void __launch_bounds__(128, 1)
attention_dflash_wmma_m64_n128_f16kv_v3_f32(
    const float* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale);

int main() {
    const int seq_len = 19520;
    const int n_heads = 16;
    const int head_dim = 128;
    
    // Q in FP32, K/V in FP16
    size_t q_size = seq_len * n_heads * head_dim * sizeof(float);
    size_t kv_size = seq_len * n_heads * head_dim * sizeof(_Float16);
    size_t out_size = seq_len * n_heads * head_dim * sizeof(float);
    
    float *d_q, *d_out;
    _Float16 *d_k, *d_v;
    
    hipMalloc((void**)&d_q, q_size);
    hipMalloc((void**)&d_k, kv_size);
    hipMalloc((void**)&d_v, kv_size);
    hipMalloc((void**)&d_out, out_size);
    
    // Zero initialize
    hipMemset(d_q, 0, q_size);
    hipMemset(d_k, 0, kv_size);
    hipMemset(d_v, 0, kv_size);
    hipMemset(d_out, 0, out_size);
    
    dim3 grid(n_heads, (seq_len + 63) / 64);
    dim3 block(128);
    size_t shared_mem = 64 * 130 * sizeof(_Float16) + 128 * 128 * sizeof(_Float16);
    float scale = 1.0f / sqrtf(head_dim);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        attention_dflash_wmma_m64_n128_f16kv_v3_f32<<<grid, block, shared_mem>>>(
            d_q, d_k, d_v, d_out, seq_len, seq_len, n_heads, n_heads, head_dim, scale);
    }
    hipDeviceSynchronize();
    
    // Benchmark
    int iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        attention_dflash_wmma_m64_n128_f16kv_v3_f32<<<grid, block, shared_mem>>>(
            d_q, d_k, d_v, d_out, seq_len, seq_len, n_heads, n_heads, head_dim, scale);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double tflops = (2.0 * 2 * seq_len * seq_len * n_heads * head_dim / 1e12) / (ms / 1000.0);
    
    printf("Vision Encoder Attention Profile\n");
    printf("================================\n");
    printf("Config: seq_len=%d, n_heads=%d, head_dim=%d\n", seq_len, n_heads, head_dim);
    printf("Memory layout: Q(FP32), K/V(FP16), Out(FP32)\n");
    printf("Kernel: attention_dflash_wmma_m64_n128_f16kv_v3_f32\n");
    printf("Grid: (%d, %d), Block: 128\n", grid.x, grid.y);
    printf("Shared memory: %zu KB\n\n", shared_mem / 1024);
    
    printf("Results:\n");
    printf("  Single call: %.2f ms\n", ms);
    printf("  Throughput: %.2f TFLOPS\n", tflops);
    printf("  42 layers total: %.2f s (%.1f%% of 29.5s vision encoder time)\n\n", 
           ms * 42 / 1000.0, (ms * 42 / 1000.0) / 29.5 * 100.0);
    
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
    return 0;
}
