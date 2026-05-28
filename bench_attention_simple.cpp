#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <chrono>

// Kernel declaration
__global__ void __launch_bounds__(128, 1)
attention_dflash_wmma_m64_n128_f16kv_v3_f32(
    const float* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale);

int main() {
    // Vision encoder shape
    const int seq_len = 19520;
    const int d_model = 1024;
    const int n_heads = 16;
    const int head_dim = 64;
    const int n_kv_heads = 16;
    
    printf("Vision Attention Benchmark (seq_len=%d, n_heads=%d)\n\n", seq_len, n_heads);
    
    // Allocate memory
    size_t q_size = seq_len * d_model * sizeof(float);
    size_t k_size = seq_len * n_kv_heads * head_dim * sizeof(_Float16);
    size_t v_size = seq_len * n_kv_heads * head_dim * sizeof(_Float16);
    size_t out_size = seq_len * d_model * sizeof(float);
    
    float *d_q, *d_out;
    _Float16 *d_k, *d_v;
    
    hipMalloc(&d_q, q_size);
    hipMalloc(&d_k, k_size);
    hipMalloc(&d_v, v_size);
    hipMalloc(&d_out, out_size);
    
    printf("Memory per layer:\n");
    printf("  Q: %.1f MB\n", q_size / 1024.0 / 1024.0);
    printf("  K: %.1f MB\n", k_size / 1024.0 / 1024.0);
    printf("  V: %.1f MB\n", v_size / 1024.0 / 1024.0);
    printf("  Out: %.1f MB\n", out_size / 1024.0 / 1024.0);
    printf("  Total: %.1f MB\n\n", (q_size + k_size + v_size + out_size) / 1024.0 / 1024.0);
    
    // LDS calculation (from kernel)
    int S_LDS_STRIDE = 130;
    size_t V_lds = 128 * 128 * sizeof(_Float16);  // 32 KB
    size_t S_lds = 64 * S_LDS_STRIDE * sizeof(_Float16);  // 16.64 KB
    size_t m_lds = 64 * sizeof(float);  // 256 bytes
    size_t l_lds = 64 * sizeof(float);  // 256 bytes
    size_t alpha_lds = 64 * sizeof(float);  // 256 bytes
    size_t total_lds = V_lds + S_lds + m_lds + l_lds + alpha_lds;
    
    printf("LDS per block:\n");
    printf("  V_lds: %.1f KB\n", V_lds / 1024.0);
    printf("  S_lds: %.1f KB\n", S_lds / 1024.0);
    printf("  Total: %.1f KB (%.1f KB available)\n\n", total_lds / 1024.0, 64.0);
    
    // Launch configuration
    dim3 grid(n_heads, (seq_len + 63) / 64);
    dim3 block(128);
    size_t shared_mem = total_lds;
    
    printf("Launch config:\n");
    printf("  Grid: (%d, %d)\n", grid.x, grid.y);
    printf("  Block: %d threads\n", block.x);
    printf("  Shared memory: %zu bytes per block\n\n", shared_mem);
    
    // Warmup
    float scale = 1.0f / sqrtf(head_dim);
    for (int i = 0; i < 5; i++) {
        attention_dflash_wmma_m64_n128_f16kv_v3_f32<<<grid, block, shared_mem>>>(
            d_q, d_k, d_v, d_out, seq_len, seq_len, n_heads, n_kv_heads, head_dim, scale);
    }
    hipDeviceSynchronize();
    
    // Benchmark
    int iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        attention_dflash_wmma_m64_n128_f16kv_v3_f32<<<grid, block, shared_mem>>>(
            d_q, d_k, d_v, d_out, seq_len, seq_len, n_heads, n_kv_heads, head_dim, scale);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    
    // Calculate FLOPS
    // QK^T: seq_len × seq_len × d_model × 2 (multiply + add)
    // SV: seq_len × seq_len × d_model × 2
    double qk_flops = 2.0 * seq_len * seq_len * d_model;
    double sv_flops = 2.0 * seq_len * seq_len * d_model;
    double total_flops = qk_flops + sv_flops;
    double tflops = (total_flops / 1e12) / (elapsed_ms / 1000.0);
    
    printf("Results:\n");
    printf("  Time per call: %.2f ms\n", elapsed_ms);
    printf("  Throughput: %.2f TFLOP/s\n", tflops);
    printf("  Calls per second: %.1f\n", 1000.0 / elapsed_ms);
    
    // 42 layers
    double total_time = elapsed_ms * 42 / 1000.0;
    printf("\nVision encoder (42 layers): %.2f s\n", total_time);
    
    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_out);
    
    return 0;
}
