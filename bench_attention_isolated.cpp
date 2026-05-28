// Isolated benchmark for attention_dflash_wmma_m64_n128_f16kv_v3_f32
// Used to profile vision encoder attention kernel in isolation
//
// Compile:
//   hipcc -O3 --offload-arch=gfx1100 -I/opt/rocm/include \
//         -L/opt/rocm/lib -lhiprtc \
//         kernels/src/attention_dflash_wmma_m64_n128_f16kv_v3.hip \
//         bench_attention_isolated.cpp -o bench_attention_v3
//
// Run:
//   ./bench_attention_v3 19520 16 128 16 100 5

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cstring>

extern "C" __launch_bounds__(128, 1)
__global__ void attention_dflash_wmma_m64_n128_f16kv_v3_f32(
    const float* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale
);

#define CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <seq_len> <n_heads> <head_dim> <n_kv_heads> <warmup> <iters>\n", argv[0]);
        printf("Example: %s 19520 16 128 16 100 5\n", argv[0]);
        return 1;
    }
    
    int seq_len = atoi(argv[1]);
    int n_heads = atoi(argv[2]);
    int head_dim = atoi(argv[3]);
    int n_kv_heads = atoi(argv[4]);
    int warmup = atoi(argv[5]);
    int iters = (argc > 6) ? atoi(argv[6]) : 100;
    
    if (head_dim != 128) {
        fprintf(stderr, "Error: head_dim must be 128 for this kernel\n");
        return 1;
    }
    
    printf("Attention v3 Benchmark\n");
    printf("=====================\n");
    printf("seq_len:    %d\n", seq_len);
    printf("n_heads:    %d\n", n_heads);
    printf("head_dim:   %d\n", head_dim);
    printf("n_kv_heads: %d\n", n_kv_heads);
    printf("warmup:     %d\n", warmup);
    printf("iters:      %d\n\n", iters);
    
    // Calculate memory requirements
    size_t q_size = seq_len * n_heads * head_dim * sizeof(float);
    size_t k_size = seq_len * n_kv_heads * head_dim * sizeof(_Float16);
    size_t v_size = seq_len * n_kv_heads * head_dim * sizeof(_Float16);
    size_t out_size = seq_len * n_heads * head_dim * sizeof(float);
    
    printf("Memory requirements:\n");
    printf("  Q:   %zu bytes (%.2f MB)\n", q_size, q_size / 1024.0 / 1024.0);
    printf("  K:   %zu bytes (%.2f MB)\n", k_size, k_size / 1024.0 / 1024.0);
    printf("  V:   %zu bytes (%.2f MB)\n", v_size, v_size / 1024.0 / 1024.0);
    printf("  Out: %zu bytes (%.2f MB)\n", out_size, out_size / 1024.0 / 1024.0);
    printf("  Total: %.2f MB\n\n", (q_size + k_size + v_size + out_size) / 1024.0 / 1024.0);
    
    // Allocate device memory
    float *d_q, *d_out;
    _Float16 *d_k, *d_v;
    
    CHECK(hipMalloc(&d_q, q_size));
    CHECK(hipMalloc(&d_k, k_size));
    CHECK(hipMalloc(&d_v, v_size));
    CHECK(hipMalloc(&d_out, out_size));
    
    // Initialize with random data
    float *h_q = (float*)malloc(q_size);
    _Float16 *h_k = (_Float16*)malloc(k_size);
    _Float16 *h_v = (_Float16*)malloc(v_size);
    
    srand(42);
    for (size_t i = 0; i < q_size / sizeof(float); i++) {
        h_q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (size_t i = 0; i < k_size / sizeof(_Float16); i++) {
        h_k[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }
    for (size_t i = 0; i < v_size / sizeof(_Float16); i++) {
        h_v[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }
    
    CHECK(hipMemcpy(d_q, h_q, q_size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_k, h_k, k_size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_v, h_v, v_size, hipMemcpyHostToDevice));
    
    // Calculate kernel parameters
    float scale = 1.0f / sqrtf((float)head_dim);
    int q_tiles = (seq_len + 63) / 64;
    dim3 grid(n_heads, q_tiles, 1);
    dim3 block(128, 1, 1);
    
    // LDS requirements
    int lds_size = 32 * seq_len * sizeof(_Float16) +  // V cache
                   128 * 130 * sizeof(_Float16) +    // S matrix
                   128 * sizeof(float);               // softmax scratch
    if (lds_size > 65536) {
        fprintf(stderr, "Error: LDS size %d exceeds 64KB limit\n", lds_size);
        return 1;
    }
    
    printf("Kernel launch config:\n");
    printf("  grid:  (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("  block: (%d, %d, %d)\n", block.x, block.y, block.z);
    printf("  LDS:   %d bytes (%.2f KB)\n\n", lds_size, lds_size / 1024.0);
    
    // Warmup
    printf("Warming up (%d iterations)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        attention_dflash_wmma_m64_n128_f16kv_v3_f32<<<grid, block, lds_size>>>(
            d_q, d_k, d_v, d_out,
            seq_len, seq_len, n_heads, n_kv_heads, head_dim, scale
        );
    }
    CHECK(hipDeviceSynchronize());
    
    // Benchmark
    printf("Benchmarking (%d iterations)...\n", iters);
    
    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));
    
    CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; i++) {
        attention_dflash_wmma_m64_n128_f16kv_v3_f32<<<grid, block, lds_size>>>(
            d_q, d_k, d_v, d_out,
            seq_len, seq_len, n_heads, n_kv_heads, head_dim, scale
        );
    }
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    
    float elapsed_ms;
    CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= iters;
    
    // Calculate performance metrics
    double elapsed_s = elapsed_ms / 1000.0;
    double flops = 2.0 * seq_len * seq_len * n_heads * head_dim * 2;  // QK + SV
    double tflops = flops / elapsed_s / 1e12;
    double bytes = (q_size + k_size + v_size + out_size);
    double gbps = bytes / elapsed_s / 1e9;
    
    printf("\nResults:\n");
    printf("  Time:        %.3f ms per call\n", elapsed_ms);
    printf("  Throughput:  %.2f TFLOP/s\n", tflops);
    printf("  Bandwidth:   %.1f GB/s\n", gbps);
    printf("  Calls/sec:   %.1f\n", 1000.0 / elapsed_ms);
    
    // Theoretical analysis
    double theo_flops = elapsed_s * 82.7e12;  // 82.7 TFLOPS for gfx1100
    double flops_util = (flops / theo_flops) * 100.0;
    
    double theo_bytes = elapsed_s * 512e9;  // 512 GB/s for gfx1100 GDDR6
    double bytes_util = (bytes / theo_bytes) * 100.0;
    
    printf("\nUtilization (gfx1100 theoretical: 82.7 TFLOPS, 512 GB/s):\n");
    printf("  Compute:  %.1f%%\n", flops_util);
    printf("  Memory:   %.1f%%\n", bytes_util);
    
    // Cleanup
    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));
    CHECK(hipFree(d_q));
    CHECK(hipFree(d_k));
    CHECK(hipFree(d_v));
    CHECK(hipFree(d_out));
    free(h_q);
    free(h_k);
    free(h_v);
    
    return 0;
}
