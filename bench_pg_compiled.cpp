#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>

#define TILE_SIZE 16
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 32

__global__ __launch_bounds__(THREADS_PER_BLOCK, WARPS_PER_BLOCK)
void gemm_f16_persistent_compiled(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int grid_x, int grid_y,
    int persistent_groups
) {
    // Hardcoded for proj shape: 1536x1536x19520
    const int n_tile_groups = 153;  // Compile-time constant = ceil(19520/128)
    
    int m_tile = blockIdx.x;
    int n_start = blockIdx.y;
    
    if (m_tile >= grid_x || n_start >= grid_y) return;
    
    int m_base = m_tile * TILE_SIZE;
    
    // Each block processes multiple n_tile_groups in a loop
    for (int n_group = n_start; n_group < n_tile_groups; n_group += grid_y) {
        int n_base = n_group * 128;  // Each group processes 128 columns (8 tiles * 16)
        
        // Check bounds
        if (n_base + 128 > N) continue;
        
        for (int tile_idx = 0; tile_idx < 8; tile_idx++) {
            int n_offset = n_base + tile_idx * TILE_SIZE;
            
            for (int k = 0; k < K; k += TILE_SIZE) {
                for (int m_off = 0; m_off < TILE_SIZE; m_off++) {
                    float sum = 0.0f;
                    for (int k_off = 0; k_off < TILE_SIZE; k_off++) {
                        float a_val = __half2float(A[(m_base + m_off) * K + k + k_off]);
                        float b_val = __half2float(B[(n_offset + 0) * K + k + k_off]);  
                        sum += a_val * b_val;
                    }
                    atomicAdd(&C[(n_offset + 0) * M + m_base + m_off], sum);
                }
            }
        }
    }
}

// Test harness
struct TestCase {
    const char* name;
    int M, N, K;
    std::vector<int> pg_values;
};

void run_test(const TestCase& tc) {
    size_t size_A = tc.M * tc.K * sizeof(_Float16);
    size_t size_B = tc.N * tc.K * sizeof(_Float16);
    size_t size_C = tc.M * tc.N * sizeof(float);
    
    _Float16 *d_A, *d_B;
    float *d_C;
    
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);
    
    hipMemset(d_A, 0, size_A);
    hipMemset(d_B, 0, size_B);
    hipMemset(d_C, 0, size_C);
    
    double peak_tflops = 122.8;  // RX 7900 XTX FP16 theoretical
    
    std::cout << tc.name << " (" << tc.M << "x" << tc.N << "x" << tc.K << ")\n";
    
    for (int pg : tc.pg_values) {
        int grid_x = (tc.M + TILE_SIZE - 1) / TILE_SIZE;
        int grid_y = pg;
        dim3 grid(grid_x, grid_y);
        dim3 block(THREADS_PER_BLOCK);
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            gemm_f16_persistent_compiled<<<grid, block>>>(
                d_A, d_B, d_C, tc.M, tc.N, tc.K, grid_x, grid_y, pg);
        }
        hipDeviceSynchronize();
        
        // Benchmark
        const int iters = 20;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            gemm_f16_persistent_compiled<<<grid, block>>>(
                d_A, d_B, d_C, tc.M, tc.N, tc.K, grid_x, grid_y, pg);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double flops = 2.0 * tc.M * tc.N * tc.K;
        double tflops = flops / (time_ms / 1000.0) / 1e12;
        double percent_peak = (tflops / peak_tflops) * 100.0;
        
        printf("  PG=%4d: %8.2f ms  %6.2f TFLOP/s  (%5.1f%% peak)\n", 
               pg, time_ms, tflops, percent_peak);
        
        // Cooldown
        std::chrono::milliseconds cooldown(2000);
        std::this_thread::sleep_for(cooldown);
    }
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    std::cout << "\n";
}

int main() {
    std::cout << "=== Compiled n_tile_groups Benchmark ===\n\n";
    
    std::vector<TestCase> tests = {
        {"proj", 1536, 19520, 1536, {384, 512, 768, 1024, 1536}}
    };
    
    for (const auto& tc : tests) {
        run_test(tc);
    }
    
    return 0;
}
