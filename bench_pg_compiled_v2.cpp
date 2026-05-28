#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>  // Added for sleep_for

#define TILE_SIZE 16
#define TILE_N 8  // Process 8 N-tiles per group (128 columns)
#define THREADS_PER_BLOCK 32

__global__ void gemm_f16_persistent_compiled(
    const _Float16* __restrict__ A,  // M x K
    const _Float16* __restrict__ B,  // N x K (transposed for better access)
    float* __restrict__ C,           // M x N
    int M, int N, int K,
    int grid_x, int grid_y
) {
    // Hardcoded for proj: 1536x19520x1536
    const int n_tile_groups = 153;  // ceil(19520/128) = 153
    
    int m_tile = blockIdx.x;
    int n_start = blockIdx.y;
    
    if (m_tile >= grid_x || n_start >= grid_y) return;
    
    int m_base = m_tile * TILE_SIZE;
    if (m_base >= M) return;
    
    int tid = threadIdx.x;
    
    // Persistent loop: each block processes multiple n_tile_groups
    for (int n_group = n_start; n_group < n_tile_groups; n_group += grid_y) {
        int n_base = n_group * (TILE_N * TILE_SIZE);  // 128 columns per group
        
        if (n_base >= N) break;
        
        // Process 8 N-tiles (128 columns) in this group
        for (int tile_idx = 0; tile_idx < TILE_N; tile_idx++) {
            int n_col = n_base + tile_idx * TILE_SIZE + (tid % TILE_SIZE);
            
            if (n_col >= N) continue;
            
            // Each thread computes one element C[m_base:m_base+TILE_SIZE, n_col]
            for (int m_off = tid / TILE_SIZE; m_off < TILE_SIZE; m_off += (THREADS_PER_BLOCK / TILE_SIZE)) {
                int m_row = m_base + m_off;
                if (m_row >= M) continue;
                
                float sum = 0.0f;
                
                // Inner product over K
                for (int k = 0; k < K; k += TILE_SIZE) {
                    // Vectorized load and compute
                    for (int k_off = 0; k_off < TILE_SIZE; k_off++) {
                        if (k + k_off < K) {
                            float a_val = __half2float(A[m_row * K + k + k_off]);
                            float b_val = __half2float(B[n_col * K + k + k_off]);  // B is N x K
                            sum += a_val * b_val;
                        }
                    }
                }
                
                C[m_row * N + n_col] = sum;
            }
        }
    }
}

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
    
    double peak_tflops = 122.8;  // RX 7900 XTX FP16
    
    std::cout << tc.name << " (" << tc.M << "x" << tc.N << "x" << tc.K << ")\n";
    
    for (int pg : tc.pg_values) {
        int grid_x = (tc.M + TILE_SIZE - 1) / TILE_SIZE;
        int grid_y = pg;
        dim3 grid(grid_x, grid_y);
        dim3 block(THREADS_PER_BLOCK);
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            gemm_f16_persistent_compiled<<<grid, block>>>(
                d_A, d_B, d_C, tc.M, tc.N, tc.K, grid_x, grid_y);
        }
        hipDeviceSynchronize();
        
        // Benchmark
        const int iters = 20;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            gemm_f16_persistent_compiled<<<grid, block>>>(
                d_A, d_B, d_C, tc.M, tc.N, tc.K, grid_x, grid_y);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double flops = 2.0 * tc.M * tc.N * tc.K;
        double tflops = flops / (time_ms / 1000.0) / 1e12;
        double percent_peak = (tflops / peak_tflops) * 100.0;
        
        printf("  PG=%4d: %8.2f ms  %6.2f TFLOP/s  (%5.1f%% peak)\n", 
               pg, time_ms, tflops, percent_peak);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
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
