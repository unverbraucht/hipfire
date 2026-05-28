#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "kernels/src/gemm_f16_wmma_mb32_opt.hip"

int main() {
    // fc2 shape: 1536x19520 x 4224x19520 -> 1536x4224
    int M = 1536, N = 19520, K = 4224;
    
    std::cout << "Testing MB32 kernel directly: " << M << "x" << N << " x " << K << "x" << N << "\n";
    
    // Allocate host memory
    std::vector<_Float16> h_w(M * K);
    std::vector<float> h_x(K * N);
    std::vector<float> h_y(M * N, 0.0f);
    
    // Initialize with simple values
    for (int i = 0; i < M * K; i++) h_w[i] = (_Float16)((i % 100) * 0.01f);
    for (int i = 0; i < K * N; i++) h_x[i] = (float)((i % 100) * 0.01f);
    
    // Allocate device memory
    _Float16 *d_w;
    float *d_x, *d_y;
    hipMalloc(&d_w, M * K * sizeof(_Float16));
    hipMalloc(&d_x, K * N * sizeof(float));
    hipMalloc(&d_y, M * N * sizeof(float));
    
    // Copy to device
    hipMemcpy(d_w, h_w.data(), M * K * sizeof(_Float16), hipMemcpyHostToDevice);
    hipMemcpy(d_x, h_x.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid((M + 31) / 32, (N + 127) / 128);
    dim3 block(32);
    
    std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: " << block.x << "\n";
    std::cout << "Blocks in Y: " << grid.y << " (should be ~153)\n";
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        gemm_f16_wmma_mb32_opt<<<grid, block, 16384>>>(d_w, d_x, d_y, M, K, N);
    }
    hipDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 20;
    for (int i = 0; i < iterations; i++) {
        gemm_f16_wmma_mb32_opt<<<grid, block, 16384>>>(d_w, d_x, d_y, M, K, N);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    double bandwidth = ((M * K * 2 + K * N * 4 + M * N * 4) / 1e9) / (ms / 1e3);
    
    std::cout << "MB32: " << ms << " ms, " << gflops << " GFLOP/s, " << bandwidth << " GB/s\n";
    
    // Copy back
    hipMemcpy(h_y.data(), d_y, M * N * sizeof(float), hipMemcpyDeviceToHost);
    
    // Verify a few values
    std::cout << "Sample outputs: " << h_y[0] << ", " << h_y[100] << ", " << h_y[1000] << "\n";
    
    // Cleanup
    hipFree(d_w);
    hipFree(d_x);
    hipFree(d_y);
    
    return 0;
}
