#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

typedef void (*launch_fn_t)(const void*, const void*, void*, int, int, int);

struct GemmShape {
    const char* name;
    int m, k, n;
};

void benchmark_kernel(launch_fn_t launcher, const char* name, 
                      const void* a, const void* b, void* c,
                      int m, int k, int n, int warmup, int iterations,
                      double peak_gbs, double peak_tflops) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        launcher(a, b, c, m, k, n);
    }
    hipDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        launcher(a, b, c, m, k, n);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    // Calculate metrics
    double flops = 2.0 * m * n * k / 1e12;
    double tflops = flops / (time_ms / 1000.0);
    double tflops_pct = (tflops / peak_tflops) * 100.0;
    
    size_t bytes = (size_t)(m * k + k * n + m * n) * sizeof(float);
    double bw_gbs = bytes / 1e9 / (time_ms / 1000.0);
    double bw_pct = (bw_gbs / peak_gbs) * 100.0;
    
    std::cout << std::setw(20) << name 
              << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " ms"
              << std::setw(12) << std::fixed << std::setprecision(1) << bw_gbs << " GB/s"
              << std::setw(8) << std::fixed << std::setprecision(1) << bw_pct << "%"
              << std::setw(10) << std::fixed << std::setprecision(1) << tflops << " TF"
              << std::setw(8) << std::fixed << std::setprecision(1) << tflops_pct << "%"
              << std::endl;
}

int main() {
    // GPU specs for gfx1100
    const double peak_gbs = 960.0;
    const double peak_tflops = 82.6;
    
    std::cout << "GPU Peak Performance: " << peak_gbs << " GB/s bandwidth, " 
              << peak_tflops << " TFLOP/s FP16\n\n";
    
    // Vision encoder shapes
    std::vector<GemmShape> shapes = {
        {"QKV projection", 4608, 1536, 19520},
        {"proj",           1536, 1536, 19520},
        {"fc1",            4224, 1536, 19520},
        {"fc2",            1536, 4224, 19520}
    };
    
    // Load libraries
    void* mb4_lib = dlopen("./libmb4.so", RTLD_LAZY);
    void* mb8_lib = dlopen("./libmb8.so", RTLD_LAZY);
    
    if (!mb4_lib || !mb8_lib) {
        std::cerr << "Failed to load kernel libraries\n";
        return 1;
    }
    
    launch_fn_t mb4_kernel = (launch_fn_t)dlsym(mb4_lib, "gemm_f16_wmma_mb4");
    launch_fn_t mb8_kernel = (launch_fn_t)dlsym(mb8_lib, "gemm_f16_wmma_mb8");
    
    if (!mb4_kernel || !mb8_kernel) {
        std::cerr << "Failed to load kernels\n";
        return 1;
    }
    
    std::cout << std::setw(20) << "Shape"
              << std::setw(15) << "Time"
              << std::setw(18) << "Bandwidth"
              << std::setw(8) << "BW%"
              << std::setw(15) << "Compute"
              << std::setw(8) << "TF%"
              << std::endl;
    std::cout << std::string(84, '=') << std::endl;
    
    for (const auto& shape : shapes) {
        // W (weights) is F16, X (input) and Y (output) are F32
        size_t a_size = (size_t)shape.m * shape.k * sizeof(_Float16);
        size_t b_size = (size_t)shape.k * shape.n * sizeof(float);
        size_t c_size = (size_t)shape.m * shape.n * sizeof(float);
        
        _Float16 *d_a;
        float *d_b, *d_c;
        
        hipMalloc(&d_a, a_size);
        hipMalloc(&d_b, b_size);
        hipMalloc(&d_c, c_size);
        
        // Benchmark MB4
        benchmark_kernel(mb4_kernel, shape.name, d_a, d_b, d_c, 
                        shape.m, shape.k, shape.n, 10, 100, 
                        peak_gbs, peak_tflops);
        
        // Benchmark MB8
        benchmark_kernel(mb8_kernel, "", d_a, d_b, d_c,
                        shape.m, shape.k, shape.n, 10, 100,
                        peak_gbs, peak_tflops);
        
        std::cout << std::string(84, '-') << std::endl;
        
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
    }
    
    dlclose(mb4_lib);
    dlclose(mb8_lib);
    
    return 0;
}
