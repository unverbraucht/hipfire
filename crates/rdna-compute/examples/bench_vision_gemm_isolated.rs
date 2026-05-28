// Isolated benchmark for vision encoder GEMM
// Tests MB8 kernel with realistic vision encoder dimensions
// Measures TFLOP/s, bandwidth, and kernel launch overhead

use rdna_compute::gpu::{Gpu, DType};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = Gpu::new()?;
    
    // Vision encoder dimensions
    let m = 1536;  // hidden dimension
    let k = 1536;  // hidden dimension
    let n = 19520; // number of patches
    
    println!("Vision GEMM benchmark: M={} K={} N={}", m, k, n);
    
    // Theoretical peak for RX 7900 XTX
    let peak_tflops = 379.0;  // FP16 peak
    let peak_bandwidth = 960.0; // GB/s
    let bytes_per_element = 2; // FP16
    
    // Create tensors
    let a = gpu.create_device_buffer(DType::F16, m * k)?;
    let b = gpu.create_device_buffer(DType::F16, k * n)?;
    let c = gpu.create_device_buffer(DType::F16, m * n)?;
    
    // Warmup
    for _ in 0..5 {
        gpu.gemm_f16_wmma_mb8(&a, &b, &c, m, k, n)?;
    }
    gpu.device_synchronize()?;
    
    // Single kernel timing (includes launch overhead)
    let start = Instant::now();
    gpu.gemm_f16_wmma_mb8(&a, &b, &c, m, k, n)?;
    gpu.device_synchronize()?;
    let single_time_us = start.elapsed().as_micros() as f64;
    
    // Batch timing (amortizes launch overhead)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        gpu.gemm_f16_wmma_mb8(&a, &b, &c, m, k, n)?;
    }
    gpu.device_synchronize()?;
    let batch_time_us = start.elapsed().as_micros() as f64 / iterations as f64;
    
    // Calculate metrics
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let bytes = ((m * k) + (k * n) + (m * n)) as u64 * bytes_per_element;
    
    let single_tflops = flops / (single_time_us * 1e-6) / 1e12;
    let batch_tflops = flops / (batch_time_us * 1e-6) / 1e12;
    
    let single_bandwidth = bytes as f64 / (single_time_us * 1e-6) / 1e9;
    let batch_bandwidth = bytes as f64 / (batch_time_us * 1e-6) / 1e9;
    
    let launch_overhead_us = single_time_us - batch_time_us;
    
    println!("\n=== Results ===");
    println!("Single kernel: {:.1} µs", single_time_us);
    println!("Batch average: {:.1} µs ({} iterations)", batch_time_us, iterations);
    println!("Launch overhead: {:.1} µs ({:.1}% of single kernel)", 
             launch_overhead_us, launch_overhead_us / single_time_us * 100.0);
    println!();
    println!("Performance (single): {:.1} TFLOP/s ({:.1}% of {:.1} peak)", 
             single_tflops, single_tflops / peak_tflops * 100.0, peak_tflops);
    println!("Performance (batch):  {:.1} TFLOP/s ({:.1}% of {:.1} peak)", 
             batch_tflops, batch_tflops / peak_tflops * 100.0, peak_tflops);
    println!();
    println!("Bandwidth (single): {:.1} GB/s ({:.1}% of {:.1} peak)", 
             single_bandwidth, single_bandwidth / peak_bandwidth * 100.0, peak_bandwidth);
    println!("Bandwidth (batch):  {:.1} GB/s ({:.1}% of {:.1} peak)", 
             batch_bandwidth, batch_bandwidth / peak_bandwidth * 100.0, peak_bandwidth);
    println!();
    println!("Arithmetic intensity: {:.1} FLOPs/byte", flops / bytes as f64);
    println!("Compute-bound threshold: {:.1} FLOPs/byte", peak_tflops * 1e12 / (peak_bandwidth * 1e9));
    
    // Analysis
    let arith_intensity = flops / bytes as f64;
    let ridge_point = peak_tflops * 1e12 / (peak_bandwidth * 1e9);
    
    println!("\n=== Analysis ===");
    if arith_intensity > ridge_point {
        println!("Workload is COMPUTE-BOUND (arithmetic intensity {:.1} > ridge point {:.1})", 
                 arith_intensity, ridge_point);
        println!("Bottleneck: GPU compute units");
        println!("Achievement: {:.1}% of peak TFLOP/s", batch_tflops / peak_tflops * 100.0);
    } else {
        println!("Workload is MEMORY-BOUND (arithmetic intensity {:.1} < ridge point {:.1})", 
                 arith_intensity, ridge_point);
        println!("Bottleneck: Memory bandwidth");
        println!("Achievement: {:.1}% of peak bandwidth", batch_bandwidth / peak_bandwidth * 100.0);
    }
    
    if launch_overhead_us > batch_time_us * 0.5 {
        println!("WARNING: Launch overhead is {:.1}% of kernel time!", 
                 launch_overhead_us / batch_time_us * 100.0);
        println!("Consider batching multiple GEMMs or using persistent kernels.");
    }
    
    Ok(())
}
