//! Test MB8 performance on QKV and FC2 shapes

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn bench_gemm(gpu: &mut Gpu, m: usize, k: usize, n: usize, label: &str, use_mb8: bool, iters: usize) -> f64 {
    let w_bytes = m * k * 2;
    let x_bytes = n * k * 4;
    let w = gpu.alloc_tensor(&[w_bytes], DType::Raw).unwrap();
    let x = gpu.alloc_tensor(&[x_bytes], DType::Raw).unwrap();
    let y = gpu.alloc_tensor(&[n * m * 4], DType::Raw).unwrap();

    // Warmup
    if use_mb8 {
        gpu.gemm_f16_wmma_mb8(&w, &x, &y, m, k, n).unwrap();
    } else {
        gpu.gemm_f16_wmma_mb4(&w, &x, &y, m, k, n).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // Benchmark
    let t0 = Instant::now();
    for _ in 0..iters {
        if use_mb8 {
            gpu.gemm_f16_wmma_mb8(&w, &x, &y, m, k, n).unwrap();
        } else {
            gpu.gemm_f16_wmma_mb4(&w, &x, &y, m, k, n).unwrap();
        }
    }
    gpu.hip.device_synchronize().unwrap();
    let us = t0.elapsed().as_micros() as f64 / iters as f64;
    
    let kernel = if use_mb8 { "MB8" } else { "MB4" };
    eprintln!("  {} {}: {:.2} ms", kernel, label, us / 1000.0);
    
    gpu.free_tensor(w).unwrap();
    gpu.free_tensor(x).unwrap();
    gpu.free_tensor(y).unwrap();
    
    us
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU");
    let iters = 10;
    
    eprintln!("QKV shape: M=4608, K=1536, N=19520");
    let mb4_qkv = bench_gemm(&mut gpu, 4608, 1536, 19520, "QKV", false, iters);
    let mb8_qkv = bench_gemm(&mut gpu, 4608, 1536, 19520, "QKV", true, iters);
    let speedup_qkv = mb4_qkv / mb8_qkv;
    eprintln!("  QKV speedup: {:.2}x\n", speedup_qkv);
    
    eprintln!("FC2 shape: M=1536, K=4224, N=19520");
    let mb4_fc2 = bench_gemm(&mut gpu, 1536, 4224, 19520, "FC2", false, iters);
    let mb8_fc2 = bench_gemm(&mut gpu, 1536, 4224, 19520, "FC2", true, iters);
    let speedup_fc2 = mb4_fc2 / mb8_fc2;
    eprintln!("  FC2 speedup: {:.2}x\n", speedup_fc2);
    
    eprintln!("Summary:");
    eprintln!("  QKV: {:.2} ms (MB4) → {:.2} ms (MB8) = {:.2}x speedup", mb4_qkv/1000.0, mb8_qkv/1000.0, speedup_qkv);
    eprintln!("  FC2: {:.2} ms (MB4) → {:.2} ms (MB8) = {:.2}x speedup", mb4_fc2/1000.0, mb8_fc2/1000.0, speedup_fc2);
    
    let ms_saved_per_layer = (mb4_qkv - mb8_qkv + mb4_fc2 - mb8_fc2) / 1000.0;
    let total_saved = ms_saved_per_layer * 24.0 / 1000.0;
    eprintln!("\nEstimated savings: {:.1} ms/layer × 24 layers = {:.2}s total", 
             ms_saved_per_layer, total_saved);
}
