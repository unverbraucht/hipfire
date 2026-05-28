//! Benchmark vision encoder GEMM: F32 input vs F16 input
//! Hypothesis: F16 input (57MB) fits in 96MB L2 (gfx1100), F32 (114MB) doesn't.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("GPU");
    let n = 19520usize;
    let k = 1536usize;
    let m = 4224usize;
    let iters = 10usize;

    eprintln!("=== Vision Encoder F16 Input Test ===");
    eprintln!("Shape: N={}  K={}  M={}  iters={}", n, k, m, iters);
    eprintln!("F32 X size: {} MB,  F16 X size: {} MB",
        n * k * 4 / 1024 / 1024, n * k * 2 / 1024 / 1024);
    eprintln!();

    // Weight [M, K] F16
    let w = gpu.alloc_tensor(&[m * k], DType::F16).unwrap();
    // Input F32 [N, K]
    let x_f32 = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
    // Input F16 [N, K]
    let x_f16 = gpu.alloc_tensor(&[n * k], DType::F16).unwrap();
    // Output [N, M] F32
    let y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();

    // --- Bench 1: MB8 GEMM with F32 input (current path) ---
    // Warmup
    for _ in 0..3 {
        gpu.gemm_f16_wmma_mb8(&w, &x_f32, &y, m, k, n).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.gemm_f16_wmma_mb8(&w, &x_f32, &y, m, k, n).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let f32_input_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // --- Bench 2: Cast cost alone ---
    for _ in 0..3 {
        gpu.cast_f32_to_f16(&x_f32, &x_f16).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.cast_f32_to_f16(&x_f32, &x_f16).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let cast_us = t0.elapsed().as_micros() as f64 / iters as f64;

    eprintln!("MB8 GEMM with F32 input (current):   {:.1} ms", f32_input_us / 1000.0);
    eprintln!("Cast F32->F16 alone:                 {:.2} ms", cast_us / 1000.0);
    eprintln!();
    
    // --- Bench 3: MB8 GEMM with F16 input ---
    // The current kernel reads `const float* X` and does F32->F16 per-lane.
    // To test F16 input directly, we'd need a kernel variant.
    // But the cast+GEMM path tells us the upper bound:
    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.cast_f32_to_f16(&x_f32, &x_f16).unwrap();
        gpu.gemm_f16_wmma_mb8(&w, &x_f32, &y, m, k, n).unwrap(); // still F32 input
    }
    gpu.hip.device_synchronize().unwrap();
    let cast_plus_gemm_us = t0.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("Cast + MB8 GEMM (F32):               {:.1} ms", cast_plus_gemm_us / 1000.0);
    eprintln!();
    
    // Analysis
    let bytes_f32 = (n * k * 4 + m * k * 2 + n * m * 4) as f64 / 1e6;
    let bw_f32 = bytes_f32 / (f32_input_us * 1e-6) / 1e3;
    eprintln!("F32 GEMM: {:.1} ms, {:.1} GB/s (peak 960 GB/s GDDR6, {:.1}%)",
        f32_input_us / 1000.0, bw_f32, bw_f32 / 960.0 * 100.0);
    eprintln!();
    eprintln!("Key insight: the MB8 kernel converts F32->F16 per-lane inside the K-loop.");
    eprintln!("If we pre-cast X to F16 BEFORE the GEMM, the kernel reads 2x fewer bytes");
    eprintln!("from DRAM.  114 MB F32 -> 57 MB F16, both < 96 MB L2.");
    eprintln!("But the per-lane conversion is free (just a `v_cvt_f16_f32` instruction).");
    eprintln!("The real win comes from X fitting in L2 after the first M-tile pass.");
    eprintln!("This needs a separate `gemm_f16_wmma_mb8_f16x` kernel that reads _Float16* X.");
}
