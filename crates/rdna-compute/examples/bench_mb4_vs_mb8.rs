//! Compare MB4 vs MB8 GEMM performance for vision encoder shapes
//! 
//! MB4: 4-way register blocking, 16×64 output panel per block
//! MB8: 8-way register blocking, 16×128 output panel per block (amortizes weight load 2× better)

use hipfire::hip::*;
use hipfire::kernels::gemm_f16_wmma::{gemm_f16_wmma_mb4, gemm_f16_wmma_mb8};
use std::time::Instant;

fn bench_gemm(m: usize, k: usize, n: usize, iterations: usize, use_mb8: bool) -> (f64, f64, f64) {
    unsafe {
        // Allocate device memory
        let mut a_dev: *mut f32 = std::ptr::null_mut();
        let mut b_dev: *mut f32 = std::ptr::null_mut();
        let mut c_dev: *mut f32 = std::ptr::null_mut();
        
        hipMalloc(&mut a_dev as *mut *mut f32 as *mut *mut std::ffi::c_void, 
                  m * k * std::mem::size_of::<f32>());
        hipMalloc(&mut b_dev as *mut *mut f32 as *mut *mut std::ffi::c_void, 
                  k * n * std::mem::size_of::<f32>());
        hipMalloc(&mut c_dev as *mut *mut f32 as *mut *mut std::ffi::c_void, 
                  m * n * std::mem::size_of::<f32>());

        // Warmup
        for _ in 0..10 {
            if use_mb8 {
                gemm_f16_wmma_mb8(b_dev, a_dev, c_dev, m, k, n);
            } else {
                gemm_f16_wmma_mb4(b_dev, a_dev, c_dev, m, k, n);
            }
        }
        hipDeviceSynchronize();

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            if use_mb8 {
                gemm_f16_wmma_mb8(b_dev, a_dev, c_dev, m, k, n);
            } else {
                gemm_f16_wmma_mb4(b_dev, a_dev, c_dev, m, k, n);
            }
        }
        hipDeviceSynchronize();
        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        // Cleanup
        hipFree(a_dev as *mut std::ffi::c_void);
        hipFree(b_dev as *mut std::ffi::c_void);
        hipFree(c_dev as *mut std::ffi::c_void);

        // Calculate metrics
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64) / 1e12;
        let tflops = flops / (time_ms / 1000.0);
        
        let bytes_transferred = ((m * k) + (k * n) + (m * n)) * std::mem::size_of::<f32>();
        let bw_eff = (bytes_transferred as f64) / 1e9 / (time_ms / 1000.0);

        (time_ms, bw_eff, tflops)
    }
}

fn main() {
    let peak_gbs = 960.0; // GB/s for gfx1100
    let peak_tflops = 82.6; // FP16 TFLOP/s for gfx1100
    
    println!("GPU Peak Performance: {:.0} GB/s bandwidth, {:.1} TFLOP/s FP16\n", 
             peak_gbs, peak_tflops);

    let shapes = vec![
        ("QKV projection", 4608, 1536, 19520),
        ("proj", 1536, 1536, 19520),
        ("fc1", 4224, 1536, 19520),
        ("fc2", 1536, 4224, 19520),
    ];

    println!("{:<20} {:>10} {:>12} {:>10} {:>12} {:>8}", 
             "Shape", "Time(ms)", "Eff BW(GB/s)", "BW %", "TFLOP/s", "Method");
    println!("{}", "=".repeat(80));

    for (name, m, k, n) in shapes {
        // MB4 benchmark
        let (time_mb4, bw_mb4, tflops_mb4) = bench_gemm(m, k, n, 100, false);
        let bw_pct_mb4 = (bw_mb4 / peak_gbs) * 100.0;
        let tflops_pct_mb4 = (tflops_mb4 / peak_tflops) * 100.0;
        
        println!("{:<20} {:>10.2} {:>12.1} {:>9.1}% {:>11.1} {:>7.1}% {:>8}",
                 name, time_mb4, bw_mb4, bw_pct_mb4, tflops_mb4, tflops_pct_mb4, "mb4");

        // MB8 benchmark
        let (time_mb8, bw_mb8, tflops_mb8) = bench_gemm(m, k, n, 100, true);
        let bw_pct_mb8 = (bw_mb8 / peak_gbs) * 100.0;
        let tflops_pct_mb8 = (tflops_mb8 / peak_tflops) * 100.0;
        
        println!("{:<20} {:>10.2} {:>12.1} {:>9.1}% {:>11.1} {:>7.1}% {:>8}",
                 "", time_mb8, bw_mb8, bw_pct_mb8, tflops_mb8, tflops_pct_mb8, "mb8");
        
        let speedup = time_mb4 / time_mb8;
        let tflops_gain = tflops_mb8 / tflops_mb4;
        println!("{:<20} {:>10} {:>12} {:>10} {:>11.2}x ({:.2}x TFLOP/s)",
                 "Speedup", "", "", "", speedup, tflops_gain);
        println!();
    }
}
