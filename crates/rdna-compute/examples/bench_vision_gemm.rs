//! Benchmark vision encoder GEMM shapes and measure effective bandwidth.
//! Tests both scattered (current) and tiled access patterns.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters = args.iter().position(|a| a == "--iters")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(10usize);

    let mut gpu = Gpu::init().expect("GPU");
    let arch = gpu.arch.clone();
    let peak_gbs = 960.0; // gfx1100 GDDR6

    // Vision encoder GEMM shapes: (M, K, N, label)
    let shapes: Vec<(&str, usize, usize, usize)> = vec![
        ("qkv  (4608x1536 x 19520)", 4608, 1536, 19520),
        ("proj (1536x1536 x 19520)", 1536, 1536, 19520),
        ("fc1  (4224x1536 x 19520)", 4224, 1536, 19520),
        ("fc2  (1536x4224 x 19520)", 1536, 4224, 19520),
    ];

    eprintln!("GPU: {} peak_bw={} GB/s iters={}", arch, peak_gbs, iters);
    eprintln!();

    for (label, m, k, n) in &shapes {
        // Allocate weight (F16) and input (F32)
        let w_bytes = m * k * 2;
        let x_bytes = n * k * 4;
        let w = gpu.alloc_tensor(&[w_bytes], DType::Raw).unwrap();
        let x = gpu.alloc_tensor(&[x_bytes], DType::Raw).unwrap();
        let y = gpu.alloc_tensor(&[n * m * 4], DType::Raw).unwrap();

        // Warmup
        gpu.gemm_f16_wmma_mb4(&w, &x, &y, *m, *k, *n).unwrap();
        gpu.hip.device_synchronize().unwrap();

        // Benchmark
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_f16_wmma_mb4(&w, &x, &y, *m, *k, *n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t0.elapsed().as_micros() as f64 / iters as f64;

        // Calculate traffic (assuming scattered loads with ~25% efficiency)
        let blocks_y = (n + 63) / 64; // NB=4 → N-tile-group=64
        let blocks_x = (m + 15) / 16;
        let k_steps = k / 16;
        let scattered_w = blocks_x as f64 * blocks_y as f64 * 512.0 / 0.25 / 1e9;
        let scattered_x = blocks_x as f64 * blocks_y as f64 * 4096.0 / 0.25 / 1e9;
        let total_scattered = scattered_w + scattered_x;
        let eff_bw = total_scattered / (us * 1e-6);

        // Flops
        let flops = 2.0 * *m as f64 * *k as f64 * *n as f64;
        let gflops = flops / (us * 1e-6) / 1e9;
        let pct_peak = gflops / (16.4 * 1e3) * 100.0; // ~16.4 TFLOPS for f16 WMMA

        eprintln!("{}: {:.1} ms  eff_bw={:.0} GB/s ({:.0}% of {:.0})",
                  label, us / 1000.0, eff_bw, eff_bw / peak_gbs * 100.0, peak_gbs);
        eprintln!("  {:.1} GFLOP/s ({:.1}% WMMA peak)  m={} k={} n={}",
                  gflops * 1e3, pct_peak, m, k, n);
        eprintln!("  scattered W={:.1} GB  X={:.1} GB  total_actual={:.1} GB",
                  scattered_w, scattered_x, total_scattered);

        gpu.free_tensor(w).unwrap();
        gpu.free_tensor(x).unwrap();
        gpu.free_tensor(y).unwrap();
    }
}
