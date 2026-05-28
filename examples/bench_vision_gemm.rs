//! Benchmark vision encoder GEMM shapes to measure effective bandwidth
//! and identify load pattern bottlenecks.

use hipfire::{DType, Gpu};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let iters = args.iter()
        .position(|a| a == "--iters")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(20);

    let mut gpu = Gpu::init()?;
    let arch = gpu.arch();
    let peak_gbps = 960.0;  // gfx1100 GDDR6

    eprintln!("GPU: {} | Peak BW: {:.0} GB/s | Iters: {}", arch, peak_gbps, iters);
    eprintln!();

    // Vision encoder shapes: [N, K] x [K, M] = [N, M]
    // N = 19520 (patches)
    // K varies by layer
    // M = output dim
    let test_shapes = vec![
        // (name, N, K, M)
        ("qkv",  19520, 1536, 4608),  // 3 heads of 1536
        ("fc1",  19520, 1536, 4096),
        ("fc2",  19520, 4096, 1536),
        ("proj", 19520, 1536, 1536),
    ];

    for (name, n, k, m) in &test_shapes {
        // Allocate tensors
        let x = gpu.alloc_tensor(&[n * k * 2], DType::F16)?;  // [N, K]
        let w = gpu.alloc_tensor(&[k * m * 2], DType::F16)?;  // [K, M]
        let y = gpu.alloc_tensor(&[n * m * 2], DType::F16)?;  // [N, M]

        // Warmup
        gpu.matmul_f16(&x, n, k, &w, k, m, &y, n, m)?;
        gpu.synchronize()?;

        // Benchmark
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.matmul_f16(&x, n, k, &w, k, m, &y, n, m)?;
        }
        gpu.synchronize()?;
        let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // Calculate metrics
        let flops = 2.0 * (n * k * m) as f64;
        let tflops = flops / (elapsed_us * 1e-6) / 1e12;

        // Memory traffic (read: X + W, write: Y)
        let bytes = (n * k + k * m + n * m) * 2;
        let e_bw = bytes as f64 / (elapsed_us * 1e-6) / 1e9;

        eprintln!("{:10} | N={:5} K={:4} M={:4} | {:.1}ms | {:.1} TFLOP/s | {:.1} GB/s ({:.1}%)",
                  name, n, k, m,
                  elapsed_us / 1000.0,
                  tflops,
                  e_bw,
                  e_bw / peak_gbps * 100.0);
    }

    Ok(())
}
