// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Microbenchmark: measure gemv_hfq4g256 decode performance on gfx906.
//! Isolates the GEMV kernel (38.7% of decode kernel time) to measure
//! its bandwidth utilization without the noise of the full decode path.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

// MI50 HBM2 theoretical peak: 262 GiB/s
const PEAK_GIB_S: f64 = 262.0;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== gemv_hfq4g256 BW probe on {arch} ===");
    eprintln!("  peak_bw = {PEAK_GIB_S} GiB/s (MI50 HBM2 theoretical)");

    // Qwen3.6-27B decode shapes (MQ4 format, K/256 groups × 136 bytes each)
    let shapes: Vec<(&str, usize, usize)> = vec![
        ("gate_up (M=17408, K=5120)", 17408, 5120),
        ("down (M=5120, K=17408)", 5120, 17408),
        ("qkv (M=13824, K=5120)", 13824, 5120),
        ("o_proj (M=5120, K=8192)", 5120, 8192),
        ("lm_head (M=248320, K=5120)", 248320, 5120),
    ];

    let iterations = 200;

    for (name, m, k) in &shapes {
        let m = *m;
        let k = *k;

        let groups_per_row = k / 256;
        let weight_bytes = m * groups_per_row * 136;
        let a = gpu.alloc_tensor(&[weight_bytes], DType::Raw).expect("alloc weights");

        // Allocate input vector (FP32)
        let x = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x");

        // Allocate output vector (FP32)
        let y = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");

        // Warmup: ensure kernel is compiled
        gpu.gemv_hfq4g256(&a, &x, &y, m, k).expect("warmup gemv");
        gpu.hip.device_synchronize().expect("sync");

        // Timed loop
        let start = Instant::now();
        for _ in 0..iterations {
            gpu.gemv_hfq4g256(&a, &x, &y, m, k).expect("gemv");
        }
        gpu.hip.device_synchronize().expect("sync");
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / iterations as f64;

        // Analytical bytes
        let weight_bytes = m * groups_per_row * 136;
        let total_bytes = weight_bytes + k * 4 + m * 4;
        let bw_gib_s = (total_bytes as f64) / (per_call_us * 1e-6) / (1024.0_f64.powi(3));
        let pct_peak = bw_gib_s / PEAK_GIB_S * 100.0;

        eprintln!(
            "  {:35} M={:>6} K={:>6}  {:>7.1} µs/call  {:>6.1} GiB/s  ({:>5.1}% peak)  weight={:.1} MiB",
            name, m, k, per_call_us, bw_gib_s, pct_peak,
            weight_bytes as f64 / (1024.0 * 1024.0)
        );

        gpu.free_tensor(a).expect("free a");
        gpu.free_tensor(x).expect("free x");
        gpu.free_tensor(y).expect("free y");
    }
}
