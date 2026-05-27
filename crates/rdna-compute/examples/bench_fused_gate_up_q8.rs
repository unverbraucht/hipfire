// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt <kaden@hipfire.dev>
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test + microbench for fused_gate_up_q8_0.
//!
//! Compares against two separate gemv_q8_0 calls (gate + up) at the
//! dots.ocr SwiGLU decode shape (gate_m=up_m=8960, K=1536).

use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
    }).collect()
}

fn make_q8_weight(m: usize, k: usize, seed: u32) -> Vec<u8> {
    let nblocks = k / 32;
    let row_bytes = nblocks * 34;
    let mut buf = vec![0u8; m * row_bytes];
    let mut s = seed;
    for r in 0..m {
        for nb in 0..nblocks {
            let off = r * row_bytes + nb * 34;
            // fake scale (f16 = 1.0 = 0x3c00)
            buf[off] = 0x00;
            buf[off + 1] = 0x3c;
            // random weights
            for i in 0..32 {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                buf[off + 2 + i] = ((s >> 8) & 0xff) as u8;
            }
        }
    }
    buf
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str, d: usize| args.iter().position(|a| a == k)
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(d);
    let gate_m = argval("--gate-m", 8960);
    let up_m = argval("--up-m", 8960);
    let k = argval("--k", 1536);
    let iters = argval("--iters", 200);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  gate_m={gate_m} up_m={up_m} K={k} iters={iters}", gpu.arch);

    // Weights
    let gate_bytes = make_q8_weight(gate_m, k, 0xc3c3);
    let up_bytes = make_q8_weight(up_m, k, 0x9696);
    let x_data: Vec<f32> = lcg(0xa5a5, k);

    let d_gate = gpu.upload_raw(&gate_bytes, &[gate_bytes.len()]).unwrap();
    let d_up = gpu.upload_raw(&up_bytes, &[up_bytes.len()]).unwrap();
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
    let d_yg = gpu.zeros(&[gate_m], DType::F32).unwrap();
    let d_yu = gpu.zeros(&[up_m], DType::F32).unwrap();

    // Baseline: 2 separate gemv_q8_0
    gpu.gemv_q8_0(&d_gate, &d_x, &d_yg, gate_m, k).unwrap();
    gpu.gemv_q8_0(&d_up, &d_x, &d_yu, up_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let ref_gate = gpu.download_f32(&d_yg).unwrap();
    let ref_up = gpu.download_f32(&d_yu).unwrap();

    // Fused
    let d_yg2 = gpu.zeros(&[gate_m], DType::F32).unwrap();
    let d_yu2 = gpu.zeros(&[up_m], DType::F32).unwrap();
    gpu.fused_gate_up_q8_0(&d_gate, &d_up, &d_x, &d_yg2, &d_yu2, gate_m, up_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let fused_gate = gpu.download_f32(&d_yg2).unwrap();
    let fused_up = gpu.download_f32(&d_yu2).unwrap();

    // Correctness
    let maxdiff_gate = ref_gate.iter().zip(&fused_gate)
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let maxdiff_up = ref_up.iter().zip(&fused_up)
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("maxdiff gate={maxdiff_gate:.2e} up={maxdiff_up:.2e}");

    // Time: 2 separate calls
    gpu.gemv_q8_0(&d_gate, &d_x, &d_yg, gate_m, k).unwrap();
    gpu.gemv_q8_0(&d_up, &d_x, &d_yu, up_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.gemv_q8_0(&d_gate, &d_x, &d_yg, gate_m, k).unwrap();
        gpu.gemv_q8_0(&d_up, &d_x, &d_yu, up_m, k).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let sep_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // Time: fused
    gpu.fused_gate_up_q8_0(&d_gate, &d_up, &d_x, &d_yg2, &d_yu2, gate_m, up_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.fused_gate_up_q8_0(&d_gate, &d_up, &d_x, &d_yg2, &d_yu2, gate_m, up_m, k).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let fused_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    eprintln!("2× gemv_q8_0:   {sep_us:.1} µs (separate)");
    eprintln!("fused gate+up:    {fused_us:.1} µs (fused)");
    eprintln!("speedup: {:.2}×", sep_us / fused_us);
}
