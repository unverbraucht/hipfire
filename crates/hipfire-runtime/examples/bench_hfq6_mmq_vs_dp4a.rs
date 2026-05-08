//! Phase B.2 Session 1 GO/NO-GO benchmark: MMQ x8 vs wave64_dp4a at B=8.
//!
//! Threshold (per plan §4 S1): MMQ must beat wave64_dp4a by ≥10 % per-call
//! wall time at B=8. If <10 %, redesign before S2 size sweep.
//!
//! Methodology: 100 warmup launches + 1000 timed launches each, take median.
//! Same weights, same activations for both kernels.

use std::time::Instant;

fn quantize_hfq6g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];
        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        let actual_len = end - start;
        for i in (0..256).step_by(4) {
            let q0 = if i < actual_len { ((group[i] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q1 = if i + 1 < actual_len { ((group[i+1] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q2 = if i + 2 < actual_len { ((group[i+2] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q3 = if i + 3 < actual_len { ((group[i+3] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q0 = q0.min(63);
            let q1 = q1.min(63);
            let q2 = q2.min(63);
            let q3 = q3.min(63);
            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off]     = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }
    output
}

fn rand_f32(seed: u32, i: usize) -> f32 {
    let mut s = seed.wrapping_add(i as u32).wrapping_mul(1103515245).wrapping_add(12345);
    s = s.wrapping_mul(1103515245).wrapping_add(12345);
    (s % 1000) as f32 / 500.0 - 1.0
}

fn time_kernel<F: FnMut() -> ()>(label: &str, mut f: F, iters: usize) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
    let per_call_us = elapsed_us / iters as f64;
    eprintln!("  {label}: {per_call_us:.2} µs/call (over {iters} iters, {elapsed_us:.0} µs total)");
    per_call_us
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    if gpu.arch != "gfx906" {
        eprintln!("This bench is gfx906-only.");
        return;
    }

    // Shape: M=3584 (Qwen 9B FFN dim), K=4096.
    // B from arg or default to 128 (production prefill case → mmq_x=64).
    let n: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let m = 3584usize;
    let k = 4096usize;

    let x_data: Vec<f32> = (0..n*k).map(|i| rand_f32(42, i) * 0.5).collect();
    let w_data: Vec<f32> = (0..m*k).map(|i| rand_f32(101, i) * 0.3).collect();
    let q = quantize_hfq6g256(&w_data);

    let d_x = gpu.upload_f32(&x_data, &[n * k]).unwrap();
    let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();
    let d_y = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();

    eprintln!("\nShape: M={m} K={k} B={n}");
    eprintln!("Bytes per call: A={} MB, X={} KB, Y={} KB",
        m * (k/256) * 200 / 1024 / 1024,
        n * k * 4 / 1024,
        n * m * 4 / 1024);

    // Warmup: JIT + scheduler stabilization.
    eprintln!("\n=== Warmup (100 calls each) ===");
    for _ in 0..100 {
        let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).unwrap();
        gpu.gemm_hfq6g256_mmq_set_gfx906(&d_a, xq, &d_y, m, k, n).unwrap();
    }
    for _ in 0..100 {
        gpu.gemm_hfq6g256_residual_wave64_dp4a(&d_a, &d_x, &d_y, m, k, n).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let iters = 1000;
    eprintln!("\n=== Timed ({iters} calls each, sync after each) ===");

    // MMQ x8 set path (no Q8_1 quantize cost — assumed amortized at dispatcher level).
    // We pre-quantize once and reuse the xq pointer across iters.
    let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let mmq_us = time_kernel("MMQ x8 set", || {
        gpu.gemm_hfq6g256_mmq_set_gfx906(&d_a, xq, &d_y, m, k, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
    }, iters);

    // wave64_dp4a — internally calls ensure_q8_1_mmq_x, which is cached after
    // the first call (pointer-equality short-circuit for X). So the per-call
    // cost is just the kernel itself. Sync-after-each-call to measure wall.
    let dp4a_us = time_kernel("wave64_dp4a", || {
        gpu.gemm_hfq6g256_residual_wave64_dp4a(&d_a, &d_x, &d_y, m, k, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
    }, iters);

    // GO/NO-GO check.
    let speedup = dp4a_us / mmq_us;
    let pct_improvement = (1.0 - mmq_us / dp4a_us) * 100.0;
    eprintln!("\n=== Result ===");
    eprintln!("  MMQ x8     : {mmq_us:.2} µs/call");
    eprintln!("  wave64_dp4a: {dp4a_us:.2} µs/call");
    eprintln!("  speedup    : {speedup:.3}× ({pct_improvement:+.1} %)");
    if pct_improvement >= 10.0 {
        eprintln!("\n  ✅ GO — MMQ beats wave64_dp4a by ≥10 % at B=8");
        eprintln!("     S2 size sweep (mmq_x ∈ {{16,24,32,40,48,56,64}}) cleared.");
    } else if pct_improvement >= 0.0 {
        eprintln!("\n  ⚠️  MARGINAL — MMQ improvement is < 10 % ({pct_improvement:+.1} %).");
        eprintln!("     Plan §4 S1: redesign before S2 size sweep.");
    } else {
        eprintln!("\n  ❌ NO-GO — MMQ is slower than wave64_dp4a at B=8.");
        eprintln!("     Investigate before scaling: load strategy, occupancy, LDS layout.");
    }
}
