//! gfx906-only correctness gate for the new HFP4G32 dp4a paths:
//!
//!   1. `gemm_hfp4g32_residual` → `gemm_hfp4g32_residual_mmq_gfx906`
//!      (prefill MMQ, batch ≥ 8 on gfx906)
//!   2. `fused_qkv_hfp4g32_dp4a` / `fused_gate_up_hfp4g32_dp4a` /
//!      `fused_qkvza_hfp4g32_dp4a` (decode wave64 dp4a paths)
//!
//! Reference: scalar `gemv_hfp4g32` (the v1 correctness anchor from
//! docs/quant-formats/hfp4.md), called once per batch row. Tolerance:
//! NRMSE < 1.5e-2 (matches the HFQ4 MMQ correctness gate at
//! `test_gfx906_mmq_correctness.rs`). The MMQ / wave64 dp4a paths
//! consume Q8_1-quantized activations while the reference consumes
//! FP32; the per-element delta is bounded by Q8_1's row_scale/255
//! quantization noise (~2e-2 absolute on weights at row_scale ~4-6),
//! so a per-element 1e-3 relative tolerance would always fail.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("arch={arch}");
    if !arch.starts_with("gfx906") {
        eprintln!("SKIP — test is gfx906-only (arch={arch})");
        return;
    }

    let mut all_pass = true;
    all_pass &= test_residual_mmq(&mut gpu);
    all_pass &= test_gemv_wave64_dp4a(&mut gpu);
    all_pass &= test_fused_qkv_dp4a(&mut gpu);
    all_pass &= test_fused_gate_up_dp4a(&mut gpu);
    all_pass &= test_fused_qkvza_dp4a(&mut gpu);

    if !all_pass {
        eprintln!("\n=== FAIL ===");
        std::process::exit(1);
    }
    eprintln!("\n=== ALL PASS ===");
}

fn row_bytes_for(k: usize) -> usize {
    16 + (k / 32) * 17
}

fn test_residual_mmq(gpu: &mut Gpu) -> bool {
    // Batch sizes that all hit the MMQ path on gfx906 (should_use_mmq
    // threshold = 8 on gfx906; batch < 8 currently falls through to
    // the gfx11/12 WMMA which fails to JIT on gfx906 — out of scope).
    let n_list = [8usize, 16, 32, 64];
    let m: usize = 512;
    let k: usize = 2048;
    let row_bytes = row_bytes_for(k);

    eprintln!("\n=== gemm_hfp4g32_residual_mmq_gfx906 ===");
    eprintln!("  M={m} K={k}");

    let w = gpu.upload_raw(&synth(m, k, 0xFF), &[m * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = make_x(max_n * k, 0x3333);
    let res_seed: Vec<f32> = (0..max_n * m)
        .map(|i| ((i as i32) % 7 - 3) as f32 * 1e-3)
        .collect();

    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_1 = gpu.alloc_tensor(&[m], DType::F32).unwrap();
    let y_ref_col = gpu.alloc_tensor(&[max_n * m], DType::F32).unwrap();

    let x_gemm = gpu.alloc_tensor(&[max_n * k], DType::F32).unwrap();
    let y_gemm = gpu.alloc_tensor(&[max_n * m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    let mut all_pass = true;
    for &n in &n_list {
        let mut ref_host: Vec<f32> = vec![0.0; n * m];
        let mut gemv_us = 0.0;
        for i in 0..n {
            gpu.hip.memcpy_htod(&x_gemv.buf, bytes_of(&x_host[i * k..(i + 1) * k])).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            gpu.gemv_hfp4g32(&w, &x_gemv, &y_1, m, k).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_us += t.elapsed().as_secs_f64() * 1e6;
            let y_1_host = gpu.download_f32(&y_1).unwrap();
            for r in 0..m {
                ref_host[i * m + r] = res_seed[i * m + r] + y_1_host[r];
            }
        }
        gpu.hip.memcpy_htod(&y_ref_col.buf, bytes_of(&ref_host)).unwrap();
        // Warm-up call to amortize hipcc JIT for this mmq_x variant.
        gpu.hip.memcpy_htod(&y_gemm.buf, bytes_of(&res_seed[..n * m])).unwrap();
        gpu.gemm_hfp4g32_residual(&w, &x_gemm, &y_gemm, m, k, n).unwrap();
        gpu.hip.device_synchronize().unwrap();

        gpu.hip.memcpy_htod(&y_gemm.buf, bytes_of(&res_seed[..n * m])).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_hfp4g32_residual(&w, &x_gemm, &y_gemm, m, k, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_us = t.elapsed().as_secs_f64() * 1e6;

        let ok = cmp_tol(gpu, &y_ref_col, &y_gemm, n, m, "res");
        eprintln!(
            "  N={n:3}  gemv×N: {:8.1} µs   mmq×1: {:8.1} µs   speedup: {:5.2}x   [{}]",
            gemv_us, gemm_us, gemv_us / gemm_us,
            if ok { "PASS" } else { "FAIL" }
        );
        all_pass &= ok;
    }
    all_pass
}

fn test_gemv_wave64_dp4a(gpu: &mut Gpu) -> bool {
    // Two shape regimes:
    //   (A) lm_head: M=248320 K=4096 — Qwen3.5-9B vocab × dim; the
    //       single biggest GEMV in decode. Q8_1 prequant is amortized
    //       across all 248k output rows.
    //   (B) Small projection: M=4096 K=1024. Q8_1 prequant cost is
    //       a larger share of the total; surfaces whether the dp4a
    //       speedup holds when amortization is weak.
    let shapes: &[(usize, usize, &str)] = &[
        (248320, 4096, "lm_head"),
        (  4096, 1024, "small"),
    ];

    eprintln!("\n=== gemv_hfp4g32_wave64_dp4a ===");

    let mut all_pass = true;
    for &(m, k, label) in shapes {
        let row_bytes = row_bytes_for(k);
        eprintln!("  [{label}]  M={m} K={k}");

        let w = gpu.upload_raw(&synth(m, k, 0x77 ^ (m as u64)), &[m * row_bytes]).unwrap();

        let x_host: Vec<f32> = make_x(k, 0x7777);
        let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();

        let y_ref = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let y_kernel = gpu.alloc_tensor(&[m], DType::F32).unwrap();

        gpu.gemv_hfp4g32_fallback(&w, &x, &y_ref, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();

        // Warm both kernels (JIT + Q8_1 prequant cache).
        gpu.gemv_hfp4g32_wave64_dp4a(&w, &x, &y_kernel, m, k).unwrap();
        gpu.gemv_hfp4g32_fallback(&w, &x, &y_ref, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();

        // Median of 5 timed runs each.
        let mut dp4a_runs = Vec::with_capacity(5);
        let mut fb_runs = Vec::with_capacity(5);
        for _ in 0..5 {
            let t = Instant::now();
            gpu.gemv_hfp4g32_wave64_dp4a(&w, &x, &y_kernel, m, k).unwrap();
            gpu.hip.device_synchronize().unwrap();
            dp4a_runs.push(t.elapsed().as_secs_f64() * 1e6);

            let t = Instant::now();
            gpu.gemv_hfp4g32_fallback(&w, &x, &y_ref, m, k).unwrap();
            gpu.hip.device_synchronize().unwrap();
            fb_runs.push(t.elapsed().as_secs_f64() * 1e6);
        }
        dp4a_runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        fb_runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let dp4a_us = dp4a_runs[2];
        let fb_us = fb_runs[2];

        let ok = cmp_tol(gpu, &y_ref, &y_kernel, 1, m, "gemv");
        eprintln!(
            "  [{label}]  fallback: {:8.1} µs   dp4a: {:8.1} µs   speedup: {:5.2}x   [{}]",
            fb_us, dp4a_us, fb_us / dp4a_us,
            if ok { "PASS" } else { "FAIL" }
        );
        all_pass &= ok;
    }
    all_pass
}

fn test_fused_qkv_dp4a(gpu: &mut Gpu) -> bool {
    let q_m: usize = 2048;
    let k_m: usize = 512;
    let v_m: usize = 512;
    let k: usize = 1024;
    let row_bytes = row_bytes_for(k);

    eprintln!("\n=== fused_qkv_hfp4g32_dp4a (decode, batch=1) ===");
    eprintln!("  q_m={q_m} k_m={k_m} v_m={v_m} K={k}");

    let w_q = gpu.upload_raw(&synth(q_m, k, 0xAA), &[q_m * row_bytes]).unwrap();
    let w_k = gpu.upload_raw(&synth(k_m, k, 0xBB), &[k_m * row_bytes]).unwrap();
    let w_v = gpu.upload_raw(&synth(v_m, k, 0xCC), &[v_m * row_bytes]).unwrap();

    let x_host: Vec<f32> = make_x(k, 0x1111);
    let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();

    let y_q_ref = gpu.alloc_tensor(&[q_m], DType::F32).unwrap();
    let y_k_ref = gpu.alloc_tensor(&[k_m], DType::F32).unwrap();
    let y_v_ref = gpu.alloc_tensor(&[v_m], DType::F32).unwrap();
    let y_q = gpu.alloc_tensor(&[q_m], DType::F32).unwrap();
    let y_k = gpu.alloc_tensor(&[k_m], DType::F32).unwrap();
    let y_v = gpu.alloc_tensor(&[v_m], DType::F32).unwrap();

    gpu.gemv_hfp4g32(&w_q, &x, &y_q_ref, q_m, k).unwrap();
    gpu.gemv_hfp4g32(&w_k, &x, &y_k_ref, k_m, k).unwrap();
    gpu.gemv_hfp4g32(&w_v, &x, &y_v_ref, v_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Warm-up to amortize JIT.
    gpu.fused_qkv_hfp4g32_dp4a(&w_q, &w_k, &w_v, &x, &y_q, &y_k, &y_v, q_m, k_m, v_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let t = Instant::now();
    gpu.fused_qkv_hfp4g32_dp4a(&w_q, &w_k, &w_v, &x, &y_q, &y_k, &y_v, q_m, k_m, v_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let dp4a_us = t.elapsed().as_secs_f64() * 1e6;

    let ok_q = cmp_tol(gpu, &y_q_ref, &y_q, 1, q_m, "q");
    let ok_k = cmp_tol(gpu, &y_k_ref, &y_k, 1, k_m, "k");
    let ok_v = cmp_tol(gpu, &y_v_ref, &y_v, 1, v_m, "v");
    let ok = ok_q && ok_k && ok_v;
    eprintln!("  dp4a: {:7.1} µs   [{}]", dp4a_us, if ok { "PASS" } else { "FAIL" });
    ok
}

fn test_fused_gate_up_dp4a(gpu: &mut Gpu) -> bool {
    let gate_m: usize = 4096;
    let up_m: usize = 4096;
    let k: usize = 1024;
    let row_bytes = row_bytes_for(k);

    eprintln!("\n=== fused_gate_up_hfp4g32_dp4a (decode, batch=1) ===");
    eprintln!("  gate_m={gate_m} up_m={up_m} K={k}");

    let w_g = gpu.upload_raw(&synth(gate_m, k, 0xDD), &[gate_m * row_bytes]).unwrap();
    let w_u = gpu.upload_raw(&synth(up_m, k, 0xEE), &[up_m * row_bytes]).unwrap();

    let x_host: Vec<f32> = make_x(k, 0x2222);
    let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();

    let y_g_ref = gpu.alloc_tensor(&[gate_m], DType::F32).unwrap();
    let y_u_ref = gpu.alloc_tensor(&[up_m], DType::F32).unwrap();
    let y_g = gpu.alloc_tensor(&[gate_m], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[up_m], DType::F32).unwrap();

    gpu.gemv_hfp4g32(&w_g, &x, &y_g_ref, gate_m, k).unwrap();
    gpu.gemv_hfp4g32(&w_u, &x, &y_u_ref, up_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();

    gpu.fused_gate_up_hfp4g32_dp4a(&w_g, &w_u, &x, &y_g, &y_u, gate_m, up_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let t = Instant::now();
    gpu.fused_gate_up_hfp4g32_dp4a(&w_g, &w_u, &x, &y_g, &y_u, gate_m, up_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let dp4a_us = t.elapsed().as_secs_f64() * 1e6;

    let ok_g = cmp_tol(gpu, &y_g_ref, &y_g, 1, gate_m, "gate");
    let ok_u = cmp_tol(gpu, &y_u_ref, &y_u, 1, up_m, "up");
    let ok = ok_g && ok_u;
    eprintln!("  dp4a: {:7.1} µs   [{}]", dp4a_us, if ok { "PASS" } else { "FAIL" });
    ok
}

fn test_fused_qkvza_dp4a(gpu: &mut Gpu) -> bool {
    let qkv_m: usize = 2048;
    let z_m: usize = 512;
    let beta_m: usize = 128;
    let alpha_m: usize = 64;
    let k: usize = 1024;
    let row_bytes = row_bytes_for(k);

    eprintln!("\n=== fused_qkvza_hfp4g32_dp4a (decode, batch=1) ===");
    eprintln!("  qkv_m={qkv_m} z_m={z_m} beta_m={beta_m} alpha_m={alpha_m} K={k}");

    let w_qkv   = gpu.upload_raw(&synth(qkv_m,   k, 0x11), &[qkv_m   * row_bytes]).unwrap();
    let w_z     = gpu.upload_raw(&synth(z_m,     k, 0x22), &[z_m     * row_bytes]).unwrap();
    let w_beta  = gpu.upload_raw(&synth(beta_m,  k, 0x33), &[beta_m  * row_bytes]).unwrap();
    let w_alpha = gpu.upload_raw(&synth(alpha_m, k, 0x44), &[alpha_m * row_bytes]).unwrap();

    let x_host: Vec<f32> = make_x(k, 0x5555);
    let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();

    let y_qkv_ref   = gpu.alloc_tensor(&[qkv_m],   DType::F32).unwrap();
    let y_z_ref     = gpu.alloc_tensor(&[z_m],     DType::F32).unwrap();
    let y_beta_ref  = gpu.alloc_tensor(&[beta_m],  DType::F32).unwrap();
    let y_alpha_ref = gpu.alloc_tensor(&[alpha_m], DType::F32).unwrap();
    let y_qkv   = gpu.alloc_tensor(&[qkv_m],   DType::F32).unwrap();
    let y_z     = gpu.alloc_tensor(&[z_m],     DType::F32).unwrap();
    let y_beta  = gpu.alloc_tensor(&[beta_m],  DType::F32).unwrap();
    let y_alpha = gpu.alloc_tensor(&[alpha_m], DType::F32).unwrap();

    gpu.gemv_hfp4g32(&w_qkv,   &x, &y_qkv_ref,   qkv_m,   k).unwrap();
    gpu.gemv_hfp4g32(&w_z,     &x, &y_z_ref,     z_m,     k).unwrap();
    gpu.gemv_hfp4g32(&w_beta,  &x, &y_beta_ref,  beta_m,  k).unwrap();
    gpu.gemv_hfp4g32(&w_alpha, &x, &y_alpha_ref, alpha_m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();

    gpu.fused_qkvza_hfp4g32_dp4a(
        &w_qkv, &w_z, &w_beta, &w_alpha, &x,
        &y_qkv, &y_z, &y_beta, &y_alpha,
        qkv_m, z_m, beta_m, alpha_m, k,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let t = Instant::now();
    gpu.fused_qkvza_hfp4g32_dp4a(
        &w_qkv, &w_z, &w_beta, &w_alpha, &x,
        &y_qkv, &y_z, &y_beta, &y_alpha,
        qkv_m, z_m, beta_m, alpha_m, k,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let dp4a_us = t.elapsed().as_secs_f64() * 1e6;

    let ok_qkv   = cmp_tol(gpu, &y_qkv_ref,   &y_qkv,   1, qkv_m,   "qkv");
    let ok_z     = cmp_tol(gpu, &y_z_ref,     &y_z,     1, z_m,     "z");
    let ok_beta  = cmp_tol(gpu, &y_beta_ref,  &y_beta,  1, beta_m,  "beta");
    let ok_alpha = cmp_tol(gpu, &y_alpha_ref, &y_alpha, 1, alpha_m, "alpha");
    let ok = ok_qkv && ok_z && ok_beta && ok_alpha;
    eprintln!("  dp4a: {:7.1} µs   [{}]", dp4a_us, if ok { "PASS" } else { "FAIL" });
    ok
}

fn cmp_tol(gpu: &mut Gpu, y_ref: &GpuTensor, y_kernel: &GpuTensor, n: usize, m: usize, label: &str) -> bool {
    let r = gpu.download_f32(y_ref).unwrap();
    let k = gpu.download_f32(y_kernel).unwrap();

    let mut max_abs: f64 = 0.0;
    let mut max_abs_ref: f64 = 0.0;
    let mut rms_err: f64 = 0.0;
    let mut rms_ref: f64 = 0.0;

    for b in 0..n {
        for row in 0..m {
            let r_v = r[b * m + row] as f64;
            let k_v = k[b * m + row] as f64;
            let abs = (r_v - k_v).abs();
            if abs > max_abs { max_abs = abs; }
            if r_v.abs() > max_abs_ref { max_abs_ref = r_v.abs(); }
            rms_err += abs * abs;
            rms_ref += r_v * r_v;
        }
    }
    let total = (n * m) as f64;
    rms_err = (rms_err / total).sqrt();
    rms_ref = (rms_ref / total).sqrt();
    let nrmse = if rms_ref > 1e-12 { rms_err / rms_ref } else { rms_err };

    // 1.5e-2 NRMSE tolerance — matches the HFQ4 MMQ correctness gate
    // floor at test_gfx906_mmq_correctness.rs. The Q8_1 activation
    // quantize introduces noise of order weight_row_scale / 255.
    let ok = nrmse < 1.5e-2;
    let status = if ok { "PASS" } else { "FAIL" };
    eprintln!(
        "  {label}: {status}  NRMSE={:.3e}  max_abs={:.3e}  max_|y_ref|={:.3e}",
        nrmse, max_abs, max_abs_ref
    );
    ok
}

fn make_x(n: usize, seed: i64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as i64).wrapping_mul(seed.wrapping_add(0x91c2_a73d)).wrapping_add(seed) & 0xFFFFFF) as f32 * 1e-7 - 0.5)
        .collect()
}

fn synth(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        let row_off = row * row_bytes;
        let rs_f32 = 0.02f32 + ((next() & 0xFF) as f32) * 1e-4;
        let rs_f16 = f32_to_f16_bits(rs_f32);
        out[row_off..row_off + 2].copy_from_slice(&rs_f16.to_le_bytes());
        let bc = blocks_per_row as u16;
        out[row_off + 4..row_off + 6].copy_from_slice(&bc.to_le_bytes());
        for b in 0..blocks_per_row {
            let bp = row_off + 16 + b * 17;
            let e = 120 + (next() & 0x7) as u8;
            out[bp] = e;
            for i in 0..16 {
                out[bp + 1 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0 { return sign; }
    if exp >= 143 { return sign | 0x7C00; }
    if exp <= 112 { return sign; }
    let new_exp = (exp - 127 + 15) as u16;
    let new_mant = (mant >> 13) as u16;
    sign | (new_exp << 10) | new_mant
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
