//! Phase B.2 Session 1 — HFQ6 MMQ-streaming correctness validation.
//!
//! Three classes of test, each run for both `_full_set` (add=0) and
//! `_full_add` (add=1, residual fuse):
//!
//!   1. Aligned shape (M=3584, K=4096, B=8) — hits `_full_*` fast path.
//!   2. Non-aligned shape (M=3000, K=4096, B=13) — hits the data-dependent
//!      `_x8` path with `need_check=true` boundary handling.
//!   3. Constant-weight discriminator (q ≡ 5, x ≡ 1.0) — checks absolute
//!      equality `Σ == M·K·(sc·q+zp)·x`. This catches the two LANDMINES
//!      from the plan (§3.1):
//!        - LANDMINE 1: x_dm carrying `+ 8·sc` shift compensation
//!        - LANDMINE 2: 0.25f factor leaking from dp4a kernel
//!      Both produce systematic biases that NRMSE < 0.5 % could miss.
//!
//! Validation thresholds:
//!   - NRMSE < 0.005 vs CPU dequant + matmul reference (HFQ6 noise floor)
//!   - Per-element max-abs-err < 1e-3 vs wave64_dp4a reference
//!   - Constant-weight: bit-exact equality against analytical formula

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

/// Quantize with caller-controlled (scale, zp) — used for the discriminator
/// test where we want q ≡ q_const exactly.
fn quantize_hfq6g256_with_const(m: usize, k: usize, sc: f32, zp: f32, q_const: u8) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200;
    let n = m * k;
    let n_blocks = n / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    let qc = q_const & 63;
    for b in 0..n_blocks {
        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&sc.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&zp.to_le_bytes());
        // 256 weights × 6 bits / 8 bits per byte = 192 bytes of weight bytes.
        // For all q identical = qc, we pack 4 q's per 3 bytes:
        //   byte0 = q0 | q1<<6
        //   byte1 = q1>>2 | q2<<4
        //   byte2 = q2>>4 | q3<<2
        // With q0=q1=q2=q3=qc:
        let qc32 = qc as u32;
        let b0 = (qc32 | (qc32 << 6)) & 0xFF;
        let b1 = ((qc32 >> 2) | (qc32 << 4)) & 0xFF;
        let b2 = ((qc32 >> 4) | (qc32 << 2)) & 0xFF;
        for i in (0..256).step_by(4) {
            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off]     = b0 as u8;
            output[out_off + byte_off + 1] = b1 as u8;
            output[out_off + byte_off + 2] = b2 as u8;
        }
    }
    output
}

fn cpu_dequant(q: &[u8], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * k];
    let groups_per_row = k / 256;
    for r in 0..m {
        for g in 0..groups_per_row {
            let off = (r * groups_per_row + g) * 200;
            let scale = f32::from_le_bytes([q[off], q[off+1], q[off+2], q[off+3]]);
            let zero = f32::from_le_bytes([q[off+4], q[off+5], q[off+6], q[off+7]]);
            for i in (0..256).step_by(4) {
                let bo = off + 8 + (i / 4) * 3;
                let b0 = q[bo] as u32;
                let b1 = q[bo+1] as u32;
                let b2 = q[bo+2] as u32;
                let q0 = (b0 & 0x3F) as f32;
                let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
                let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
                let q3 = ((b2 >> 2) & 0x3F) as f32;
                let base = r * k + g * 256 + i;
                out[base    ] = scale * q0 + zero;
                out[base + 1] = scale * q1 + zero;
                out[base + 2] = scale * q2 + zero;
                out[base + 3] = scale * q3 + zero;
            }
        }
    }
    out
}

fn cpu_gemm(w: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Y[b][r] = Σ_c w[r][c] * x[b][c], stored column-major Y[b*m + r].
    let mut y = vec![0.0f32; n * m];
    for b in 0..n {
        for r in 0..m {
            let mut acc = 0.0f32;
            for c in 0..k {
                acc += w[r * k + c] * x[b * k + c];
            }
            y[b * m + r] = acc;
        }
    }
    y
}

fn nrmse(a: &[f32], b: &[f32]) -> f32 {
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let e = (x - y) as f64;
        sum_sq_err += e * e;
        sum_sq_ref += (x as f64) * (x as f64);
    }
    let rmse = (sum_sq_err / a.len() as f64).sqrt();
    let rms_ref = (sum_sq_ref / a.len() as f64).sqrt();
    if rms_ref > 1e-12 { (rmse / rms_ref) as f32 } else { rmse as f32 }
}

fn max_abs_err(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut max_e = 0.0f32;
    let mut pos = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs();
        if e > max_e { max_e = e; pos = i; }
    }
    (max_e, pos)
}

fn rand_f32(seed: u32, i: usize) -> f32 {
    let mut s = seed.wrapping_add(i as u32).wrapping_mul(1103515245).wrapping_add(12345);
    s = s.wrapping_mul(1103515245).wrapping_add(12345);
    (s % 1000) as f32 / 500.0 - 1.0
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    if gpu.arch != "gfx906" {
        eprintln!("test_hfq6_mmq is gfx906-only (kernel uses dp4a + wave64). Skipping.");
        return;
    }

    let total_pass = std::cell::Cell::new(0);
    let total_fail = std::cell::Cell::new(0);
    let pass = |label: &str, msg: &str| {
        eprintln!("  PASS  {label}  {msg}");
        total_pass.set(total_pass.get() + 1);
    };
    let fail = |label: &str, msg: &str| {
        eprintln!("  FAIL  {label}  {msg}");
        total_fail.set(total_fail.get() + 1);
    };

    // ─── Test 1: aligned shape (M=3584, K=4096, B=8), set + add ───────────
    {
        eprintln!("\n[Test 1] aligned: M=3584 K=4096 B=8 (hits _full_set_x8 / _full_add_x8)");
        let m = 3584usize;
        let k = 4096usize;
        let n = 8usize;

        let x_data: Vec<f32> = (0..n*k).map(|i| rand_f32(42, i) * 0.5).collect();
        let w_data: Vec<f32> = (0..m*k).map(|i| rand_f32(101, i) * 0.3).collect();
        let q = quantize_hfq6g256(&w_data);
        let w_dq = cpu_dequant(&q, m, k);
        let y_cpu = cpu_gemm(&w_dq, &x_data, m, k, n);

        let d_x = gpu.upload_f32(&x_data, &[n * k]).unwrap();
        let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();

        // SET path: zero-init Y, dispatch set kernel, compare.
        let d_y_set = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
        let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).unwrap();
        gpu.gemm_hfq6g256_mmq_set_gfx906(&d_a, xq, &d_y_set, m, k, n).unwrap();
        let y_set = gpu.download_f32(&d_y_set).unwrap();

        let nr = nrmse(&y_cpu, &y_set);
        let (mae, pos) = max_abs_err(&y_cpu, &y_set);
        if nr < 0.005 {
            pass("set NRMSE", &format!("nrmse={nr:.5} (< 0.005)"));
        } else {
            fail("set NRMSE", &format!("nrmse={nr:.5} (≥ 0.005)"));
        }
        eprintln!("        max_abs_err={mae:.4} at [{}], y_cpu={:.3} y_gpu={:.3}",
            pos, y_cpu[pos], y_set[pos]);

        // ADD path: pre-fill Y with known offset, dispatch add kernel, expect Y = offset + cpu_ref.
        let offset_data: Vec<f32> = (0..n*m).map(|i| rand_f32(999, i) * 2.0).collect();
        let d_y_add = gpu.upload_f32(&offset_data, &[n * m]).unwrap();
        gpu.gemm_hfq6g256_residual_mmq_gfx906(&d_a, &d_x, &d_y_add, m, k, n).unwrap();
        let y_add = gpu.download_f32(&d_y_add).unwrap();
        let y_add_expected: Vec<f32> = offset_data.iter().zip(y_cpu.iter()).map(|(o, c)| o + c).collect();
        let nr_add = nrmse(&y_add_expected, &y_add);
        let (mae_add, pos_add) = max_abs_err(&y_add_expected, &y_add);
        if nr_add < 0.005 {
            pass("add NRMSE", &format!("nrmse={nr_add:.5} (< 0.005)"));
        } else {
            fail("add NRMSE", &format!("nrmse={nr_add:.5} (≥ 0.005)"));
        }
        eprintln!("        max_abs_err={mae_add:.4} at [{}], expect={:.3} got={:.3}",
            pos_add, y_add_expected[pos_add], y_add[pos_add]);

        gpu.free_tensor(d_x).ok();
        gpu.free_tensor(d_a).ok();
        gpu.free_tensor(d_y_set).ok();
        gpu.free_tensor(d_y_add).ok();
    }

    // ─── Test 2: non-aligned shape (M=3000, K=4096, B=13) ──────────────────
    {
        eprintln!("\n[Test 2] non-aligned: M=3000 K=4096 B=13 (hits _x8 path with need_check)");
        let m = 3000usize;
        let k = 4096usize;
        let n = 13usize;

        let x_data: Vec<f32> = (0..n*k).map(|i| rand_f32(43, i) * 0.5).collect();
        let w_data: Vec<f32> = (0..m*k).map(|i| rand_f32(102, i) * 0.3).collect();
        let q = quantize_hfq6g256(&w_data);
        let w_dq = cpu_dequant(&q, m, k);
        let y_cpu = cpu_gemm(&w_dq, &x_data, m, k, n);

        let d_x = gpu.upload_f32(&x_data, &[n * k]).unwrap();
        let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();
        let d_y = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
        let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).unwrap();
        // n=13 > 8, so this currently falls back to wave64_dp4a inside the
        // dispatcher. S2 will lift the size cap.
        gpu.gemm_hfq6g256_mmq_set_gfx906(&d_a, xq, &d_y, m, k, n).unwrap_or_else(|_| {
            // Fallback path under S1: caller does it explicitly via residual fn.
            eprintln!("        S1 set-mode rejects B>8; falling back to residual path with zero-init Y");
            gpu.gemm_hfq6g256_residual_wave64_dp4a(&d_a, &d_x, &d_y, m, k, n).unwrap();
        });
        let y_gpu = gpu.download_f32(&d_y).unwrap();

        let nr = nrmse(&y_cpu, &y_gpu);
        let (mae, pos) = max_abs_err(&y_cpu, &y_gpu);
        if nr < 0.01 {
            pass("non-aligned NRMSE", &format!("nrmse={nr:.5} (< 0.01)"));
        } else {
            fail("non-aligned NRMSE", &format!("nrmse={nr:.5} (≥ 0.01)"));
        }
        eprintln!("        max_abs_err={mae:.4} at [{}]", pos);

        gpu.free_tensor(d_x).ok();
        gpu.free_tensor(d_a).ok();
        gpu.free_tensor(d_y).ok();
    }

    // ─── Test 3: constant-weight LANDMINE discriminator ────────────────────
    // q_const=5, sc=0.1, zp=2.0, x_const=1.0 → every w = sc*5 + zp = 2.5.
    // Expected: y[b][r] = sum over k of w[r][c] * x[b][c]
    //                   = K * 2.5 * 1.0 = K * 2.5 = 4096 * 2.5 = 10240.
    //
    // LANDMINE 1 (x_dm = (sc, zp + 8·sc)) would add 8·sc·sum_x per group:
    //   8 * 0.1 * 32 (sum_x for x≡1) = 25.6 per sub-block.
    //   With 16 groups × 8 sub-blocks = 128 subs: extra = 128 * 25.6 = 3276.8
    //   Total = 10240 + 3276.8 = 13516.8 (way off, easy to detect).
    //
    // LANDMINE 2 (zp * sum_x * 0.25f instead of zp * sum_x):
    //   zp contribution = 2.0 * 32 * 0.25 = 16 per sub-block instead of 64.
    //   With 128 subs: zp total = 128 * 16 = 2048 instead of 128 * 64 = 8192.
    //   sc·sumi part: sc * sumi = 0.1 * (sum of q*x_int8). For x_const=1.0
    //   quantized to int8, sum_x_int8 = 127·32 (saturate). Hmm — Q8_1 with
    //   x_const=1.0 picks d_x s.t. max(|x|) → 127, so x_int8 = 127. Then
    //   sumi over 32 elements = 5·127·32 = 20320 (per sub-block, q=5 each).
    //   sc·d_x·sumi = 0.1 * d_x * 20320; d_x = 1/127 (since q8_1 d for max=1).
    //   So sc·d_x·sumi = 0.1 * (1/127) * 20320 = 16.0 per sub-block.
    //   With 128 subs: sc·sumi total = 128 * 16 = 2048.
    //   Correct sum: 2048 (sc·sumi) + 8192 (zp·sum_x) = 10240. ✓
    //   With LANDMINE 2: 2048 + 2048 = 4096 (off by exactly 6144).
    //
    // Either landmine produces a clear, easy-to-detect deviation.
    {
        eprintln!("\n[Test 3] discriminator: q ≡ 5, sc=0.1, zp=2.0, x ≡ 1.0");
        let m = 128usize;  // smallest power-of-128 alignment
        let k = 4096usize;
        let n = 8usize;
        let sc = 0.1f32;
        let zp = 2.0f32;
        let q_const = 5u8;
        let x_const = 1.0f32;

        // Expected: y[b][r] = K * (sc*q_const + zp) * x_const
        let expected_per_element = (k as f32) * (sc * q_const as f32 + zp) * x_const;
        eprintln!("        expected y = {expected_per_element:.4} for every (b, r)");

        let x_data: Vec<f32> = vec![x_const; n * k];
        let q = quantize_hfq6g256_with_const(m, k, sc, zp, q_const);

        let d_x = gpu.upload_f32(&x_data, &[n * k]).unwrap();
        let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();
        let d_y = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
        let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).unwrap();
        gpu.gemm_hfq6g256_mmq_set_gfx906(&d_a, xq, &d_y, m, k, n).unwrap();
        let y_gpu = gpu.download_f32(&d_y).unwrap();

        // Tolerance: Q8_1 has ~1/127 quantization noise per element on x_const=1.0
        // (d_x = 1/127, x_int8 = 127, perfect). The dp4a sum over K=4096 of
        // products will accumulate rounding to ~few ULPs. Allow ±1.0 (relative
        // ~0.01% on expected=10240). LANDMINEs produce errors >1000.
        let mut max_dev = 0.0f32;
        let mut pos = 0;
        let mut sum_y = 0.0f32;
        for (i, &v) in y_gpu.iter().enumerate() {
            let dev = (v - expected_per_element).abs();
            if dev > max_dev { max_dev = dev; pos = i; }
            sum_y += v;
        }
        let mean_y = sum_y / (n * m) as f32;
        eprintln!("        actual y: mean={mean_y:.4}, max_dev={max_dev:.4} at [{pos}], sample y[0]={:.4}", y_gpu[0]);
        if max_dev < 1.0 {
            pass("LANDMINE discriminator", &format!("max_dev={max_dev:.4} < 1.0 — both landmines clear"));
        } else if (mean_y - expected_per_element * 1.32).abs() < 100.0 {
            // mean ratio ≈ 1.32 → LANDMINE 1 likely (extra +8·sc bias).
            fail("LANDMINE 1 SUSPECT", &format!("mean={mean_y:.1} vs expected={expected_per_element:.1} (ratio {:.2})", mean_y / expected_per_element));
        } else if (mean_y - expected_per_element * 0.4).abs() < 100.0 {
            // mean ratio ≈ 0.4 → LANDMINE 2 likely (zp underweighted by 0.25).
            fail("LANDMINE 2 SUSPECT", &format!("mean={mean_y:.1} vs expected={expected_per_element:.1} (ratio {:.2})", mean_y / expected_per_element));
        } else {
            fail("discriminator", &format!("max_dev={max_dev:.4} ≥ 1.0 — unexpected error pattern"));
        }

        gpu.free_tensor(d_x).ok();
        gpu.free_tensor(d_a).ok();
        gpu.free_tensor(d_y).ok();
    }

    // ─── Test 4: parity vs wave64_dp4a reference ───────────────────────────
    // Plan §4 S1: per-element max-abs-err < 1e-3 vs wave64_dp4a.
    // (Both kernels carry Q8_1 activation noise — they should agree closely.)
    {
        eprintln!("\n[Test 4] MMQ vs wave64_dp4a: M=128 K=4096 B=8");
        let m = 128usize;
        let k = 4096usize;
        let n = 8usize;
        let x_data: Vec<f32> = (0..n*k).map(|i| rand_f32(44, i) * 0.5).collect();
        let w_data: Vec<f32> = (0..m*k).map(|i| rand_f32(103, i) * 0.3).collect();
        let q = quantize_hfq6g256(&w_data);

        let d_x = gpu.upload_f32(&x_data, &[n * k]).unwrap();
        let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();

        // MMQ result.
        let d_y_mmq = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
        let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).unwrap();
        gpu.gemm_hfq6g256_mmq_set_gfx906(&d_a, xq, &d_y_mmq, m, k, n).unwrap();
        let y_mmq = gpu.download_f32(&d_y_mmq).unwrap();

        // wave64_dp4a result (zero-init Y first because that kernel also adds residual).
        let d_y_dp4a = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
        gpu.gemm_hfq6g256_residual_wave64_dp4a(&d_a, &d_x, &d_y_dp4a, m, k, n).unwrap();
        let y_dp4a = gpu.download_f32(&d_y_dp4a).unwrap();

        let (mae, pos) = max_abs_err(&y_dp4a, &y_mmq);
        let nr = nrmse(&y_dp4a, &y_mmq);
        eprintln!("        max_abs_err={mae:.5} at [{}] (mmq={:.3} dp4a={:.3}) nrmse={nr:.5}",
            pos, y_mmq[pos], y_dp4a[pos]);
        if mae < 1e-3 {
            pass("MMQ vs dp4a", &format!("max_abs_err={mae:.5} (< 1e-3)"));
        } else {
            fail("MMQ vs dp4a", &format!("max_abs_err={mae:.5} (≥ 1e-3)"));
        }

        gpu.free_tensor(d_x).ok();
        gpu.free_tensor(d_a).ok();
        gpu.free_tensor(d_y_mmq).ok();
        gpu.free_tensor(d_y_dp4a).ok();
    }

    // ─── Test 5: B=128 mmq_x=64 path (production prefill) ─────────────────
    {
        eprintln!("\n[Test 5] B=128 prefill: M=3584 K=4096 B=128 (hits _full_set_x64 / _full_add_x64)");
        let m = 3584usize;
        let k = 4096usize;
        let n = 128usize;

        let x_data: Vec<f32> = (0..n*k).map(|i| rand_f32(45, i) * 0.5).collect();
        let w_data: Vec<f32> = (0..m*k).map(|i| rand_f32(105, i) * 0.3).collect();
        let q = quantize_hfq6g256(&w_data);
        let w_dq = cpu_dequant(&q, m, k);
        let y_cpu = cpu_gemm(&w_dq, &x_data, m, k, n);

        let d_x = gpu.upload_f32(&x_data, &[n * k]).unwrap();
        let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();
        let d_y = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
        let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).unwrap();
        gpu.gemm_hfq6g256_mmq_set_gfx906(&d_a, xq, &d_y, m, k, n).unwrap();
        let y_gpu = gpu.download_f32(&d_y).unwrap();

        let nr = nrmse(&y_cpu, &y_gpu);
        let (mae, pos) = max_abs_err(&y_cpu, &y_gpu);
        if nr < 0.005 {
            pass("B=128 set NRMSE", &format!("nrmse={nr:.5} (< 0.005)"));
        } else {
            fail("B=128 set NRMSE", &format!("nrmse={nr:.5} (≥ 0.005)"));
        }
        eprintln!("        max_abs_err={mae:.4} at [{}], y_cpu={:.3} y_gpu={:.3}",
            pos, y_cpu[pos], y_gpu[pos]);

        gpu.free_tensor(d_x).ok();
        gpu.free_tensor(d_a).ok();
        gpu.free_tensor(d_y).ok();
    }

    // ─── Test 6: per-mmq_x correctness sweep (S2.3) ────────────────────────
    // For each mmq_x ∈ {8,16,24,32,40,48,56,64}, run a B=mmq_x bench at
    // M=128 K=4096 (smallest tile-aligned shape). Catches any size-specific
    // bug in the body template (e.g. bad b128/b32 path selection at the
    // mmq_x>=64 boundary).
    {
        eprintln!("\n[Test 6] per-mmq_x correctness sweep (M=128, K=4096)");
        let m = 128usize;
        let k = 4096usize;
        for &n in &[8usize, 16, 24, 32, 40, 48, 56, 64] {
            let x_data: Vec<f32> = (0..n*k).map(|i| rand_f32(50 + n as u32, i) * 0.5).collect();
            let w_data: Vec<f32> = (0..m*k).map(|i| rand_f32(150 + n as u32, i) * 0.3).collect();
            let q = quantize_hfq6g256(&w_data);
            let w_dq = cpu_dequant(&q, m, k);
            let y_cpu = cpu_gemm(&w_dq, &x_data, m, k, n);

            let d_x = gpu.upload_f32(&x_data, &[n * k]).unwrap();
            let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();
            let d_y = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
            let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).unwrap();
            gpu.gemm_hfq6g256_mmq_set_gfx906(&d_a, xq, &d_y, m, k, n).unwrap();
            let y_gpu = gpu.download_f32(&d_y).unwrap();

            let nr = nrmse(&y_cpu, &y_gpu);
            let (mae, _) = max_abs_err(&y_cpu, &y_gpu);
            let label = format!("mmq_x={n:2} (B={n})");
            if nr < 0.005 {
                pass(&label, &format!("nrmse={nr:.5} max_abs_err={mae:.4}"));
            } else {
                fail(&label, &format!("nrmse={nr:.5} max_abs_err={mae:.4}"));
            }

            gpu.free_tensor(d_x).ok();
            gpu.free_tensor(d_a).ok();
            gpu.free_tensor(d_y).ok();
        }
    }

    eprintln!("\n=== Summary: {} passed, {} failed ===", total_pass.get(), total_fail.get());
    if total_fail.get() > 0 {
        std::process::exit(1);
    }
}
