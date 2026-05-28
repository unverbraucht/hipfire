//! v6 NaN diagnostic: dump per-head, per-q_tile output to localize the issue.
//!
//! Usage: cargo run --release --example test_v6_diag

use rdna_compute::{DType, Gpu};

fn run_with_data(gpu: &mut Gpu, q: &[f32], k: &[f32], v: &[f32],
                 b: usize, l: usize, n_heads: usize, hd: usize, label: &str) {
    let total_q = b * n_heads * hd;
    let total_kv = l * n_heads * hd;

    let d_q = gpu.upload_f32(&q[..total_q], &[total_q]).unwrap();
    let d_k = gpu.upload_f32(&k[..total_kv], &[total_kv]).unwrap();
    let d_v = gpu.upload_f32(&v[..total_kv], &[total_kv]).unwrap();

    let d_k_f16 = gpu.alloc_tensor(&[total_kv], DType::F16).unwrap();
    let d_v_f16 = gpu.alloc_tensor(&[total_kv], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
    gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();

    // Run v5
    let d_out_v5 = gpu.zeros(&[total_q], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out_v5, b, l, n_heads, n_heads, hd,
    ).unwrap();

    // Run v6
    let d_out_v6 = gpu.zeros(&[total_q], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out_v6, b, l, n_heads, n_heads, hd,
    ).unwrap();

    gpu.hip.device_synchronize().unwrap();

    let out_v5 = gpu.download_f32(&d_out_v5).unwrap();
    let out_v6 = gpu.download_f32(&d_out_v6).unwrap();

    // Per-head NaN analysis
    let n_q_tiles = (b + 63) / 64;
    eprintln!("\n=== {} — B={} L={} heads={} hd={} ===", label, b, l, n_heads, hd);
    eprintln!("q_tiles={}, total_q_out={}", n_q_tiles, total_q);

    let mut total_nan_v6 = 0;
    let mut total_zero_v6 = 0;

    for head in 0..n_heads {
        let mut nan_count = 0;
        let mut zero_count = 0;
        let mut max_abs_v5 = 0.0f32;
        let mut max_abs_v6 = 0.0f32;
        let mut first_nan_qt = -1i32;
        let mut first_ok_qt = -1i32;

        for qt in 0..n_q_tiles {
            let q_start = qt * 64;
            for row in 0..64.min(b - q_start) {
                for d in 0..hd {
                    let idx = (q_start + row) * n_heads * hd + head * hd + d;
                    if idx >= total_q { continue; }
                    let v5 = out_v5[idx];
                    let v6_val = out_v6[idx];

                    if v6_val.is_nan() {
                        nan_count += 1;
                        if first_nan_qt < 0 { first_nan_qt = qt as i32; }
                    } else if v6_val == 0.0 {
                        zero_count += 1;
                    } else if first_ok_qt < 0 {
                        first_ok_qt = qt as i32;
                    }

                    max_abs_v5 = max_abs_v5.max(v5.abs());
                    if v6_val.is_finite() {
                        max_abs_v6 = max_abs_v6.max(v6_val.abs());
                    }
                }
            }
        }
        total_nan_v6 += nan_count;
        total_zero_v6 += zero_count;

        let elements_per_head = b * hd;
        if head < 3 || nan_count > 0 {
            eprintln!("  head {head:2}: v6 nan={nan_count}/{elements_per_head} zero={zero_count} | v5 max={max_abs_v5:.6} v6 max={max_abs_v6:.6} | first_nan_qt={first_nan_qt} first_ok_qt={first_ok_qt}");
        }
    }

    if total_nan_v6 == 0 {
        let mut max_diff = 0.0f32;
        let mut mismatches = 0;
        for i in 0..total_q {
            let diff = (out_v5[i] - out_v6[i]).abs();
            max_diff = max_diff.max(diff);
            if diff > 0.01 { mismatches += 1; }
        }
        eprintln!("  PASS: max_diff={max_diff:.6} mismatches={mismatches}");
    } else {
        eprintln!("  FAIL: {} NaN values out of {} total ({} zero)", total_nan_v6, total_q, total_zero_v6);

        // Dump first few v5 vs v6 values for head 0, qt 0
        eprintln!("  First 32 values (head=0, q_tile=0):");
        for i in 0..32.min(total_q / n_heads) {
            let idx = i; // head=0, row=0, dim=i
            let v5 = out_v5[idx];
            let v6_val = out_v6[idx];
            eprintln!("    [{i:2}] v5={v5:12.6} v6={v6_val:12.6} {}",
                      if v6_val.is_nan() { "NaN!" } else { "" });
        }
    }

    gpu.free_tensor(d_q);
    gpu.free_tensor(d_k);
    gpu.free_tensor(d_v);
    gpu.free_tensor(d_k_f16);
    gpu.free_tensor(d_v_f16);
    gpu.free_tensor(d_out_v5);
    gpu.free_tensor(d_out_v6);
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);
    let nh = 12usize;
    let hd = 128usize;

    // Test 1: tiny B=64, L=64 (1 q_tile, 1 k_tile — minimal case)
    let b = 64; let l = 64;
    let q: Vec<f32> = (0..b*nh*hd).map(|i| ((i as f32)*0.1).sin()).collect();
    let k: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).cos()).collect();
    let v: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).sin()*0.5).collect();
    run_with_data(&mut gpu, &q, &k, &v, b, l, nh, hd, "B=64 L=64 synthetic");

    // Test 2: B=64, L=128 (1 q_tile, 2 k_tiles)
    let b = 64; let l = 128;
    let q: Vec<f32> = (0..b*nh*hd).map(|i| ((i as f32)*0.1).sin()).collect();
    let k: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).cos()).collect();
    let v: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).sin()*0.5).collect();
    run_with_data(&mut gpu, &q, &k, &v, b, l, nh, hd, "B=64 L=128 (2 k_tiles)");

    // Test 3: B=128, L=128 (2 q_tiles, 2 k_tiles)
    let b = 128; let l = 128;
    let q: Vec<f32> = (0..b*nh*hd).map(|i| ((i as f32)*0.1).sin()).collect();
    let k: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).cos()).collect();
    let v: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).sin()*0.5).collect();
    run_with_data(&mut gpu, &q, &k, &v, b, l, nh, hd, "B=128 L=128 (2x2)");

    // Test 4: B=64, L=19520 (1 q_tile, 153 k_tiles)
    let b = 64; let l = 19520;
    let q: Vec<f32> = (0..b*nh*hd).map(|i| ((i as f32)*0.1).sin()).collect();
    let k: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).cos()).collect();
    let v: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).sin()*0.5).collect();
    run_with_data(&mut gpu, &q, &k, &v, b, l, nh, hd, "B=64 L=19520 (1 qt, 153 kt)");

    // Test 5: B=128, L=19520
    let b = 128; let l = 19520;
    let q: Vec<f32> = (0..b*nh*hd).map(|i| ((i as f32)*0.1).sin()).collect();
    let k: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).cos()).collect();
    let v: Vec<f32> = (0..l*nh*hd).map(|i| ((i as f32)*0.1).sin()*0.5).collect();
    run_with_data(&mut gpu, &q, &k, &v, b, l, nh, hd, "B=128 L=19520 (2 qt, 153 kt)");

    // Test 6: Real captured data
    if let Ok(meta) = std::fs::read("/tmp/attn_capture/meta.bin") {
        let n_patches = meta[0] as usize | (meta[1] as usize) << 8 | (meta[2] as usize) << 16;
        let q_bytes = std::fs::read("/tmp/attn_capture/q_f32.bin").unwrap();
        let k_bytes = std::fs::read("/tmp/attn_capture/k_f32.bin").unwrap();
        let v_bytes = std::fs::read("/tmp/attn_capture/v_f32.bin").unwrap();
        let q: Vec<f32> = q_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        let k: Vec<f32> = k_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        let v: Vec<f32> = v_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        run_with_data(&mut gpu, &q, &k, &v, n_patches, n_patches, nh, hd, "Real capture data");
    }
}
