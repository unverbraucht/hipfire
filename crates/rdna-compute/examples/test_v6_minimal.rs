//! Minimal v6 isolation test: tiny shapes to diagnose NaN source.
//!
//! Tests v6 with various B, L, and data magnitudes to determine:
//! 1. Does v6 work with B=64 (single q_tile)?
//! 2. Does it fail with larger B at the same data?
//! 3. Does magnitude matter?

use rdna_compute::{DType, Gpu};

fn test_case(gpu: &mut Gpu, b: usize, l: usize, n_heads: usize, scale_q: f32, scale_k: f32, label: &str) {
    let hd = 128;
    let total_q = b * n_heads * hd;
    let total_k = l * n_heads * hd;

    let q: Vec<f32> = (0..total_q).map(|i| ((i as f32) * 0.0037).sin() * scale_q).collect();
    let k: Vec<f32> = (0..total_k).map(|i| ((i as f32) * 0.0053).cos() * scale_k).collect();
    let v: Vec<f32> = (0..total_k).map(|i| ((i as f32) * 0.0029).cos() * scale_k * 0.5).collect();

    let d_q = gpu.upload_f32(&q, &[total_q]).unwrap();
    let d_k = gpu.upload_f32(&k, &[total_k]).unwrap();
    let d_v = gpu.upload_f32(&v, &[total_k]).unwrap();

    let d_k_f16 = gpu.alloc_tensor(&[total_k], DType::F16).unwrap();
    let d_v_f16 = gpu.alloc_tensor(&[total_k], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
    gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();

    let d_out = gpu.zeros(&[total_q], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_heads, hd,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let out = gpu.download_f32(&d_out).unwrap();
    let nan_count = out.iter().filter(|x| x.is_nan()).count();
    let inf_count = out.iter().filter(|x| x.is_infinite()).count();
    let zero_count = out.iter().filter(|x| **x == 0.0).count();
    let max_abs = out.iter().map(|x| x.abs()).filter(|x| x.is_finite()).fold(0.0f32, f32::max);
    let mean = out.iter().filter(|x| x.is_finite()).map(|x| x.abs()).sum::<f32>() / out.iter().filter(|x| x.is_finite()).count() as f32;

    let verdict = if nan_count > 0 { "NaN" } else if inf_count > 0 { "INF" } else if out.iter().all(|x| *x == 0.0) { "ALL-ZERO" } else { "OK" };
    eprintln!("{label:>50} B={b:>5} L={l:>5} scale=Q{k:>5.2}K{v:>5.2}  → {verdict:>8} nan={nan_count} inf={inf_count} zero={zero_count} max={max_abs:.4} mean={mean:.6}");

    gpu.free_tensor(d_q);
    gpu.free_tensor(d_k);
    gpu.free_tensor(d_v);
    gpu.free_tensor(d_k_f16);
    gpu.free_tensor(d_v_f16);
    gpu.free_tensor(d_out);
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}\n", gpu.arch);
    let nh = 12usize;

    // Single q_tile tests
    eprintln!("=== Single q_tile (B=64) ===");
    test_case(&mut gpu, 64, 64,   nh, 1.0, 1.0, "B=64 L=64 small");
    test_case(&mut gpu, 64, 64,   nh, 4.0, 4.0, "B=64 L=64 real-mag");
    test_case(&mut gpu, 64, 128,  nh, 1.0, 1.0, "B=64 L=128 (2 K-tiles)");
    test_case(&mut gpu, 64, 19520, nh, 1.0, 1.0, "B=64 L=19520 (many K-tiles)");

    // Two q_tiles
    eprintln!("\n=== Two q_tiles (B=128) ===");
    test_case(&mut gpu, 128, 128,   nh, 1.0, 1.0, "B=128 L=128 small");
    test_case(&mut gpu, 128, 19520, nh, 1.0, 1.0, "B=128 L=19520 many K-tiles");
    test_case(&mut gpu, 128, 19520, nh, 4.0, 4.0, "B=128 L=19520 real-mag");

    // Full vision shape
    eprintln!("\n=== Full vision shape ===");
    test_case(&mut gpu, 19520, 19520, nh, 1.0, 1.0, "Full shape small");
    test_case(&mut gpu, 19520, 19520, nh, 4.0, 4.0, "Full shape real-mag");

    // Exact data from capture
    eprintln!("\n=== Captured real data (replay) ===");
    let meta = std::fs::read("/tmp/attn_capture/meta.bin")
        .unwrap_or_else(|_| { eprintln!("No capture found, skipping"); return vec![0,0,0,0] });
    let n_patches = meta[0] as usize | (meta[1] as usize) << 8 | (meta[2] as usize) << 16;
    let total = n_patches * nh * 128;
    let q_bytes = std::fs::read("/tmp/attn_capture/q_f32.bin").unwrap();
    let k_bytes = std::fs::read("/tmp/attn_capture/k_f32.bin").unwrap();
    let v_bytes = std::fs::read("/tmp/attn_capture/v_f32.bin").unwrap();

    let q: Vec<f32> = q_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
    let k: Vec<f32> = k_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
    let v: Vec<f32> = v_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();

    eprintln!("Q range: [{:.3}, {:.3}]", q.iter().cloned().fold(f32::INFINITY, f32::min), q.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    eprintln!("K range: [{:.3}, {:.3}]", k.iter().cloned().fold(f32::INFINITY, f32::min), k.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Test with real Q/K data but smaller B
    for b_test in [64, 128, 1024, n_patches] {
        let n_test = b_test * nh * 128;
        let n_kv = if b_test == n_patches { n_patches } else { b_test };
        let n_kv_test = n_kv * nh * 128;

        let d_q = gpu.upload_f32(&q[..n_test.min(q.len())], &[n_test.min(q.len())]).unwrap();
        let d_k = gpu.upload_f32(&k[..n_kv_test.min(k.len())], &[n_kv_test.min(k.len())]).unwrap();
        let d_v = gpu.upload_f32(&v[..n_kv_test.min(v.len())], &[n_kv_test.min(v.len())]).unwrap();

        let d_k_f16 = gpu.alloc_tensor(&[n_kv_test.min(k.len())], DType::F16).unwrap();
        let d_v_f16 = gpu.alloc_tensor(&[n_kv_test.min(v.len())], DType::F16).unwrap();
        gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
        gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();

        let act_b = n_test.min(q.len()) / (nh * 128);
        let act_l = n_kv_test.min(k.len()) / (nh * 128);
        let act_n = act_b * nh * 128;

        let d_out = gpu.zeros(&[act_n], DType::F32).unwrap();
        gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(
            &d_q, &d_k_f16, &d_v_f16, &d_out, act_b, act_l, nh, nh, 128,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();

        let out = gpu.download_f32(&d_out).unwrap();
        let nan_count = out.iter().filter(|x| x.is_nan()).count();
        let max_abs = out.iter().map(|x| x.abs()).filter(|x| x.is_finite()).fold(0.0f32, f32::max);
        let verdict = if nan_count > 0 { "NaN" } else { "OK" };
        eprintln!("  Real data B={act_b:5d} L={act_l:5d}: {verdict} nan={nan_count}/{act_n} max={max_abs:.6}");

        gpu.free_tensor(d_q); gpu.free_tensor(d_k); gpu.free_tensor(d_v);
        gpu.free_tensor(d_k_f16); gpu.free_tensor(d_v_f16); gpu.free_tensor(d_out);
    }
}
