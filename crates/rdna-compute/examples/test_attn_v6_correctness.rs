//! Correctness test: compare v5 vs v6 attention output on identical input.
//!
//! Usage:
//!   ./target/release/examples/test_attn_v6_correctness

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let b = 19520usize;
    let l = 19520usize;
    let n_heads = 12usize;
    let n_kv_heads = 12usize;
    let hd = 128usize;

    // Test with edge-case data patterns
    // Pattern 1: all zeros (tests softmax with all-zero QK^T)
    let q: Vec<f32> = vec![0.0f32; b * n_heads * hd];
    let k: Vec<f32> = vec![0.0f32; l * n_kv_heads * hd];
    let v: Vec<f32> = vec![1.0f32; l * n_kv_heads * hd];

    let d_q = gpu.upload_f32(&q, &[b * n_heads * hd]).unwrap();
    let d_k = gpu.upload_f32(&k, &[l * n_kv_heads * hd]).unwrap();
    let d_v = gpu.upload_f32(&v, &[l * n_kv_heads * hd]).unwrap();

    let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
    let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
    gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();

    // v5 output
    let d_out_v5 = gpu.zeros(&[b * n_heads * hd], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out_v5, b, l, n_heads, n_kv_heads, hd,
    ).unwrap();

    // v6 output
    let d_out_v6 = gpu.zeros(&[b * n_heads * hd], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out_v6, b, l, n_heads, n_kv_heads, hd,
    ).unwrap();

    gpu.hip.device_synchronize().unwrap();

    let out_v5 = gpu.download_f32(&d_out_v5).unwrap();
    let out_v6 = gpu.download_f32(&d_out_v6).unwrap();

    // Compare
    let n = b * n_heads * hd;
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mismatches = 0usize;

    for i in 0..n {
        let a = out_v5[i];
        let b = out_v6[i];
        let diff = (a - b).abs();
        let rel = if a.abs() > 1e-6 { diff / a.abs() } else { diff };
        max_diff = max_diff.max(diff);
        max_rel = max_rel.max(rel);
        sum_diff += diff;
        if diff > 0.01 {
            mismatches += 1;
            if mismatches <= 10 {
                eprintln!("  MISMATCH [{i}]: v5={a:.6} v6={b:.6} diff={diff:.6}");
            }
        }
    }

    eprintln!("\n=== v5 vs v6 comparison (B={b} L={l} heads={n_heads}) ===");
    eprintln!("  max abs diff: {max_diff:.6}");
    eprintln!("  max rel diff: {max_rel:.6}");
    eprintln!("  mean abs diff: {}", sum_diff / n as f32);
    eprintln!("  mismatches (>0.01): {mismatches}/{n}");

    if max_diff < 0.01 {
        eprintln!("  VERDICT: PASS (outputs match within tolerance)");
    } else {
        eprintln!("  VERDICT: FAIL (outputs differ)");
    }
}
