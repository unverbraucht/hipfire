//! Correctness test: compare v5 vs v6 vs v6b attention output.
//!
//! Modes:
//! 1. Synthetic (default): generates random data
//! 2. Replay: loads captured tensors from disk (--replay DIR)
//!
//! Exit 0 = all kernels match v5. Exit 1 = NaN or mismatch detected.

use rdna_compute::{DType, Gpu};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let replay_dir = args.iter().position(|a| a == "--replay").map(|i| args[i + 1].clone());

    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    if let Some(dir) = replay_dir {
        run_replay(&mut gpu, &dir);
    } else {
        run_synthetic(&mut gpu);
    }
}

fn run_replay(gpu: &mut Gpu, dir: &str) {
    eprintln!("Replay mode: loading from {dir}");
    let path = std::path::PathBuf::from(dir);

    let meta = std::fs::read(path.join("meta.bin")).expect("meta.bin");
    let n_patches = meta[0] as usize | (meta[1] as usize) << 8 | (meta[2] as usize) << 16;
    let n_heads = 12usize;
    let hd = 128usize;
    let total = n_patches * n_heads * hd;

    let q_bytes = std::fs::read(path.join("q_f32.bin")).expect("q_f32.bin");
    let k_bytes = std::fs::read(path.join("k_f32.bin")).expect("k_f32.bin");
    let v_bytes = std::fs::read(path.join("v_f32.bin")).expect("v_f32.bin");

    let q: Vec<f32> = q_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
    let k: Vec<f32> = k_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
    let v: Vec<f32> = v_bytes.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();

    assert_eq!(q.len(), total);
    assert_eq!(k.len(), total);
    assert_eq!(v.len(), total);

    eprintln!("Q range: [{:.3}, {:.3}]", q.iter().cloned().fold(f32::INFINITY, f32::min), q.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    eprintln!("K range: [{:.3}, {:.3}]", k.iter().cloned().fold(f32::INFINITY, f32::min), k.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    run_with_data(gpu, &q, &k, &v, n_patches, n_patches, n_heads, hd);
}

fn run_synthetic(gpu: &mut Gpu) {
    let b = 19520usize;
    let l = 19520usize;
    let n_heads = 12usize;
    let hd = 128usize;
    let scale = 5.0f32;

    let q: Vec<f32> = (0..b * n_heads * hd).map(|i| ((i as f32) * 0.0037).sin() * scale).collect();
    let k: Vec<f32> = (0..l * n_heads * hd).map(|i| ((i as f32) * 0.0053).cos() * scale).collect();
    let v: Vec<f32> = (0..l * n_heads * hd).map(|i| ((i as f32) * 0.0029).tan().max(-3.0).min(3.0) * scale * 0.5).collect();

    eprintln!("Synthetic mode: B={b} L={l} heads={n_heads} hd={hd}");
    run_with_data(gpu, &q, &k, &v, b, l, n_heads, hd);
}

fn run_with_data(gpu: &mut Gpu, q: &[f32], k: &[f32], v: &[f32],
                 b: usize, l: usize, n_heads: usize, hd: usize) {
    let total = b * n_heads * hd;

    let d_q = gpu.upload_f32(q, &[total]).unwrap();
    let d_k = gpu.upload_f32(k, &[total]).unwrap();
    let d_v = gpu.upload_f32(v, &[total]).unwrap();

    let d_k_f16 = gpu.alloc_tensor(&[total], DType::F16).unwrap();
    let d_v_f16 = gpu.alloc_tensor(&[total], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
    gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();

    // v5 (reference)
    let d_out_v5 = gpu.zeros(&[total], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out_v5, b, l, n_heads, n_heads, hd,
    ).unwrap();

    // v6 (split d_half, n_tile=128, s_acc[8])
    let d_out_v6 = gpu.zeros(&[total], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out_v6, b, l, n_heads, n_heads, hd,
    ).unwrap();

    // v6b (split d_half, n_tile=64, s_acc[4])
    let d_out_v6b = gpu.zeros(&[total], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v6b_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out_v6b, b, l, n_heads, n_heads, hd,
    ).unwrap();

    gpu.hip.device_synchronize().unwrap();

    let out_v5 = gpu.download_f32(&d_out_v5).unwrap();
    let out_v6 = gpu.download_f32(&d_out_v6).unwrap();
    let out_v6b = gpu.download_f32(&d_out_v6b).unwrap();

    let pass_v6 = compare(&out_v5, &out_v6, "v6", b, n_heads, hd);
    let pass_v6b = compare(&out_v5, &out_v6b, "v6b", b, n_heads, hd);

    eprintln!("\n=== SUMMARY ===");
    eprintln!("  v5:  REFERENCE");
    eprintln!("  v6:  {}", if pass_v6 { "PASS" } else { "FAIL" });
    eprintln!("  v6b: {}", if pass_v6b { "PASS" } else { "FAIL" });

    if !pass_v6 || !pass_v6b {
        std::process::exit(1);
    }
}

fn compare(out_ref: &[f32], out_test: &[f32], label: &str,
           n_patches: usize, n_heads: usize, hd: usize) -> bool {
    let n = n_patches * n_heads * hd;
    let mut max_diff = 0.0f32;
    let mut max_val_ref = 0.0f32;
    let mut max_val_test = 0.0f32;
    let mut nan_ref = 0usize;
    let mut nan_test = 0usize;
    let mut mismatches = 0usize;

    for i in 0..n {
        let a = out_ref[i];
        let b = out_test[i];
        if a.is_nan() { nan_ref += 1; }
        if b.is_nan() { nan_test += 1; }
        max_val_ref = max_val_ref.max(a.abs());
        max_val_test = max_val_test.max(b.abs());
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        if diff > 0.02 && !a.is_nan() && !b.is_nan() {
            mismatches += 1;
        }
    }

    eprintln!("\n=== v5 vs {} (patches={} heads={} hd={}) ===", label, n_patches, n_heads, hd);
    eprintln!("  v5:   max_abs={:.6} nan={}", max_val_ref, nan_ref);
    eprintln!("  {}:  max_abs={:.6} nan={}", label, max_val_test, nan_test);
    eprintln!("  max abs diff: {:.6}", max_diff);
    eprintln!("  mismatches (>0.01): {}/{}", mismatches, n);

    if nan_test > 0 {
        eprintln!("  VERDICT: FAIL ({} produced {} NaN values)", label, nan_test);
        false
    } else if max_diff < 0.01 {
        eprintln!("  VERDICT: PASS");
        true
    } else {
        eprintln!("  VERDICT: FAIL (outputs differ)");
        false
    }
}
