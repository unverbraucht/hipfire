//! v6b diagnostic: check VGPR, scratch, correctness for n_tile=64 variant.

use hipfire_runtime::{Gpu, DType};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let use_v6b = args.iter().any(|a| a == "--v6b");
    let use_v6 = args.iter().any(|a| a == "--v6");
    let use_v5 = args.iter().any(|a| a == "--v5");

    let mut gpu = Gpu::init().expect("Gpu::init");

    // Same test parameters as v6 diagnostic
    let b = 8192;
    let l = 8192;
    let n_heads = 16;
    let n_kv_heads = 16;
    let head_dim = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Allocate tensors
    let q_size = b * n_heads * head_dim;
    let k_size = l * n_kv_heads * head_dim;
    let v_size = l * n_kv_heads * head_dim;

    let q = gpu.alloc_d(&[q_size], DType::F32).unwrap_d().unwrap();
    let k = gpu.alloc_d(&[k_size], DType::F32).unwrap_d().unwrap();
    let v = gpu.alloc_d(&[v_size], DType::F32).unwrap_d().unwrap();
    let out = gpu.alloc_d(&[q_size], DType::F32).unwrap_d().unwrap();

    // Fill with deterministic data
    let q_h: Vec<f32> = (0..q_size).map(|i| ((i % 37) as f32 - 18.0) * 0.01).collect();
    let k_h: Vec<f32> = (0..k_size).map(|i| ((i % 41) as f32 - 20.0) * 0.01).collect();
    let v_h: Vec<f32> = (0..v_size).map(|i| ((i % 43) as f32 - 21.0) * 0.01).collect();

    q.copy_from_host(&q_h).unwrap();
    k.copy_from_host(&k_h).unwrap();
    v.copy_from_host(&v_h).unwrap();

    // Cast K/V to F16
    let k_f16 = gpu.alloc_d(&[k_size], DType::F16).unwrap_d().unwrap();
    let v_f16 = gpu.alloc_d(&[v_size], DType::F16).unwrap_d().unwrap();
    gpu.cast_f32_to_f16(&k, &k_f16).unwrap();
    gpu.cast_f32_to_f16(&v, &v_f16).unwrap();

    // Run the kernel
    println!("Running attention kernel (b={}, l={}, n_heads={}, head_dim={})", b, l, n_heads, head_dim);

    if use_v6b {
        println!("Using v6b (n_tile=64, 2-pass d_half)");
        gpu.attention_dflash_wmma_m64_n32_f16kv_v6b_f32(
            &q, &k_f16, &v_f16, &out,
            b, l, n_heads, n_kv_heads, head_dim,
        ).unwrap();
    } else if use_v6 {
        println!("Using v6 (n_tile=64, fixed s_acc[8] -> s_acc[4] with 8 chunks overflow)");
        gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(
            &q, &k_f16, &v_f16, &out,
            b, l, n_heads, n_kv_heads, head_dim,
        ).unwrap();
    } else if use_v5 {
        println!("Using v5 (n_tile=128, 2-pass d_half with 8 chunks)");
        gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
            &q, &k_f16, &v_f16, &out,
            b, l, n_heads, n_kv_heads, head_dim,
        ).unwrap();
    } else {
        eprintln!("Specify --v5, --v6, or --v6b");
        return;
    }

    gpu.device_synchronize().unwrap();

    // Check for NaN in output
    let out_h: Vec<f32> = out.copy_to_host_vec().unwrap();
    let nan_count = out_h.iter().filter(|x| x.is_nan()).count();
    let total = out_h.len();
    let nan_pct = 100.0 * nan_count as f64 / total as f64;

    println!("Output: {} elements, {} NaN ({:.2}%)", total, nan_count, nan_pct);

    if nan_count == 0 {
        println!("✓ No NaN detected");
    } else {
        println!("✗ NaN detected!");
        // Print first few values
        println!("First 20 values: {:?}", &out_h[..20.min(total)]);
    }
}
