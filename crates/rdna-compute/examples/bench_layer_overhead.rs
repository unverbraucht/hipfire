//! Measure each kernel group in the qwen2 decode loop individually.
//! Simulates one layer's worth of kernel calls to find what eats time
//! outside of GEMV + attention.
use rdna_compute::{DType, Gpu};

fn main() {
    let iters = 200usize;
    let mut gpu = Gpu::init().expect("GPU init");

    let hidden = 1536usize;
    let n_heads = 12usize;
    let n_kv_heads = 2usize;
    let head_dim = 128usize;
    let interm = 8960usize;

    let x = gpu.zeros(&[hidden], DType::F32).unwrap();
    let tmp = gpu.zeros(&[hidden], DType::F32).unwrap();
    let out = gpu.zeros(&[hidden], DType::F32).unwrap();
    let gate = gpu.zeros(&[interm], DType::F32).unwrap();
    let up = gpu.zeros(&[interm], DType::F32).unwrap();
    let ffn_hidden = gpu.zeros(&[interm], DType::F32).unwrap();
    let norm = gpu.zeros(&[hidden], DType::F32).unwrap();

    // Warm up
    for _ in 0..10 { gpu.rmsnorm_f32(&x, &norm, &tmp, 1e-5).unwrap(); }
    gpu.hip.device_synchronize().unwrap();

    // (a) RMSNorm alone
    let t = std::time::Instant::now();
    for _ in 0..iters { gpu.rmsnorm_f32(&x, &norm, &tmp, 1e-5).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let rmsnorm_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // (b) add_inplace_f32 alone (residual add)
    for _ in 0..10 { gpu.add_inplace_f32(&x, &out).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters { gpu.add_inplace_f32(&x, &out).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let add_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // (c) silu_mul_f32
    for _ in 0..10 { gpu.silu_mul_f32(&gate, &up, &ffn_hidden).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters { gpu.silu_mul_f32(&gate, &up, &ffn_hidden).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let silu_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // (d) rope_f32
    let kv_dim = n_kv_heads * head_dim;
    let q_dim = n_heads * head_dim;
    let q_buf = gpu.zeros(&[q_dim], DType::F32).unwrap();
    let k_buf = gpu.zeros(&[kv_dim], DType::F32).unwrap();
    let pos_val = 100i32;
    let pos = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&pos, &pos_val.to_ne_bytes()).unwrap();
    for _ in 0..10 { gpu.rope_f32(&q_buf, &k_buf, &pos, n_heads, n_kv_heads, head_dim, 1e6).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters { gpu.rope_f32(&q_buf, &k_buf, &pos, n_heads, n_kv_heads, head_dim, 1e6).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let rope_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // (e) kv_cache_write_f32
    let max_seq = 12000usize;
    let cache = gpu.alloc_tensor(&[max_seq * kv_dim], DType::F32).unwrap();
    for _ in 0..10 { gpu.kv_cache_write(&cache, &k_buf, &pos, kv_dim).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters { gpu.kv_cache_write(&cache, &k_buf, &pos, kv_dim).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let kvcache_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // (f) bias_add_f32 (q bias, k bias, v bias)
    let qbias = gpu.zeros(&[q_dim], DType::F32).unwrap();
    for _ in 0..10 { gpu.bias_add_f32(&q_buf, &qbias, 1, q_dim).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters { gpu.bias_add_f32(&q_buf, &qbias, 1, q_dim).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let bias_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // Summary
    let misc_per_layer_us = rmsnorm_us * 2.0 + add_us * 2.0 + silu_us + rope_us + kvcache_us * 2.0 + bias_us * 3.0;
    let misc_per_token_us = misc_per_layer_us * 28.0;
    eprintln!("rmsnorm_f32 (hidden=1536): {rmsnorm_us:.2} µs");
    eprintln!("add_inplace_f32 (1536):   {add_us:.2} µs");
    eprintln!("silu_mul_f32 (interm=8960): {silu_us:.2} µs");
    eprintln!("rope_f32 (q+k hd=128):    {rope_us:.2} µs");
    eprintln!("kv_cache_write_f32:       {kvcache_us:.2} µs");
    eprintln!("bias_add_f32:             {bias_us:.2} µs");
    eprintln!("---");
    eprintln!("misc per layer: {misc_per_layer_us:.1} µs");
    eprintln!("misc per token (28×): {misc_per_token_us:.0} µs = {:.1} ms", misc_per_token_us / 1000.0);
}
