//! Minimal harness: run v4 attention exactly once, for rocprofv2 PMC profiling.
//!
//! Usage:
//!   rocprofv2 -i /tmp/pmc.txt -d /tmp/pmc_out -- \
//!     ./target/release/examples/bench_attn_v4_only

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let b = 19520usize;
    let l = 19520usize;
    let n_heads = 12usize;
    let n_kv_heads = 12usize;
    let hd = 128usize;
    eprintln!("shape: B={b} L={l} n_heads={n_heads} hd={hd}");

    // Small deterministic data — content doesn't matter for PMC counters.
    let q = vec![0.1f32; b * n_heads * hd];
    let k = vec![0.1f32; l * n_kv_heads * hd];
    let v = vec![0.1f32; l * n_kv_heads * hd];

    let d_q = gpu.upload_f32(&q, &[b * n_heads * hd]).unwrap();
    let d_k = gpu.upload_f32(&k, &[l * n_kv_heads * hd]).unwrap();
    let d_v = gpu.upload_f32(&v, &[l * n_kv_heads * hd]).unwrap();
    let d_out = gpu.zeros(&[b * n_heads * hd], DType::F32).unwrap();

    let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
    let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
    gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();

    // Warm-up (JIT compile the kernel).
    gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out,
        b, l, n_heads, n_kv_heads, hd,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Profiled dispatch — single call.
    gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
        &d_q, &d_k_f16, &d_v_f16, &d_out,
        b, l, n_heads, n_kv_heads, hd,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();

    eprintln!("done");
}
