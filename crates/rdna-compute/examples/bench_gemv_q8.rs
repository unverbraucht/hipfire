// gemv_q8_0 effective bandwidth at dots.ocr decode shapes (batch=1 GEMV).
// Q8_0 weight = M rows × (K/32) blocks × 34 bytes (2-byte f16 scale + 32 i8).
use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let peak = 960.0_f64; // gfx1100 GDDR6 ~960 GB/s
    // (M, K, label) — the dots.ocr text-decoder GEMV shapes (hidden=1536, interm=8960).
    let shapes = [
        (1536usize, 1536usize, "qkv/o (M=1536,K=1536)"),
        (8960, 1536, "gate/up (M=8960,K=1536)"),
        (1536, 8960, "down   (M=1536,K=8960)"),
        (151936, 1536, "lm_head(M=151936,K=1536)"),
    ];
    let n_iter = 300;
    eprintln!("GPU: {}  gemv_q8_0 bandwidth (peak ~{:.0} GB/s)", gpu.arch, peak);
    for (m, k, label) in shapes {
        let nbytes = m * (k / 32) * 34;
        let wbytes: Vec<u8> = (0..nbytes).map(|i| (i * 131 + 7) as u8).collect();
        let a = gpu.upload_raw(&wbytes, &[nbytes]).unwrap();
        let x = gpu.upload_f32(&vec![0.5f32; k], &[k]).unwrap();
        let y = gpu.zeros(&[m], DType::F32).unwrap();
        gpu.gemv_q8_0(&a, &x, &y, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..n_iter { gpu.gemv_q8_0(&a, &x, &y, m, k).unwrap(); }
        gpu.hip.device_synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / n_iter as f64;
        let gbs = (nbytes as f64 + (k * 4) as f64) / (us * 1e-6) / 1e9;
        eprintln!("  {label:28} {us:7.1} us  {gbs:6.1} GB/s  {:4.1}% peak", gbs / peak * 100.0);
        let _ = (gpu.free_tensor(a), gpu.free_tensor(x), gpu.free_tensor(y));
    }
}
