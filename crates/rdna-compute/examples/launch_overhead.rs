use rdna_compute::{DType, Gpu};

fn main() {
    let iters = 1000usize;
    let mut gpu = Gpu::init().expect("GPU init");
    let x = gpu.zeros(&[1024], DType::F32).unwrap();
    let y = gpu.zeros(&[1024], DType::F32).unwrap();

    for _ in 0..100 { gpu.add_inplace_f32(&x, &y).unwrap(); }
    gpu.hip.device_synchronize().unwrap();

    let t = std::time::Instant::now();
    for _ in 0..iters { gpu.add_inplace_f32(&x, &y).unwrap(); }
    let launch_cost_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    gpu.hip.device_synchronize().unwrap();

    let t = std::time::Instant::now();
    for _ in 0..iters { gpu.add_inplace_f32(&x, &y).unwrap(); }
    gpu.hip.device_synchronize().unwrap();
    let full_cost_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let n_gemvs = 7 * 28;
    let est_gpu_ms = n_gemvs as f64 * 200.0 / 1000.0;
    let est_launch_ms = n_gemvs as f64 * launch_cost_us / 1000.0;
    let save_per_fuse = launch_cost_us;
    eprintln!("launch-only:    {:.2} µs (no sync)", launch_cost_us);
    eprintln!("launch+sync:    {:.2} µs (synced)", full_cost_us);
    eprintln!("estimated GPU compute (196 gemvs × 200µs): {:.1} ms", est_gpu_ms);
    eprintln!("estimated launch overhead: {:.2} ms", est_launch_ms);
    eprintln!("fusing 1 kernel saves: {:.1} µs / token", save_per_fuse);
}
