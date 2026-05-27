// Correctness check: gemm_f16_wmma_mb4 (transposed [N,M] output) vs the proven
// gemm_f16_wmma ([M,N]) + transpose, for the vision shapes — full weight and
// sub_offset weight halves (the fc13 fc1/fc3 split).
use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5 }).collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    // Real dots.ocr fc13 shape: M2 = 2*interm = 8448, K = embed = 1536,
    // N = patches (use 19520, the smoke image). Overridable via argv for sweeps.
    let args: Vec<String> = std::env::args().collect();
    let av = |k: &str, d: usize| args.iter().position(|a| a == k).map(|i| args[i+1].parse().unwrap()).unwrap_or(d);
    let interm = av("--interm", 4224); let k = av("--k", 1536); let n = av("--n", 19520);
    let m2 = 2 * interm;

    let w_f32 = lcg(1, m2 * k);
    let x_f32 = lcg(2, n * k);
    // Upload W as F16 (via F32→F16 cast), X as F32.
    let w_tmp = gpu.upload_f32(&w_f32, &[m2 * k]).unwrap();
    let w = gpu.alloc_tensor(&[m2, k], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&w_tmp, &w).unwrap();
    let x = gpu.upload_f32(&x_f32, &[n, k]).unwrap();

    // Reference: gemm_f16_wmma → yt[M2, N], transpose → ref[N, M2].
    let yt = gpu.zeros(&[m2 * n], DType::F32).unwrap();
    gpu.gemm_f16_wmma(&w, &x, &yt, m2, k, n).unwrap();
    let refy = gpu.zeros(&[n, m2], DType::F32).unwrap();
    gpu.transpose_f32(&yt, &refy, m2, n).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let refv = gpu.download_f32(&refy).unwrap(); // [n, m2]

    // mb4 on FULL weight → [N, M2].
    let mb_full = gpu.zeros(&[n * m2], DType::F32).unwrap();
    gpu.gemm_f16_wmma_mb4(&w, &x, &mb_full, m2, k, n).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let fullv = gpu.download_f32(&mb_full).unwrap();
    let d_full = refv.iter().zip(&fullv).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    println!("mb4 FULL  vs ref: maxdiff = {d_full:.3e}");

    // mb4 on sub_offset halves (fc1 rows 0:interm, fc3 rows interm:2interm).
    let fc1 = w.sub_offset(0, interm * k);
    let fc3 = w.sub_offset(interm * k, interm * k);
    let g = gpu.zeros(&[n * interm], DType::F32).unwrap();
    let u = gpu.zeros(&[n * interm], DType::F32).unwrap();
    gpu.gemm_f16_wmma_mb4(&fc1, &x, &g, interm, k, n).unwrap();
    gpu.gemm_f16_wmma_mb4(&fc3, &x, &u, interm, k, n).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let gv = gpu.download_f32(&g).unwrap();
    let uv = gpu.download_f32(&u).unwrap();
    // ref gate = refv[n][0:interm], ref up = refv[n][interm:2interm].
    let mut d_g = 0.0f32; let mut d_u = 0.0f32;
    for nn in 0..n {
        for m in 0..interm {
            d_g = d_g.max((gv[nn*interm + m] - refv[nn*m2 + m]).abs());
            d_u = d_u.max((uv[nn*interm + m] - refv[nn*m2 + interm + m]).abs());
        }
    }
    println!("mb4 fc1-sub vs ref gate: maxdiff = {d_g:.3e}");
    println!("mb4 fc3-sub vs ref up:   maxdiff = {d_u:.3e}");
}
