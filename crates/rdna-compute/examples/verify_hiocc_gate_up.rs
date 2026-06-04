// Correctness verification: original vs hiocc variant of gemm_gate_up_mq4g256_lloyd_wmma
// Compares output of both kernels element-wise.

use rdna_compute::{DType, Gpu};

const HIOCC_SRC: &str = include_str!(
    "../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_hiocc.gfx1151.hip"
);
const ORIGINAL_SRC: &str = include_str!(
    "../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.gfx1151.hip"
);

fn f32_to_f16_bytes(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    let h = if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else if exp - 127 + 15 < 1 {
        sign << 15
    } else if exp - 127 + 15 > 30 {
        (sign << 15) | (0x1f << 10)
    } else {
        let ne = (exp - 127 + 15) as u16;
        let m13 = mant & 0x1fff;
        let mut nm = (mant >> 13) as u16;
        if m13 > 0x1000 || (m13 == 0x1000 && (nm & 1) != 0) { nm += 1; }
        let mut eb = ne;
        if nm == 0x400 { nm = 0; eb += 1; }
        (sign << 15) | (eb << 10) | nm
    };
    h.to_le_bytes()
}

fn build_mq4(m: usize, k: usize, seed: u32) -> Vec<u8> {
    let groups = k / 256;
    let mut data = vec![0u8; m * groups * 160];
    let mut r = seed as u64;
    for row in 0..m {
        for g in 0..groups {
            let off = (row * groups + g) * 160;
            for i in 0..16 {
                r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = 0.5 + ((r >> 32) as f32) * 0.001;
                let b = f32_to_f16_bytes(v);
                data[off + i * 2..off + i * 2 + 2].copy_from_slice(&b);
            }
            for p in 0..32 {
                let mut pk = 0u32;
                for _ in 0..8 {
                    r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    pk = (pk >> 4) | (((r >> 32) as u32 & 0xFu32) << 28);
                }
                data[off + 32 + p * 4..off + 32 + p * 4 + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    data
}

fn run_kernel(
    src: &str,
    module_name: &str,
    func_name: &str,
    a_gate: &[u8],
    a_up: &[u8],
    x_u8: &[u8],
    gm: usize,
    um: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut gpu = Gpu::init_with_device(0).unwrap();
    gpu.ensure_kernel_public(module_name, src, func_name).unwrap();

    let d_ag = gpu.upload_raw(a_gate, &[a_gate.len()]).unwrap();
    let d_au = gpu.upload_raw(a_up, &[a_up.len()]).unwrap();
    let d_x = gpu.upload_raw(x_u8, &[n, k]).unwrap();
    let d_yg = gpu.zeros(&[n, gm], DType::F32).unwrap();
    let d_yu = gpu.zeros(&[n, um], DType::F32).unwrap();

    let total_m = gm + um;
    let row_tiles = ((total_m + 15) / 16) as u32;
    let batch_tiles = ((n + 15) / 16) as u32;

    let make_kernargs = |ag_p, au_p, x_p, yg_p, yu_p, gm_v, um_v, k_v, n_v| {
        let mut b = Vec::new();
        b.extend_from_slice(&(ag_p as u64).to_le_bytes());
        b.extend_from_slice(&(au_p as u64).to_le_bytes());
        b.extend_from_slice(&(x_p as u64).to_le_bytes());
        b.extend_from_slice(&(yg_p as u64).to_le_bytes());
        b.extend_from_slice(&(yu_p as u64).to_le_bytes());
        b.extend_from_slice(&(gm_v as i32).to_le_bytes());
        b.extend_from_slice(&(um_v as i32).to_le_bytes());
        b.extend_from_slice(&(k_v as i32).to_le_bytes());
        b.extend_from_slice(&(n_v as i32).to_le_bytes());
        b
    };

    let mut ka = make_kernargs(
        d_ag.buf.as_ptr(), d_au.buf.as_ptr(), d_x.buf.as_ptr(),
        d_yg.buf.as_ptr(), d_yu.buf.as_ptr(), gm, um, k, n,
    );
    gpu.launch_kernel_blob(func_name, [row_tiles, batch_tiles, 1], [32, 1, 1], 0, &mut ka).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let yg = gpu.download_f32(&d_yg).unwrap();
    let yu = gpu.download_f32(&d_yu).unwrap();
    (yg, yu)
}

fn main() {
    let shapes = [
        (27648usize, 8192usize, 64usize, "27b-gate+up, N=64"),
        (6912usize, 8192usize, 64usize, "9b-gate+up, N=64"),
        (27648usize, 8192usize, 16usize, "27b-gate+up, N=16"),
        (27648usize, 8192usize, 1usize, "27b-gate+up, N=1"),
    ];

    for &(m, k, n, desc) in &shapes {
        let gm = m / 2;
        let um = m - gm;
        let a_gate = build_mq4(gm, k, 42);
        let a_up = build_mq4(um, k, 99);
        let x_f32: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.001).collect();
        let x_u8: Vec<u8> = x_f32.iter().flat_map(|&v| f32_to_f16_bytes(v)).collect();

        println!("=== {} (M={} K={} N={}) ===", desc, m, k, n);

        let (yg_orig, yu_orig) = run_kernel(
            ORIGINAL_SRC, "gemm_gate_up_orig_gfx1151",
            "gemm_gate_up_mq4g256_lloyd_wmma",
            &a_gate, &a_up, &x_u8, gm, um, k, n,
        );

        let (yg_hiocc, yu_hiocc) = run_kernel(
            HIOCC_SRC, "gemm_gate_up_hiocc_gfx1151",
            "gemm_gate_up_mq4g256_lloyd_wmma",
            &a_gate, &a_up, &x_u8, gm, um, k, n,
        );

        // Compare
        let total_gate = n * gm;
        let total_up = n * um;
        let mut max_abs_gate = 0.0f32;
        let mut max_abs_up = 0.0f32;
        let mut mismatches_gate = 0usize;
        let mut mismatches_up = 0usize;
        let tol = 1e-4f32;

        for i in 0..total_gate {
            let diff = (yg_orig[i] - yg_hiocc[i]).abs();
            if diff > max_abs_gate { max_abs_gate = diff; }
            if diff > tol { mismatches_gate += 1; }
        }
        for i in 0..total_up {
            let diff = (yu_orig[i] - yu_hiocc[i]).abs();
            if diff > max_abs_up { max_abs_up = diff; }
            if diff > tol { mismatches_up += 1; }
        }

        let pass = mismatches_gate == 0 && mismatches_up == 0;
        println!(
            "  gate: max_abs={:.2e}, mismatches={}/{}",
            max_abs_gate, mismatches_gate, total_gate
        );
        println!(
            "  up:   max_abs={:.2e}, mismatches={}/{}",
            max_abs_up, mismatches_up, total_up
        );
        println!("  RESULT: {}", if pass { "PASS ✓" } else { "FAIL ✗" });
    }
}
