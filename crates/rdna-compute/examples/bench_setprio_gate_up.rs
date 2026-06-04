// A/B/C benchmark: original vs s_setprio variant A vs s_setprio variant B
// of gemm_gate_up_mq4g256_lloyd_wmma on gfx1151.
// Issue #344: FeatherOps s_setprio investigation.
//
// Variant A: s_setprio 1/0 around codebook VRAM→LDS loads (per group)
// Variant B: s_setprio 1/0 around inner K-loop weight reads (per kt iteration)
//
// Run: cargo run --release --example bench_setprio_gate_up -p rdna-compute

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const SETPRIO_A_SRC: &str = include_str!(
    "../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_setprio.gfx1151.hip"
);
const SETPRIO_B_SRC: &str = include_str!(
    "../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_setprioB.gfx1151.hip"
);
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
        if m13 > 0x1000 || (m13 == 0x1000 && (nm & 1) != 0) {
            nm += 1;
        }
        let mut eb = ne;
        if nm == 0x400 {
            nm = 0;
            eb += 1;
        }
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
                r = r
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = 0.5 + ((r >> 32) as f32) * 0.001;
                let b = f32_to_f16_bytes(v);
                data[off + i * 2..off + i * 2 + 2].copy_from_slice(&b);
            }
            for p in 0..32 {
                let mut pk = 0u32;
                for _ in 0..8 {
                    r = r
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    pk = (pk >> 4) | (((r >> 32) as u32 & 0xFu32) << 28);
                }
                data[off + 32 + p * 4..off + 32 + p * 4 + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    data
}

fn run_variant(
    label: &str,
    module_name: &str,
    src: &str,
    func_name: &str,
    a_gate: &[u8],
    a_up: &[u8],
    x_u8: &[u8],
    gm: usize,
    um: usize,
    k: usize,
    n: usize,
    warmup: usize,
    runs: usize,
) -> f64 {
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

    // Warmup
    for _ in 0..warmup {
        let mut ka = make_kernargs(
            d_ag.buf.as_ptr(), d_au.buf.as_ptr(), d_x.buf.as_ptr(),
            d_yg.buf.as_ptr(), d_yu.buf.as_ptr(), gm, um, k, n,
        );
        gpu.launch_kernel_blob(func_name, [row_tiles, batch_tiles, 1], [32, 1, 1], 0, &mut ka).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let mut times_us: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut ka = make_kernargs(
            d_ag.buf.as_ptr(), d_au.buf.as_ptr(), d_x.buf.as_ptr(),
            d_yg.buf.as_ptr(), d_yu.buf.as_ptr(), gm, um, k, n,
        );
        let t0 = Instant::now();
        gpu.launch_kernel_blob(func_name, [row_tiles, batch_tiles, 1], [32, 1, 1], 0, &mut ka).unwrap();
        gpu.hip.device_synchronize().unwrap();
        times_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[runs / 2];
    let w_bytes = ((gm + um) * (k / 256) * 160) as f64;
    let total = w_bytes + (n * k * 2) as f64 + (n * (gm + um) * 4) as f64;
    let gbps = total / (median_us / 1_000_000.0) / 1_073_741_824.0;

    println!(
        "  {:40}  median={:8.1} µs  {:5.1} GiB/s",
        label, median_us, gbps
    );
    median_us
}

fn main() {
    let shapes = [
        (27648usize, 8192usize, 64usize, "27b-gate+up, N=64 (prefill batch)"),
        (6912usize, 8192usize, 64usize, "9b-gate+up, N=64 (prefill batch)"),
        (27648usize, 8192usize, 16usize, "27b-gate+up, N=16 (small batch)"),
        (27648usize, 8192usize, 1usize, "27b-gate+up, N=1 (decode-like)"),
    ];

    let warmup = 3;
    let runs = 11;

    for &(m, k, n, desc) in &shapes {
        let gm = m / 2;
        let um = m - gm;
        let a_gate = build_mq4(gm, k, 42);
        let a_up = build_mq4(um, k, 99);
        let x_f32: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.001).collect();
        let x_u8: Vec<u8> = x_f32.iter().flat_map(|&v| f32_to_f16_bytes(v)).collect();

        println!("\n=== M={} K={} N={} — {} ===", m, k, n, desc);

        let orig = run_variant(
            "ORIGINAL (no setprio)",
            "gemm_gate_up_orig_gfx1151",
            ORIGINAL_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma",
            &a_gate, &a_up, &x_u8, gm, um, k, n, warmup, runs,
        );

        let _spa = run_variant(
            "SETPRIO-A (around codebook loads)",
            "gemm_gate_up_setprioA_gfx1151",
            SETPRIO_A_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma",
            &a_gate, &a_up, &x_u8, gm, um, k, n, warmup, runs,
        );

        let _spb = run_variant(
            "SETPRIO-B (around inner K-loop)",
            "gemm_gate_up_setprioB_gfx1151",
            SETPRIO_B_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma",
            &a_gate, &a_up, &x_u8, gm, um, k, n, warmup, runs,
        );

        let hiocc = run_variant(
            "HIOCC (launch_bounds 32,10)",
            "gemm_gate_up_hiocc_gfx1151",
            HIOCC_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma",
            &a_gate, &a_up, &x_u8, gm, um, k, n, warmup, runs,
        );

        let delta_pct = (orig - hiocc) / orig * 100.0;
        println!(
            "  >>> HIOCC vs orig: {:+.1}%  ({:.1} vs {:.1} µs)",
            delta_pct, orig, hiocc
        );
    }
}
