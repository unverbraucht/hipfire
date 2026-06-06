// WMMA prefill kernel tiling sweep for gfx1100.
// Sweeps K-unroll × launch_bounds on the MQ4-Lloyd gate_up kernel.
// Issue #344: FeatherOps autotune tiling investigation.
//
// Run: cargo run --release --example bench_wmma_tiling_sweep -p rdna-compute

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const ORIGINAL_SRC: &str = include_str!(
    "../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.hip"
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

/// Generate a K-unrolled variant of the MQ4-Lloyd gate_up WMMA kernel.
/// K_UNROLL is the number of K-tiles per inner iteration (1, 2, 4, 8, or 16).
fn generate_k_unroll_variant(k_unroll: usize, min_blocks: usize) -> String {
    let n_tiles = 16 / k_unroll;
    let mut src = String::new();

    src.push_str(include_str!("wmma_sweep_prefix.inc"));

    // Generate the K-loop
    src.push_str(&format!(
        "        for (int kt = 0; kt < 16; kt += {}) {{\n",
        k_unroll
    ));

    // Front-load all nibble-pack reads
    for t in 0..k_unroll {
        src.push_str(&format!(
            "            const unsigned char* dp{t} = dp + (kt + {t}) * 8;\n",
            t = t
        ));
        src.push_str(&format!(
            "            unsigned int pk0a_{t} = *(const unsigned int*)dp{t};\n",
            t = t
        ));
        src.push_str(&format!(
            "            unsigned int pk1a_{t} = *(const unsigned int*)(dp{t} + 4);\n",
            t = t
        ));
        // Load B-tile
        src.push_str(&format!(
            "            half16_t b_{t} = *(const half16_t*)(x_base + (kt + {t}) * 16);\n",
            t = t
        ));
    }

    src.push_str("\n");

    // Generate DQ + WMMA for each tile
    for t in 0..k_unroll {
        src.push_str("            half16_t a_reg;\n");
        src.push_str("#define DQ(i, pk, sh) a_reg[i] = cb_lds[cb_base + (((pk) >> (sh)) & 0xFu)]\n");
        src.push_str(&format!(
            "            DQ(0,  pk0a_{t},  0); DQ(1,  pk0a_{t},  4); DQ(2,  pk0a_{t},  8); DQ(3,  pk0a_{t}, 12);\n"
        ));
        src.push_str(&format!(
            "            DQ(4,  pk0a_{t}, 16); DQ(5,  pk0a_{t}, 20); DQ(6,  pk0a_{t}, 24); DQ(7,  pk0a_{t}, 28);\n"
        ));
        src.push_str(&format!(
            "            DQ(8,  pk1a_{t},  0); DQ(9,  pk1a_{t},  4); DQ(10, pk1a_{t},  8); DQ(11, pk1a_{t}, 12);\n"
        ));
        src.push_str(&format!(
            "            DQ(12, pk1a_{t}, 16); DQ(13, pk1a_{t}, 20); DQ(14, pk1a_{t}, 24); DQ(15, pk1a_{t}, 28);\n"
        ));
        src.push_str("#undef DQ\n");
        src.push_str(&format!(
            "            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_{t}, acc);\n\n",
            t = t
        ));
    }

    src.push_str("        }\n");

    src.push_str(include_str!("wmma_sweep_suffix.inc"));

    // Replace launch_bounds
    let src = src.replace(
        "__launch_bounds__(32, 2)",
        &format!("__launch_bounds__(32, {})", min_blocks),
    );

    src
}

fn run_variant(
    label: &str,
    module_name: &str,
    src: &str,
    a_gate: &[u8],
    a_up: &[u8],
    x_u8: &[u8],
    gm: usize,
    um: usize,
    k: usize,
    n: usize,
    warmup: usize,
    runs: usize,
) -> Option<f64> {
    let mut gpu = match Gpu::init_with_device(0) {
        Ok(g) => g,
        Err(_) => return None,
    };
    if gpu.ensure_kernel_public(module_name, src, "gemm_gate_up_mq4g256_lloyd_wmma").is_err() {
        eprintln!("  {} — JIT compile FAILED, skipping", label);
        return None;
    }

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

    for _ in 0..warmup {
        let mut ka = make_kernargs(
            d_ag.buf.as_ptr(), d_au.buf.as_ptr(), d_x.buf.as_ptr(),
            d_yg.buf.as_ptr(), d_yu.buf.as_ptr(), gm, um, k, n,
        );
        let _ = gpu.launch_kernel_blob(
            "gemm_gate_up_mq4g256_lloyd_wmma",
            [row_tiles, batch_tiles, 1], [32, 1, 1], 0, &mut ka,
        );
    }
    let _ = gpu.hip.device_synchronize();

    let mut times_us: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut ka = make_kernargs(
            d_ag.buf.as_ptr(), d_au.buf.as_ptr(), d_x.buf.as_ptr(),
            d_yg.buf.as_ptr(), d_yu.buf.as_ptr(), gm, um, k, n,
        );
        let t0 = Instant::now();
        let _ = gpu.launch_kernel_blob(
            "gemm_gate_up_mq4g256_lloyd_wmma",
            [row_tiles, batch_tiles, 1], [32, 1, 1], 0, &mut ka,
        );
        let _ = gpu.hip.device_synchronize();
        times_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[runs / 2];
    let w_bytes = ((gm + um) * (k / 256) * 160) as f64;
    let total = w_bytes + (n * k * 2) as f64 + (n * (gm + um) * 4) as f64;
    let gbps = total / (median_us / 1_000_000.0) / 1_073_741_824.0;

    println!("  {:45}  median={:8.1} µs  {:5.1} GiB/s", label, median_us, gbps);
    Some(median_us)
}

fn main() {
    let shapes = [
        (27648usize, 8192usize, 64usize, "27b N=64 (prefill)"),
        (6912usize, 8192usize, 64usize, "9b N=64 (prefill)"),
        (27648usize, 8192usize, 16usize, "27b N=16 (small batch)"),
        (27648usize, 8192usize, 1usize, "27b N=1 (decode-like)"),
    ];

    let k_unrolls = [1, 2, 4, 8, 16];
    let min_blocks_list = [2, 4, 6, 8, 10, 12, 16];

    let warmup = 3;
    let runs = 11;

    println!("WMMA tiling sweep — gfx1100 — K-unroll × launch_bounds");
    println!("{} runs (median), {} warmup\n", runs, warmup);

    for &(m, k, n, desc) in &shapes {
        let gm = m / 2;
        let um = m - gm;
        let a_gate = build_mq4(gm, k, 42);
        let a_up = build_mq4(um, k, 99);
        let x_f32: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.001).collect();
        let x_u8: Vec<u8> = x_f32.iter().flat_map(|&v| f32_to_f16_bytes(v)).collect();

        println!("=== M={} K={} N={} — {} ===", m, k, n, desc);

        // Baseline (original K2, lb 32,2)
        let baseline = run_variant(
            "BASELINE (K2, lb=2)",
            "sweep_baseline",
            ORIGINAL_SRC,
            &a_gate, &a_up, &x_u8, gm, um, k, n, warmup, runs,
        ).unwrap_or(f64::NAN);

        // Sweep K-unroll at default lb=2
        for &ku in &k_unrolls {
            if ku == 2 { continue; } // already baseline
            let src = generate_k_unroll_variant(ku, 2);
            let label = format!("K{} lb=2", ku);
            let med = run_variant(
                &label,
                &format!("sweep_k{}_lb2", ku),
                &src,
                &a_gate, &a_up, &x_u8, gm, um, k, n, warmup, runs,
            ).unwrap_or(f64::NAN);
            if !med.is_nan() && !baseline.is_nan() {
                let delta = (baseline - med) / baseline * 100.0;
                println!("    >>> vs baseline: {:+.1}% ({:.1} vs {:.1} µs)", delta, baseline, med);
            }
        }

        // Sweep launch_bounds at K2
        for &lb in &min_blocks_list {
            if lb == 2 { continue; }
            let mut src = ORIGINAL_SRC.to_string();
            src = src.replace("__launch_bounds__(32, 2)", &format!("__launch_bounds__(32, {})", lb));
            let label = format!("K2 lb={}", lb);
            let med = run_variant(
                &label,
                &format!("sweep_k2_lb{}", lb),
                &src,
                &a_gate, &a_up, &x_u8, gm, um, k, n, warmup, runs,
            ).unwrap_or(f64::NAN);
            if !med.is_nan() && !baseline.is_nan() {
                let delta = (baseline - med) / baseline * 100.0;
                println!("    >>> vs baseline: {:+.1}% ({:.1} vs {:.1} µs)", delta, baseline, med);
            }
        }

        // Best combo: sweep K-unroll at best lb from above
        // (We'll do K4 at lb=8 as the likely sweet spot)
        for &ku in &[4, 8] {
            for &lb in &[6, 8, 10] {
                let src = generate_k_unroll_variant(ku, lb);
                let label = format!("K{} lb={}", ku, lb);
                let med = run_variant(
                    &label,
                    &format!("sweep_k{}_lb{}", ku, lb),
                    &src,
                    &a_gate, &a_up, &x_u8, gm, um, k, n, warmup, runs,
                ).unwrap_or(f64::NAN);
                if !med.is_nan() && !baseline.is_nan() {
                    let delta = (baseline - med) / baseline * 100.0;
                    println!("    >>> vs baseline: {:+.1}% ({:.1} vs {:.1} µs)", delta, baseline, med);
                }
            }
        }

        println!();
    }
}
