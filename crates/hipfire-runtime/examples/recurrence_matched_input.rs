//! Phase 3a probe: feed HF transformers' bit-exact q/k/v/α/β tensors into
//! hipfire's `gated_delta_net_f32`, capture the per-position output, write to
//! HIPFIRE_DUMP_LA_STAGES format. Compare against HF's stage-13 dump to
//! localize whether hipfire's recurrence kernel itself diverges from HF's
//! `chunk_gated_delta_rule`, isolated from upstream input drift.
//!
//! Method: state starts at zero (Option-a clean reset). For each position
//! 0..N, upload HF's stage-11 (q), stage-12 (k), stage-10 (v), stage-6
//! (gated α), stage-7 (gated β) into per-position scratch buffers. Run
//! hipfire's recurrence one step. Download `s.dn_attn_out` (stage 13).
//! Repeat for all positions in sequence; the recurrence's state accumulates
//! across positions identically to inference.
//!
//! Outputs a HIPFIRE_DUMP_LA_STAGES-format dump file with stage_id=13 records
//! for layer_idx=0 (Q3.5-0.8B's first LA layer). `compare_la_stages.py` can
//! diff this against the existing HF dump's stage 13 to produce per-position
//! rL2 vs HF transformers BF16.
//!
//! Q3.5-0.8B dims (hard-coded; only model audited at Phase 3a):
//!   n_v_heads = n_k_heads = 16
//!   head_v_dim = head_k_dim = 128
//!   v_dim = k_dim = 2048
//!   state_size = 16 * 128 * 128 = 262144
//!
//! Usage:
//!   recurrence_matched_input --hf-dump <path> --out <path> [--positions N]

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::PathBuf;

const N_V_HEADS: usize = 16;
const HEAD_DIM: usize = 128;
const KV_DIM: usize = N_V_HEADS * HEAD_DIM; // 2048
const STATE_SIZE: usize = N_V_HEADS * HEAD_DIM * HEAD_DIM; // 262144
const HEADER_BYTES: usize = 32;

/// Per-record header layout matches `dump_la_stage` in qwen35.rs:
/// `[u32×8] layer_idx, pos, stage_id, n_elems, 0, 0, 0, 0`
fn read_record(buf: &[u8], offset: usize) -> Option<(u32, u32, u32, u32, usize)> {
    if offset + HEADER_BYTES > buf.len() {
        return None;
    }
    let hdr = &buf[offset..offset + HEADER_BYTES];
    let layer = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    let pos = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    let stage = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
    let n_elems = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let next = offset + HEADER_BYTES + n_elems * 4;
    if next > buf.len() {
        return None;
    }
    Some((layer, pos, stage, n_elems as u32, next))
}

fn read_f32(buf: &[u8], offset: usize, n_elems: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_elems];
    let nbytes = n_elems * 4;
    let dst = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, nbytes)
    };
    dst.copy_from_slice(&buf[offset + HEADER_BYTES..offset + HEADER_BYTES + nbytes]);
    out
}

fn write_record(
    f: &mut std::fs::File,
    layer_idx: u32,
    pos: u32,
    stage_id: u32,
    data: &[f32],
) {
    let hdr: [u32; 8] = [layer_idx, pos, stage_id, data.len() as u32, 0, 0, 0, 0];
    for w in hdr.iter() {
        f.write_all(&w.to_le_bytes()).unwrap();
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    f.write_all(bytes).unwrap();
}

fn print_usage() -> ! {
    eprintln!(
        "Usage:\n  recurrence_matched_input --hf-dump <path> --out <path> [--positions N=2048]"
    );
    std::process::exit(1);
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut hf_dump: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut n_positions: usize = 2048;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--hf-dump" => {
                hf_dump = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--positions" => {
                n_positions = argv[i + 1].parse().expect("--positions must be integer");
                i += 2;
            }
            "-h" | "--help" => print_usage(),
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
            }
        }
    }
    let hf_dump = hf_dump.unwrap_or_else(|| print_usage());
    let out_path = out_path.unwrap_or_else(|| print_usage());

    // ----- read HF dump file -----
    eprintln!("reading HF dump: {}", hf_dump.display());
    let mut hf_data = Vec::new();
    std::fs::File::open(&hf_dump)
        .expect("open hf dump")
        .read_to_end(&mut hf_data)
        .expect("read hf dump");
    eprintln!("  {} MB", hf_data.len() as f64 / 1_048_576.0);

    // Index records: per (stage, pos) → byte offset
    // For Phase 3a we only care about stages 6, 7, 10, 11, 12 at layer 0
    let mut by_stage_pos: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::new();
    let mut cursor = 0;
    let mut n_records = 0;
    while let Some((layer, pos, stage, _n, next)) = read_record(&hf_data, cursor) {
        if layer == 0 && matches!(stage, 6 | 7 | 10 | 11 | 12) {
            by_stage_pos.insert((stage, pos), cursor);
        }
        cursor = next;
        n_records += 1;
    }
    eprintln!(
        "  indexed {} records, {} stage/pos keys for layer 0 (stages 6/7/10/11/12)",
        n_records,
        by_stage_pos.len()
    );

    // ----- GPU init -----
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("arch={}", gpu.arch);

    // ----- pre-allocate scratch buffers -----
    let q_buf = gpu.zeros(&[KV_DIM], rdna_compute::DType::F32).expect("alloc q");
    let k_buf = gpu.zeros(&[KV_DIM], rdna_compute::DType::F32).expect("alloc k");
    let v_buf = gpu.zeros(&[KV_DIM], rdna_compute::DType::F32).expect("alloc v");
    let alpha_buf = gpu.zeros(&[N_V_HEADS], rdna_compute::DType::F32).expect("alloc alpha");
    let beta_buf = gpu.zeros(&[N_V_HEADS], rdna_compute::DType::F32).expect("alloc beta");
    let state = gpu.zeros(&[STATE_SIZE], rdna_compute::DType::F32).expect("alloc state");
    let output = gpu.zeros(&[KV_DIM], rdna_compute::DType::F32).expect("alloc output");

    // ----- output file -----
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir parent");
    }
    let mut out_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&out_path)
        .expect("open out");
    eprintln!("writing per-position stage-13 to {}", out_path.display());

    // ----- per-position recurrence -----
    let t0 = std::time::Instant::now();
    for pos in 0..n_positions {
        // Read HF inputs at this position
        let q_off = by_stage_pos
            .get(&(11, pos as u32))
            .copied()
            .unwrap_or_else(|| panic!("missing stage 11 at pos {pos}"));
        let k_off = by_stage_pos
            .get(&(12, pos as u32))
            .copied()
            .unwrap_or_else(|| panic!("missing stage 12 at pos {pos}"));
        let v_off = by_stage_pos
            .get(&(10, pos as u32))
            .copied()
            .unwrap_or_else(|| panic!("missing stage 10 at pos {pos}"));
        let alpha_off = by_stage_pos
            .get(&(6, pos as u32))
            .copied()
            .unwrap_or_else(|| panic!("missing stage 6 at pos {pos}"));
        let beta_off = by_stage_pos
            .get(&(7, pos as u32))
            .copied()
            .unwrap_or_else(|| panic!("missing stage 7 at pos {pos}"));

        let q_data = read_f32(&hf_data, q_off, KV_DIM);
        let k_data = read_f32(&hf_data, k_off, KV_DIM);
        let v_data = read_f32(&hf_data, v_off, KV_DIM);
        let alpha_data = read_f32(&hf_data, alpha_off, N_V_HEADS);
        let beta_data = read_f32(&hf_data, beta_off, N_V_HEADS);

        // Upload to pre-allocated GPU buffers (avoid per-position alloc)
        let bytes_kv = unsafe {
            std::slice::from_raw_parts(q_data.as_ptr() as *const u8, q_data.len() * 4)
        };
        gpu.hip.memcpy_htod(&q_buf.buf, bytes_kv).expect("upload q");
        let bytes_kv = unsafe {
            std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4)
        };
        gpu.hip.memcpy_htod(&k_buf.buf, bytes_kv).expect("upload k");
        let bytes_kv = unsafe {
            std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4)
        };
        gpu.hip.memcpy_htod(&v_buf.buf, bytes_kv).expect("upload v");
        let bytes_ab = unsafe {
            std::slice::from_raw_parts(alpha_data.as_ptr() as *const u8, alpha_data.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&alpha_buf.buf, bytes_ab)
            .expect("upload alpha");
        let bytes_ab = unsafe {
            std::slice::from_raw_parts(beta_data.as_ptr() as *const u8, beta_data.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&beta_buf.buf, bytes_ab)
            .expect("upload beta");

        // Run hipfire's recurrence (state accumulates in-place)
        gpu.gated_delta_net_f32(
            &q_buf, &k_buf, &v_buf, &alpha_buf, &beta_buf,
            &state, &output, 1, N_V_HEADS, HEAD_DIM,
        )
        .expect("gated_delta_net_f32");

        // Download output, write to dump file as stage 13
        let out_data = gpu.download_f32(&output).expect("download output");
        write_record(&mut out_file, 0, pos as u32, 13, &out_data);

        if pos == 0 || (pos + 1) % 256 == 0 {
            eprintln!(
                "  pos {:4}/{}: {:.1}s elapsed",
                pos + 1,
                n_positions,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    out_file.flush().unwrap();
    let size_mb = std::fs::metadata(&out_path).expect("stat").len() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "wrote {} ({:.1} MB) in {:.1}s",
        out_path.display(),
        size_mb,
        t0.elapsed().as_secs_f64()
    );
}
