//! D0.4 — isolated rmsnorm probe (engine-drift residual audit pattern-hunt).
//!
//! Reads stage-0 records (LA-block input residual) from an
//! HIPFIRE_DUMP_LA_STAGES-format dump produced by either the HF transformers
//! oracle (`scripts/dump_hf_la_stages.py`) or hipfire's
//! `dump_qwen35_hidden_states`. For each position, uploads the input vector,
//! runs a single rmsnorm kernel against the model's `input_layernorm` weight,
//! downloads the output, and writes it as a stage-1 record to the output
//! dump file. Compare against the same HF dump's stage-1 records with
//! `scripts/compare_la_stages.py --stages 1` to get per-position rL2.
//!
//! The point is to isolate ONE kernel's intrinsic drift from upstream noise.
//! Full-forward stage-1 measures ~0.0055 rL2 because stage-0 itself is at
//! 0.0053; isolated stage-1 measures rmsnorm's contribution given bit-exact
//! input.
//!
//! Variant selection: `--kernel <entry>`. The kernel must live in
//! `kernels/src/rmsnorm.hip` (or any module compiled into RMSNORM_SRC) and
//! must have the signature
//!   `(const float* x, const float* weight, float* out, int n, float eps)`.
//! Day 0/1 entries:
//!   rmsnorm_f32           — baseline (default)
//!   rmsnorm_f32_f64acc    — D0.1: fp64 sum-of-squares + rsqrt
//!
//! Usage:
//!   rmsnorm_isolated --model <hfq> --hf-dump <hf-stages.bin> --out <out.bin>
//!                    [--layer N=0] [--n-positions N=2048]
//!                    [--kernel rmsnorm_f32] [--eps 1e-6]
//!                    [--shared-bytes-per-thread N]    (auto: 8 if kernel ends `_f64acc`, else 4)

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::PathBuf;

use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

const HEADER_BYTES: usize = 32;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let f = if exp == 0 {
        if frac == 0 { 0.0 } else {
            let f = frac as f32 / 1024.0;
            f * 2f32.powi(-14)
        }
    } else if exp == 0x1f {
        if frac == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        let f = 1.0 + (frac as f32 / 1024.0);
        f * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -f } else { f }
}

/// Header layout matches `dump_la_stage` in qwen35.rs:
/// `[u32×8] layer_idx, pos, stage_id, n_elems, 0, 0, 0, 0`
fn read_record(buf: &[u8], offset: usize) -> Option<(u32, u32, u32, usize, usize)> {
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
    Some((layer, pos, stage, n_elems, next))
}

fn read_f32(buf: &[u8], offset: usize, n_elems: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_elems];
    let nbytes = n_elems * 4;
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, nbytes) };
    dst.copy_from_slice(&buf[offset + HEADER_BYTES..offset + HEADER_BYTES + nbytes]);
    out
}

fn write_record(f: &mut std::fs::File, layer: u32, pos: u32, stage: u32, data: &[f32]) {
    let hdr: [u32; 8] = [layer, pos, stage, data.len() as u32, 0, 0, 0, 0];
    for w in hdr.iter() {
        f.write_all(&w.to_le_bytes()).unwrap();
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    f.write_all(bytes).unwrap();
}

fn print_usage() -> ! {
    eprintln!(
        "Usage:\n  rmsnorm_isolated --model <hfq> --hf-dump <path> --out <path>\n\
         \x20                  [--layer N=0] [--n-positions N=2048]\n\
         \x20                  [--kernel rmsnorm_f32] [--eps 1e-6]\n\
         \x20                  [--shared-bytes-per-thread N]\n\
         \x20                  [--stage-out STAGE=1]"
    );
    std::process::exit(1);
}

fn load_attn_norm_weight(hfq: &HfqFile, layer: usize) -> Vec<f32> {
    let suffix = format!("layers.{layer}.input_layernorm.weight");
    // Try the language-model-prefixed name first (Qwen3.5 multimodal nesting),
    // fall back to the bare name. Mirrors load_norm_weight in qwen35.rs.
    let candidates = [
        format!("model.language_model.{suffix}"),
        format!("model.{suffix}"),
        suffix.clone(),
    ];
    let mut found: Option<(u8, Vec<u8>)> = None;
    for name in &candidates {
        if let Some((info, data)) = hfq.tensor_data_vec(name) {
            eprintln!("  loaded weight: {name} (qt={}, {} bytes)", info.quant_type, data.len());
            found = Some((info.quant_type, data));
            break;
        }
    }
    let (qt, data) = found.unwrap_or_else(|| {
        panic!(
            "input_layernorm.weight not found for layer {layer}; tried: {:?}",
            candidates
        )
    });
    let mut f32_data: Vec<f32> = match qt {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => panic!("expected F16/F32 (qt=1|2), got qt={qt}"),
    };
    // Qwen3.5 RMSNorm convention: stored as offset from 1.0
    // (forward = x * rsqrt(var+eps) * (1 + weight))
    for v in &mut f32_data {
        *v += 1.0;
    }
    f32_data
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut hf_dump: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut layer: usize = 0;
    let mut n_positions: usize = 2048;
    let mut kernel_entry: String = "rmsnorm_f32".to_string();
    let mut eps: f32 = 1e-6;
    let mut shared_bytes_per_thread: Option<u32> = None;
    let mut stage_out: u32 = 1;
    let mut dump_weight: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--hf-dump" => { hf_dump = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--out" => { out_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--layer" => { layer = argv[i + 1].parse().expect("--layer integer"); i += 2; }
            "--n-positions" => { n_positions = argv[i + 1].parse().expect("--n-positions integer"); i += 2; }
            "--kernel" => { kernel_entry = argv[i + 1].clone(); i += 2; }
            "--eps" => { eps = argv[i + 1].parse().expect("--eps float"); i += 2; }
            "--shared-bytes-per-thread" => {
                shared_bytes_per_thread = Some(argv[i + 1].parse().expect("--shared-bytes-per-thread u32"));
                i += 2;
            }
            "--stage-out" => { stage_out = argv[i + 1].parse().expect("--stage-out u32"); i += 2; }
            "--dump-weight" => { dump_weight = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "-h" | "--help" => print_usage(),
            other => { eprintln!("unknown arg: {other}"); print_usage(); }
        }
    }
    let model = model.unwrap_or_else(|| print_usage());
    let hf_dump = hf_dump.unwrap_or_else(|| print_usage());
    let out_path = out_path.unwrap_or_else(|| print_usage());

    // Auto-pick shared-mem-per-thread from kernel name if caller didn't override.
    // The convention here mirrors rmsnorm_batched (dispatch.rs:12690): an
    // f64-accumulating variant needs 8 bytes per thread; everything else needs 4.
    let shared_bpt = shared_bytes_per_thread.unwrap_or_else(|| {
        if kernel_entry.ends_with("_f64acc") { 8 } else { 4 }
    });

    eprintln!("rmsnorm_isolated:");
    eprintln!("  model:        {}", model.display());
    eprintln!("  hf-dump:      {}", hf_dump.display());
    eprintln!("  out:          {}", out_path.display());
    eprintln!("  layer:        {layer}");
    eprintln!("  n-positions:  {n_positions}");
    eprintln!("  kernel entry: {kernel_entry}");
    eprintln!("  eps:          {eps}");
    eprintln!("  shared bpt:   {shared_bpt}");
    eprintln!("  stage out:    {stage_out}");

    // ----- read HF dump file -----
    let mut hf_data = Vec::new();
    std::fs::File::open(&hf_dump)
        .expect("open hf dump")
        .read_to_end(&mut hf_data)
        .expect("read hf dump");
    eprintln!("hf dump: {} MB", hf_data.len() as f64 / 1_048_576.0);

    // Index stage-0 records at requested layer by position.
    let mut by_pos: std::collections::HashMap<u32, (usize, usize)> =
        std::collections::HashMap::new();
    let mut cursor = 0usize;
    let mut n_total = 0usize;
    let mut dim_observed: Option<usize> = None;
    while let Some((lyr, pos, stage, n_elems, next)) = read_record(&hf_data, cursor) {
        if (lyr as usize) == layer && stage == 0 {
            by_pos.insert(pos, (cursor, n_elems));
            if dim_observed.map(|d| d != n_elems).unwrap_or(false) {
                panic!("stage-0 n_elems varies across positions (got {n_elems}, expected {})",
                       dim_observed.unwrap());
            }
            dim_observed = Some(n_elems);
        }
        cursor = next;
        n_total += 1;
    }
    let dim = dim_observed.unwrap_or_else(|| {
        panic!("no stage-0 records found for layer {layer} in {}", hf_dump.display())
    });
    eprintln!("  indexed {n_total} records total; {} stage-0 records at layer {layer}, dim={dim}",
              by_pos.len());

    if n_positions > by_pos.len() {
        eprintln!("  WARNING: requested {n_positions} positions, only {} available; clamping",
                  by_pos.len());
        n_positions = by_pos.len();
    }

    // ----- open hfq + load attn_norm weight -----
    let hfq = HfqFile::open(&model).expect("open model");
    let weight_f32 = load_attn_norm_weight(&hfq, layer);
    assert_eq!(
        weight_f32.len(), dim,
        "weight len {} != dim {}", weight_f32.len(), dim
    );

    // Optional weight dump (D0.3 needs the effective weight as f32 bytes).
    if let Some(dw_path) = &dump_weight {
        if let Some(parent) = dw_path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir parent of --dump-weight");
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(
                weight_f32.as_ptr() as *const u8,
                weight_f32.len() * 4,
            )
        };
        std::fs::write(dw_path, bytes).expect("write --dump-weight");
        eprintln!("wrote effective weight (post +1.0 offset) to {}",
                  dw_path.display());
    }

    // ----- GPU init + compile chosen kernel -----
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch={}", gpu.arch);
    gpu.ensure_kernel_public("rmsnorm", rdna_compute::RMSNORM_SRC, &kernel_entry)
        .unwrap_or_else(|e| panic!("ensure_kernel {kernel_entry}: {e}"));

    let x_buf = gpu.zeros(&[dim], rdna_compute::DType::F32).expect("alloc x");
    let w_buf = gpu.upload_f32(&weight_f32, &[dim]).expect("upload weight");
    let out_buf = gpu.zeros(&[dim], rdna_compute::DType::F32).expect("alloc out");

    let block_size: u32 = 256u32.min(dim as u32);
    let shared_mem: u32 = block_size * shared_bpt;
    eprintln!("launch: grid=[1,1,1] block=[{block_size},1,1] shared={shared_mem}");

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

    // ----- per-position rmsnorm launch -----
    let t0 = std::time::Instant::now();
    let mut sorted_pos: Vec<u32> = by_pos.keys().copied().collect();
    sorted_pos.sort_unstable();
    let sorted_pos = &sorted_pos[..n_positions];

    let x_ptr = x_buf.buf.as_ptr();
    let w_ptr = w_buf.buf.as_ptr();
    let out_ptr = out_buf.buf.as_ptr();
    let n_val: i32 = dim as i32;

    for (i_iter, &pos) in sorted_pos.iter().enumerate() {
        let (offset, n_elems) = by_pos[&pos];
        debug_assert_eq!(n_elems, dim);
        let x_host = read_f32(&hf_data, offset, n_elems);
        let bytes = unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4)
        };
        gpu.hip.memcpy_htod(&x_buf.buf, bytes).expect("upload x");

        let mut kernargs = hip_bridge::KernargBlob::new();
        kernargs.push_ptr(x_ptr);
        kernargs.push_ptr(w_ptr);
        kernargs.push_ptr(out_ptr);
        kernargs.push_i32(n_val);
        kernargs.push_f32(eps);
        gpu.launch_kernel_blob(
            &kernel_entry,
            [1, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            kernargs.as_mut_slice(),
        )
        .expect("launch rmsnorm");

        let out_host = gpu.download_f32(&out_buf).expect("download out");
        write_record(&mut out_file, layer as u32, pos, stage_out, &out_host);

        if i_iter == 0 || (i_iter + 1) % 256 == 0 {
            eprintln!(
                "  pos {:4}/{n_positions}: {:.1}s elapsed",
                i_iter + 1,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    out_file.flush().unwrap();
    let size_mb =
        std::fs::metadata(&out_path).expect("stat").len() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "wrote {} ({:.1} MB) in {:.1}s",
        out_path.display(),
        size_mb,
        t0.elapsed().as_secs_f64()
    );
}
