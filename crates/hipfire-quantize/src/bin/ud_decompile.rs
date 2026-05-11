//! ud_decompile — extract per-tensor bit allocation from a llama.cpp GGUF
//! (typically an Unsloth Dynamic UD-Q4_K_XL or UD-Q4_K_M).
//!
//! Phase A Step 6b (per `docs/plans/qwen35-mq4-quality-gap.md` §5): the cheap
//! shortcut for L5d (per-tensor bit allocation). Instead of computing
//! `Σ(Act²)` + ZD score ourselves (which requires the Step 4 imatrix
//! collector), we read Unsloth's chosen quant type per tensor out of their
//! published GGUF and map it onto our format family.
//!
//! What this DOES recover:
//!   - Per-tensor bit allocation decisions (which tensors got bumped to
//!     Q5_K / Q6_K / Q8 / FP16; which stayed at the default Q4_K).
//!   - Aggregate bpw budget across tensor classes.
//!   - A JSON sidecar the quantizer can consume as a `--kmap-file` override
//!     to mimic Unsloth's choices on hipfire format equivalents.
//!
//! What this does NOT recover:
//!   - Per-channel `Σ act²` values for L5c (activation-weighted LS).
//!     Unsloth's chosen scales are the *result* of imatrix-aware LS — the
//!     imatrix itself is consumed at quant-time and not recoverable from
//!     the quantized weights in any tractable way. Step 5 (the dominant
//!     lever) still needs the Step 4 imatrix data.
//!
//! Usage:
//!   ud_decompile --input <path-to.gguf> \
//!                [--output <path-to.kmap.json>=<input-stem>.kmap.json] \
//!                [--summary-only]
//!
//! Output: JSON sidecar with schema:
//!   {
//!     "schema_version": 1,
//!     "source_gguf": "<input path>",
//!     "source_gguf_sha256": "<sha256>",
//!     "source_meta": { "general.architecture": "...", "general.name": "...", ... },
//!     "summary": {
//!       "total_tensors": <N>,
//!       "total_params": <P>,
//!       "type_distribution": {"Q4_K": {...}, "Q6_K": {...}, ...},
//!       "promoted_tensors": [...]   // anything above the modal 4-bit class
//!     },
//!     "per_tensor": [
//!       {"name": "...", "ggml_type": "Q4_K", "shape": [...], "params": N,
//!        "suggested_hipfire_qt": "MQ4G256"},
//!       ...
//!     ]
//!   }

// `gguf_input.rs` is a module of the parent crate (main.rs siblings, not a
// library). Cargo's bin layout doesn't share modules with main.rs by default,
// so reach it via #[path]. This keeps ud_decompile self-contained without
// promoting the crate to lib+bin (which would touch every consumer of
// `hipfire-quantize`).
#[path = "../gguf_input.rs"]
mod gguf_input;
use gguf_input::{GgmlType, GgufFile, MetaValue};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn print_usage() {
    eprintln!(
        "Usage:\n  ud_decompile --input <path-to.gguf> [--output <path>=<input-stem>.kmap.json] [--summary-only]\n\
         \n\
         Decompiles a llama.cpp / Unsloth GGUF into a hipfire per-tensor bit-allocation\n\
         sidecar. See docs/plans/qwen35-mq4-quality-gap.md §5 Step 6b for the lever\n\
         this implements."
    );
}

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    summary_only: bool,
}

fn parse_args() -> Args {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut summary_only = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().map(PathBuf::from),
            "--output" => output = it.next().map(PathBuf::from),
            "--summary-only" => summary_only = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown arg: {a}");
                print_usage();
                std::process::exit(2);
            }
        }
    }
    let input = input.unwrap_or_else(|| {
        print_usage();
        std::process::exit(2);
    });
    Args { input, output, summary_only }
}

/// Map a llama.cpp GGML quant type to the closest hipfire qt name.
///
/// The mapping is **format-family approximate**, not byte-equivalent. The goal
/// is "preserve UD's per-tensor importance ordering on the hipfire format
/// family" — i.e., tensors UD chose to bump get bumped here too. Exact-match
/// is impossible because llama.cpp's K-quants (Q4_K / Q5_K / Q6_K) carry
/// per-32 sub-block scales that hipfire's MQ-family doesn't, so we substitute
/// the closest-bpw hipfire equivalent.
///
/// Format-family rationale per row:
///   - F32 / F16 / BF16 → F16 (norms, biases, scalars; hipfire stores these
///     uncompressed too)
///   - Q4_0 / Q4_1 → HFQ4G256 (unrotated INT4 — closest match by spirit)
///   - Q5_0 / Q5_1 → HFQ6G256 (we don't ship a 5-bit; bump to 6-bit floor)
///   - Q8_0 / Q8_1 / Q8_K → Q8_F16 (same bpw, same role)
///   - Q2K / Q3K → MQ3G256 (smaller is research-grade; Q2/Q3K are Unsloth's
///     "compress hard" levels — fall back to our MQ3 family)
///   - Q4K → MQ4G256 (the *default*; equivalent role)
///   - Q5K → MQ6G256 (no 5-bit family member — bump to 6-bit)
///   - Q6K → MQ6G256 (same bpw, equivalent role)
fn suggested_hipfire_qt(ggml: GgmlType) -> &'static str {
    match ggml {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => "F16",
        GgmlType::Q4_0 | GgmlType::Q4_1 => "HFQ4G256",
        GgmlType::Q5_0 | GgmlType::Q5_1 => "HFQ6G256",
        GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::Q8K => "Q8_F16",
        GgmlType::Q2K | GgmlType::Q3K => "MQ3G256",
        GgmlType::Q4K => "MQ4G256",
        GgmlType::Q5K | GgmlType::Q6K => "MQ6G256",
        // IQ family (Unsloth UD recipes use IQ4_XS / IQ3_S / IQ2_XS for
        // "promoted-down" tensors). Map to the closest-bpw hipfire family:
        //   IQ4_XS (4.25 bpw) → MQ4G256 (4.25 bpw) — exact match in bpw
        //   IQ4_NL (4.5 bpw)  → HFQ4G256 (4.5 bpw)  — same bpw, unrotated
        //   IQ3_S (3.4375)    → MQ3G256 (3.25 bpw)  — closest 3-bit
        //   IQ3_XXS (3.0625)  → MQ3G256
        //   IQ2_* / IQ1_*     → MQ3G256 floor (we don't ship 2-bit yet; mark
        //                       for promotion when sub-3-bit family lands)
        //   Iq1M / TQ family  → MQ3G256 floor (same reasoning)
        GgmlType::Iq4Xs => "MQ4G256",
        GgmlType::Iq4Nl => "HFQ4G256",
        GgmlType::Iq3S | GgmlType::Iq3Xxs => "MQ3G256",
        GgmlType::Iq2Xxs | GgmlType::Iq2Xs | GgmlType::Iq2S
        | GgmlType::Iq1S | GgmlType::Iq1M
        | GgmlType::Tq1_0 | GgmlType::Tq2_0 => "MQ3G256",
    }
}

/// Heuristic bpw per ggml type. Matches llama.cpp's documented numbers
/// (per `tools/quantize/README.md`); useful for the summary table.
fn ggml_bpw(ggml: GgmlType) -> f32 {
    match ggml {
        GgmlType::F32 => 32.0,
        GgmlType::F16 | GgmlType::BF16 => 16.0,
        GgmlType::Q4_0 => 4.5,
        GgmlType::Q4_1 => 5.0,
        GgmlType::Q5_0 => 5.5,
        GgmlType::Q5_1 => 6.0,
        GgmlType::Q8_0 | GgmlType::Q8_1 => 8.5,
        GgmlType::Q2K => 2.625,
        GgmlType::Q3K => 3.4375,
        GgmlType::Q4K => 4.5,
        GgmlType::Q5K => 5.5,
        GgmlType::Q6K => 6.5625,
        GgmlType::Q8K => 8.5,
        GgmlType::Iq2Xxs => 2.0625,
        GgmlType::Iq2Xs => 2.3125,
        GgmlType::Iq3Xxs => 3.0625,
        GgmlType::Iq1S => 1.5625,
        GgmlType::Iq4Nl => 4.5,
        GgmlType::Iq3S => 3.4375,
        GgmlType::Iq2S => 2.5625,
        GgmlType::Iq4Xs => 4.25,
        GgmlType::Iq1M => 1.75,
        GgmlType::Tq1_0 => 1.6875,
        GgmlType::Tq2_0 => 2.0625,
    }
}

fn ggml_name(ggml: GgmlType) -> &'static str {
    match ggml {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::BF16 => "BF16",
        GgmlType::Q4_0 => "Q4_0",
        GgmlType::Q4_1 => "Q4_1",
        GgmlType::Q5_0 => "Q5_0",
        GgmlType::Q5_1 => "Q5_1",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q8_1 => "Q8_1",
        GgmlType::Q2K => "Q2_K",
        GgmlType::Q3K => "Q3_K",
        GgmlType::Q4K => "Q4_K",
        GgmlType::Q5K => "Q5_K",
        GgmlType::Q6K => "Q6_K",
        GgmlType::Q8K => "Q8_K",
        GgmlType::Iq2Xxs => "IQ2_XXS",
        GgmlType::Iq2Xs => "IQ2_XS",
        GgmlType::Iq3Xxs => "IQ3_XXS",
        GgmlType::Iq1S => "IQ1_S",
        GgmlType::Iq4Nl => "IQ4_NL",
        GgmlType::Iq3S => "IQ3_S",
        GgmlType::Iq2S => "IQ2_S",
        GgmlType::Iq4Xs => "IQ4_XS",
        GgmlType::Iq1M => "IQ1_M",
        GgmlType::Tq1_0 => "TQ1_0",
        GgmlType::Tq2_0 => "TQ2_0",
    }
}

/// Convert a small set of useful provenance metadata keys to JSON. The full
/// GGUF metadata table can be huge (Unsloth GGUFs carry per-token tokenizer
/// data, etc.) — restrict to the fields a future quantizer wants for
/// provenance + reproducibility.
fn extract_provenance_meta(gguf: &GgufFile) -> Value {
    let keys = [
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.basename",
        "general.size_label",
        "general.quantization_version",
        "general.organization",
        "general.finetune",
        "general.languages",
        // Architecture-specific dims (Qwen3.5 / Qwen3.6 / Llama)
        "qwen3.embedding_length",
        "qwen3moe.embedding_length",
        "llama.embedding_length",
        "qwen3.block_count",
        "qwen3moe.block_count",
        "llama.block_count",
    ];
    let mut out = serde_json::Map::new();
    for k in keys {
        if let Some(v) = gguf.meta(k) {
            let json_v = match v {
                MetaValue::U8(x) => json!(x),
                MetaValue::I8(x) => json!(x),
                MetaValue::U16(x) => json!(x),
                MetaValue::I16(x) => json!(x),
                MetaValue::U32(x) => json!(x),
                MetaValue::I32(x) => json!(x),
                MetaValue::F32(x) => json!(x),
                MetaValue::Bool(x) => json!(x),
                MetaValue::String(x) => json!(x),
                MetaValue::U64(x) => json!(x),
                MetaValue::I64(x) => json!(x),
                MetaValue::F64(x) => json!(x),
                // Arrays in metadata are typically token-tables; emit as
                // empty marker rather than inlining megabytes of data.
                MetaValue::Array(_) => json!("<array, omitted>"),
            };
            out.insert(k.to_string(), json_v);
        }
    }
    Value::Object(out)
}

fn main() {
    let args = parse_args();
    if !args.input.exists() {
        eprintln!("error: input file not found: {}", args.input.display());
        std::process::exit(1);
    }

    eprintln!("ud_decompile: opening {} ...", args.input.display());
    let gguf = GgufFile::open(&args.input).unwrap_or_else(|e| {
        eprintln!("error: failed to parse GGUF: {e}");
        std::process::exit(1);
    });
    eprintln!("ud_decompile: {} tensors loaded", gguf.tensors.len());
    // sha256: deliberately omitted — users can `sha256sum <input>` if needed.
    // Keeps this tool dependency-free (no sha2/hex crate adds).
    let sha256: String = String::new();

    // ── Aggregate per ggml type ────────────────────────────────────────────
    let mut type_counts: BTreeMap<&'static str, (usize, usize, f32)> = BTreeMap::new();
    let mut total_params: usize = 0;
    let mut total_bytes: u64 = 0;
    for t in &gguf.tensors {
        let n = t.numel();
        total_params += n;
        let bytes = t.dtype.tensor_bytes(n);
        total_bytes += bytes as u64;
        let entry = type_counts.entry(ggml_name(t.dtype)).or_insert((0, 0, 0.0));
        entry.0 += 1;             // tensor count
        entry.1 += n;             // param count
        entry.2 += bytes as f32;  // total bytes (track as f32 for percentage math)
    }

    // ── Identify the modal 4-bit class; tensors above it are "promoted" ──
    // Modal is the type whose param share is biggest among the 4-bit family
    // {Q4_0, Q4_1, Q4_K, IQ4_XS, IQ4_NL}. Unsloth UD-Q4_K_XL has Q4_K as the
    // modal class with some tensors bumped to Q5_K/Q6_K/Q8_0/F32 ("promoted")
    // and some demoted to IQ3_S/IQ2_XXS for non-critical tensors. If no
    // 4-bit class exists (e.g. an FP16 anchor file), modal stays None.
    let four_bit_class = ["Q4_K", "IQ4_XS", "Q4_0", "Q4_1", "IQ4_NL"];
    let modal_qt: Option<&str> = four_bit_class
        .iter()
        .filter_map(|&qt| type_counts.get(qt).map(|c| (qt, c.1)))
        .max_by_key(|(_qt, params)| *params)
        .map(|(qt, _)| qt);
    let promoted_threshold_bpw = modal_qt
        .and_then(|qt| {
            // Use the BPW of the modal type as the floor; anything strictly above is "promoted".
            match qt {
                "Q4_K" => Some(ggml_bpw(GgmlType::Q4K)),
                "IQ4_XS" => Some(ggml_bpw(GgmlType::Iq4Xs)),
                "Q4_0" => Some(ggml_bpw(GgmlType::Q4_0)),
                "Q4_1" => Some(ggml_bpw(GgmlType::Q4_1)),
                "IQ4_NL" => Some(ggml_bpw(GgmlType::Iq4Nl)),
                _ => None,
            }
        })
        .unwrap_or(0.0);

    // ── Per-tensor JSON ────────────────────────────────────────────────────
    let mut per_tensor: Vec<Value> = Vec::with_capacity(gguf.tensors.len());
    let mut promoted_names: Vec<(String, &'static str)> = Vec::new();
    for t in &gguf.tensors {
        let tname = ggml_name(t.dtype);
        let bpw = ggml_bpw(t.dtype);
        if bpw > promoted_threshold_bpw {
            promoted_names.push((t.name.clone(), tname));
        }
        per_tensor.push(json!({
            "name": t.name,
            "ggml_type": tname,
            "ggml_bpw": bpw,
            "shape": t.shape,
            "params": t.numel(),
            "bytes": t.dtype.tensor_bytes(t.numel()),
            "suggested_hipfire_qt": suggested_hipfire_qt(t.dtype),
        }));
    }
    promoted_names.sort_by(|a, b| a.0.cmp(&b.0));

    // ── Type-distribution summary ──────────────────────────────────────────
    let mut type_distribution = serde_json::Map::new();
    for (name, (tensor_count, param_count, byte_count)) in &type_counts {
        type_distribution.insert(name.to_string(), json!({
            "tensors": tensor_count,
            "params": param_count,
            "bytes": *byte_count as u64,
            "fraction_of_params": *param_count as f64 / total_params as f64,
        }));
    }

    let aggregate_bpw = (total_bytes as f64 * 8.0) / total_params as f64;

    let summary = json!({
        "total_tensors": gguf.tensors.len(),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "aggregate_bpw": aggregate_bpw,
        "modal_4bit_class": modal_qt,
        "type_distribution": type_distribution,
        "promoted_tensor_count": promoted_names.len(),
        "promoted_tensors": promoted_names.iter()
            .map(|(n, qt)| format!("{n}: {qt}"))
            .collect::<Vec<_>>(),
    });

    let provenance = extract_provenance_meta(&gguf);

    let root = json!({
        "schema_version": 1,
        "tool": "ud_decompile",
        "source_gguf": args.input.display().to_string(),
        "source_gguf_sha256": sha256,
        "source_meta": provenance,
        "summary": summary,
        "per_tensor": per_tensor,
    });

    // ── Console summary (always) ───────────────────────────────────────────
    eprintln!();
    eprintln!("=== Source ===");
    eprintln!("  path:    {}", args.input.display());
    if !sha256.is_empty() {
        eprintln!("  sha256:  {}", sha256);
    }
    if let Some(arch) = provenance.get("general.architecture").and_then(|v| v.as_str()) {
        eprintln!("  arch:    {arch}");
    }
    if let Some(name) = provenance.get("general.name").and_then(|v| v.as_str()) {
        eprintln!("  name:    {name}");
    }
    eprintln!();
    eprintln!("=== Aggregate ===");
    eprintln!("  total tensors:   {}", gguf.tensors.len());
    eprintln!("  total params:    {total_params}");
    eprintln!("  total bytes:     {total_bytes}");
    eprintln!("  aggregate bpw:   {aggregate_bpw:.4}");
    if let Some(qt) = modal_qt {
        eprintln!("  modal 4-bit:     {qt}");
    }
    eprintln!();
    eprintln!("=== Type distribution ===");
    eprintln!("  {:<8} {:>8} {:>14} {:>10}", "type", "tensors", "params", "share");
    eprintln!("  {:-<8} {:->8} {:->14} {:->10}", "", "", "", "");
    let mut entries: Vec<_> = type_counts.iter().collect();
    entries.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (name, (tensors, params, _bytes)) in &entries {
        let share = (*params as f64 / total_params as f64) * 100.0;
        eprintln!("  {:<8} {:>8} {:>14} {:>9.2}%", name, tensors, params, share);
    }

    // Only meaningful when there IS a modal 4-bit class. For a BF16 / F16
    // anchor file every weight tensor is trivially "above modal 0 bpw" — skip
    // that noise.
    if modal_qt.is_some() && !promoted_names.is_empty() {
        eprintln!();
        eprintln!("=== Promoted tensors ({} above modal {}) ===",
            promoted_names.len(),
            modal_qt.unwrap_or("?")
        );
        // Group by suggested hipfire qt for legibility.
        let mut by_qt: BTreeMap<&'static str, Vec<&str>> = BTreeMap::new();
        for (name, qt) in &promoted_names {
            by_qt.entry(*qt).or_default().push(name.as_str());
        }
        for (qt, names) in &by_qt {
            eprintln!("  [{qt}] ({} tensors)", names.len());
            for n in names {
                eprintln!("    {n}");
            }
        }
    }

    // ── Write sidecar JSON ─────────────────────────────────────────────────
    if !args.summary_only {
        let output_path = args.output.unwrap_or_else(|| {
            let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
            args.input.with_file_name(format!("{stem}.kmap.json"))
        });
        let s = serde_json::to_string_pretty(&root).expect("serialize json");
        let mut f = File::create(&output_path).unwrap_or_else(|e| {
            eprintln!("error: failed to create {}: {e}", output_path.display());
            std::process::exit(1);
        });
        f.write_all(s.as_bytes()).expect("write json");
        eprintln!();
        eprintln!("=== Output ===");
        eprintln!("  wrote {} ({} bytes)", output_path.display(), s.len());
    }
}
