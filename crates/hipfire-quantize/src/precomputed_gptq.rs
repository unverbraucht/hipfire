//! Reader for the Python+GPU GPTQ manifest (see `scripts/gptq_gpu.py`).
//!
//! When `hipfire-quantize` is invoked with `--precomputed-gptq-path <dir>`,
//! the GPTQ math has already happened on a GPU box (CUDA or HIP/ROCm);
//! this module reads
//! the resulting manifest and serves as the byte/grid source for the
//! MQ4G256 packing step. The Rust quantize path skips AWQ pre-scale,
//! FWHT rotation, Hessian/Cholesky, and column-sequential OBS for any
//! tensor that has a manifest entry.
//!
//! Manifest directory layout (per `docs/plans/gptq_cuda.md` §1):
//!   <dir>/
//!     weights.safetensors        # all tensors; MQ4G256 ones are
//!                                # post-AWQ-scale + post-FWHT-rotate
//!                                # + post-GPTQ-update (BF16).
//!     awq_scales.safetensors     # F16 vectors keyed by
//!                                # "<weight_name>.awq_scale". Present
//!                                # only for AWQ-eligible tensors.
//!     frozen_grids.safetensors   # F16 [n_blocks, 2] keyed by
//!                                # "<weight_name>.grids". One entry per
//!                                # MQ4G256-eligible tensor.
//!     manifest.json              # alpha, damp settings, source md5, etc.
//!
//! Lookup contract:
//!   * `weight_bytes(name)` returns the BF16 bytes for any tensor name
//!     present in the manifest, or `None` (caller falls back to original
//!     input safetensors — but in `--precomputed-gptq-path` mode this
//!     should never happen because the manifest must passthrough every
//!     tensor — caller fails loud per `expect_all_present`).
//!   * `frozen_grids(name)` returns the per-256-block grids for a
//!     MQ4G256-eligible tensor, or `None` (caller falls through to plain
//!     `quantize_mq4g256` RTN).
//!   * `awq_scale(name)` returns the F16 scale vector to emit as the
//!     `<name>.awq_scale.weight` sidecar in the `.hfq`, or `None` (no
//!     sidecar emitted = runtime uses non-AWQ kernel).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use serde::Deserialize;

use crate::gptq::BlockGrid;

/// Local F16 → F32 (matches `main.rs::f16_to_f32` byte-for-byte). Inlined
/// here to keep this module independent of `main` for unit testing.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

/// Errors specific to manifest loading. Per-tensor lookup failures are
/// `Option::None` returns; this enum is for whole-manifest faults.
#[derive(Debug)]
pub enum ManifestError {
    Io(std::io::Error),
    MissingFile(String),
    InvalidJson(String),
    InvalidSafetensors(String),
    SchemaVersion(u64),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Io(e) => write!(f, "I/O error: {e}"),
            ManifestError::MissingFile(p) => write!(f, "missing manifest file: {p}"),
            ManifestError::InvalidJson(s) => write!(f, "invalid manifest.json: {s}"),
            ManifestError::InvalidSafetensors(s) => write!(f, "invalid safetensors: {s}"),
            ManifestError::SchemaVersion(v) => write!(
                f, "unsupported manifest schema_version {v} (this build expects 1)",
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

impl From<std::io::Error> for ManifestError {
    fn from(e: std::io::Error) -> Self { ManifestError::Io(e) }
}

/// Minimal record per safetensors tensor — name → (dtype, shape, byte
/// offsets into the mmap). Re-implemented here (rather than reusing
/// `main.rs::SafetensorsFile`) so this module compiles independently
/// for testing.
#[derive(Debug, Clone, Deserialize)]
struct StMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct StFile {
    _file: File,
    mmap: Mmap,
    header_size: usize,
    tensors: HashMap<String, StMeta>,
}

impl StFile {
    fn open(path: &Path) -> Result<Self, ManifestError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            return Err(ManifestError::InvalidSafetensors(
                format!("{} truncated: < 8 bytes", path.display()),
            ));
        }
        let header_len = LittleEndian::read_u64(&mmap[0..8]) as usize;
        if mmap.len() < 8 + header_len {
            return Err(ManifestError::InvalidSafetensors(
                format!("{}: header_len={header_len} exceeds file size {}",
                        path.display(), mmap.len()),
            ));
        }
        let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
            .map_err(|e| ManifestError::InvalidSafetensors(
                format!("{}: header UTF-8: {e}", path.display())
            ))?;
        let raw: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| ManifestError::InvalidSafetensors(
                format!("{}: header JSON: {e}", path.display())
            ))?;
        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = raw {
            for (k, v) in map {
                if k == "__metadata__" { continue; }
                let meta: StMeta = serde_json::from_value(v)
                    .map_err(|e| ManifestError::InvalidSafetensors(
                        format!("{}: tensor {k} meta: {e}", path.display())
                    ))?;
                tensors.insert(k, meta);
            }
        }
        Ok(Self {
            _file: file,
            mmap,
            header_size: 8 + header_len,
            tensors,
        })
    }

    fn raw_bytes(&self, name: &str) -> Option<(&StMeta, &[u8])> {
        let meta = self.tensors.get(name)?;
        let start = self.header_size + meta.data_offsets[0];
        let end = self.header_size + meta.data_offsets[1];
        Some((meta, &self.mmap[start..end]))
    }

    fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

/// Minimal manifest.json schema — schema_version + the tuning knobs we
/// echo into the `.hfq` provenance. We do not deserialize the per-tensor
/// stats array (it can be hundreds of records) — kept as opaque JSON if
/// needed for debugging.
#[derive(Debug, Clone, Deserialize)]
pub struct ManifestMeta {
    pub schema_version: u64,
    pub source_model_dir: String,
    pub hessian_path: String,
    pub imatrix_path: Option<String>,
    pub alpha: f32,
    #[serde(default)]
    pub awq_f1_only: bool,
    /// Quantization bit width: 3 (MQ3G256) or 4 (MQ4G256, default).
    /// Defaults to 4 for backward-compat with schema_version=1 manifests
    /// that pre-date the field.
    #[serde(default = "default_n_bits")]
    pub n_bits: u8,
    pub gptq_initial_damp_ratio: f64,
    pub gptq_max_damp_multiplier: f64,
}

fn default_n_bits() -> u8 { 4 }

/// Loaded manifest with all three safetensors files + parsed JSON meta.
pub struct PrecomputedGptq {
    pub meta: ManifestMeta,
    weights: StFile,
    awq_scales: StFile,
    frozen_grids: StFile,
    pub dir: PathBuf,
}

impl std::fmt::Debug for PrecomputedGptq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrecomputedGptq")
            .field("dir", &self.dir)
            .field("alpha", &self.meta.alpha)
            .field("n_weights", &self.weights.tensors.len())
            .field("n_awq_scales", &self.awq_scales.tensors.len())
            .field("n_frozen_grids", &self.frozen_grids.tensors.len())
            .finish()
    }
}

impl PrecomputedGptq {
    pub fn open(dir: &Path) -> Result<Self, ManifestError> {
        let weights_path = dir.join("weights.safetensors");
        let awq_path = dir.join("awq_scales.safetensors");
        let grids_path = dir.join("frozen_grids.safetensors");
        let manifest_path = dir.join("manifest.json");
        for p in [&weights_path, &awq_path, &grids_path, &manifest_path] {
            if !p.exists() {
                return Err(ManifestError::MissingFile(p.display().to_string()));
            }
        }
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let meta: ManifestMeta = serde_json::from_str(&manifest_json)
            .map_err(|e| ManifestError::InvalidJson(e.to_string()))?;
        if meta.schema_version != 1 && meta.schema_version != 2 {
            return Err(ManifestError::SchemaVersion(meta.schema_version));
        }
        if meta.n_bits != 3 && meta.n_bits != 4 {
            return Err(ManifestError::InvalidJson(
                format!("unsupported n_bits={} (only 3 and 4 are wired up)", meta.n_bits)
            ));
        }
        Ok(Self {
            meta,
            weights: StFile::open(&weights_path)?,
            awq_scales: StFile::open(&awq_path)?,
            frozen_grids: StFile::open(&grids_path)?,
            dir: dir.to_path_buf(),
        })
    }

    /// Validation: every expected tensor from the source model must
    /// appear in the manifest's weights file. Fail loud on missing —
    /// per plan §3 step 8.
    ///
    /// `expected` is the iterator of names the input safetensors
    /// directory exposes. We check both directions: anything the input
    /// has must be in the manifest, AND we report unknown extras in
    /// the manifest (warning only — they'll silently passthrough).
    pub fn validate_against<'a>(
        &self,
        expected: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), Vec<String>> {
        let manifest_names: std::collections::HashSet<&str> =
            self.weights.keys().collect();
        let mut missing = Vec::new();
        let mut expected_set = std::collections::HashSet::new();
        for n in expected {
            expected_set.insert(n);
            if !manifest_names.contains(n) {
                missing.push(n.to_string());
            }
        }
        for n in &manifest_names {
            if !expected_set.contains(n) {
                eprintln!("[precomputed-gptq] manifest carries extra tensor {n} not in input model (will be ignored)");
            }
        }
        if missing.is_empty() { Ok(()) } else { Err(missing) }
    }

    /// Returns the BF16 bytes for a tensor name, or None if the
    /// manifest doesn't carry it (caller falls back to original
    /// input — but `validate_against` should have caught the gap
    /// at startup).
    pub fn weight_bf16(&self, name: &str) -> Option<&[u8]> {
        let (meta, bytes) = self.weights.raw_bytes(name)?;
        assert!(meta.dtype == "BF16" || meta.dtype == "F16" || meta.dtype == "F32",
            "manifest weight {name} has unsupported dtype {}", meta.dtype);
        // The orchestrator emits BF16 per the format spec; F16/F32
        // would indicate an older or modified manifest — accept them
        // for forward-compat, the caller converts to f32 via `to_f32`.
        Some(bytes)
    }

    /// Returns the dtype + shape of a manifest weight as a forwarding
    /// helper when the caller needs to upcast or reshape.
    pub fn weight_meta(&self, name: &str) -> Option<(&str, &[usize])> {
        let meta = self.weights.tensors.get(name)?;
        Some((meta.dtype.as_str(), meta.shape.as_slice()))
    }

    /// Frozen per-256-block grids for `<name>.grids`. Returns Vec<BlockGrid>
    /// upcast to f64 for the FP64 pack pipeline.
    ///
    /// Storage layout in safetensors: shape `[n_blocks, 2]` F16, with
    /// column 0 = scale, column 1 = min_val (matching
    /// `scripts/gptq_gpu_pkg/algo.py::compute_frozen_block_grids`).
    pub fn frozen_grids(&self, weight_name: &str) -> Option<Vec<BlockGrid>> {
        let key = format!("{weight_name}.grids");
        let (meta, bytes) = self.frozen_grids.raw_bytes(&key)?;
        if meta.shape.len() != 2 || meta.shape[1] != 2 {
            eprintln!("[precomputed-gptq] {key}: unexpected shape {:?}, expected [n_blocks, 2]; ignoring", meta.shape);
            return None;
        }
        let n_blocks = meta.shape[0];
        let mut out = Vec::with_capacity(n_blocks);
        match meta.dtype.as_str() {
            "F16" => {
                for b in 0..n_blocks {
                    let off = b * 4;  // 2 F16 values = 4 bytes
                    let scale_bits = LittleEndian::read_u16(&bytes[off..off + 2]);
                    let min_bits = LittleEndian::read_u16(&bytes[off + 2..off + 4]);
                    out.push(BlockGrid {
                        scale: f16_to_f32(scale_bits) as f64,
                        min_val: f16_to_f32(min_bits) as f64,
                    });
                }
            }
            "F32" => {
                for b in 0..n_blocks {
                    let off = b * 8;
                    out.push(BlockGrid {
                        scale: LittleEndian::read_f32(&bytes[off..off + 4]) as f64,
                        min_val: LittleEndian::read_f32(&bytes[off + 4..off + 8]) as f64,
                    });
                }
            }
            other => {
                eprintln!("[precomputed-gptq] {key}: unsupported dtype {other}; ignoring");
                return None;
            }
        }
        Some(out)
    }

    /// F16 byte payload for the `<weight_name>.awq_scale.weight` sidecar
    /// emission. Manifest stores under key `<weight_name>.awq_scale`;
    /// Rust sidecar naming adds the trailing `.weight` to match what
    /// the runtime loader expects (see `hfq.rs::load_awq_scale`).
    pub fn awq_scale_f16_bytes(&self, weight_name: &str) -> Option<Vec<u8>> {
        let key = format!("{weight_name}.awq_scale");
        let (meta, bytes) = self.awq_scales.raw_bytes(&key)?;
        if meta.dtype != "F16" {
            eprintln!("[precomputed-gptq] {key}: expected F16, got {}; passing through anyway", meta.dtype);
        }
        // Defensive copy — we want the bytes to live independently of
        // the mmap once the manifest is closed.
        Some(bytes.to_vec())
    }

    /// True if the manifest carries an AWQ scale for this weight name.
    pub fn has_awq_scale(&self, weight_name: &str) -> bool {
        let key = format!("{weight_name}.awq_scale");
        self.awq_scales.tensors.contains_key(&key)
    }
}
