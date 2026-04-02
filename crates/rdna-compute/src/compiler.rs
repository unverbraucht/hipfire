//! Compile HIP kernels to code objects (.hsaco) via hipcc.
//! Supports pre-compiled .hsaco blobs for deployment without ROCm SDK.

use hip_bridge::HipResult;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Compiles HIP kernel sources to code objects, with caching.
/// Tries pre-compiled blobs first (kernels/compiled/{arch}/), falls back to hipcc.
pub struct KernelCompiler {
    cache_dir: PathBuf,
    arch: String,
    compiled: HashMap<String, PathBuf>,
    precompiled_dir: Option<PathBuf>,
}

impl KernelCompiler {
    pub fn new(arch: &str) -> HipResult<Self> {
        let cache_dir = std::env::temp_dir().join("hipfire_kernels");
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to create cache dir: {e}"))
        })?;

        // Probe for pre-compiled kernels: try exe-relative, then CWD-relative
        let precompiled_dir = std::env::current_exe().ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .map(|dir| dir.join("kernels").join("compiled").join(arch))
            .filter(|p| p.is_dir())
            .or_else(|| {
                let cwd_path = PathBuf::from("kernels/compiled").join(arch);
                if cwd_path.is_dir() { Some(cwd_path) } else { None }
            });

        if let Some(ref dir) = precompiled_dir {
            eprintln!("  pre-compiled kernels: {}", dir.display());
        }

        Ok(Self {
            cache_dir,
            arch: arch.to_string(),
            compiled: HashMap::new(),
            precompiled_dir,
        })
    }

    /// Compile a HIP kernel source string. Returns path to .hsaco file.
    /// Tries pre-compiled blob first, falls back to runtime hipcc compilation.
    pub fn compile(&mut self, name: &str, source: &str) -> HipResult<&Path> {
        if self.compiled.contains_key(name) {
            return Ok(&self.compiled[name]);
        }

        // Try pre-compiled .hsaco first
        if let Some(ref dir) = self.precompiled_dir {
            let precompiled = dir.join(format!("{name}.hsaco"));
            if precompiled.exists() {
                self.compiled.insert(name.to_string(), precompiled);
                return Ok(&self.compiled[name]);
            }
        }

        // Fall back to runtime compilation via hipcc
        let src_path = self.cache_dir.join(format!("{name}.hip"));
        let obj_path = self.cache_dir.join(format!("{name}.hsaco"));
        let hash_path = self.cache_dir.join(format!("{name}.hash"));

        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        self.arch.hash(&mut hasher);
        let src_hash = format!("{:016x}", hasher.finish());

        let cache_valid = obj_path.exists() && hash_path.exists()
            && std::fs::read_to_string(&hash_path).unwrap_or_default() == src_hash;

        if !cache_valid {
            std::fs::write(&src_path, source).map_err(|e| {
                hip_bridge::HipError::new(0, &format!("failed to write kernel source: {e}"))
            })?;

            let _ = std::fs::remove_file(&obj_path);

            let output = Command::new("hipcc")
                .args([
                    "--genco",
                    &format!("--offload-arch={}", self.arch),
                    "-O3",
                    "-o",
                    obj_path.to_str().unwrap(),
                    src_path.to_str().unwrap(),
                ])
                .output()
                .map_err(|e| {
                    hip_bridge::HipError::new(0, &format!("failed to run hipcc: {e}"))
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("hipcc compilation failed for {name}:\n{stderr}"),
                ));
            }

            let _ = std::fs::write(&hash_path, &src_hash);
        }

        self.compiled.insert(name.to_string(), obj_path);
        Ok(&self.compiled[name])
    }
}
