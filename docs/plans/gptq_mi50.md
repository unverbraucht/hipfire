# GPTQ on AMD Instinct MI50 (gfx906) — hardware migration plan

**Status**: partial bring-up complete (2026-05-17). Triggered by the
wall-time hit on 27B (Stage B 9.8h + Stage C ~5h on dual RTX 5070 Ti)
— see `docs/plans/gptq_cuda.md` §11. MI50's HPC-tier FP64 + 32 GB HBM2
is a much better fit for this FP64-bound, VRAM-hungry workload than
consumer Blackwell.

This doc covers MIGRATING the existing quantize-time pipeline onto
MI50 hardware. **The runtime target (.hfq consumer) is unchanged** —
RDNA1/2/3 consumer cards remain the inference target; MI50 is offline
quant-prep only.

**See §11 for empirical findings** from the actual bring-up on this
host — sections 1-10 were written speculatively before hardware
contact and several of their assumptions (matching mixa3607 docker
recommendation, dual-MI50 capacity, host-rocBLAS sufficiency) have
been falsified or refined in §11. The §11 findings supersede the
plan when they conflict.

## 1. Motivation: where consumer Blackwell hurts

From `gptq_cuda.md` §11.3-11.4, the 27B end-to-end on dual RTX 5070 Ti:

| Stage | Bottleneck | Wall |
|---|---|---:|
| A. Imatrix (llama.cpp) | -ngl 28 / 64 layers; CPU-portion dominates | 3.3h |
| B. Hessian collect (HF forward) | 36 of 64 layers on CPU → safetensors mmap page-fault thrashing | 9.8h |
| C. GPTQ pipeline | FP64 Cholesky + OBS column loop at K=17408 on 1:64 FP64 ratio | ~5h |
| **Total** | | **~18h** |

Two compounding problems:

1. **Not enough VRAM** to hold the 27B BF16 model on the GPUs. 32 GB
   total (2× 16 GB) vs 54 GB needed → ~56% of layers on CPU. The
   forward pass page-faults safetensors weights per-layer per-sequence.
2. **Consumer Blackwell FP64 is artificially gimped** at 1:64 of FP32
   throughput (~0.5 TFLOPS FP64 per card). The K=17408 Cholesky +
   OBS loop is bound on this.

The Hessian collection is the worst offender (9.8h) because BOTH
problems compound — page-faulting model weights AND running BF16
forward through them.

## 2. Why MI50 fixes both

| | RTX 5070 Ti (current) | Instinct MI50 |
|---|---|---|
| VRAM per card | 16 GB GDDR7 | **32 GB HBM2** (2×) |
| 2× config total VRAM | 32 GB | **64 GB** — fits 27B BF16 fully |
| Memory bandwidth | 896 GB/s | ~1 TB/s |
| FP32 | 32 TFLOPS | 13.3 TFLOPS |
| **FP64** | ~0.5 TFLOPS (1:64) | **6.6 TFLOPS (1:2 HPC ratio)** — **~13× faster** |
| BF16 matrix accel | Tensor Cores | none (CDNA1 pre-matrix-cores) |
| ROCm support | n/a (NVIDIA) | gfx906, last fully-supported ROCm 5.7 (mid-2024) |
| Used-market price | $700+ | $200-400 |

**Key wins per stage:**

- **Stage B (Hessian collection)**: 2× MI50 = 64 GB → **all 27B layers
  on GPU, zero CPU offload**. Even without BF16 matrix cores (the
  forward falls back to FP32 internally), this kills the page-fault
  thrash that dominates current wall time. Estimated **9.8h → 2-3h**.
- **Stage C (GPTQ pipeline)**: pure PyTorch linalg, FP64 1:2 ratio →
  ~13× faster Cholesky + OBS. Estimated **5h → ~45-90 min**.
- **Stage A (Imatrix)**: llama.cpp can run with `-ngl 64` (full
  offload) since 27B BF16 fits. Estimated **3.3h → ~1-1.5h**.

**Estimated full 27B end-to-end on 2× MI50: ~5-6h vs current ~18h.**

## 3. Compatibility audit: what runs on gfx906?

Our pipeline parts:

| Component | Pure-PyTorch? | gfx906 status |
|---|---|---|
| `gptq_cuda.py` (Stage C) | yes — `cholesky_ex`, `solve_triangular`, `cholesky_inverse`, matmul, indexing only | ✓ all via MIOpen/rocBLAS |
| `collect_hessian.py` (Stage B) | yes — `transformers` + `accelerate` `device_map="auto"` | ✓ HF backend-agnostic; `cuda:0`/`cuda:1` map to ROCm devices |
| `safetensors` + `numpy` + `mmap` | pure Python/CPU | ✓ |
| Imatrix via `llama.cpp` | C++ with HIP backend | ✓ build with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906` |
| `fla` (Flash Linear Attention) | Triton kernels | ⚠️ Triton's ROCm/HIP backend supports gfx906 but needs verification; fallback to reference impl works (just slower) |
| `causal_conv1d` | CUDA extension | ⚠️ `causal-conv1d-hip` fork exists; may need build-from-source for gfx906 |
| Rust `hipfire-quantize` (Stage D) | CPU only | ✓ |

**Even if `fla` and `causal_conv1d` aren't available** for gfx906, the
forward falls back to reference PyTorch implementations. For Qwen3.6-27B
with 48 linear-attn layers this would slow Stage B somewhat — but
**not having CPU offload still dominates the win**. The 9.8h current
Stage B is overwhelmingly CPU-page-fault-bound, not compute-bound.

## 4. ROCm support: the mixa3607/ML-gfx906 path

AMD officially dropped gfx906 from ROCm 6+ (mid-2024). The community
project `github.com/mixa3607/ML-gfx906` maintains gfx906-built PyTorch
Docker images precisely for this reason — keeping the architecture
alive for ML/HPC workloads.

Images include:
- PyTorch built against the last gfx906-compatible ROCm (5.7)
- numpy, transformers, accelerate, safetensors typically pre-installed
- ROCm runtime + MIOpen + rocBLAS for gfx906

**Practical:** mount /data, the hipfire repo, and ~/.hipfire into the
container; run scripts/gptq_cuda.py + scripts/collect_hessian.py from
inside. Same code as the CUDA path — torch routes through HIP
transparently.

## 5. Migration steps

Assumes the MI50 host has 2× MI50, /data NFS mount, and Docker with
amdgpu device passthrough.

```
# 1. Pull image
docker pull mixa3607/ml-gfx906:pytorch-rocm5.7-latest  # exact tag TBD

# 2. Run container with GPU + filesystem mounts
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=16g \
  -v /data:/data \
  -v $HOME/.hipfire:/root/.hipfire \
  -v $HOME/git/hipfire:/hipfire \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  mixa3607/ml-gfx906:pytorch-rocm5.7-latest \
  bash

# 3. Inside container — verify torch sees both MI50s
cd /hipfire
python -c "
import torch
print(torch.cuda.device_count(), 'devices')
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i),
          f'{torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB')
"
# Expect: 2 devices, both 'AMD Instinct MI50 / Vega 20', 32.0 GB each

# 4. Install our few extra deps (gguf reader, etc.)
pip install gguf safetensors packaging
# Optional fast paths for linear-attn — may or may not build cleanly on gfx906:
pip install causal_conv1d || true     # ok if it fails
pip install flash-linear-attention || true  # ok if it fails

# 5. Verify FP64 Cholesky works on gfx906 (K=17408 stress test)
python - <<'PY'
import torch
torch.manual_seed(0)
for dev in ['cuda:0', 'cuda:1']:
    a = torch.randn(17408, 17408, dtype=torch.float64, device=dev)
    h = a @ a.T + 17408 * torch.eye(17408, dtype=torch.float64, device=dev)
    l = torch.linalg.cholesky(h)
    h_inv = torch.cholesky_inverse(l)
    err = (h_inv @ h - torch.eye(17408, dtype=torch.float64, device=dev)).abs().max().item()
    print(f"{dev} K=17408 cholesky+inverse residual = {err:.3e}")
PY
# Expect: residuals ~1e-9 to 1e-7. Anything larger means rocBLAS has issues
# at this K and we'd need to investigate (potentially patch via batched
# panel-Cholesky or fall back to k_max ~10000 with chunked Hessians).

# 6. Build llama-imatrix with HIP backend (in container)
cd /tmp && git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-imatrix -j

# 7. Smoke test on 4B (we have all inputs already)
cd /hipfire
python scripts/gptq_cuda.py \
  --input /data/models/qwen/Qwen3.5-4B \
  --hessian /data/hipfire-refs/qwen3.5-4b-bf16.hessian.bin \
  --imatrix benchmarks/quality-baselines/refs/qwen3.5-4b-bf16.imatrix.gguf \
  --alpha 0.55 --bits 4 \
  --output /tmp/4b-mi50-smoke \
  --devices cuda:0 cuda:1 --limit 8 -v
# Expect: ~5-10 min for 8 tensors, MSEs in the 1e-6 range matching the
# CUDA-path 4B reference (commit 353576e produced KLD-validated 4B)
```

If steps 5-7 pass, scale to 27B per the existing `gptq_9b_overnight.sh`
template — just change paths.

## 6. Risks to verify before committing time

Each is best validated on a small smoke test (steps 5-7 above) before
running 27B end-to-end.

1. **`torch.linalg.cholesky_ex` at K=17408 on rocBLAS.** CUDA path was
   already tested (RTX 5070 Ti, residual ~1e-13). ROCm could have edge
   cases at large K. Step 5 in §5 above is the canary.
2. **`torch.cholesky_inverse` available.** It's just `potri`; should be
   in rocBLAS. Step 5 checks it.
3. **`device_map="auto"` with `max_memory={0: '30GiB', 1: '30GiB', 'cpu': '60GiB'}`
   on 2× MI50.** 27B BF16 (~54 GB) should fit entirely on the two GPUs;
   no CPU offload needed. But accelerate's heuristics may need tuning.
4. **fla / causal_conv1d available.** If both fail to install, the
   forward pass uses reference PyTorch — slower per-layer but doesn't
   block. Worth checking pip-build experience.
5. **`logits_to_keep=1` honored by Qwen3_5 transformers model on ROCm.**
   Should be backend-agnostic but worth verifying — first OOM on the
   RTX 5070 Ti path was lm_head materializing full logits.
6. **HIP/Triton kernel cache permanence.** First forward pass may pay
   significant JIT compile cost (we saw ~5 min Triton compile on 5070
   Ti's first run). On ROCm this is typically smoother since rocBLAS
   has fewer JIT paths.

## 7. Cost / benefit

| | Cost | Benefit |
|---|---|---|
| Hardware | $400-800 for 2× used MI50 | Permanent 3-4× speedup on every future quant job |
| Setup time | ~1 day (Docker, build llama.cpp, smoke test) | One-time |
| Per-run wall savings | n/a on first run | 27B: 18h → ~5h. 9B: 1.5h → ~30 min. 4B: 25 min → ~10 min |
| Power | MI50 ≈ 300W ea. = 600W vs 5070 Ti 250W ea. = 500W | Similar; offset by shorter wall |

Worth it if you expect to do >5 more big-model quants. Not worth it
for a one-off 27B; the current 5070 Ti path will finish in 18h.

## 8. Open questions

1. **Does fla's Triton-on-HIP backend work on gfx906?** Triton 3.x has
   AMD support but mostly tested on CDNA2+ (gfx90a, gfx940). gfx906
   may need a build flag. If not available, the reference fallback
   doesn't break anything but Stage B is slower (maybe 50% rather than
   the 70% reduction we'd hope for).
2. **Hessian sidecar locality:** the 126 GB 27B Hessian sits on /data
   NFS. Per-tensor read is 1.2 GB. On MI50 with PCIe 4.0 ×16 = 32 GB/s
   theoretical, but NFS gates at ~100 MB/s. Same I/O bottleneck as
   5070 Ti path. Worth investigating local Hessian cache or async
   prefetch (worth ~20% Stage C wall).
3. **MI50 power + cooling:** server-class card, no fan, needs proper
   airflow. Not a Docker concern but a host concern.

## 9. Decision: when to migrate

**Trigger conditions to actually pull the trigger:**

- Quantize budget over the next 3 months > ~50h (likely if Qwen3.5-32B,
  Qwen3.6-35B-A3B, or 70B-class models enter scope)
- Or: a 27B re-quant becomes necessary (e.g. new AWQ alpha tuning,
  new GPTQ damp schedule, new quant format like Lloyd-Max MQ3) — each
  re-run is 18h on current hardware vs ~5h on MI50

**Until then:** the current 5070 Ti pipeline is sufficient. The
performance patches landed for 27B (memmap accumulators, multi-pass,
memory-frugal Cholesky, etc. — see `gptq_cuda.md` §11.2) mean the
slowness is structural to consumer-Blackwell FP64 + VRAM, not fixable
in software.

## 10. Companion proposal: rename `gptq_cuda` → `gptq_gpu`

**Status**: not done. Worth doing as part of MI50 migration prep, or
as a standalone cleanup whenever the GPU is idle (~30 min mechanical
work including verification).

### Motivation

§3 above audits that the entire pipeline is backend-agnostic — every
GPU op is `torch.*` or `torch.cuda.*`, and `torch.cuda.*` is the
shared API surface for both NVIDIA CUDA and AMD ROCm/HIP (PyTorch's
design choice for compatibility). The current naming
(`scripts/gptq_cuda.py`, `scripts/gptq_cuda_pkg/`) misleadingly
suggests NVIDIA-only when in fact the code runs unmodified on MI50
via the mixa3607 ROCm path.

Renaming makes the portability explicit, removes a friction point for
anyone reading the code and wondering "do I need NVIDIA hardware to
contribute?", and avoids the awkward `scripts/gptq_cuda.py` invocation
on a ROCm box.

### Scope

Mechanical file moves + import updates:

| Path before | Path after |
|---|---|
| `scripts/gptq_cuda.py` | `scripts/gptq_gpu.py` |
| `scripts/gptq_cuda_pkg/` (dir + 6 files) | `scripts/gptq_gpu_pkg/` |

Reference updates:

- Imports in renamed `gptq_gpu.py` (top-level + 2 lazy imports inside
  fallback paths)
- Imports in `scripts/tests/test_gptq_algo.py`, `scripts/tests/test_pipeline.py`
- Path references in `scripts/mq3_sweep_4b.sh`, `scripts/gptq_9b_overnight.sh`
- Doc pointers in `docs/plans/gptq_cuda.md`, `docs/plans/gptq_mi50.md`
- Rust comments in `crates/hipfire-quantize/src/main.rs` (~107) and
  `crates/hipfire-quantize/src/precomputed_gptq.rs` (~1, ~304)

Functional code is unaffected — the rename touches only paths and
identifiers, not algorithm or API.

### Verification after rename

```
PYTHONPATH=scripts ./.venv-cuda/bin/python \
    -m unittest scripts/tests/test_gptq_algo.py \
                scripts/tests/test_pipeline.py
# Expect: 30/30 tests pass

cargo build --release -p hipfire-quantize
# Expect: clean build (only pre-existing dead-code warnings)
```

### Timing — when NOT to do it

If a `gptq_gpu.py` (formerly `gptq_cuda.py`) process is currently
running, the rename mid-flight risks crashing it if a lazy import path
fires after the rename:

- `gptq_cuda.py:336` — `from gptq_cuda_pkg.algo import (...)` inside
  the RTN-fallback branch (triggers when a tensor has no Hessian
  entry — at 27B this never fires because all eligible tensors have
  Hessians, but at 4B/9B with vision encoder tensors it does)
- `gptq_cuda.py:372` — `from gptq_cuda_pkg.algo import (...)` inside
  the `except CholeskyFailedError` block (triggers on adaptive-damping
  failure — hasn't fired in any production run so far, but possible)

A running process has the OLD module paths in `sys.modules` from
startup; if it lazy-imports the OLD path after we rename the dir,
ImportError. Two mitigations:

1. **Defer**: do the rename when no `gptq_*.py` process is in flight.
2. **Temporary symlink** during the run: `ln -sfn gptq_gpu_pkg scripts/gptq_cuda_pkg`
   so the old import path still resolves until the running process
   finishes. Drop the symlink in a follow-up commit.

For typical re-quant runs the rename takes ~30 min; just wait for the
GPU to clear.

### Acceptance criteria

- 30/30 Python tests pass
- Rust build clean
- One representative end-to-end smoke (4B Stage C+D on the renamed
  `gptq_gpu.py`) produces a byte-identical `.hfq` to the pre-rename
  baseline — md5 match against `~/.hipfire/quantized/qwen3.5-4b.mq4-cuda.hfq`
  (`bf4063ded4182d8b5a7cd275c06641e5`)

## 11. Empirical findings — 2026-05-17 bring-up on the dev host

**TL;DR for future readers:** the §1-10 plan was substantially wrong
about gfx906 being a viable GPTQ quantize-time target. rocBLAS FP64
`cholesky_ex` is silently broken at K≥6144 on gfx906 in every
available PyTorch build (§11.4-§11.6). The §10 `gptq_cuda` → `gptq_gpu`
rename shipped successfully and is independently useful. The
quantize-time MI50 migration is parked (§11.8); the 5070 Ti box
remains primary. Re-check `probe_cholesky_only.py` after any ROCm
upgrade.

Sections 1-10 were written before hardware contact. This section
records what actually happened on the dev host (Ryzen 5 4650G +
**1× MI50 32 GB** + Renoir iGPU). The §10 rename (`gptq_cuda` →
`gptq_gpu`) shipped here.

### 11.1 Hardware reality vs §1-2 assumptions

- **Single MI50, not dual.** Host has 1× MI50 32 GB visible as
  `cuda:0`. The Renoir iGPU shows up as `cuda:1` (16.5 GB shared
  system RAM, 7 CUs, gfx90c) — usable for CPU offload tier but not
  a real second compute device.
- **VRAM capacity for 27B BF16 (54 GB) is NOT met** on a single 32 GB
  card. 27B work as described in §2 requires either a second MI50,
  CPU offload (back to the page-fault problem we were trying to
  avoid), or a quantized model as input. **9B and smaller fit
  comfortably.**
- **Host already runs ROCm 6.4.3** with full gfx906 support in system
  rocBLAS (`/opt/rocm/lib/rocblas/library/` has `TensileLibrary_lazy_gfx906.dat`
  + 156 gfx906 kernel objects). ROCm 6.4.4 in the mixa3607 PyTorch
  vers-compatibility table is the upper supported pin; host 6.4.3 is
  the same minor.

### 11.2 PyTorch wheel choice (§4 supersession)

§4 said "use mixa3607/ML-gfx906 Docker images because AMD dropped
gfx906 in ROCm 6+". The mixa README's PyTorch compatibility table
contradicts this — gfx906 is currently tested through ROCm 7.2.0
with PyTorch 2.9.0+ — but stock PyPI `torch==2.9.1+rocm6.4` ships a
**bundled rocBLAS** at `torch/lib/rocblas/library/` whose
`TensileLibrary_lazy_*.dat` set **excludes gfx906** entirely (only
gfx908/90a/942/1030/1100-1102/1200-1201 are present). So:

| Path | Result |
|---|---|
| Stock `pip install torch==2.9.1+rocm6.4` | Aborts at first DGEMM with `Cannot read .../TensileLibrary.dat: No such file or directory for GPU arch : gfx906` |
| Stock wheel + `ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library` | Loads system rocBLAS kernels but throws `hipErrorInvalidDeviceFunction` — bundled `librocblas.so` cannot load system-compiled code objects (HIP runtime ABI mismatch) |
| Mixa3607 `pytorch-gfx906:v2.9.0-rocm-6.4.4` image | Works; rocBLAS in the image is built with `AMDGPU_TARGETS=...gfx906...` |
| Mixa3607 `vllm-gfx906:0.20.1-rocm-6.3.3-aiinfos` image (already cached on host) | Works (torch 2.11.0a0+rocm6.3); used for bring-up |

**Resolution: use a mixa3607 image. Do not attempt to patch a stock
PyPI wheel** — torch's bundled rocBLAS and the system rocBLAS are not
ABI-interchangeable at the code-object level.

### 11.3 Docker vs host venv

§5 prescribes the full Docker recipe and §10 wishfully imagines a
host venv after rename. The actual experience:

- Docker is **the** path. The cached `mixa3607/vllm-gfx906:0.20.1-rocm-6.3.3-aiinfos`
  image runs the renamed `gptq_gpu.py` end-to-end without modification.
- A "wheel extraction onto host venv" is theoretically possible — pull
  the image, `docker cp` the `site-packages/torch/` directory out, run
  against host ROCm — but in practice the wheel + bundled rocBLAS +
  HIP runtime ABI is one tight unit and any cross-version mixing
  blows up at kernel-load time. Not worth the debugging budget.
- The `.venv-cuda` references in `scripts/gptq_9b_overnight.sh` and
  `scripts/mq3_sweep_4b.sh` are now load-bearing-stale — they invoke
  `./.venv-cuda/bin/python scripts/gptq_gpu.py` which would not work
  on this host. Wrap with a Docker invocation when used on MI50.
  Cleanup deferred — the launchers are wrapper scripts, not the
  load-bearing math.

### 11.4 FP64 numerics on gfx906 rocBLAS — the K-scaling wall

The §6 canary list checked the wrong thing. What actually matters:

**FP64 Cholesky + cholesky_inverse residuals on cuda:0 (MI50):**

| K | `\|H_inv·H − I\|_∞` | Notes |
|---:|---:|---|
| 4096 | 1.38e-14 | machine-precision, expected |
| 17408 | 8.65e-2 | **OFF BY ORDERS OF MAGNITUDE** |

rocBLAS emits `WARNING: Device memory allocation size is too small
for TRSM; TRSM performance is degraded` at the larger K. Combined
with the residual, this looks like a workspace-undersize-driven
precision regression in `cholesky_inverse`'s internal TRSM at large
K on gfx906. Not isolated to one of `cholesky` or `cholesky_inverse`
— the factorization (`L L^T == H`) checks clean but the inverse-
verification fails.

**Concrete impact on 4B smoke (limit=8 tensors, alpha=0.55, bits=4):**

- 7/8 tensors quantized successfully. MSE 1e-6 to 1e-5 range
  (matching §5 canary criterion).
- **1/8 tensor — `model.language_model.layers.0.mlp.down_proj.weight`
  (K=9216) — hit `CholeskyFailedError: Second Cholesky on H_inv
  failed; H_inv lost PSD due to FP drift in matmul`** and the script
  fell back to RTN-equivalent packing (FWHT rotation + per-block RTN,
  no Hessian-aware error redistribution). This is `gptq_gpu_pkg/algo.py:283-288`
  exception path — `compute_damped_inv_cholesky_upper` does
  `chol(H+λI) → cholesky_inverse → chol(H_inv)` and the second chol
  is where gfx906 falls down.

**The K-cutoff is somewhere between 4096 (clean) and 9216 (fails on
real data).** The actual cliff depends on Hessian condition number
plus rocBLAS workspace heuristics, not just K. For 9B (K_max=12288)
expect more frequent fallbacks; for 27B (K_max=17408) expect the
down_proj layers to all fall back.

### 11.5 Resolution options tried — none work

Probed under `docs/investigations/2026-05-17-mi50-bringup/`:

| Variant | Result |
|---|---|
| A. Baseline `cholesky_inverse → chol(H_inv)` | `info=8161` (the production failure) |
| B. + `H_inv = 0.5*(H_inv + H_inv.T)` symmetrize | **same `info=8161`** |
| C. + symmetrize + 1e-8 diagonal ridge | **same `info=8161`** |
| C2. + symmetrize + 1e-6 diagonal ridge | **same `info=8161`** |
| D. `solve_triangular(L, eye) → L_inv.T @ L_inv → chol` | **same `info=8161`** |

All four variants return the **exact same** error code on the real
layer-0 down_proj Hessian (K=9216). That fingerprint means it's not
a numerical-stability question — the rocBLAS gfx906 kernels themselves
are broken at this K.

Follow-up probe (`probe_cholesky_only.py`, K-sweep with clean
diagonal-dominant random PSD):

| K | `cholesky_ex` info | `L L^T vs H` residual | Verdict |
|---:|---:|---:|---|
| 2048 | 0 | 3.1e-15 | ✓ FP64-clean |
| 4096 | 0 | 3.6e-15 | ✓ FP64-clean |
| 6144 | 0 | **1.2e-2** | ✗ silently wrong by 10¹² |
| 8192 | 0 | **7.9e-3** | ✗ silently wrong |
| 9216 | 0 | **7.5e-3** | ✗ silently wrong |
| 10240 | 0 | **5.9e-3** | ✗ silently wrong |
| 12288 | 0 | **4.2e-3** | ✗ silently wrong |
| 9216 (2·I trivial case) | 0 | 4.4e-16 | ✓ trivial PSD survives |

**The conclusion: rocBLAS FP64 `cholesky_ex` is fundamentally broken
on gfx906 at K≥6144** in the only PyTorch build that exposes it
(`mixa3607/vllm-gfx906:0.20.1-rocm-6.3.3`, torch 2.11.0a0). It returns
`info=0` (success) but produces an L that bears no resemblance to the
true factorization. **The production 4B GPTQ run output is therefore
suspect for every tensor processed at K≥6144**, not just the ones that
loudly failed the second Cholesky step.

### 11.6 No viable PyTorch build on gfx906 for this workload

Re-probed in the matched `mixa3607/pytorch-gfx906:v2.9.0-rocm-6.4.4`
image to check whether the ROCm 6.4 rocBLAS fixes the kernel bug:

```
torch 2.9.0a0+git0fabc3b  hip 6.4.43484
GPU cholesky_ex: RuntimeError: requires compiling PyTorch with MAGMA
CPU cholesky_ex: RuntimeError: requires compiling PyTorch with LAPACK
```

The matched-version build ships without **either** MAGMA or LAPACK.
`torch.linalg.cholesky` simply cannot be called in this build — neither
on GPU nor on CPU. So:

| PyTorch build | GPU Cholesky | CPU Cholesky |
|---|---|---|
| Stock PyPI torch+rocm6.4 | rocBLAS has no gfx906 Tensile | — |
| `vllm-gfx906:0.20.1-rocm-6.3.3` (cached) | Silently wrong at K≥6144 | No LAPACK |
| `pytorch-gfx906:v2.9.0-rocm-6.4.4` (matched) | No MAGMA | No LAPACK |

There is no off-the-shelf path to a correct GPTQ Cholesky on gfx906
with the available builds.

### 11.7 Resolution options that remain (all expensive)

1. **Build PyTorch from source** with both LAPACK and MAGMA enabled,
   then either (a) rely on CPU LAPACK for Cholesky (K=12288 FP64 ~30-60s
   per tensor × ~200 tensors = 1-2h added per quant — undermines the
   speedup motivation) or (b) hope MAGMA's Cholesky is built on a code
   path that doesn't hit the rocBLAS kernel bug.
2. **Hand-roll panel Cholesky in PyTorch ops** (block factorization
   using only `tril`/`mm`/`addmm`). Bypasses rocBLAS `potrf` entirely.
   2-3 day implementation; will be slow at runtime; numerical accuracy
   is on us to verify.
3. **Call rocSOLVER directly** via ctypes / a Rust extension —
   rocSOLVER's `Dpotrf` is the underlying primitive PyTorch wraps. If
   the bug is in rocSOLVER itself the bypass doesn't help; if it's in
   PyTorch's MAGMA path, this could work. Untested.
4. **Wait for ROCm 7.x** or for upstream patches. Mixa README's table
   shows ROCm 7.2 + torch 2.10/2.11 is currently green — but until
   we verify the Cholesky-at-K bug is fixed there, that's wishcasting.

### 11.8 Decision: park gfx906 quantize-path exploration

**Verdict: gfx906 is not viable as a GPTQ quantize-time host with
available tooling as of 2026-05-17.** The original motivation (faster
27B Stage C) is moot here anyway:

- 1× MI50 = 32 GB VRAM, not enough for 27B BF16 (54 GB) without
  CPU offload — defeating the §1 wall-time argument.
- Even at 4B/9B, the rocBLAS Cholesky bug silently corrupts GPTQ
  output for every K≥6144 tensor. The 4B run's manifest at
  `/local/hipfire/gptq-precomputed/4b-mi50/` is kept as evidence of
  the corruption pattern, not for production use.

**Recommendation: keep the 5070 Ti box as the primary GPTQ quantize
host.** gfx906 retains its value for runtime inference (`.hfq`
consumer) per CLAUDE.md; the renamed `scripts/gptq_gpu.py` is now
provably portable to ROCm and will work on future AMD generations
(gfx908/90a/942 CDNA with HPC FP64, RDNA3+ wave32) once a non-buggy
rocBLAS is in play, but **not on gfx906 today**.

The reproducer scripts in `docs/investigations/2026-05-17-mi50-bringup/`
are kept so the next attempt has a clean go/no-go gate before sinking
time in. Re-run `probe_cholesky_only.py` after any rocBLAS upgrade;
if the K=9216 `L L^T vs H` residual drops back to ~1e-14, the bug is
fixed and the rest of the path becomes worth retrying.

### 11.9 Wall-time observed on the partial 4B run

Full 4B Stage C (355 eligible tensors, alpha=0.55, bits=4) on 1× MI50
cuda:0 completed in **27.7 min** (1659.9 s) — `[done] 219 GPTQ, 136
RTN fallback, 248 AWQ sidecars`. Of the 136 RTN fallbacks: 29 are
K=9216 `down_proj` Cholesky failures (real defect), 107 are vision-
encoder tensors that have no Hessian sidecar entry by design
(`linear_fc1`, `linear_fc2`, `proj`, `qkv` from the multimodal
front-end — RTN is the correct path for those).

This timing is **misleading as a benchmark** — per §11.4 the 219
"GPTQ-ok" tensors include K≥6144 tensors whose silently-garbage L was
used to update weights. So the run completed but its output is
not byte-correct. Compared to the §2 dual-MI50 ~10-min projection,
single-card wall scales as expected given (a) 1 card vs 2, (b) docker
overhead per invocation. The wall-time isn't the problem; the
numerical correctness is.

### 11.10 Decision: what to keep from the original plan

| Section | Status | Notes |
|---|---|---|
| §1 motivation | Valid | The "consumer Blackwell hurts" framing is unchanged |
| §2 hardware comparison | **Single-MI50 only**; capacity numbers for 2× are aspirational | Get a second MI50 before serious 27B work |
| §3 compatibility audit | Mostly valid | Update: torch 2.9+ with ROCm 6.4 works (not just 5.7-via-Docker) |
| §4 mixa3607 path | Valid | But §4 framed it as "rebuilt for legacy support" — actually still actively maintained, validated through ROCm 7.2 |
| §5 migration steps | Partially valid | Steps 1-3 are unchanged (docker pull + run). Step 5 K=17408 canary IS the canary that catches the bug — re-purpose it as a go/no-go gate, not a smoke test. Step 6 llama.cpp HIP build is unchanged. Step 7 smoke "worked" but produced numerically wrong output — do not trust the §10 md5-match acceptance criterion until §11.4 is resolved |
| §6 risks | **§6.1 + §6.6 are realized; deeper issue surfaced** | rocBLAS Cholesky at K≥6144 silently produces garbage on gfx906 (see §11.4-§11.5). Not a workspace tuning issue — kernel-level bug |
| §7 cost/benefit | **Reversed at gfx906** | The savings story assumed Cholesky works. It doesn't. The 5070 Ti retains both correctness and wall-time advantage on this hardware until rocBLAS is fixed |
| §8 open questions | Q1+Q2 now mostly answered | Q1 fla: untested. Q2 hessian locality: still applies. New Q3: does the bug persist in ROCm 7.x? (would re-open the path) |
| §9 decision criteria | Stands but moot for gfx906 | A second MI50 still doesn't fix the Cholesky bug, and consumer CDNA2+ (gfx90a+) is where mixa's tooling is best maintained anyway |
| §10 rename | **Done** | Shipped 2026-05-17. md5 of `bf4063ded4182d8b5a7cd275c06641e5` baseline is the reference; produced-on-MI50 output does NOT match it because of §11.4 corruption, not because of the rename |

### 11.11 Docker recipe used during bring-up (supersedes §5 step 2)

For one-shot quantize runs:

```
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --shm-size=8g \
  --network=none \
  -v $HOME/git/hipfire:/hipfire \
  -v /data/models/qwen/Qwen3.5-4B:/model:ro \
  -v /data/hipfire-refs:/refs:ro \
  -v /local/hipfire/gptq-precomputed/4b-mi50:/out \
  -w /hipfire \
  --entrypoint /bin/bash \
  mixa3607/vllm-gfx906:0.20.1-rocm-6.3.3-aiinfos \
  -c '
PYTHONPATH=scripts HIP_VISIBLE_DEVICES=0 python3 scripts/gptq_gpu.py \
    --input /model \
    --hessian /refs/qwen3.5-4b-bf16.hessian.bin \
    --imatrix /hipfire/benchmarks/quality-baselines/refs/qwen3.5-4b-bf16.imatrix.gguf \
    --alpha 0.55 --bits 4 \
    --output /out \
    --devices cuda:0 \
    --verbose
'
```

`HIP_VISIBLE_DEVICES=0` is required — the iGPU is `cuda:1` in the
container and would otherwise also be available. The script would
try to round-robin onto it and OOM. (If a second MI50 lands, drop
this and use `--devices cuda:0 cuda:1`.)

`--network=none` is paranoia — the container has no network needs
once the model and references are mounted. Saves a syscall surface.

The `mixa3607/vllm-gfx906:0.20.1-rocm-6.3.3-aiinfos` tag is the one
already cached on this host (31 GB, mostly vLLM that we don't use).
The smaller `mixa3607/pytorch-gfx906:v2.9.0-rocm-6.4.4` would be a
better fit (matches the plan's target torch version exactly); not
yet pulled — open question whether the K=9216 second-Cholesky issue
is also present there.

## 12. References
