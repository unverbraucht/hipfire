# GPTQ on AMD Instinct MI50 (gfx906) — hardware migration plan

**Status**: proposal. Not started. Triggered by the wall-time hit on
27B (Stage B 9.8h + Stage C ~5h on dual RTX 5070 Ti) — see
`docs/plans/gptq_cuda.md` §11. MI50's HPC-tier FP64 + 32 GB HBM2 is
a much better fit for this FP64-bound, VRAM-hungry workload than
consumer Blackwell.

This doc covers MIGRATING the existing quantize-time pipeline onto
MI50 hardware. **The runtime target (.hfq consumer) is unchanged** —
RDNA1/2/3 consumer cards remain the inference target; MI50 is offline
quant-prep only.

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

## 10. References

- `docs/plans/gptq_cuda.md` — the canonical pipeline plan; §11 covers
  what was needed for 27B that informs this migration's gains
- `github.com/mixa3607/ML-gfx906` — gfx906-ported PyTorch Docker images
- Hipfire CLAUDE.md — gfx906 is a supported runtime arch, MI50 quant
  output (.hfq) is reusable on the same hardware family
