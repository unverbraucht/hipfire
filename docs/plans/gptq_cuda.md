# GPTQ on CUDA — handoff plan for the RTX 5070 Ti box

**Status**: not started. Branch `feat/mq-v2-quant-format` at HEAD `b709375c`.
Remote agent works on `feat/mq-v2-quant-format-cuda` branched from this HEAD;
PR back when done.

## Mission

Replace the ~14h CPU-bound GPTQ inner loop (in `crates/hipfire-quantize/src/gptq.rs`)
with a CUDA implementation. Target: full 9B GPTQ pass in **1-3h wall**.

Math must stay byte-faithful to the post-OBS-fix algorithm at commit
`687aa2d0` (`fix(gptq): OBS propagation uses correct upper-Cholesky-of-H_inv`).
That commit was a 2-week investigation that fixed a real correctness bug;
do **not** regress it.

## Hardware here

- **2× RTX 5070 Ti, 16 GB GDDR7 each** (Blackwell consumer, `sm_120`, Jan 2026)
- 32 GB system RAM (tight; stream the Hessian sidecar — see §4.4)
- **PyTorch ≥ 2.6 stable** required for `sm_120`. PyTorch 2.4/2.5 wheels
  ship with CUDA 12.4 which predates Blackwell consumer; needs 12.6+.
  If 2.6 stable not yet packaged, use nightly with
  `TORCH_CUDA_ARCH_LIST="12.0"` env var. Verify:
  ```
  python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
  ```

## Architecture (decided)

Split:
- **Python/CUDA**: GPTQ math only. Loads BF16 → loads Hessian → AWQ pre-scale →
  FWHT rotate → Hessian symmetrize + transform → WEIGHT-mode actorder →
  GPTQ inner loop (FP64 Cholesky + propagation with frozen grids) →
  emit **manifest** of (GPTQ-updated weights + AWQ scale vectors + frozen
  per-block grids).
- **Rust (existing hipfire-quantize, new flag)**: format-conversion only.
  New `--precomputed-gptq-path <dir>` consumes the manifest, applies
  MQ4G256 packing using the frozen grids (skips its own AWQ + FWHT +
  GPTQ math), writes `awq_scale.weight` sidecars from the manifest,
  emits `.hfq`.

This split keeps the tested `.hfq` writer in Rust and isolates the
slow O(K³) GPU-friendly math in Python.

### Manifest format (the Python→Rust contract)

A single directory with three `.safetensors` files + one `manifest.json`:

```
<output-dir>/
  weights.safetensors        # BF16 GPTQ-updated weights (post-AWQ-scale, post-FWHT-rotate)
                             # tensor names = ORIGINAL Qwen 3.5 names (e.g. "model.layers.0.linear_attn.in_proj_qkv.weight")
                             # includes ALL tensors from the original model (passthrough for non-MQ4G256)
  awq_scales.safetensors     # F16 awq_scale vectors, length K
                             # tensor names = "<weight_name>.awq_scale" — present ONLY for AWQ-eligible tensors
  frozen_grids.safetensors   # per-256-block (scale, min_val) pairs, F16
                             # tensor names = "<weight_name>.grids", shape [n_blocks, 2]
  manifest.json              # { alpha, gptq_damp, gptq_max_damp, hessian_path, source_model_md5, validated_at }
```

Rust `--precomputed-gptq-path <dir>` semantics:
- Mutually exclusive with `--awq`, `--gptq`, `--imatrix` (error if any are set)
- Reads `weights.safetensors` as the input model bytes
- For each MQ4G256-eligible tensor:
  - Skip AWQ pre-scale (already baked into weights)
  - Skip FWHT rotate (already baked)
  - Skip Hessian/Cholesky/propagation (already done in Python)
  - Use frozen grids from `frozen_grids.safetensors` to pack the 4-bit values
  - If `<tensor>.awq_scale` exists in `awq_scales.safetensors`, write it as
    `<tensor>.awq_scale.weight` F16 sidecar in the `.hfq`
- Validate at load time: every Qwen 3.5 9B expected tensor name is present
  in `weights.safetensors`. Fail loud on missing.

## 1. Files to read first (in order)

| File | Why |
|---|---|
| `crates/hipfire-quantize/src/gptq.rs` | The reference algorithm. Read `gptq_pipeline_mq4g256` (line 618), `gptq_column_sequential` (line 769), `weight_mode_actorder` (line 723), `compute_damped_inv_cholesky_upper`, `symmetrize_in_place` (line 508). All math is `Mat<f64>`. |
| `crates/hipfire-quantize/src/hessian_io.rs` | Hessian sidecar reader. |
| `docs/plans/gptq-hessian-format.md` §3 | **Canonical** Hessian format spec. Use this, not any inline summary. Note: doc says "~6 GB" but actual on-disk size is **33 GB** for 9B — file size is authoritative. |
| `scripts/collect_hessian.py` | The Python that wrote the sidecar. Existing PyTorch infra to crib from. |
| `crates/hipfire-quantize/src/main.rs:515-543` | `cpu_fwht_256` + `gen_fwht_signs`. Seeds: signs1=42, signs2=1042. |
| `crates/hipfire-quantize/src/main.rs:2794-2841` | `compute_awq_scales`, `awq_pre_scale_weights`. The AWQ math, geo-mean normalization in log space. |
| `crates/hipfire-quantize/src/main.rs:2912-2960` | `awq_eligible` whitelist. Match exactly. |
| `crates/hipfire-runtime/src/hfq.rs` | `.hfq` writer + reader. `awq_scale.weight` sidecar protocol. |

## 2. Math reference (Cholesky-direct OBS, FP64, with actorder)

For each MQ4G256 weight `W: [M, K]` and its Hessian `H_unrot: [K, K]`:

```python
# All computation in FP64 on CUDA. FP32 has insufficient precision at K=12288.

# 1. AWQ pre-scale (per imatrix; identity for non-AWQ-eligible tensors)
#    s[j] = RMS_act[j] ** alpha   (NOT alpha/2 — see §4.1)
#    geo-mean normalize: s = s / exp(mean(log(s)))
s = compute_awq_scales(in_sum2, alpha)        # length K, geo-mean = 1
W_awq = W * s[None, :]                         # broadcast multiply

# 2. FWHT-256 rotation per row (signs seeds 42 + 1042)
W_rot = fwht_rotate_per_row(W_awq)

# 3. Transform Hessian to match rotated+scaled basis (REQUIRED — see §4.2)
H = (1.0 / s[:, None]) * H_unrot * (1.0 / s[None, :])    # AWQ-rescale, H' = diag(1/s) H diag(1/s)
H = fwht_per_256_similarity(H)                            # R · H' · R^T per 256-block
H = 0.5 * (H + H.T)                                       # symmetrize — FP drift compensation

# 4. WEIGHT-mode actorder (REQUIRED — see §4.3)
perm = argsort(diag(H), descending=True)                  # permutation indices, length K
H = H[perm][:, perm]                                       # P^T H P
W_rot = W_rot[:, perm]                                     # permute columns of W
inv_perm = argsort(perm)                                   # for un-permuting at the end

# 5. Damped Cholesky → H_inv → upper-Cholesky-of-H_inv (the OBS fix from 687aa2d0)
damp = 0.01 * mean(diag(H))
max_damp = 1.0 * mean(diag(H))
while damp <= max_damp:
    L_H, info = torch.linalg.cholesky_ex(H + damp * eye(K), upper=False)
    if info == 0:
        break
    damp *= 10
# else: cap damp, accept whatever Cholesky returns; if still bad, fall back to CPU for this tensor

L_H_inv = torch.linalg.solve_triangular(L_H, eye(K), upper=False)      # lower-tri
H_inv = L_H_inv.T @ L_H_inv                                             # K×K symmetric
L_HI, info = torch.linalg.cholesky_ex(H_inv, upper=False)
U = L_HI.T                                                              # U^T @ U = H_inv  ← THE OBS FIX

# 6. Compute frozen per-256-block grids from W_rot (BEFORE the column loop)
frozen_grids = compute_block_grids(W_rot, group_size=256)               # shape [M, n_blocks, 2]

# 7. Sequential column-quantize with propagation (frozen grids)
for j in range(K):
    Q[:, j] = quantize_mq4_with_frozen_grid(W_rot[:, j], frozen_grids[:, j // 256])
    err = (W_rot[:, j] - dequant(Q[:, j], frozen_grids[:, j // 256])) / U[j, j]
    W_rot[:, j+1:] -= err[:, None] * U[j, j+1:][None, :]

# 8. Un-permute back to original column order
W_final = W_rot[:, inv_perm]
frozen_grids_final = frozen_grids[:, inv_perm // 256]       # careful: grids are per-block, see §4.5
```

### The OBS fix in one sentence

`U` satisfies `U^T·U = (P^T (H + λI) P)^-1`, NOT `U·U^T = H_inv`. The
earlier "L_H^{-T}" variant was wrong. Construct `U` as
`lower_cholesky(H_inv).T`, where `H_inv` is explicitly materialized.

The Rust function `compute_damped_inv_cholesky_upper` in `gptq.rs:162+`
already does this correctly. Mirror its structure.

## 3. Implementation steps

Realistic estimate: **3-5 days end-to-end to first KLD anchor.**
The plan author's prior "1-2 day" estimate was optimistic — it ignored
parity debugging, FP64-vs-FP32 numerical issues, sm_120 driver issues,
and per-tensor quality auditing.

### Day 0 — bootstrap + math parity (1-2 days)

1. **Environment setup** (~2h)
   - `python -m venv .venv-cuda && source .venv-cuda/bin/activate`
   - `pip install 'torch>=2.6' safetensors transformers` (or nightly +
     `TORCH_CUDA_ARCH_LIST="12.0"`)
   - `python -c "import torch; print(torch.cuda.get_device_name(0))"`
     must show "RTX 5070 Ti" without warnings

2. **OBS unit test in FP64** (~3h)
   - Port `compute_damped_inv_cholesky_upper` to PyTorch.
   - 16×16 unit test: random PSD H, identity permutation. Compute `U` via
     Rust (extract values from `gptq.rs:obs_propagation_ratios_match_direct_h_inv`)
     and your Python. Assert `max|U_rust - U_python| < 1e-9` (FP64).
   - Then with non-identity perm: assert same property holds for
     `P^T (H + λI) P`.

3. **Per-tensor pipeline (no AWQ, no FWHT)** (~4h)
   - `scripts/gptq_cuda.py`. Given (BF16 weight, FP32 Hessian, alpha=0,
     identity actorder), produce GPTQ-updated BF16 weight.
   - Test on ONE 9B tensor: layer 0 `out_proj` (non-AWQ, K=4096).
     Compare against current Rust output: relative weight diff
     `max|w_rust - w_python| / max|w_rust| < 1e-6`. Floats won't be
     bit-identical (Rust faer vs cuBLAS) but should agree at FP64.

4. **Add AWQ + FWHT** (~4h)
   - Implement `compute_awq_scales` in PyTorch. **Geo-mean normalization
     in log space** (see `main.rs:2805-2820`).
   - Implement FWHT-256 in PyTorch with signs1=42, signs2=1042.
   - Implement `apply_awq_hessian` (`diag(1/s) H diag(1/s)`) and
     `fwht_per_256_similarity` (`R H R^T` per 256-block).
   - **Symmetrize after**: `H = 0.5 * (H + H.T)`.
   - Test on layer 0 `in_proj_qkv` (AWQ-eligible, K=4096). Same weight
     parity criterion as step 3.

5. **Add actorder** (~3h)
   - `perm = torch.argsort(torch.diag(H), descending=True)`
   - Apply to W and H, run inner loop in permuted order, un-permute at
     end.
   - Re-test layer 0 in_proj_qkv. Output should now match Rust to
     within FP64 precision.

### Day 1 — full 9B + manifest writer (1 day)

6. **Full 9B sweep** (~4h)
   - Iterate all MQ4G256-eligible tensors in lexicographic order
     (matches Rust). For each:
     - Load weight (BF16) + Hessian record from sidecar (stream, don't mmap)
     - Run GPTQ on cuda:0 or cuda:1 (round-robin, simple `dist[i] = devices[i % 2]`)
     - Save updated weight to in-memory dict
   - **Per-tensor quality gate**: track MSE(quant, original) per tensor.
     If any tensor's MSE > 10× median MSE, log loudly — likely actorder
     or grid bug for that tensor's distribution.
   - Two-card parallelism, simplest version: alternate-tensor assignment.
     Each card runs independently; CPU collects results in tensor order.

7. **Write manifest** (~2h)
   - `weights.safetensors`: GPTQ-updated BF16 weights + passthrough of
     ALL other tensors (norms, embeddings, lm_head etc.) — full model
     state. Tensor names = original Qwen 3.5 names exactly.
   - `awq_scales.safetensors`: per-eligible-tensor F16 `s` vectors.
   - `frozen_grids.safetensors`: per-eligible-tensor F16 (scale, min_val)
     per 256-block.
   - `manifest.json`: alpha, damp settings, source md5, validation hashes.

### Day 2 — Rust `--precomputed-gptq-path` flag (1 day)

8. **Add flag to `hipfire-quantize`** (~4h)
   - CLI parse: `--precomputed-gptq-path <dir>`. Mutually exclusive with
     `--gptq`, `--awq`, `--imatrix` (error message if combined).
   - At load time: validate every expected tensor name from the original
     Qwen 3.5 9B model is present in `weights.safetensors`. Fail loud.
   - In the MQ4G256 quant path: skip `apply_awq_prescale`, skip
     `gptq_pipeline_mq4g256`, instead call a new
     `quantize_mq4g256_with_frozen_grids(weights, frozen_grids)` that
     does packing only.
   - For AWQ-eligible tensors: read the corresponding `<name>.awq_scale`
     from `awq_scales.safetensors`, write as F16 sidecar (matches what
     the current Rust path does).

9. **End-to-end run + KLD** (~2h)
   - `python scripts/gptq_cuda.py --input <hf snapshot> --hessian
     refs/qwen3.5-9b-bf16.hessian.bin --imatrix refs/qwen3.5-9b-bf16.imatrix.gguf
     --output ~/.hipfire/gptq-precomputed/qwen3.5-9b-mq4-awq-gptq-q8conv/`
   - `hipfire-quantize --precomputed-gptq-path ~/.hipfire/gptq-precomputed/qwen3.5-9b-mq4-awq-gptq-q8conv/
     --output ~/.hipfire/models/qwen3.5-9b.mq4-awq-gptq-q8conv-cuda --format mq4`
   - `eval_hipfire` at n=512 q8-KV. Compare against anchor 0.1842.

### Day 3-4 — buffer for parity debugging

Reserved. The FP64-vs-FP32 numerical gap, sm_120 driver edge cases,
and per-tensor outlier debugging will eat time. Don't pretend they
won't.

## 4. Sharp edges, in detail

### 4.1 AWQ exponent formula

The effective formula is `s[j] = RMS_act[j] ^ alpha`, geo-mean normalized
to 1. The Rust code operates on `in_sum2[j] = N_tok · RMS_act[j]²`,
hence the `half_alpha = alpha * 0.5` factor in the code — that's
`(in_sum2)^(alpha/2)`, which equals `N_tok^(alpha/2) · RMS^alpha`,
and the `N_tok^(alpha/2)` constant cancels under geo-mean normalization.
**Do not express the formula as `RMS^(alpha/2)`** — that's a 2× exponent
error vs the intended math.

### 4.2 Hessian-domain transforms (REQUIRED — easy to miss)

The Hessian sidecar was collected on **unrotated, unscaled** activations.
GPTQ must operate on Hessian in the **same basis** as the weights it
updates. So before the GPTQ inner loop:

```
H_target = R · diag(1/s) · H_unrot · diag(1/s) · R^T
```

where R is the per-256-block FWHT operator. Then symmetrize:
`H = 0.5 * (H + H.T)`. The FWHT-per-256-similarity is row-pass then
col-pass; FP drift makes the result slightly non-symmetric (O(ε·K)).
Rust does this explicitly at `gptq.rs:643-650`. Mirror it.

### 4.3 WEIGHT-mode actorder

Sort columns by descending `diag(H_target)` and process in that order:

```python
perm = torch.argsort(torch.diag(H_target), descending=True)
H_target = H_target[perm][:, perm]      # P^T H P
W_rot = W_rot[:, perm]
# ... GPTQ inner loop in permuted order ...
W_final = W_result[:, inv_perm]         # un-permute
```

Then the OBS property is `U^T · U = (P^T (H + λI) P)^-1`. Reference:
`gptq.rs:weight_mode_actorder` line 723, used by `gptq_column_sequential`
line 769.

### 4.4 RAM streaming — 32 GB system RAM is tight

Don't load the entire 9B BF16 model AND the 33 GB Hessian sidecar
simultaneously. Pattern:

```python
# Stream weights from safetensors: each .safetensors layer file mmapped, BF16 tensor at a time
# Stream Hessians from sidecar: parse the next record only when needed
for tensor_name in eligible_tensors_in_order:
    W = load_one_tensor_bf16(model_path, tensor_name)       # ~30-200 MB
    H = load_one_hessian(hessian_path, tensor_name)         # ~100-600 MB FP32
    W_updated = gptq_one_tensor(W, H, ...)                  # on GPU
    save_one_tensor_to_output(W_updated, manifest_output)   # write + close
    del W, H, W_updated
```

System RAM peak should stay under ~4 GB. GPU peak per-tensor stays
under 4 GB (FP64 makes Cholesky workspaces 2× the FP32 size).

### 4.5 Frozen grids must be permuted alongside W

The grids are computed BEFORE the column loop on the post-rotation
weight. They're indexed `[m, block]` where `block = j // 256`. If you
permute W columns by `perm`, the grids must follow the SAME permutation
(grouped by 256). At un-permute time, both W and grids un-permute back
to original column order. Easiest: index grids by ORIGINAL column j
all the way through and never permute the grid array.

### 4.6 `torch.linalg.cholesky_ex`, not `cholesky`

`cholesky` throws on non-PD; the exception serializes the CUDA stream
and is slow under retry. `cholesky_ex` returns `(L, info)` non-throwing.
Use it in the damp-retry loop.

### 4.7 CUDA memory hygiene

Over 224 tensors with H, L_H, L_H_inv, H_inv, L_HI, U each at ~600 MB
peak in FP64, allocator fragmentation is predictable. Either
`torch.cuda.empty_cache()` between tensors (modest overhead) or
pre-allocate a workspace buffer per device and reuse. Pick one,
document the choice.

### 4.8 Two-card streams

`tensor.to('cuda:1')` synchronizes the source by default. Use
`non_blocking=True` + explicit `torch.cuda.Stream` per device.
Synchronize only at end-of-tensor (when saving result back to CPU).

### 4.9 AWQ whitelist (full list)

Apply AWQ pre-scaling ONLY to weights whose name ends with one of, or
contains:

```
q_proj.weight, k_proj.weight, v_proj.weight,
qkv_proj.weight, wqkv.weight,
gate_proj.weight, up_proj.weight, w_gate.weight, w_up.weight,
gate_up_proj.weight,
.in_proj_   (substring; covers in_proj_qkv, in_proj_a, in_proj_b, in_proj_z)
mlp.gate.weight (the MoE router — separate from gate_proj)
router.weight
```

Reference: `main.rs:2912-2960`. Non-eligible (e.g., `out_proj`,
`down_proj`, `o_proj`, `wo`) get identity scales `s = [1.0; K]` —
they STILL go through GPTQ, just without AWQ pre-scale.

## 5. Validation

### 5.1 0.8B parity (the critical pre-9B gate)

The 0.8B Hessian sidecar exists at
`benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.hessian.bin` (2.2 GB).

Steps:
1. Quantize 0.8B via current Rust path: `hipfire-quantize ... --awq --gptq ...`
2. Quantize 0.8B via your Python+CUDA path + new `--precomputed-gptq-path`
3. Compare on resulting `.hfq`:
   - Per-tensor weight bytes: `max|w_rust - w_python| / max|w_rust| < 1e-6` (FP64)
   - Per-tensor MSE(quant, original): match within 5%
4. Run `eval_hipfire` n=512 q8-KV on both. **KLD bootstrap CIs must overlap**.

The weight-diff threshold is the sanity gate (fail-loud on order-of-magnitude
divergence). The KLD CI overlap is the actual acceptance.

### 5.2 9B end-to-end

Run `eval_hipfire` n=512 q8-KV on the CUDA-path `.hfq`. Compare KLD
against the AWQ-only anchor **0.1842** (master doc §1.1f).

**Acceptance is a clean measurement**, not a win:
- KLD < 0.1842 → Stage B is a win at 9B
- KLD ≥ 0.1842 (within CI overlap) → Stage B is a wash at 9B; this is
  the same outcome observed at 0.8B and is a legitimate publishable result
- KLD ≫ 0.1842 OR gibberish output → math/handoff regression, debug

### 5.3 Per-tensor MSE outlier check

KLD is a global metric. A single broken tensor can be masked. During
the Python loop, log per-tensor MSE. If any tensor's MSE exceeds
**10× the median**, halt and inspect — likely actorder or frozen-grid
indexing bug on that tensor's distribution.

## 6. Pitfalls / known sharp edges (quick reference)

1. **FP64 everywhere** for Cholesky + propagation. FP32 fails at K=12288.
2. **Hessian-domain transforms** (§4.2) — easy to miss; silently breaks GPTQ.
3. **Actorder permutation** — easy to miss; degrades KLD without crashing.
4. **AWQ exponent** is `alpha`, not `alpha/2` in RMS terms.
5. **AWQ sidecar emission** — Rust must write `awq_scale.weight` sidecars
   even with `--precomputed-gptq-path`, sourced from the manifest. Without
   them the runtime kernel runs the non-AWQ activation path on
   AWQ-pre-scaled weights → catastrophic.
6. **Frozen grids** — compute once from rotated weights, never re-derive
   per column.
7. **Symmetrize H after FWHT** (`gptq.rs:508 symmetrize_in_place`).
8. **`cholesky_ex` not `cholesky`** for the damp retry loop.
9. **Per-tensor Hessian streaming** (§4.4) — don't mmap, don't load all
   at once.
10. **Tensor name validation** at Rust load time — fail loud on missing.
11. **Hessian sidecar format** — use `gptq-hessian-format.md` §3 as
    canonical, not any inline summary. Doc claims "~6 GB"; actual file
    is 33 GB on disk — file size wins.

## 7. Ecosystem implications

This CUDA path is **NVIDIA-only**. It cannot run on the gfx1100/ROCm
host that the rest of the hipfire stack targets. After this lands:

- Quantize has two paths: Rust (cross-platform, ~14h on this gfx1100)
  and Python+CUDA (NVIDIA-only, 1-3h on the 5070 Ti box).
- The RTX 5070 Ti box becomes the only machine that can run the fast
  quantize path.
- AMD-only developers must use the Rust path or get NVIDIA hardware
  for quantize-development tasks.

This is an informed tradeoff (the RTX hardware exists, the math is
GPU-friendly, the cost-per-experiment justifies it), but should not
become a load-bearing assumption for any cross-vendor feature.

## 8. Hand-back

When the CUDA path runs end-to-end + matches Rust on 0.8B parity test:

1. Commit on `feat/mq-v2-quant-format-cuda`:
   - `scripts/gptq_cuda.py`
   - `crates/hipfire-quantize/src/main.rs` changes for `--precomputed-gptq-path`
   - `crates/hipfire-quantize/src/gptq.rs` changes if any (probably none —
     the path bypasses GPTQ math entirely)
   - **Create** `scripts/gptq_9b_overnight.sh` (currently only at
     `/tmp/`; promote it and switch to the CUDA path)
2. Update `docs/plans/kld-measurements-master.md` §1.1f with the 9B GPTQ
   anchor (whether it beats 0.1842 or not, plus the per-tensor MSE table)
3. Open PR from `feat/mq-v2-quant-format-cuda` into
   `feat/mq-v2-quant-format`
4. Tag for review here

Branch protocol: mainline `feat/mq-v2-quant-format` stays on
this-machine work; only blocking-bugfix commits cherry-pick between
branches mid-flight.

## 9. Reference: relevant commits

| Commit | What |
|---|---|
| `e51a3cd9` | Phase 2a — AWQ runtime infrastructure (runtime side that consumes AWQ scales) |
| `a4265ce4` | Phase 2b — AWQ forward-pass call-site wiring |
| `6711308d` | AWQ whitelist — only pre-scale weights whose runtime path applies the inverse |
| `5a91c027` | Stage B 2.5a — GPTQ pipeline helpers (FWHT + pack) |
| `bd1ca3e1` | Stage B 2.5b — wire `--gptq` flag into MQ4G256 branch |
| `dcf3b18a` | Stage B 2.1-2.4 — GPTQ algorithm core (Rust + faer) |
| `0cf5cc45` | Pre-launch review fixes — single-pass Cholesky, damp clamp, FP-asym scrub |
| `a9dde0de` | Cholesky-direct perf opt — drop dense H_inv + serial solve(I) |
| **`687aa2d0`** | **The OBS-Cholesky FIX — `U^T·U = H_inv` math. CRITICAL.** |
| `0ab8575a` | Per-tensor clamp counter for diagnostic |
| `6e358a11` | TensorSpill path-uniqueness + write_hfq non-panicking error |
| `b709375c` | (HEAD) CLAUDE.md note re: clean-rebuild reset for token-attractor garbage |

## 10. What NOT to do

- **Do NOT use AutoGPTQ.** Doesn't speak FWHT-256, doesn't speak AWQ
  pre-scaling, doesn't have frozen-grids semantics. Output layout would
  not match `.hfq`. Mentioned for completeness — don't pursue.
- **Do NOT re-derive the OBS math** from Frantar 2210.17323 paper. The
  paper's notation lets you draw wrong conclusions about which Cholesky
  upper-triangular variant is correct. Commit `687aa2d0` is the
  authoritative fix. Mirror `gptq.rs::compute_damped_inv_cholesky_upper`.
- **Do NOT skip the 0.8B parity test.** It's the only way to be sure
  the math is right before sinking 1-3h of GPU time on 9B.
- **Do NOT change the `.hfq` format.** Rust handles that.
- **Do NOT use `/tmp/` for the manifest output.** AGENTS.md rule 4.
  Use `~/.hipfire/gptq-precomputed/` or `benchmarks/artifacts/gptq-cuda/`.
- **Do NOT skip the AWQ sidecar emission step.** Without `awq_scale.weight`
  in the `.hfq`, the runtime breaks (kernel path divergence + magnitude
  mismatch on activations).
