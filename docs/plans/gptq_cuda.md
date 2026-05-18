# GPTQ on CUDA — handoff plan for the RTX 5070 Ti box

**Status**: not started. Branch `feat/mq-v2-quant-format` at HEAD `1df2a24d`.
Remote agent works on `feat/mq-v2-quant-format-cuda` branched from this HEAD;
PR back when done.

## Mission

Replace the ~14h CPU-bound GPTQ inner loop (in `crates/hipfire-quantize/src/gptq.rs`)
with a CUDA implementation. Target: full 9B GPTQ pass in **1-3h wall**.

Math must stay byte-faithful to the post-OBS-fix algorithm at commit
`687aa2d0` (`fix(gptq): OBS propagation uses correct upper-Cholesky-of-H_inv`).
That commit was a 2-week investigation that fixed a real correctness bug;
do **not** regress it.

**Anchor to beat** (commits `9ca8d900` + `1df2a24d`, F2 AWQ whitelist
expansion + cross-arch alpha-sweep reproduction):

| Config | sidecars | KLD | NLL | PPL |
|---|---:|---:|---:|---:|
| F1 α=0.5 (old anchor, superseded) | 184 | 0.1725 | 2.2551 | 9.54 |
| **F2 α=0.55** (current ship default, gfx906 + gfx1151 confirmed) | 248 | 0.1830 | **2.1730** | **8.79** |

**Acceptance metric: paired-t on per-chunk NLL, not KLD.** AWQ-class
improvements redistribute probability mass toward the true next token
without changing the top-K divergence shape vs BF16, so they're flat
on KLD and strongly visible on NLL. F1→F2 at α=0.5: KLD Δ ≈ 0
(t=−0.01) but NLL Δ = −0.0678 (t=**−13.32**, p<10⁻³⁰). KLD-only
ranking would have shipped α=0.5 and missed the gain. See
master-doc §6 rule 9.

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
#    Default alpha = 0.55 (gfx906 + gfx1151 PPL optimum, F2 whitelist).
#    See `crates/hipfire-quantize/src/main.rs:awq_eligible` for the expanded
#    F2 whitelist (248 sidecars on 9B: input + output projections).
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

9. **End-to-end run + NLL paired-t** (~2h)
   - `python scripts/gptq_cuda.py --input <hf snapshot>
     --hessian refs/qwen3.5-9b-bf16.hessian.bin
     --imatrix refs/qwen3.5-9b-bf16.imatrix.gguf
     --alpha 0.55
     --output ~/.hipfire/gptq-precomputed/qwen3.5-9b-mq4-awq-gptq-q8conv-f2/`
   - `hipfire-quantize --precomputed-gptq-path ~/.hipfire/gptq-precomputed/qwen3.5-9b-mq4-awq-gptq-q8conv-f2/
     --output ~/.hipfire/models/qwen3.5-9b.mq4-awq-gptq-q8conv-f2-cuda --format mq4`
   - `eval_hipfire` at n=512 q8-KV. Compare via NLL paired-t against
     F2 α=0.55 anchor (KLD 0.1830 / NLL 2.1730 / PPL 8.79). See §5.2.

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

### 4.9 AWQ whitelist (F2 expansion, full list)

Apply AWQ pre-scaling to weights whose name ends with one of, or
contains:

**Input-side (F1, 184 sidecars):**
```
q_proj.weight, k_proj.weight, v_proj.weight,
qkv_proj.weight, wqkv.weight,
gate_proj.weight, up_proj.weight, w_gate.weight, w_up.weight,
gate_up_proj.weight,
.in_proj_   (substring; covers in_proj_qkv, in_proj_a, in_proj_b, in_proj_z)
mlp.gate.weight, router.weight    (MoE router — two distinct patterns)
```

**Output-side (F2 addition, +64 → 248 sidecars on 9B):**
```
o_proj.weight, wo.weight,             (full-attention output projection)
out_proj.weight,                       (linear-attention output projection)
down_proj.weight, w_down.weight        (MLP down projection)
```

Reference: `main.rs:2912-2960` (F2 expansion in commit `9ca8d900`).
Only embeddings, layer norms, lm_head, and conv1d weights stay out of
AWQ — those go through GPTQ with identity scales `s = [1.0; K]`.

**Runtime kernels for output-side AWQ are different from input-side.**
Input-side: AWQ divide happens BEFORE the FWHT in
`fused_rmsnorm_mq_rotate_awq`. Output-side: divide happens at different
points in the activation flow, hence the F2-new kernels
`rotate_x_mq_awq.hip` and `fused_silu_mul_mq_rotate_awq.hip`. This
matters only for Rust runtime dispatch; **the quantize-time math is
identical across whitelist entries** — multiply weight columns by
`s = RMS_act^alpha`, geo-mean-normalized. Python doesn't need to know
which kernel the runtime will use.

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
4. Run `eval_hipfire` n=512 q8-KV on both. **Paired-t on per-chunk NLL
   must be |t| < 2.0** (the two paths produce statistically
   indistinguishable per-chunk NLL). KLD bootstrap CIs should also
   overlap, but NLL paired-t is the load-bearing check — see master-doc
   §6 rule 9 for why.

The weight-diff threshold is the sanity gate (fail-loud on order-of-magnitude
divergence). The NLL paired-t is the actual acceptance. KLD parity is
informational.

### 5.2 9B end-to-end

Run `eval_hipfire` n=512 q8-KV on the CUDA-path `.hfq`. Compare against
the **F2 α=0.55 anchor** (master doc §1.1h/i):

| Anchor (no-GPTQ baseline) | KLD | NLL | PPL |
|---|---:|---:|---:|
| **F2 α=0.55 (current ship default)** | 0.1830 | 2.1730 | 8.79 |

**Acceptance is a clean measurement**, not necessarily a win:
- NLL paired-t < 0 with |t| > 3 (GPTQ-on-F2 strictly better per-chunk)
  → Stage B is a win at 9B
- |NLL paired-t| ≤ 3 (within noise) → Stage B is a wash at 9B
  (consistent with 0.8B finding); legitimate publishable result
- NLL paired-t > 0 with |t| > 3 (GPTQ-on-F2 strictly worse) → math/
  handoff regression, debug
- Any gibberish at chat decode → handoff regression, debug

**Report both KLD and NLL paired-t**; do NOT lead with KLD because the
KLD-PPL inversion at α=0.55 means KLD-only ranking can pick the wrong
direction. Per master-doc §6 rule 9, paired-t on per-chunk NLL is
primary; KLD is secondary informational.

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
12. **NLL paired-t is the acceptance signal, not KLD** (master-doc §6
    rule 9). At α≈0.55 the KLD-PPL inversion can flip the apparent
    ranking. Always report both, lead with NLL paired-t.
13. **eval_hipfire segfault-on-exit with F2 kernels** (`9ca8d900` known
    issue on gfx1151): all three sweep evals on gfx1151 exited with
    SIGSEGV AFTER writing the kldseq + slice-mean line. **Data is valid**
    (verified per-chunk + paired-t). Suspected `Drop` ordering issue
    with the new F2 AWQ-aware kernels in the dispatch table. Doesn't
    block measurement; do not panic when you see it. Separate cleanup PR.

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
2. Update `docs/plans/kld-measurements-master.md` with a new section
   (§1.1j or similar) for the 9B GPTQ-on-F2 anchor — include KLD,
   NLL paired-t vs F2 α=0.55, PPL, per-tensor MSE outlier list, and
   the per-arch numbers if both gfx906 and gfx1151 are run
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
| `589de2e5` | CLAUDE.md note re: clean-rebuild reset for token-attractor garbage |
| **`9ca8d900`** | **F2 — AWQ on output-side projections. New runtime kernels (`rotate_x_mq_awq`, `fused_silu_mul_mq_rotate_awq`). Sidecar count 184→248. NLL paired-t vs KLD methodology change.** |
| `0c7aaeed` | AWQ on K-map MQ4/MQ6 mix — −20.1% KLD vs kmd2 baseline on gfx1151 |
| `1df2a24d` | (HEAD) F2 α-sweep reproduces on gfx1151 — α=0.55 sweet spot arch-portable |

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

## 11. Phase 8 — 27B endpoint (post-9B follow-up, 2026-05-16/17)

Added after Phases 0-7 (9B) shipped. Covers the changes needed to scale
the same pipeline from 9B (hidden=4096, intermediate=12288) to
Qwen3.6-27B (hidden=5120, intermediate=17408, 64 hybrid-arch layers).
Most of the math is unchanged; the surprises were memory + I/O.

### 11.1 Why 27B is harder than 9B

| Constraint | 9B impact | 27B impact |
|---|---|---|
| K_max for Cholesky | 12288 | **17408** — FP64 K×K = 2.4 GB, peak compute graph hits 15.7 GB on a 16 GB card |
| Total Hessian sidecar | 33 GB | **126 GB** — fits on /data (244 GB free) but won't fit in system RAM, ever |
| Per-tensor accumulator RAM | ~600 MB (K=12288 max) | **1.2 GB** (K=17408 max); × 64 layers = ~77 GB just on down_projs |
| Model BF16 | 18 GB (fits on 2× 16 GB GPU) | **54 GB** — must spread across 2× 16 GB GPU + CPU offload |

### 11.2 Patches landed for the 27B path

All on `feat/mq-v2-quant-format-cuda`. Cherry-pickable into other models
without changing semantics for ≤9B (defaults preserve 9B behavior).

| Commit | Change | Why |
|---|---|---|
| `9860597` | `collect_hessian.py --accumulator-dir` (memmap) + `--n-passes N` (multi-pass) | 126 GB Hessian doesn't fit in 32 GB RAM. Memmap accumulators to disk + multi-pass over disjoint layer subsets bound per-pass RAM to ~30 GB |
| `16b9551` | `collect_hessian.py --max-gpu-mem` | `device_map="auto"` without an explicit cap packs everything to GPU first, OOMs during `_init_weights`'s FP32 cast. Cap to 14 GiB/card → forces ~31 layers to CPU |
| `5c50d16` | `collect_hessian.py` always passes `logits_to_keep=1` | Default forward runs lm_head on all 2048 positions → [B, 2048, 248320] BF16 ≈ 1 GB on whichever device lm_head lives on, OOMs the GPU. Hessian collector never uses logits anyway |
| `2c453fb` | Memory-frugal `compute_damped_inv_cholesky_upper` | K=17408 cholesky workspace peaked at ~17 GB. Switch `solve_triangular → cholesky_inverse`, in-place damp on diagonal, aggressive `del`, consume `h_unrot` instead of cloning, pre-permute outside helper. Drops peak to 15.7 GB |

### 11.3 Stage breakdown for 27B (measured wall on dual RTX 5070 Ti)

| Stage | Tool | Wall | Output |
|---|---|---:|---|
| A. Imatrix | `llama-imatrix -ngl 28` (28 GPU + 36 CPU layers) on Qwen3.6-27B-BF16 GGUF, 128 chunks × 2048 ctx | **3.3h** | `benchmarks/quality-baselines/refs/qwen3.6-27b-bf16.imatrix.gguf` (13.6 MB, PPL 7.92 ± 0.07 on wikitext-2) |
| B. Hessian collection | `collect_hessian.py --n-passes 4 --accumulator-dir ~/.hipfire/hessian-acc-27b/ --max-gpu-mem 14GiB --n-sequences 32` | **9.8h** (4× ~2h passes + 18min merge) | `/data/hipfire-refs/qwen3.6-27b-bf16.hessian.bin` (125.83 GB) |
| C. GPTQ pipeline | `gptq_cuda.py --alpha 0.55 --bits 4 --devices cuda:0 cuda:1` | **~5h** (in flight; K=17408 down_projs at ~130s each are the cost driver) | `/data/hipfire/precomputed-gptq/qwen3.6-27b-mq4-awq-gptq-f2/` (~52 GB manifest) |
| D. Rust pack | `hipfire-quantize --format mq4 --precomputed-gptq-path <manifest>` | ~15 min | `~/.hipfire/quantized/qwen3.6-27b.mq4-awq-gptq-f2-cuda.hfq` (~14 GB) |

**Stage B note (--n-sequences 32):** the canonical hipfire calibration
is 128 sequences × 2048 ctx. 27B at 128 seq × `--n-passes 4` was on
track for ~60h on this hardware due to CPU-offload page-fault thrashing;
dropped to 32 seq to fit in an overnight window. The Hessian is a
smoothed expectation that converges fast; 65k tokens vs 262k tokens
should be within ~5% of asymptotic quality (untested at 27B scale,
but the 9B equivalent at 128 seq had per-tensor MSE in the 1e-6 range
and 27B's pass-by-pass per-tensor MSEs look similar so far).

### 11.4 Bottleneck on RTX 5070 Ti for 27B Stage C

K=17408 OBS column-sequential loop is the dominant cost. Per-tensor
~130s breakdown:

| Phase | ~Time | Why |
|---|---:|---|
| Hessian read from NFS (1.2 GB) | 5-12s | NFS ~100 MB/s, OS prefetch helps |
| Cholesky + cholesky_inverse | ~10s | FP64 matmul, 5×10¹² ops on 0.5 TFLOPS FP64 (Blackwell consumer 1:64 ratio) |
| **OBS column loop** | **~100s** | 17408 serial steps × scatter-subtract `M × (K-step)` FP64 values — GPU memory bandwidth + Python overhead bound |

**Implication for sizing:** for 35B+ models with K>20k, this scaling
gets prohibitive on consumer cards. See `docs/plans/gptq_mi50.md` for
a hardware alternative (Instinct MI50 = 1:2 FP64 ratio, ~13× faster).

## 12. Phase 9 — MQ3 (3-bit) parameterization + 4-variant sweep

Added 2026-05-17 to address the question: does GPTQ + AWQ activation-
aware uplift make uniform 3-bit grids viable, sidestepping the need
for Lloyd-Max non-uniform codebooks?

### 12.1 Pipeline parameterization

Commit `2c453fb` makes the bit-width a pipeline parameter. The same
GPTQ math applies; only the quant grid + bit packing differ:

| Bits | Levels | scale formula | Block layout | Rust pack function |
|---|---:|---|---|---|
| 4 | 16 | `range / 15` | 136 B (4B scale + 4B min + 128B nibble pack) | `gptq::pack_mq4g256_from_rotated_f64` |
| 3 | 8 | `range / 7` | 104 B (4B scale + 4B min + 32 chunks × 3B cross-byte) | `gptq::pack_mq3g256_from_rotated_f64` |

CLI: `gptq_cuda.py --bits {3, 4}`. Manifest schema bumped to v2;
v1 manifests imply `n_bits=4` for backward-compat. Rust's
`precomputed_gptq.rs::ManifestMeta.n_bits` reads it and `main.rs`'s
MQ4G256 branch dispatches on the value (no separate MQ3 branch needed
— same precomputed-fast-path code runs with a different packer).

### 12.2 The 4-variant sweep

`scripts/mq3_sweep_4b.sh` runs all four cells of the calibration matrix
at MQ3 on Qwen3.5-4B:

| Variant | --hessian | --imatrix | Hypothesis |
|---|---|---|---|
| **RTN** | no | no | Pure round-to-nearest. Master-doc §5 reports collapse on every locally-tested model. Baseline; should be unusable. |
| **AWQ-only** | no | yes | AWQ pre-scale + RTN pack. Activation balancing without Hessian-aware error propagation. Mid-tier. |
| **GPTQ-only** | yes | no | Hessian-aware OBS without grid pre-scale. The other agent's proposal. Tests whether GPTQ's error redistribution alone tames 3-bit quantization noise. |
| **AWQ + GPTQ** | yes | yes | Full pipeline. Best-case for uniform MQ3. If this collapses too, Lloyd-Max non-uniform codebooks (`quantize_mq3g256_lloyd`) are necessary at 3-bit. |

Wall per variant on 4B: ~25 min Stage C + ~2 min Stage D. Total ~1h40m.
Outputs land in `~/.hipfire/quantized/mq3-sweep/`, then ship to
gfx1100 for coherence-gate.

### 12.3 Decision tree from sweep results

Run coherence-gate on each .hfq:

- **All 4 collapse** → uniform MQ3 is unviable as a quant format at any
  calibration. Path D Lloyd-Max non-uniform codebooks are required for
  3-bit. Reuse existing `quantize_mq3g256_lloyd` (Rust path) — but
  it doesn't yet integrate with the precomputed-GPTQ manifest, so a
  separate PR.
- **Only AWQ+GPTQ is coherent** → uniform MQ3 is salvageable but needs
  the full activation-aware stack. Default the runtime to require
  AWQ+GPTQ for any MQ3 .hfq.
- **GPTQ-only is coherent without AWQ** → simpler quant story; the
  other agent's proposal works. Lloyd-Max becomes optional optimization.
- **AWQ-only is enough** → simpler still; Hessian collection isn't
  necessary at 3-bit. Saves ~10h per model on 27B-class quants.

The empirical answer drives whether 9B and 27B get MQ3 quants and which
variant.

### 12.4 Future: MQ3 on 9B and 27B

Same flow once the 4B sweep nominates a winner:

- 9B: existing 9B Hessian (`/data/hipfire-refs/qwen3.5-9b-bf16.hessian.bin`,
  33 GB) + imatrix already on disk. Re-run `gptq_cuda.py --bits 3` ~65 min.
- 27B: existing 27B Hessian (`/data/hipfire-refs/qwen3.6-27b-bf16.hessian.bin`,
  126 GB) + imatrix already on disk. Re-run `gptq_cuda.py --bits 3` ~5h.

No Hessian re-collection needed — the Hessian only depends on the
calibration corpus and the BF16 model, not on the quant bit-width.

## 13. Vast.ai cloud runbook — user-provided SSH, V100 32 GB

Added 2026-05-18 after a successful Qwen3.6-27B end-to-end on a rented
vast.ai V100 32 GB box. The full pipeline (Stage B + C + D) ran ~11.5h
wall on a single card, supervised from this machine over SSH with the
user's ssh-agent forwarded. This section captures the hand-off pattern
+ the V100 (`sm_70`) specifics that ate time, so the next run is faster
and a sub-`sm_75` card is avoided when possible.

### 13.1 Hardware sizing

- **`sm_75` floor.** sm_70 (V100) is borderline: PyTorch nightly wheels
  routinely ship sm_75+ only, the new `fla-core` Triton fast-path can't
  compile for sm_70, and the cu13.x wheels shipped on most vast templates
  omit sm_70 kernels entirely (silent `cudaErrorNoKernelImageForDevice`).
  **Rent a card sm_75 or newer** — T4 (sm_75), RTX 2080/3090/4090, A6000,
  A100, A40, L4, H100 all qualify and side-step this whole class of
  breakage. We used V100 only because it was the cheapest 32 GB card on
  vast at the time.
- **FP64 matters.** Stage C is K=17408 Cholesky-bound on 27B; ~130-160s
  per `down_proj` tensor at FP64. V100 has 1:2 FP32:FP64 (excellent);
  consumer Blackwell `sm_120` (RTX 5070 Ti / 5090) is **1:64** and Stage C
  blows out to ~10-15h. MI50 = 1:2 but ROCm `cholesky_ex` is broken at
  K≥6144 (see `gptq_mi50.md`). Pick a card with FP64 ≥ 1:8 or rent more
  of them.
- **VRAM ≥ 24 GB** to fit the K=17408 workspace without two-card juggling.
  32 GB comfortable. 16 GB needs `--max-gpu-mem` + CPU offload (slow path).
- **Host RAM ≥ 192 GB** if Stage B will offload (any BF16 model >16 GB on
  a single 16 GB card). 27B Stage B on V100 peaked at **152 GB RSS**.
- **`/workspace` ≥ 256 GB** for 27B. Tight even then — see §13.4.

### 13.2 Bring-up (user side, before this agent attaches)

1. Spin up vast instance; note SSH port + host.
2. `ssh-add ~/.ssh/id_rsa` in your local shell (vast's `Welcome to
   vast.ai` banner is normal).
3. On the instance: `git clone <fork>/hipfire && cd hipfire && git
   checkout feat/mq-v2-quant-format-cuda` (or whichever branch carries
   the parameterized pipeline + Stage D Rust support).
4. Verify Python ≥ 3.12 and rustup present. The "PyTorch 2.x devel"
   templates work after a torch reinstall (§13.3); skip the "Jupyter"
   templates — they preinstall conflicting CUDA stacks.
5. **Pin the working torch wheel** (V100 only — see §13.3):
   `pip install 'torch==2.7.1+cu126' --index-url https://download.pytorch.org/whl/cu126`
6. **Purge `fla-core` entirely** if preinstalled: `pip uninstall -y
   fla-core`. transformers' `qwen3_5` model_type hard-imports fla; once
   gone, `is_flash_linear_attention_available()` returns False and the
   PyTorch reference attention path runs (slower but correct on any card).
7. Paste the instance's ssh command into your local shell + tell this
   agent the agent socket path (`echo $SSH_AUTH_SOCK`). Without that
   path none of the SSH-driven steps below work.

### 13.3 V100 / sm_70 specifics (the hassle to avoid)

- **Torch wheel:** torch 2.11 (cu13.0) ships sm_75+ only. Symptom:
  `cudaErrorNoKernelImageForDevice` on first `cuda.is_available()`
  call, otherwise the instance "looks fine." Fix: install
  `torch==2.7.1+cu126` via the explicit cu126 wheel index. 2.7.x is
  the last branch packaging sm_70 stably.
- **`fla-core` Triton compile fails on sm_70:** `PassManager::run
  failed` during JIT of the first attention kernel. fla-core 0.5.0+
  also requires torch ≥ 2.7. transformers imports fla unconditionally
  for qwen3_5; purging it bypasses the import.
- **`ptrace_scope` is read-only in vast containers:** py-spy can't
  attach across user namespaces (`Failed to copy Py_Version symbol /
  Permission denied`). Diagnose hangs via `/proc/<pid>/stat`,
  `/proc/<pid>/task/*/stat`, `/proc/<pid>/io`. Steadily-growing
  `utime+stime` with `wchan=0` and high R-thread count = CPU-bound,
  not deadlocked.
- **`/venv/main` lives on a 16 GB overlay**, NOT on `/workspace`.
  pip-installing torch on top of the existing venv exhausts it fast.
  Run `pip cache purge` before any heavy install; if you need more,
  install to a fresh venv under `/workspace/.venv-cuda/` and skip the
  system venv.

### 13.4 Disk layout for 27B (256 GB `/workspace`)

```
/workspace/
  hipfire/                              # cloned repo (~500 MB w/ build)
  .hf_home/hub/models--Qwen--Qwen3.6-27B/  # ~54 GB
  hipfire-refs/<model>.hessian.bin      # ~118 GB; deletable after Stage C
  gptq-precomputed/<model>-<variant>/   # ~53 GB manifest; deletable after Stage D
  <model>.mq4-awq-gptq-f2-<host>.hfq    # ~14 GB final artifact
  pipeline.log, 27b-stage-{a,b,c,d}.log
```

Peak: 54 + 118 + 53 + 14 + overhead ≈ **245 GB on 256 GB partition**.
Delete the Hessian the moment Stage C is verified to give Stage D
headroom. If you can't, rent a bigger `/workspace`.

### 13.5 Driving from this side — the SSH agent dance

The user's ssh-agent socket from their local shell is the load-bearing
piece. When they restart their shell (tmux respawn, terminal reopen),
the old socket goes stale and SSH stops working from our side. Recovery:

```bash
ssh-add -l   # → "Error connecting to agent: No such file or directory"
ls -la /tmp/ssh-*/agent.* 2>/dev/null      # find a live one
export SSH_AUTH_SOCK=/tmp/ssh-<latest>/agent.<pid>
ssh-add -l   # → lists id_rsa, ready
```

When neither succeeds, ask the user to `eval "$(ssh-agent -s)" &&
ssh-add ~/.ssh/id_rsa` in the shell that spawned this session, then
re-share `$SSH_AUTH_SOCK`. If they're AFK, the supervisor loop simply
waits — the vast box keeps running.

### 13.6 Detached, supervised stages

Each long stage runs in a detached `tmux` session so the SSH transport
can die without killing the job:

```bash
tmux new-session -d -s stage-b \
  "cd /workspace/hipfire && /venv/main/bin/python scripts/collect_hessian.py \
     --model <snapshot> --output /workspace/hipfire-refs/<model>.hessian.bin \
     --n-sequences 32 --ctx-len 2048 --corpus <slice> \
     --device cuda --dtype bfloat16 --max-gpu-mem 14GiB \
     --max-cpu-mem 600GiB --n-passes 1 \
     > /workspace/27b-stage-b.log 2>&1"
```

Gotchas hit in this run:

- **Use absolute venv paths** inside the tmux command — `/venv/main/bin/python`,
  not `python`. A detached tmux inherits no shell rc; `python` won't
  resolve. Same for `cargo`: `source /root/.cargo/env` first or use
  `/root/.cargo/bin/cargo`.
- **Redirect inside the tmux invocation**, not outside. `tmux new -d
  "cmd" > log` captures *tmux's* (empty) stdout, not the cmd's. The
  `>` must live inside the quoted command string.
- **`tmux capture-pane -t <session> -p`** is the non-attaching way to
  peek at live output.
- **Sessions terminate when the command exits.** If you want
  post-completion output (md5sum, final size, "DONE" marker), append
  it inside the command **and** `tee` to a file — once bash exits the
  pane is gone, `capture-pane` returns `no server running`.

### 13.7 Monitoring the long stages

**Stage B** emits a progress line **every 8 seqs**, not every seq. Logs
silent for 60-70 min while `etime` and CPU keep climbing is the normal
pattern, not a hang. Diagnose:

```bash
grep -cE "^      seq [0-9]+/[0-9]+" /workspace/27b-stage-b.log
ps -p <pid> -o stat,pcpu,rss,etime
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
grep wchar /proc/<pid>/io       # bytes written to stdout
```

When `wchar` stops growing **and** `utime+stime` in `/proc/<pid>/stat`
also stops → genuine hang. Both growing → slow but progressing.

**Stage C** emits one line per tensor (cheap to monitor):

```bash
grep -cE "^  \[[ 0-9]+/[0-9]+\]" stage-c.log
grep -E "(RTN fallback|CholeskyFailed|OutOfMemory|Traceback)" stage-c.log
```

**Stage D** emits little. Verify by `.hfq` file size growing toward the
expected size + md5sum capture.

### 13.8 Recovery: Stage C flag mismatch

Encountered in this run: the V100's checkout had an older `gptq_gpu.py`
that didn't recognize the driver's `--checkpoint-interval` /
`--watchdog-timeout-sec` flags. Symptom: Stage C exits in <1s with
`argparse: unrecognized arguments`. Recovery without code changes — the
Hessian sidecar is already on disk so nothing is lost:

```bash
tmux new -d -s stage-c \
  "cd /workspace/hipfire && /venv/main/bin/python scripts/gptq_gpu.py \
     --input <snapshot> \
     --hessian /workspace/hipfire-refs/<model>.hessian.bin \
     --imatrix <imatrix.gguf> \
     --alpha 0.55 --bits 4 \
     --output /workspace/gptq-precomputed/<model>-mq4-awq-gptq-f2 \
     --devices cuda:0 \
     -v > /workspace/27b-stage-c.log 2>&1"
```

If your local has newer fixes, `git fetch origin && git reset --hard
origin/<branch>` on the box first — but if vast's outbound git auth
fails (we hit this), strip the unsupported flags and re-launch instead.

### 13.9 Cleanup + hand-back

After Stage D produces a verified `.hfq`:

```bash
md5sum /workspace/<model>.hfq                       # capture this BEFORE deleting anything
scp -P <port> root@<host>:/workspace/<model>.hfq <local-dest>/
# Then on the box:
rm /workspace/hipfire-refs/<model>.hessian.bin       # 118 GB
rm -rf /workspace/gptq-precomputed/<model>-*/        # 53 GB
rm -rf /workspace/.hf_home/hub                        # 54 GB
df -h /workspace
```

**Generic Hessian compression is not worth doing.** Empirical test on a
1 GB slice of the 27B Hessian: `zstd -3` gives 1.071× ratio (8 GB saved
on 118 GB), `zstd -19` gives 1.073× ratio — same compressed size, 20×
slower. FP64 mantissa noise is near-maximum byte-entropy. If you want a
smaller Hessian for future iteration (e.g. MQ3 retry without redoing
Stage B), change the dump format instead — FP32 + upper-triangle only
gets the on-disk Hessian to ~30 GB. That's a `collect_hessian.py` patch,
not a post-processing thing.

### 13.10 Measured 27B wall on V100 32 GB (2026-05-17/18)

| Stage | Wall | Notes |
|---|---:|---|
| Bring-up (torch reinstall + fla purge + HF download) | ~30 min | one-time per instance |
| B. Hessian (32 seq × 2048 ctx, 1 pass) | **5h 32m** | CPU/GPU hybrid; forward pass dominates |
| C. GPTQ → manifest (496 GPTQ + 10 RTN-fallback + 496 AWQ) | **5h 32m** | K=17408 down_projs at ~130-160s each are the cost driver |
| D. Rust pack (cargo build 57s + pack ~5 min) | **6 min** | one-shot |
| **Total** | **~11.5h** | + bring-up |

Stage C composition is the headline quality signal: **only 10 of 506
eligible tensors fell to RTN-fallback at 4-bit** (2%), vs ~30% at 3-bit
on Qwen3.5-4B. The OBS pipeline scales cleanly to 27B at MQ4.

On an sm_75+ card with comparable FP64 throughput, bring-up disappears
and the realistic next-time wall is **~11h pure compute** for a 27B-class
GPTQ. For 9B, scale down by ~3× (smaller K, fewer layers); ~3-4h end to
end is the right expectation.
