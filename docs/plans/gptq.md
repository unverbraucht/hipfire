# Stage B — GPTQ on MQ4 (implementation plan, v2)

**Status:** plan, pre-implementation (revised after adversarial review).
**Date:** 2026-05-12 (v2 incorporates findings from `gptq_plan_rev_claude.md`, `gptq_plan_rev_gemini.md`, `gptq_plan_rev_glm5.md`, consolidated in `gptq_plan_rev_synthesis.md`).
**Predecessor:** Stage A (AWQ on MQ4) — shipped + validated at commit `6594709d`. 9B measured at −32.6% above-floor KLD vs mq4-base, −5.0% PPL. Stage B is designed to STACK on Stage A, not replace it.
**Predicted lever value (revised):** +5-15% additional KLD-above-floor reduction (NOT the 15-25% PPL improvement the literature reports for GPTQ-on-RTN — AWQ already captured much of that lever). Stacks with AWQ's per-channel pre-scaling.
**Wire format:** no changes. Same `.hfq` MQ4G256 quant_type 13 blocks. Stage B only changes the *content* of the quantized values, not the format.

---

## v1 → v2 review cycle summary

Three independent adversarial reviews surfaced 21 issues across math, infrastructure, performance, and cosmetics. v2 incorporates all critical + major findings:

- **Math fixes:** Option β (FWHT-rotated GPTQ) is v1, not v2. Hessian must be transformed for both AWQ scaling AND FWHT rotation: `H_target = FWHT_per_256 · diag(1/s) · H_unrot · diag(1/s) · FWHT_per_256^T`. Per-block (scale, min_val) is FROZEN before GPTQ to avoid circular dependency.
- **Architecture fix:** `imatrix_collect.rs` is a llama.cpp subprocess wrapper — cannot be extended for Hessian collection. v2 introduces a new Python script (`scripts/collect_hessian.py`) using HF transformers for the calibration forward pass.
- **Algorithm additions:** WEIGHT-mode actorder (free quality lever), FP64 Cholesky via `faer` crate, max condition number cap with fallback to plain MQ4, explicit asymmetric per-element quantize formula.
- **Scope changes:** GPTQ now covers ALL MQ4G256 tensors, not just AWQ-eligible ones (non-AWQ tensors get a separate Hessian without `/s`).
- **Estimate revisions:** wall time 10-12 days (was 7), LOC ~1500-2000 Rust + ~100 Python (was 850), predicted 9B post-Stage-B KLD 0.70-0.74 (was 0.62-0.66).
- **Rejected:** my N2 (block_size=256) — both Gemini and GLM5 disagreed. Keep block_size=128 per GPTQ paper. The Cholesky tile size is independent of the MQ4 quantization grid.

Full validation/rejection trace per finding: `docs/plans/gptq_plan_rev_synthesis.md`.

---

## 1. Background

### 1.1 GPTQ algorithm

GPTQ (Frantar, Ashkboos, Hoefler, Alistarh 2023, arXiv 2210.17323) is a **post-training quantization algorithm** that minimizes per-tensor reconstruction error under an activation-aware loss:

```
min_W'  E_u [ ‖W·u − W'·u‖² ]   subject to W' having INT4 codewords
```

where `u` is the actual input fed to the matmul kernel — for hipfire MQ4G256+AWQ, `u = FWHT_per_256(x/s)`, NOT the original `x`. The Hessian `H = E[u·u^T]` is the activation-aware metric that GPTQ uses to weight per-column error propagation.

GPTQ quantizes columns sequentially (column by column in the K-axis). After quantizing column `k`, the remaining columns are updated to *compensate* for the rounding error of column `k`, weighted by the inverse Hessian's off-diagonal entries — total reconstruction error against `u` is minimized in the L2 sense, not the per-element sense. The compensation is computed via the Cholesky factorization of `H + λI` (standard Optimal-Brain-Surgeon derivation).

The result is a quantized weight tensor that produces near-identical output to the original FP16/BF16 weight on the calibration distribution. Per-tensor reconstruction error increases vs RTN (we're minimizing activation-space error, not weight-space error), but model-level KLD drops substantially.

### 1.2 Composability with AWQ — three-step weight transform

For hipfire's Stage A (AWQ) + Stage B (GPTQ) stack, the weight transform at quantize time is:

```
W (BF16)
  → W_awq = W · diag(s)                            [Stage A: AWQ pre-scaling]
  → W_rot = FWHT_per_256_along_K(W_awq)            [offline rotation, baked into stored weights]
  → W_gptq = GPTQ(W_rot, H_target, fixed_scales)   [Stage B: column-sequential update]
  → W_quant = round_to_per_256_block_grid(W_gptq)  [INT4 codewords + (scale, min_val) per 256-block]
```

Where:
- `s[j]` = AWQ scale per input channel (computed once from imatrix).
- `H_target` = `FWHT_per_256 · diag(1/s) · H_unrot · diag(1/s) · FWHT_per_256^T` (Hessian transformed into the AWQ-divided, FWHT-rotated basis — see §3 for the derivation).
- `fixed_scales` = per-256-block (scale, min_val) computed from `W_rot` BEFORE the GPTQ loop and FROZEN through the loop. Avoids the circular dependency where post-GPTQ weights would change the per-block min/max (per GLM5 C1).

At runtime, the AWQ-aware kernel divides `x/s`, the FWHT runs per 256-block, the matmul against `W_quant` produces output matching `W·x` modulo quantization noise. The math identity `(W_quant · diag(s)) · (FWHT(x/s)) ≈ W·x` holds because both AWQ's pre-scaling and FWHT's orthogonal transform are absorbed into `W_quant`.

### 1.3 Stage C scaffolding reuse — partial

MR-GPTQ (Egiazarian et al. 2509.23202) extends GPTQ for MXFP4 (= hipfire's MFP4G32). Stage C reuses **the GPTQ inner column-update + Hessian-collection infrastructure**, but the outer loop and per-block scale optimization are net new:

- **Reused (~60% of Stage B code):** Hessian collection script, FP64 Cholesky, column-sequential update, AWQ-Hessian transform, asymmetric quantize step.
- **Net new for Stage C:** E8M0 range mapping (Appendix H of MR-GPTQ paper), MSE-optimized grid alternating optimization (column-update interleaved with per-tensor block-scale tuning), MFP4 codeword grid.

Cost estimate after Stage B lands: ~1-2 weeks additional.

---

## 2. Phased plan + revised cost estimate

### Phase 1 — Hessian collection via Python script (2-3 days, ~30 min runtime per model)

**Why Python, not Rust:** the existing `crates/hipfire-runtime/examples/imatrix_collect.rs` is a llama.cpp subprocess wrapper (line 151: `Command::new(&args.llama_imatrix_bin)`). It has no native forward pass, no activation hooks. Extending it for Hessian collection is architecturally impossible without one of: (A) forking llama.cpp's imatrix tool to add Hessian accumulation [invasive, maintenance burden], (B) building a hipfire-native BF16 host forward pass [5-10 days new work, no existing infrastructure], or (C) using a Python/PyTorch tool [1-2 days, leverages mature HF transformers + torch].

v2 chooses (C). The Hessian is `E[u·u^T]` — a mathematical expectation independent of inference-engine internals. As long as we calibrate on the same tokens hipfire will see (which the Python script ensures by using HF tokenizer matching the model), the resulting Hessian is engine-agnostic. The 0.46% tokenizer disagreement rate between llama.cpp and hipfire (per issue-113 §126) is acceptable for the Hessian since it's a smoothed expectation over many tokens.

**Deliverables:**

1. New script `scripts/collect_hessian.py` (~100 LOC):
   - Inputs: BF16 model dir (HF format), calibration corpus (text file, sub-sampled to 128 sequences × 2048 tokens — Gemini 2.1 recommendation matching GPTQ paper's scale), output sidecar path.
   - For each `nn.Linear` whose name is on the GPTQ-eligible list (all MQ4G256 tensors per Topic 10 — q/k/v/qkv, gate/up/down, in_proj_*, o_proj/out_proj/down_proj, router): register a forward hook that accumulates `x.T @ x` into a per-tensor running sum.
   - Run forward pass over the calibration tokens (no gradient, no labels — just collect activations).
   - After the pass, divide by token count → `H_tensor = (1/N) · Σ x_t · x_t^T`.
   - Serialize all tensor Hessians to a single binary file with a JSON header.
   - Tokenizer: HF AutoTokenizer for the same model. Calibration corpus: wikitext-2-train (existing AWQ imatrix corpus) sub-sampled to 128 × 2048 = 262144 tokens.
   - Estimated runtime: ~30 min on a GPU host with the BF16 model loaded (most time is the forward pass, not the outer-product accumulation).

2. New module `crates/hipfire-runtime/src/hessian_io.rs` (~250 LOC) — read side: binary file format spec, mmap-based per-tensor lookup, FP64 deserialization. The Hessian binary format is also documented in `docs/plans/gptq-hessian-format.md`.

3. Sidecar binary format:
   - Header: 16-byte magic `HFHS` (Hipfire Hessian Sidecar) + 4-byte version + 8-byte total tensor count.
   - Per-tensor record: 4-byte name length + name string + 4-byte expert_idx (default 0; reserved for Stage B.1 MoE expert Hessians per GLM5 N7) + 4-byte K dimension + 8-byte fp32-or-fp64 flag (1 = FP32, 2 = FP64) + `K*K * sizeof(scalar)` bytes Hessian matrix.
   - Storage as FP32 (4 bytes/entry, 6 GB worst case on 9B) is sufficient — we promote to FP64 only for Cholesky.

4. Unit test: collect Hessian on TinyLlama (1.1B BF16), verify symmetry (`H[i,j] == H[j,i]` to FP32 precision), and that diagonal `H[j,j]` matches imatrix's `in_sum2[j] / n_tokens` to within ε.

### Phase 2 — GPTQ algorithm in the quantizer (5-6 days)

New module: `crates/hipfire-quantize/src/gptq.rs` (~500-600 LOC). Dependency: `faer` crate for FP64 Cholesky and triangular solve (no BLAS/LAPACK dependency, pure Rust).

**Pseudocode (per tensor, post-AWQ-prescale, post-FWHT-rotate, pre-MQ4-quant):**

```
Input:  W_rot (M×K, post-AWQ-prescaled + FWHT-rotated FP32 weights)
        H_unrot (K×K, raw Hessian from Python collector)
        s (K, AWQ scale vector; identity vector for non-AWQ-eligible tensors)
        block_size = 128       // GPTQ Cholesky-inverse tile (NOT the FWHT block)
        damp = 0.01 * mean(diag(H_unrot))
        max_cond = 1e8         // condition-number cap (per Gemini §3)

# Step 0: pre-compute fixed per-256-block (scale, min_val) from W_rot
# These are FROZEN through the GPTQ loop to avoid circular dependency
# (per GLM5 C1 extension)
n_blocks = M * K / 256
fixed_grids = Vec<(scale, min_val); n_blocks>
for b in 0..n_blocks:
    block_vals = W_rot.flat[b*256 .. (b+1)*256]
    fixed_grids[b] = (range / 15.0, min(block_vals))

# Step 1: transform Hessian into the AWQ-divided, FWHT-rotated basis
# (per Gemini 1.1 + GLM5 M0 — offline transformation in one pass)
H_awq = diag(1/s) @ H_unrot @ diag(1/s)            // AWQ rescaling
H_target = FWHT_per_256_similarity(H_awq)          // similarity transform under per-256 FWHT
# H_target is the Hessian in the same basis as W_rot

# Step 2: WEIGHT-mode actorder (per Claude C3 + Gemini 1.3)
# Permute columns of W_rot and H_target by descending diag(H_target).
# Save permutation; un-permute weights after the loop. No g_idx stored.
perm = argsort(diag(H_target), descending=True)
W_p = W_rot[:, perm]
H_p = H_target[perm, perm]

# Step 3: damping + FP64 Cholesky
H_damped = H_p + damp * I_K          // promote to FP64
cond = cholesky_condition_estimate(H_damped)
if cond > max_cond:
    log("tensor X has condition number {cond} > {max_cond} — skipping GPTQ, using plain MQ4")
    return quantize_mq4g256_from_rotated(W_rot, fixed_grids)   // fallback
L = cholesky(H_damped)               // FP64, lower triangular, via faer

# Step 4: column-sequential GPTQ loop (block_size=128 tile)
W_q = W_p.copy().to_fp64()
for col_start in 0..K step block_size:
    block_cols = col_start..(col_start + block_size)
    for j in block_cols:
        # Per-element quantize using FROZEN per-256-block (scale, min_val).
        # Each element W_q[i, j] belongs to flat-block b = (i*K + j) / 256.
        for i in 0..M:
            b = (i * K + (perm^-1)[j]) / 256       // use original column index for block lookup
            scale, min_val = fixed_grids[b]
            q = clamp(round((W_q[i, j] - min_val) / scale), 0, 15)
            W_q[i, j] = q * scale + min_val
        # OBS error propagation within this tile
        err = (W_p[:, j] - W_q[:, j]) / L[j, j]
        for k in (j+1)..(col_start + block_size):
            W_q[:, k] -= err * L[j, k] / L[k, k]   // ratio per GPTQ Algorithm 1
    # Inter-tile error propagation (GPTQ paper's cross-block update)
    block_err = (W_p[:, block_cols] - W_q[:, block_cols])
    W_q[:, (col_start + block_size)..K] -= block_err @ L_inv_inter_block   // see GPTQ §3.2

# Step 5: un-permute and convert back to FP32
W_p_unperm = W_q[:, argsort(perm)].to_fp32()

# Step 6: final MQ4G256 storage using the same FROZEN grids
emit quantize_mq4g256_from_rotated_with_fixed_grids(W_p_unperm, fixed_grids)
```

**Critical implementation notes:**

- **Hessian transformation (Step 1):** The per-256 FWHT is block-diagonal — `FWHT_per_256(M)` applies the same 256×256 Hadamard to each consecutive block of 256 rows/cols. Implementing `H_target[i, j] = Σ_k Σ_l FWHT[block_i, k] · H_awq[k, l] · FWHT[block_j, l]` reduces to per-block matrix products. Complexity: O(K² · 256) per tensor, ~50 ms per 4096×4096 H on host — negligible vs the K³/3 Cholesky.
- **Asymmetric quantize step (Topic 5):** `q = round((w − min_val) / scale)`, then `dequant = q * scale + min_val`. Both scale and min_val frozen per block.
- **Per-block lookup with permutation (Step 4):** the actorder permutation operates on COLUMNS of the K axis, but the per-256-block grid is indexed by FLAT row-major position `i*K + j_original`. We must use the un-permuted column index `(perm^-1)[j]` to look up the block. This is awkward but correct.
- **Inter-tile update efficiency:** computes outer product `block_err @ L_inv_block` against the remaining-tiles tile of L. Standard GPTQ implementation; faer's matmul is fine.
- **Adaptive damping:** if Cholesky fails (H + λI not PD even after promotion to FP64), increase λ by 10× and retry, up to λ = 1.0 × mean(diag(H)). Beyond that, fall through to plain MQ4 (with warning).
- **Coverage:** ALL MQ4G256 tensors (per Topic 10), not just AWQ-eligible. For non-AWQ tensors, `s = identity vector` so `diag(1/s) = I` and the Hessian rescaling step is a no-op; the rest of the algorithm is identical.
- **Memory cap:** rayon parallelism limited to `min(N_cores, ceil(available_RAM / max_H_FP64_size))` to prevent OOM. For 9B: max H is K=4096 → 16M entries × 8 bytes FP64 = 128 MB. 32 cores parallel = 4 GB — fits easily. For 27B: max H is K=12288 → 1.2 GB. Cap parallelism at 8 cores → 9.6 GB. (Gemini 2.2)

**CLI flags:**

```
--gptq <hessian-path>           # enable GPTQ on the provided Hessian sidecar
--gptq-block-size <N>           # default 128 (GPTQ paper default)
--gptq-damp <f>                 # default 0.01 * mean(diag(H))
--gptq-actorder <none|weight>   # default 'weight' (free quality lever)
--gptq-max-cond <f>             # default 1e8
```

**Per-tensor diagnostic output (printed in "Quantization Summary" block):**
- Input MSE vs original BF16
- Post-GPTQ MSE vs original BF16 (should be lower for tensors where GPTQ succeeded)
- Effective damping λ used
- Cholesky condition number estimate
- Number of MQ4 quant clamps (sign that the grid was too tight)
- Did GPTQ fall through to plain MQ4 (condition cap exceeded)?

**Unit tests:**
- GPTQ on a 64×64 toy weight + identity Hessian → byte-identical to RTN (no compensation should fire when H is identity).
- GPTQ on a 64×64 with a known diagonal Hessian → matches closed-form per-channel weighted-LS.
- GPTQ on a 256×256 with full Hessian → matched against a Python reference (numpy-based GPTQ).
- WEIGHT-mode actorder: permutation matches `argsort(diag(H))` descending, un-permutation recovers original order.
- Asymmetric quant: per-element quantize formula matches a hand-computed reference for a 16-element column with known (scale, min_val).
- **AWQ+GPTQ integration test (per GLM5 N8):** synthetic [M=256, K=512] tensor with known AWQ scales + Hessian; verify that AWQ + GPTQ + FWHT + MQ4 has lower reconstruction error than AWQ + FWHT + MQ4 alone.
- Stack test: AWQ-eligible vs non-AWQ tensor — verify the Hessian rescaling step is identity-on-identity for non-AWQ.

### Phase 3 — Validation cohort (1 day)

Same cohort harness as Stage A, with one new variant.

**Cohort spec (4 rows on 9B):**

```
qwen35-9b-q8f16          ~/.hipfire/models/qwen3.5-9b.q8f16                          gfx1100
qwen35-9b-mq4-base       ~/.hipfire/models/qwen3.5-9b.mq4-base-2026-05-12            gfx1100
qwen35-9b-mq4-awq        ~/.hipfire/models/qwen3.5-9b.mq4-awq-loaderfix-2026-05-12   gfx1100
qwen35-9b-mq4-awq-gptq   ~/.hipfire/models/qwen3.5-9b.mq4-awq-gptq                   gfx1100
```

**Decision tree (revised per Topic 9):**

| Outcome | Action |
|---|---|
| ΔKLD < −5% above-floor vs AWQ alone | Stage B is a real lever. Ship. Proceed to Stage C (MR-GPTQ on MFP4). |
| ΔKLD ∈ [−5%, +2%] vs AWQ alone | Expected. AWQ already captured the activation-aware lever; ship Stage B as quality-neutral or marginal-positive. |
| ΔKLD > +2% vs AWQ alone | Regression. Bisect: Hessian collection (run Python collector standalone), GPTQ inner loop (toy tensor test), AWQ-Hessian transform, FWHT-similarity transform, per-block scale freeze, actorder. |

**Predicted 9B post-Stage-B numbers (revised per Topic 9):**

| Variant | Predicted KLD | Predicted PPL |
|---|---:|---:|
| q8f16 (engine floor) | 0.5735 (measured) | 13.383 (measured) |
| mq4-base | 0.8165 (measured) | 15.063 (measured) |
| mq4-awq | 0.7373 (measured) | 14.303 (measured) |
| **mq4-awq-gptq (predicted)** | **0.71-0.73** | **13.9-14.1** |

Stage B is predicted to close 5-15% of the remaining above-floor gap (0.164 nats → ~0.14-0.16). The original predicted "60-80% closure" was an artifact of double-counting AWQ's contribution when applying GPTQ-on-RTN literature numbers.

### Phase 4 — Stage C scaffolding (deferred)

MR-GPTQ for MFP4G32 reuses Stage B's Hessian collection + Cholesky + column-sequential infrastructure. New work for Stage C (~1-2 weeks):
- E8M0 range mapping (Appendix H of MR-GPTQ paper).
- MSE-optimized grid alternating optimization (column-update interleaved with per-tensor block-scale tuning).
- MFP4 codeword grid (FP4 E2M1 with UE8M0 per-32 scales).

---

## 3. Risks + unknowns

| Risk | Mitigation |
|---|---|
| Calibration corpus shift between Hessian and inference distribution | Reuse wikitext-2-train (same corpus as Stage A AWQ imatrix). If downstream tasks show domain mismatch, fall back to AutoGPTQ-style C4 calibration as Stage B.1. |
| 6 GB Hessian sidecar disk cost on 9B | Acceptable on local SSD. NFS-resident sidecars are slower to load but still feasible. Stage B.1 could shard by tensor name for partial loads. |
| AWQ + GPTQ math identity breaks | The math identity holds: AWQ pre-scales weights, GPTQ optimizes the AWQ-scaled + FWHT-rotated representation, runtime divides activations by AWQ scale (no change). The Hessian rescaling step (`H_awq = diag(1/s) · H · diag(1/s)`) ensures GPTQ optimizes for the actual runtime input distribution. Verified by §1.2 derivation. |
| GPTQ destabilizes per-block FWHT scale calibration | Per-256-block min-max scale is computed BEFORE GPTQ runs (frozen grids per GLM5 C1) and reused unchanged after. No destabilization possible. |
| Hessian damping too aggressive → no benefit from GPTQ | Adaptive damping with effective-λ logging. If λ exceeds 1.0 × mean(diag(H)), fall through to plain MQ4. Bench `λ ∈ {0.001, 0.01, 0.1}` if v1 lift is < 5%. |
| Python script tokenizer disagrees with hipfire's tokenizer | Hessian is a smoothed expectation; 0.46% token-position disagreement is averaged out. Per-tensor `H` values differ by <1% between tokenizers — not material for GPTQ's per-column weighting. |
| FP32 → FP64 Cholesky memory overhead | For 9B max K=4096: H_FP64 = 128 MB per tensor. With 32-core rayon, peak 4 GB RAM. Acceptable. For 27B max K=12288: H_FP64 = 1.2 GB. Cap parallelism at 8 cores. |
| Python dependency in build pipeline | `scripts/collect_hessian.py` is an offline tool, not part of the daemon/runtime build. Same precedent as existing `scripts/fetch-eval-refs.sh` which uses Python via `.venv/bin/python3`. Document as optional dependency for Stage B quantizers; ship without if user accepts no Hessian = no GPTQ. |
| Sub-sampled calibration (128 sequences) misses tail tokens | GPTQ paper, AWQ paper, and llm-compressor all use ~128 sequences. Effective for outlier capture. Validate empirically on the 9B cohort. |
| Per-block scale freeze suboptimal post-GPTQ | The fixed scales are computed from pre-GPTQ rotated weights. Post-GPTQ weights are perturbed (by at most the quantization error magnitude). The scale's min/max is set by extreme values, which are unlikely to be GPTQ's compensation targets (those tend to be moderate per-channel adjustments). Suboptimality bound: at most ~5% of the quantization grid width on a few outlier blocks. |

---

## 4. Sequencing + revised wall-time budget

| Item | Wall time | Dependencies | Deliverable |
|---|---:|---|---|
| Phase 1.1 — `scripts/collect_hessian.py` implementation | 2 days | Python + HF transformers + torch installed | Standalone script |
| Phase 1.2 — Hessian file format spec | 0.5 day | 1.1 | `docs/plans/gptq-hessian-format.md` |
| Phase 1.3 — `hessian_io.rs` (Rust reader) | 1 day | 1.2 | Module + unit tests |
| Phase 1.4 — Test run: collect 9B Hessians, validate diag matches imatrix | 0.5 day | 1.1-1.3 | `qwen3.5-9b-bf16.hessian.bin` artifact (~6 GB) |
| Phase 2.1 — `faer` integration + FP64 Cholesky helper | 0.5 day | None | `gptq.rs` Cholesky + condition-estimate helpers |
| Phase 2.2 — Hessian transformation (AWQ + FWHT) | 1 day | 2.1 | `transform_hessian_for_gptq()` function |
| Phase 2.3 — GPTQ column-sequential implementation | 2 days | 2.1-2.2 | `gptq_column_sequential()` function |
| Phase 2.4 — WEIGHT-mode actorder | 0.5 day | 2.3 | Sort + un-permute logic |
| Phase 2.5 — CLI wiring + per-tensor diagnostics | 0.5 day | 2.3 | Flags + summary block |
| Phase 2.6 — Unit tests (incl. AWQ+GPTQ integration test) | 1.5 days | 2.3-2.5 | 7+ tests passing |
| Phase 2.7 — Python-reference cross-validation | 1 day | 2.6 | Test matching numpy GPTQ output on toy data |
| Phase 3 — 9B cohort + decision write-up | 1 day | 2.5 | Cohort results + findings.md |
| **Total Stage B wall time** | **~12 days** | | |

Calendar plan: kick off Phase 1.1 immediately after this plan revision lands. Phase 2.1 can start in parallel with Phase 1.4. Phase 3 cohort runs unattended overnight. Ship in 2-2.5 weeks calendar.

---

## 5. Code-touch surface (revised)

| File | Change | Approx LOC |
|---|---|---:|
| `scripts/collect_hessian.py` (new) | HF + torch + numpy Hessian collector, binary file writer | +100 LOC Python |
| `crates/hipfire-runtime/src/hessian_io.rs` (new) | Read/write sidecar Hessian format + mmap streaming | +250 LOC Rust |
| `crates/hipfire-quantize/src/gptq.rs` (new) | GPTQ column-sequential + FP64 Cholesky + Hessian transforms + actorder + asymmetric quantize | +500-600 LOC Rust |
| `crates/hipfire-quantize/src/main.rs` | CLI flags (`--gptq`, `--gptq-block-size`, `--gptq-damp`, `--gptq-actorder`, `--gptq-max-cond`), wire into MQ4G256 branch after AWQ prescale, before FWHT+quant | +60 LOC Rust |
| `crates/hipfire-quantize/Cargo.toml` | Add `faer = "0.x"` dependency | +1 line |
| `docs/plans/gptq-hessian-format.md` (new) | Sidecar binary file format spec | new doc |
| Unit tests | `gptq.rs` test module, `hessian_io.rs` test module, Python reference comparison, AWQ+GPTQ integration | +500-600 LOC Rust |
| **Total** | | ~1500-2000 LOC Rust + ~100 LOC Python + 1 new doc |

No changes to: runtime crates (no kernel changes), wire format (no `.hfq` schema changes), benchmark harness (cohort runner works unchanged).

---

## 6. Open questions resolved (v2)

1. **Block size for GPTQ inner update.** **Decision: 128** (GPTQ paper default). Bench 128 vs 256 in Stage B.1 if time permits. The Cholesky-inverse tile is independent of the MQ4 quantization grid.
2. **Hessian damping schedule.** **Decision: adaptive** — start with `λ = 0.01 * mean(diag(H))`, multiply by 10× if Cholesky fails, up to `λ = 1.0 * mean(diag(H))`. Beyond that, skip GPTQ for that tensor with logged warning.
3. **AWQ-before-vs-after-GPTQ ordering.** **Decision: AWQ first.** AWQ pre-scaling is applied at quantize time before GPTQ. GPTQ optimizes the AWQ-scaled, FWHT-rotated representation. Runtime AWQ inverse path is unchanged.
4. **Hessian collection corpus.** **Decision: wikitext-2-train sub-sampled to 128 sequences × 2048 tokens (~262k tokens).** Matches GPTQ paper's scale. C4 as Stage B.1 if downstream domain mismatch appears.
5. **MoE experts in v1.** **Decision: dense + linear_attn only.** v1 sidecar format reserves an `expert_idx` field for future MoE expert Hessians (GLM5 N7). Stage B.1 will populate it when the conditional-on-router accumulation lands.

---

## 7. References

- **GPTQ paper:** Frantar, Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (arXiv 2210.17323, ICLR 2023).
- **MR-GPTQ paper:** Egiazarian, Castro, Kuznedelev, ..., Alistarh, "MR-GPTQ: Micro-Rotated GPTQ for MXFP4 Quantization" (arXiv 2509.23202).
- **AWQ paper:** Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (arXiv 2306.00978, MLSys 2024).
- **vLLM GPTQ (consumption-side reference):** `/home/kread/mygit/vllm/vllm/model_executor/layers/quantization/gptq.py` — inference path for GPTQ-quantized weights.
- **compressed-tensors `ActivationOrdering`:** `/home/kread/mygit/compressed-tensors/src/compressed_tensors/quantization/quant_args.py` — WEIGHT vs GROUP actorder enum.
- **faer crate:** https://github.com/sarah-quinones/faer-rs — Rust linear algebra, FP64 Cholesky.
- **In-tree:**
  - `docs/plans/awq_hipfire.md` — Stage A AWQ integration (committed 6594709d).
  - `docs/plans/awq_fix_claude.md` — Stage A bug-hunt + measured results.
  - `docs/plans/qwen35-mq4-quality-gap.md` — Phase A overall plan.
  - `docs/plans/gptq_plan_rev_claude.md` — first-round Claude adversarial review.
  - `gptq_plan_rev_gemini.md` — Gemini adversarial review.
  - `gptq_plan_rev_glm5.md` — GLM-5 adversarial review.
  - `docs/plans/gptq_plan_rev_synthesis.md` — three-review validation/rejection consolidation.
  - `crates/hipfire-runtime/examples/imatrix_collect.rs` — existing imatrix subprocess wrapper (NOT extended for Hessian).
  - `crates/hipfire-quantize/src/main.rs` — quantizer entry point + Stage A AWQ wiring.
