# Stage B — GPTQ on MQ4 (implementation plan)

**Status:** plan, pre-implementation.
**Date:** 2026-05-12.
**Predecessor:** Stage A (AWQ on MQ4) — shipped + validated at commit `6594709d`. 9B measured at −32.6% above-floor KLD vs mq4-base, −5.0% PPL. Stage B is designed to STACK on Stage A, not replace it.
**Predicted lever value:** +15–25% additional PPL improvement (paper data on Q4 INT4 — Frantar et al. 2210.17323). Stacks additively with AWQ's pre-channel scaling.
**Wire format:** no changes. Same `.hfq` MQ4G256 quant_type 13 blocks. Stage B only changes the *content* of the quantized values, not the format.

---

## 1. Background — what GPTQ does and why it stacks with AWQ

### 1.1 GPTQ in one paragraph

GPTQ (Frantar, Ashkboos, Hoefler, Alistarh 2023, arXiv 2210.17323) is a **post-training quantization algorithm** that minimizes per-tensor reconstruction error under an activation-aware loss:

```
min_W'  E_x [ || W·x − W'·x ||² ]   subject to W' having INT4 codewords
```

Standard per-tensor min-max quantization (what hipfire's `quantize_mq4g256` currently does) minimizes `|| W − W' ||²` — element-wise weight-space error. That ignores the input distribution: a channel with low activation magnitude can take large quantization error without hurting model output; a channel with high activation magnitude needs small quantization error.

GPTQ replaces the per-tensor min-max objective with one that:
1. **Collects the per-tensor input Hessian** `H = E_x [x · x^T]` (a `[K, K]` matrix) during a calibration pass.
2. **Quantizes columns sequentially** (column by column in the K-axis). After quantizing column `k`, the remaining columns are updated to *compensate* for the rounding error of column `k`, weighted by the Hessian's off-diagonal entries — so the total reconstruction error against `x` is minimized in the L2 sense, not the per-element sense.
3. Uses Cholesky factorization of `H + λI` for numerical stability and a closed-form per-column update (the "OBS" — Optimal Brain Surgeon — derivation).

The result is a quantized weight tensor that produces near-identical output to the original FP16/BF16 weight on the calibration distribution. Per-tensor reconstruction error increases vs RTN (because we're minimizing activation-space error, not weight-space error), but model-level KLD drops substantially.

### 1.2 Why GPTQ stacks with AWQ

AWQ pre-scales weights per input channel: `W' = W · diag(s)`. This is a *static* rescaling — it doesn't depend on the loss function. AWQ's job is to make outlier-activation channels' weights survive quantization (by amplifying them before quantization → relative error is preserved).

GPTQ optimizes *how* the quantization rounds happen — but takes the input weight matrix as given. So the natural composition is:

1. Quantize-time pass 1 (AWQ): `W_awq = W · diag(s)` where `s[j] = (RMS_act[j])^α`.
2. Quantize-time pass 2 (GPTQ): apply GPTQ's column-by-column update on `W_awq` to get `W_gptq_awq`.
3. Quantize-time pass 3 (FWHT + MQ4): rotate per 256-block, quantize each block to INT4.

At runtime, the AWQ inverse `x/s` and the FWHT `FWHT(x/s)` cancel as before (Stage A math). GPTQ's column-by-column updates already happened — the stored `W_gptq_awq` is just a different (better-conditioned) `W'` that the runtime is oblivious to.

**The order matters.** GPTQ should run AFTER AWQ pre-scaling because GPTQ's Hessian-aware updates are computed on the weight matrix that will actually get quantized — i.e., on `W · diag(s)`, not on `W` alone.

**Open question — should GPTQ run before or after the offline FWHT?** Two options:

- **Option A (GPTQ first):** `W → W_awq → W_gptq_awq → FWHT(W_gptq_awq) → quant`. GPTQ sees the unrotated weights; the FWHT happens after.
- **Option B (FWHT first):** `W → W_awq → FWHT(W_awq) → W_gptq_fwht_awq → quant`. GPTQ sees the rotated weights, optimizes for them.

Option B is the right choice — GPTQ should optimize the *final* representation that will get quantized. The MQ4 per-block min-max + INT4 codewords operate on the FWHT-rotated buffer, so GPTQ should target that. The Hessian also needs to be in the rotated coordinate system. Concretely: `H_rotated = FWHT_per_256(H · FWHT_per_256^T)` — but since FWHT is orthogonal, this is just the same H expressed in a different basis. Equivalently, collect H in the input (unrotated) coordinate, but apply GPTQ's column-by-column update on `FWHT(W_awq)` with `H` rotated to match.

Section 3 below proposes the concrete algorithm; the choice between collecting H pre- or post-rotation is a small implementation detail with a closed-form transformation between them.

### 1.3 Why stages C (MR-GPTQ on MFP4) needs Stage B first

MR-GPTQ (Egiazarian et al. 2509.23202) is GPTQ + E8M0 range mapping + MSE-optimized grid alternating optimization, specifically targeting MXFP4 (= hipfire's MFP4G32). MR-GPTQ's GPTQ leg is the same algorithm as plain GPTQ — the differentiator is the wraparound for the FP4 element format + per-block UE8M0 scale.

So Stage B builds the Hessian-collection scaffolding + GPTQ inner loop for MQ4G256. Stage C reuses the scaffolding, changes the per-tensor optimization to handle FP4 codewords + UE8M0 scales. Stage C lift on MFP4G32: ~5.5pp recovery improvement per the paper (RTN 87.83% → MR-GPTQ 93.31% on Llama-3.1-8B). Stage B lift on MQ4G256: predicted +15-25% PPL beyond AWQ, but we'll measure.

---

## 2. Phased plan + cost estimate

### Phase 1 — Hessian collection (1–2 days, CPU + GPU)

Extend `crates/hipfire-runtime/examples/imatrix_collect.rs` to ALSO dump the per-tensor full Hessian `H = (1/n) Σ_t x_t · x_t^T` in addition to the diagonal (which is what current imatrix carries).

**Design choices:**

- **Coverage:** only for tensors that will be GPTQ-quantized — i.e., the Stage A whitelist (`q/k/v_proj`, `gate/up_proj`, `in_proj_*`, `gate_up_proj`, `mlp.gate.weight`). Skipping `o_proj`/`out_proj`/`down_proj` saves ~30% of Hessians; we can revisit if Option B (AWQ for those tensors) lands later.
- **Storage:** per-tensor `H` is `[K, K]` FP32. For K=1024 → 4 MiB; for K=12288 (9B's MLP intermediate via gate_up_proj) → 576 MiB. **Total for 9B:** ~6 GB worst case, ~3 GB if we shard by tensor and only collect what's needed. Sidecar file: `<model>.hessian.bin` next to the imatrix.
- **Storage format:** binary float32 matrix per tensor, prefixed with a JSON header (matching the kldref / imatrix pattern). Each Hessian preceded by a 4-byte length + tensor-name string + 4-byte K + the `K*K*4` bytes. No compression for v1; FP32 is reasonable for v1.
- **Calibration corpus:** same as current imatrix (wikitext-2-train, ~125k tokens). 125k tokens × ~32 layers × ~K=1024 average → ~4×10⁹ Hessian outer-product accumulations per tensor. At ~10 GFLOPS sustained on host CPU (cache-friendly outer products), ~6 minutes per tensor — but parallelism over tensors brings this down. Plan ~30 minutes wall on 32-core CPU.
- **Numerical conditioning:** Hessian is symmetric PSD by construction. For Cholesky in Phase 2 we need `H + λI` with `λ ≈ 0.01 * mean(diag(H))` (standard GPTQ damping). Add `λ` at Cholesky time, not collection time.

**Concrete deliverables:**

1. New `--hessian-out <path>` flag on `imatrix_collect`.
2. New helper `collect_per_tensor_hessian` that hooks the same forward pass that imatrix uses, accumulates `H` on the host (no GPU Hessian — host CPU is fine for this scale).
3. Binary file format spec at `docs/plans/gptq-hessian-format.md` (sibling to `kldref-format`).
4. Unit test: collect Hessian on a 4-layer toy model, verify symmetry and that `H[j,j]` matches imatrix's `in_sum2[j] / n_tokens`.

### Phase 2 — GPTQ algorithm in the quantizer (3–5 days, CPU)

Implement GPTQ's column-by-column update inside `hipfire-quantize`. New CLI flag: `--gptq <path-to-hessian.bin>` (mutually compatible with `--awq`).

**Pseudocode (per tensor, post-AWQ-prescaling, pre-FWHT/MQ4-quant):**

```
Input:  W (M×K, post-AWQ-prescaled FP32)
        H (K×K, precomputed Hessian)
        block_size = 128  // GPTQ inner block (NOT the 256 FWHT block)
        damp = 0.01 * mean(diag(H))

# Damping + Cholesky factorization
H_damped = H + damp * I_K
L = cholesky(H_damped)     // K×K lower triangular
H_inv = L^-T · L^-1        // we only need the Cholesky-inverse columns

# Sequential column-quantization
W_q = W.copy()  // will hold the quantized result (still FP32-valued, INT4-grid-aligned)
for col_start in 0..K step block_size:
    block_cols = col_start..(col_start + block_size)
    for j in block_cols:
        # Quantize column j to MQ4's INT4 grid (per-256-FWHT-block min-max)
        # NOTE: at this point W_q is in unrotated coords; the actual MQ4
        # quantize happens later in quantize_mq4g256 after the offline FWHT.
        # GPTQ here is doing element-wise nearest-INT4-after-FWHT.
        # See §3 for the FWHT-integrated derivation.
        q_j = quantize_one_column_to_mq4_grid(W_q[:, j])
        err = (W_q[:, j] - q_j) / L_diag_inv[j]
        # Propagate error to remaining columns in this block
        W_q[:, j] = q_j
        for k in (j+1)..(col_start + block_size):
            W_q[:, k] -= err * L_inv_block[j, k]
    # Propagate inter-block error update (the GPTQ Cholesky off-diagonal trick)
    W_q[:, (col_start + block_size)..K] -= err_accumulated · L_inv_inter_block

# After the loop, W_q is quantized + Hessian-aware-error-compensated.
# Then proceed with normal FWHT + MQ4 storage:
emit quantize_mq4g256(W_q)
```

**Critical implementation details:**

- **Block-wise GPTQ:** the inner block size (128) is a numerical / cache-tuning knob from the GPTQ paper, NOT the same as MQ4's 256-element FWHT block. The inner block determines how much of the Cholesky-inverse we touch per error-propagation pass. Setting to 128 matches the paper's default.
- **MQ4-grid quantization step:** for plain GPTQ on per-tensor uniform INT4, this is `round((w - min) / scale)`. For MQ4G256, the grid is determined by the 256-element FWHT block's min/max AFTER FWHT — so the quantize-one-column step needs to know which 256-block it's in and what that block's per-tensor scale will be. **Two options here:**
  - **Option α (split GPTQ):** run GPTQ column-by-column in unrotated coords, with the "quantize step" using a *placeholder* grid (e.g., the unrotated weight's per-tensor min/max). Then FWHT + MQ4-quant the result. This is the simplest; quality is suboptimal because GPTQ's quantization target isn't the actual MQ4 grid.
  - **Option β (integrated):** rotate everything first. `W_awq → FWHT_per_256(W_awq)`, then GPTQ on FWHT'd weights with `H_rotated` (similarity transform of H under per-256 FWHT). Per-256-block GPTQ blocks naturally align with MQ4's per-256-block min-max. This is more accurate but requires re-deriving the Cholesky / error propagation in the rotated basis.
  - **Choice:** start with Option α for v1 (simpler, faster to implement). Land it, measure 9B AWQ+GPTQ. If lift is < 10% beyond AWQ alone, revisit with Option β. Per the GPTQ paper, Option α (placeholder grid) recovers most of the lift; the rotated/integrated version is mostly defensive.
- **Hessian I/O:** read the per-tensor `H` lazily from disk via mmap or pread. Don't load all 6 GB at once. Process tensors in arbitrary order — GPTQ is per-tensor, embarrassingly parallel across tensors (via rayon).
- **Numerical conditioning:** if Cholesky fails (H + λI not PD even after damping), increase λ adaptively. Log the per-tensor effective λ. If λ exceeds 1.0 × diag mean, skip GPTQ for that tensor and fall through to plain MQ4 (with a warning) — extreme conditioning suggests bad calibration coverage.

**Concrete deliverables:**

1. `gptq_column_sequential` function in `crates/hipfire-quantize/src/gptq.rs` (new file).
2. `--gptq <path>` flag wired into `main.rs` MQ4G256 branch, after AWQ pre-scaling and before `quantize_mq4g256`.
3. Per-tensor diagnostic output: input MSE, post-GPTQ MSE, effective damping, Cholesky condition number. Print in the "Quantization Summary" block.
4. Unit tests: GPTQ on a 64×64 toy weight + identity Hessian (should be byte-identical to RTN); GPTQ on a 64×64 with a known diagonal Hessian (should match closed-form per-channel weighted-LS); GPTQ on a small full Hessian matched against a Python reference (numpy-based GPTQ).

### Phase 3 — Validation cohort (1 day)

Same cohort harness as Stage A, with one new variant.

**Cohort spec (4 rows on 9B):**

```
qwen35-9b-q8f16          ~/.hipfire/models/qwen3.5-9b.q8f16                          gfx1100
qwen35-9b-mq4-base       ~/.hipfire/models/qwen3.5-9b.mq4-base-2026-05-12            gfx1100
qwen35-9b-mq4-awq        ~/.hipfire/models/qwen3.5-9b.mq4-awq-loaderfix-2026-05-12   gfx1100
qwen35-9b-mq4-awq-gptq   ~/.hipfire/models/qwen3.5-9b.mq4-awq-gptq                   gfx1100
```

**Decision tree (after eval):**

| Outcome | Action |
|---|---|
| GPTQ ΔKLD < −10% above-floor vs AWQ alone | Stage B is a real lever. Ship. Proceed to Stage C (MR-GPTQ on MFP4). |
| GPTQ ΔKLD ∈ [−10%, +5%] vs AWQ alone | Marginal. Investigate Option β (integrated FWHT+GPTQ) or tune block_size / damping. |
| GPTQ ΔKLD > +5% vs AWQ alone | Regression. Hessian collection or column-update logic is buggy; bisect. |

Predicted: −15-25% beyond AWQ, putting 9B at KLD ~0.62-0.66 (above-floor ~0.05-0.09 nats, vs Q8 floor 0.5735). That would close ~60-80% of the original mq4-base above-floor gap (0.243 nats).

### Phase 4 — Stage C scaffolding (deferred)

MR-GPTQ for MFP4G32 reuses the Hessian + GPTQ column-update from Stage B. New work for Stage C:
- E8M0 range mapping (Appendix H of MR-GPTQ paper) — the differentiator vs plain GPTQ.
- MSE-optimized grid alternating optimization (column-update interleaved with per-tensor block-scale tuning).

Cost estimate after Stage B lands: ~1-2 weeks.

---

## 3. Risks + unknowns

| Risk | Mitigation |
|---|---|
| Hessian collection wall time > 1h | Sample subset of calibration tokens (paper uses ~128 sequences); rerun with full corpus only if quality bench warrants |
| 6 GB sidecar disk cost | Could be reduced by collecting per-tensor `H` only for the K largest tensors (top-50% by size) and falling through to RTN+AWQ for the rest. Defer until measurement shows this matters. |
| Option α (split GPTQ) underperforms Option β | Implement α first (cheap), measure, and re-derive β only if needed. The paper claims α recovers ~80%+ of the win. |
| GPTQ + AWQ interaction breaks the math identity | The math holds: AWQ pre-scales weights, GPTQ optimizes those pre-scaled weights' representation, runtime divides activations by AWQ scale (no change). No interaction beyond "GPTQ sees scaled weights" which is intentional. |
| GPTQ destabilizes per-block FWHT scale calibration | Per-256-block min-max scale is computed by `quantize_mq4g256` AFTER GPTQ writes its output. As long as GPTQ output values are still in the expected dynamic range, the per-block scale chooser adapts. Worst case is a slightly suboptimal scale; correctness intact. |
| Hessian damping too aggressive → no benefit from GPTQ | Start with paper default `λ = 0.01 * mean(diag(H))`. Sweep `λ ∈ {0.001, 0.01, 0.1}` on one bench if v1 lift is < 5%. |
| Calibration corpus shift between Hessian and inference distribution | wikitext-2 generalizes well; same data we used for AWQ imatrix. If we ever see a calibration-vs-evaluation domain mismatch, the issue would surface in BOTH AWQ and GPTQ — and likely AWQ first. |

---

## 4. Sequencing + wall-time budget

| Item | Wall time | Dependencies | Deliverable |
|---|---:|---|---|
| Phase 1.1 — Hessian collection in `imatrix_collect` | 1 day | None | `--hessian-out` flag, `qwen3.5-9b-bf16.hessian.bin` sidecar artifact |
| Phase 1.2 — Hessian file format spec | 2 hours | 1.1 | `docs/plans/gptq-hessian-format.md` |
| Phase 1.3 — Unit tests | 0.5 day | 1.1 | Toy-model Hessian symmetry + diagonal-matches-imatrix |
| Phase 2.1 — GPTQ column-update implementation | 2 days | 1.1 | `gptq.rs` module |
| Phase 2.2 — CLI wiring + per-tensor diagnostics | 0.5 day | 2.1 | `--gptq <path>` flag in `main.rs` |
| Phase 2.3 — Unit tests + Python-reference cross-validation | 1 day | 2.1 | 3+ tests passing; bit-exact vs numpy reference on toy data |
| Phase 3 — 9B cohort + decision write-up | 1 day | 2.2 | Cohort results table + `awq_gptq_postfix_findings.md` |
| **Total Stage B wall time** | **~7 days** | | |

Calendar plan: kick off Phase 1.1 immediately after this plan lands. Phase 2.1 starts after 1.3 unit tests are green. Phase 3 cohort runs unattended overnight. Ship in 1-1.5 weeks calendar.

---

## 5. Code-touch surface

| File | Change | Approx LOC |
|---|---|---:|
| `crates/hipfire-runtime/examples/imatrix_collect.rs` | Extend with `--hessian-out` flag + per-tensor outer-product accumulation | +120 LOC |
| `crates/hipfire-runtime/src/hessian_io.rs` (new) | Read/write sidecar Hessian binary file format | +150 LOC |
| `crates/hipfire-quantize/src/gptq.rs` (new) | GPTQ column-sequential algorithm + Cholesky helpers | +250 LOC |
| `crates/hipfire-quantize/src/main.rs` | `--gptq <path>` flag, wire into MQ4G256 branch after AWQ prescale | +30 LOC |
| `docs/plans/gptq-hessian-format.md` (new) | Sidecar binary file format spec | new doc |
| Unit tests | `gptq.rs` test module, `hessian_io.rs` test module, `imatrix_collect.rs` Hessian-output test | +300 LOC |
| **Total** | | ~850 LOC + 1 new doc |

No changes to: runtime crates (no kernel changes), wire format (no `.hfq` schema changes), benchmark harness (cohort runner works unchanged).

---

## 6. Open design questions

1. **Block size for GPTQ inner update.** Paper uses 128. The MQ4 FWHT block is 256. Does it make sense to align these (block_size=256)? Worth a single-bench A/B once v1 works.
2. **Hessian damping schedule.** Constant `λ` (paper default) vs adaptive (increase if Cholesky fails). Default to adaptive with logged effective λ.
3. **Sequencing AWQ before vs after GPTQ in the quantize chain.** Default: AWQ first, GPTQ on AWQ-scaled weights. The reverse order doesn't compose cleanly (GPTQ would optimize on un-AWQ-scaled weights; AWQ post-multiply would re-introduce noise). **Decision (2026-05-12): deferred — adversarial review will revisit before implementation.**
4. **Hessian collection corpus.** Reuse wikitext-2-train via existing `imatrix_collect` integration vs commit to a separate calibration set. **Decision (2026-05-12): v1 reuses wikitext-2-train (same corpus already integrated into `imatrix_collect`).** Documented risk: if downstream tasks show calibration-vs-eval domain mismatch, run a separate AutoGPTQ-style C4 calibration as Stage B.1. The bug-surface area of standing up a second calibration pipeline is high relative to expected marginal gain — defer.
5. **MoE experts in v1.** Per-expert input distribution is conditional on the router; collecting per-expert Hessians needs the routing decisions, which adds complexity. **Decision (2026-05-12): v1 dense + linear_attn only; MoE experts deferred to Stage B.1.** Calibration dataset choice for the eventual MoE expert Hessian collection: also wikitext-2-train (see Q4); use an off-the-shelf dataset for consistency rather than building a Qwen3.5-specific one (no Qwen3.5-specific corpus is publicly available, and the marginal benefit of arch-specific calibration is small per the AWQ paper).

---

## 7. References

- **GPTQ paper:** Frantar, Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (arXiv 2210.17323, ICLR 2023).
- **MR-GPTQ paper:** Egiazarian, Castro, Kuznedelev, ..., Alistarh, "MR-GPTQ: Micro-Rotated GPTQ for MXFP4 Quantization" (arXiv 2509.23202).
- **AWQ paper:** Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (arXiv 2306.00978, MLSys 2024).
- **In-tree:**
  - `docs/plans/awq_hipfire.md` — Stage A AWQ integration (committed 6594709d).
  - `docs/plans/awq_fix_claude.md` — Stage A bug-hunt + measured results.
  - `docs/plans/qwen35-mq4-quality-gap.md` — Phase A overall plan.
  - `crates/hipfire-runtime/examples/imatrix_collect.rs` — Tier 2 imatrix collector (extends naturally).
  - `crates/hipfire-quantize/src/main.rs` — quantizer entry point + Stage A AWQ wiring.
