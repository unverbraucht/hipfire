# Adversarial review of `docs/plans/gptq.md`

**Reviewer:** Claude (self-review of own plan, with cross-references to vLLM + compressed-tensors source).
**Date:** 2026-05-12.
**Verdict:** plan is structurally sound but has **three material issues that should block implementation as-written**, plus several smaller corrections. Recommend revising the plan before kicking off Phase 1.1.

References consulted during this review:
- `/home/kread/mygit/vllm/vllm/model_executor/layers/quantization/gptq.py`
- `/home/kread/mygit/vllm/vllm/model_executor/layers/quantization/utils/gptq_utils.py`
- `/home/kread/mygit/compressed-tensors/src/compressed_tensors/quantization/quant_args.py` (ActivationOrdering enum)
- `/home/kread/mygit/compressed-tensors/src/compressed_tensors/quantization/lifecycle/initialize.py` (g_idx init)
- GPTQ paper, arXiv 2210.17323 (re-read §3, §B for implementation specifics).
- AutoGPTQ reference (referenced via vllm + compressed-tensors consumption side; the actual algorithm lives in llm-compressor / AutoGPTQ upstream).

---

## CRITICAL issues (would silently produce wrong output if implemented as-written)

### C1 — Option α has a *math mismatch* between Hessian basis and quantization basis

The plan §2.2 proposes "Option α (split GPTQ)" as v1: collect Hessian on the *unrotated* input `x/s`, run GPTQ on the *unrotated* weights, then FWHT + quantize the GPTQ output.

This is mathematically wrong, not just suboptimal.

GPTQ's column-update formula derives from the loss `|| W·u − W'·u ||²_E[u·uᵀ]` where `u` is the actual matmul input. In hipfire, the matmul input is `FWHT(x/s)`, not `x/s`. So the right Hessian is:

```
H_correct = E[ FWHT(x/s) · FWHT(x/s)ᵀ ]   (per-256-block FWHT, so this is a block-diagonal similarity transform of E[(x/s)·(x/s)ᵀ])
```

Because FWHT is orthogonal *per-256-block* (not over the full K axis), `H_correct = (Q ⊗ I) · H_unrot · (Q ⊗ I)ᵀ` where `Q` is the 256×256 Hadamard. The diagonal *sum* is preserved (Parseval), but individual entries differ. Concretely:

- Off-diagonal entries `H[i,j]` for `i, j` *within the same 256-block* are non-zero in the rotated basis even if `H_unrot[i,j]` was sparse.
- Off-diagonal entries *across* 256-blocks are unchanged.

If GPTQ uses `H_unrot` but quantizes weights that will be matmul'd against `FWHT(x/s)`, the column-update propagates error using the WRONG covariance structure. The error compensation is misaligned with the actual loss surface. Empirically this might still help, but it might also actively hurt — and the lift would be unpredictable.

**Plan says** (§1.2 last paragraph): "the choice between collecting H pre- or post-rotation is a small implementation detail with a closed-form transformation between them."

**The closed-form transformation IS that block-similarity transform.** Skipping it (Option α) means using the wrong Hessian. The plan's claim that "α recovers ~80% of the lift per the paper" is also wrong — the GPTQ paper doesn't address rotated-quantized formats; the 80% number was about block_size choice, not about a rotated-vs-unrotated Hessian. The paper was published before MXFP4 / FWHT-rotated quants existed.

**Fix**: make Option β (FWHT-rotated GPTQ) the v1 design, not the v2. The cost of doing it right is modest:
- Collect Hessian on `FWHT_per_256(x/s)` activations. Same outer-product accumulator, just feed it the rotated post-AWQ activation. Implementation: hook *after* the existing fused_rmsnorm_rotate_mq_awq runs in the calibration forward pass.
- Run GPTQ on `FWHT_per_256(W_awq)`. Same column-update kernel; the per-column "quantize to MQ4 grid" step is now block-aligned with the 256-element FWHT block boundary, which makes the per-256-block min/max determination straightforward (the block IS the unit of quantization in rotated coords).
- Output is the quantized rotated weights — directly store as MQ4 codewords with per-256-block scales. No second FWHT pass needed.

This is also conceptually cleaner: hipfire's MQ4G256 is fundamentally a "FWHT-rotated INT4 representation," and GPTQ should optimize in that representation.

### C2 — Per-256-block min/max grid breaks GPTQ's "quantize one column at a time" assumption (Option α only)

This is downstream of C1. In standard GPTQ (per-tensor or per-group symmetric INT4), the quantization grid for column `j` is determined either before the GPTQ loop (per-tensor min/max) or per group (per-128-column group min/max). The "quantize column j to INT4" step is well-defined: you have a grid, you snap to it.

For hipfire MQ4G256 in Option α (unrotated GPTQ then FWHT-then-quantize):
- "Quantize column j to INT4" uses a *placeholder grid* (plan says: "the unrotated weight's per-tensor min/max").
- After all K columns are quantized, FWHT is applied per 256-block.
- The per-256-block min/max is recomputed AFTER FWHT, on the GPTQ-modified weights.
- The actual stored INT4 codewords differ from what GPTQ "thought" it was storing.

This means GPTQ's error propagation is computing error against a quantization step that never actually happens in the stored format. The compensation is dead — you compensate for placeholder-grid error, then re-quantize to a different grid.

**Fix**: with Option β (per C1), the per-column quantize step uses the per-256-block min/max in rotated coords, which IS the final MQ4G256 grid. The error propagation operates on the same grid the format actually stores. Math correct.

### C3 — Activation ordering ("actorder") is a free quality lever the plan ignores

vLLM's GPTQ config + compressed-tensors' `ActivationOrdering` enum exposes three modes:
- **WEIGHT** (alias: `static`): permute columns by activation magnitude during calibration only; save weights in re-ordered form; no runtime change. *Slight* accuracy gain.
- **GROUP** (alias: `dynamic`): permute and save a `g_idx` parameter; runtime reorders columns. Higher accuracy, +latency.
- **None**: no permutation.

The plan doesn't mention actorder at all. WEIGHT-mode actorder is essentially free for hipfire — it just changes the column-processing order inside GPTQ's loop, doesn't change the storage format, doesn't change the runtime path. AutoGPTQ defaults to `desc_act=True`. compressed-tensors' default is None but their llm-compressor pipeline recommends WEIGHT-mode for accuracy-critical workloads.

**Fix**: add WEIGHT-mode actorder to v1. Two-line change in the GPTQ inner loop (sort columns by `diag(H)` descending before processing, sort back at the end). Skip GROUP-mode — would require `.hfq` format change + runtime kernel work, deferred to Stage B.1.

---

## MAJOR issues (would cause measurable underperformance or implementation pain)

### M1 — MQ4G256 is ASYMMETRIC (zero-point); most GPTQ reference implementations are SYMMETRIC

vLLM's GPTQ doc says: `"sym": True  # GPTQ typically uses symmetric quantization`. compressed-tensors defaults `symmetric: bool = True`. AutoGPTQ and llm-compressor's defaults are symmetric too.

Hipfire's MQ4G256 stores `(scale, min_val)` per 256-block (see `quantize_mq4g256` in `crates/hipfire-quantize/src/main.rs:535`) — asymmetric quantization with a per-block zero offset.

The GPTQ algorithm itself works for asymmetric quant — you just optimize over both scale and zero. But most reference implementations don't include zero-point optimization. The plan needs to specify how the per-column quantize step computes (scale, zero) per 256-block in the GPTQ loop.

**Fix**: in the plan's Phase 2 deliverables, add a sub-task: extend the per-256-block min/max chooser to be GPTQ-aware (or equivalently, factor the existing `quantize_mq4g256`'s grid-computing code so GPTQ can call it per-column). Reference: GPTQ paper §B includes asymmetric quant derivation; AutoGPTQ supports `sym=False`; check those for the exact per-column update formula with zero.

### M2 — Numerical precision: FP32 Cholesky on K=12288 is unstable

GPTQ's Cholesky factorization of `H + λI` is the numerically sensitive step. For K=12288 (9B's MLP intermediate after AWQ gate_up_proj), the condition number of `H` can be 1e6+ even after damping. FP32 Cholesky on a 12288×12288 matrix with condition number 1e6 routinely produces non-PSD output or accumulated rounding errors that corrupt later columns.

Reference: AutoGPTQ + llm-compressor both use FP64 for Cholesky. compressed-tensors's quantize step uses FP32 only because the actual Cholesky lives upstream.

**Plan says**: "Hessian is symmetric PSD by construction" — true at infinite precision, false in FP32 after damping.

**Fix**: do Cholesky in FP64. Convert H to FP64 in-place after damping (one-time per tensor, ~10s extra for K=12288 — 8GB temporarily). Run the column-update inner loop in FP64 too. Only cast back to FP32 when writing the final quantized weights. Memory cost: K² × 8 bytes = 1.2 GB for largest tensor — fits in RAM, no I/O changes.

### M3 — Wall-time estimate undercounts Hessian collection

The plan says: "~30 minutes wall on 32-core CPU."

For Qwen3.5-9B's largest tensors:
- Each token contributes a K×K outer product. For K=12288 (the MLP gate_up_proj input dim... wait, 9B's intermediate is 12288, so down_proj input is 12288, but down_proj is excluded from GPTQ. The largest GPTQ-eligible tensor is gate_up_proj with input dim = hidden = 4096 for 9B. Let me verify — 9B has hidden=4096, intermediate=12288).
- Actually for 9B: gate_proj input = 4096, output = 12288. So GPTQ's K-axis on gate_proj = 4096. Hessian = 16M entries × 4 bytes = 64 MB per gate_proj tensor.
- 9B has 32 layers × ~5 GPTQ-eligible tensors = ~160 tensors.
- Outer product per token at K=4096: 16M FMA ops. 125k tokens × 16M = 2×10¹² ops per tensor.
- Sustained 50 GFLOPS on FP32 AVX2 outer products (cache-friendly): 40s per tensor sequential.
- 160 tensors / 32 cores parallel: ~3-5 min if all cores are saturated. But the forward pass itself isn't parallel across tensors — it's *sequential within a layer*. So Hessian collection has to run inside the existing imatrix forward pass.

The actual constraint is the forward-pass wall time (current `imatrix_collect` for 9B BF16: ~30 min wall on the gfx1100). Adding outer-product accumulation per layer adds ~constant overhead per token to the host CPU, won't dominate. Realistic: 35-45 min for full Hessian collection.

But: **the 6 GB disk-write cost.** Writing 6 GB of FP32 to local SSD: ~20s. To NFS `/data`: much slower (we just measured 100s of MB/s). If hessian.bin lives on NFS, the write step adds 1-2 min.

**Fix**: not really a fix — just be realistic. Update the wall-time estimate to ~45-60 min for full Phase 1.

### M4 — LOC estimate is low by ~50%

The plan: ~850 LOC + 1 new doc.

Cross-checking against compressed-tensors structure for a comparable feature:
- `compressed_tensors/quantization/quant_args.py` (ActivationOrdering + the related metadata): 500 LOC.
- `compressed_tensors/quantization/lifecycle/initialize.py` (g_idx init): 200+ LOC.
- `compressed_tensors/compressors/pack_quantized/base.py` (packed loading): 300+ LOC.
- AutoGPTQ's `fasterquant` (the actual algorithm, single function): ~250 LOC.

For hipfire's equivalent:
- `gptq.rs` (algorithm + helpers, with FP64 Cholesky from Rust nalgebra or manual): ~400-500 LOC, not 250.
- `hessian_io.rs` (binary format + read/write + tensor name handling): ~200-250 LOC.
- `imatrix_collect.rs` extension (hook per-tensor outer-product into forward pass): ~150-200 LOC, plus refactoring the existing layered loops.
- CLI wiring + arg parsing (`--gptq`, `--gptq-block-size`, `--gptq-actorder`, `--gptq-damp`): ~50 LOC.
- Tests: ~400 LOC for proper coverage (toy-model end-to-end, FP32-vs-FP64 cross-check, asymmetric quant, actorder=weight, AWQ+GPTQ stack, FWHT-rotated H transform).

**Realistic: ~1200-1500 LOC.** Plan should bump from 850.

### M5 — The "decision tree" outcome thresholds aren't well-calibrated

The plan §2.3 says:
- < −10% above-floor → ship
- [−10%, +5%] → marginal, investigate
- > +5% → bug

For context, Stage A AWQ achieved −32.6% above-floor on 9B. The GPTQ paper claims +15-25% lift on Q4 INT4 — but that's *without* AWQ. AWQ already captured the activation-aware lever. The marginal GPTQ benefit *on top of AWQ* is much smaller in literature: 5-10% typical, occasionally 15%.

**Fix**: revise the decision tree thresholds:
- ΔKLD < −5% above-floor vs AWQ alone → real lever, ship.
- ΔKLD ∈ [−5%, +2%] → expected if AWQ already captured most of the win; ship as quality-neutral or cite as evidence the two levers don't compose strongly.
- ΔKLD > +2% (regression) → bug.

And the predicted post-Stage-B 9B result should be revised from "KLD ~0.62-0.66" (plan) to "KLD ~0.71-0.73" — Stage A took us from 0.8165 to 0.7373; Stage B gets us a further 5-10% reduction in above-floor noise (0.1638 → ~0.14-0.16 → total KLD ~0.71-0.73). The plan's prediction was based on AWQ + GPTQ each at their literature max independently; in practice they share a fraction of the gain.

---

## MINOR issues (worth fixing but not blocking)

### N1 — Hessian collection coverage rule is misstated

Plan §2.1: "only for tensors that will be GPTQ-quantized — i.e., the Stage A whitelist." The Stage A whitelist is for AWQ pre-scaling, and excludes `o_proj` / `out_proj` / `down_proj`. But GPTQ in principle benefits ALL quantized weights, not just pre-AWQ-scaled ones. The plan's design *is* "GPTQ-on-AWQ-eligible-only" because the runtime composability constraint is the same as AWQ. But it's worth saying this explicitly:

> "GPTQ runs on the same whitelist as AWQ because GPTQ's quantization assumes a specific quant grid post-FWHT — the runtime weight matrix is `FWHT(W_awq)` for AWQ-eligible tensors and `FWHT(W)` for others. We could in principle run GPTQ on non-AWQ weights too (just `FWHT(W)`), but that path is deferred because (a) the o_proj/down_proj weights would benefit less from GPTQ if AWQ already wasn't applied, and (b) the runtime kernels for those tensors don't change either way."

### N2 — block_size choice rationale is thin

Plan §2.2: "the GPTQ paper's default" (128). For hipfire, aligning block_size with the FWHT block (256) might be the right call because the per-256-block min/max grid is *the* quantization unit. A non-aligned block_size means GPTQ's inner block crosses MQ4G256 grid boundaries, which complicates the "quantize this column to MQ4's per-256-block grid" step inside the inner loop.

**Fix**: in v1, use block_size = 256 (= FWHT block size). Rationale: alignment with the actual quantization unit, no cross-block contamination of error propagation. Bench a 128-vs-256 A/B in Stage B.1 if 256 underperforms.

### N3 — The "Stage B unblocks Stage C" claim isn't quite right

Plan §1.3 implies Stage C just reuses Stage B's GPTQ inner loop + adds E8M0 range mapping. But MR-GPTQ paper's algorithm has multiple structural differences:
- E8M0 range mapping changes the per-block scale codeword from FP16 (Stage B's MQ4) to UE8M0 (Stage C's MFP4).
- MSE-optimized grid alternating optimization is a *different* outer loop than plain GPTQ.
- Block-wise Hadamard size (k ∈ {16, 32, 64, 128}) is a hyperparameter; hipfire's MFP4G32 is FWHT-256.

Stage C is a significant new algorithm, not just "Stage B + E8M0 mapping." Plan should be more honest about this.

**Fix**: tone down §1.3's "Stage C reuses scaffolding" to "Stage C reuses the Hessian-collection scaffolding and GPTQ's column-update *inner step*; the outer-loop and per-block scale optimization are net new."

### N4 — Two cosmetic issues in the pseudocode

In §2.2 pseudocode:
- `H_inv = L^-T · L^-1` — this is the formula for `H⁻¹`, not what GPTQ actually needs. GPTQ uses `L⁻¹` columns directly for the inverse Hessian's row scaling, not the full inverse. The pseudocode is misleading.
- `err = (W_q[:, j] - q_j) / L_diag_inv[j]` — should be `err = (W_q[:, j] - q_j) / L[j, j]` (the j-th diagonal of L, after damping/Cholesky). The naming `L_diag_inv` confuses readers.

**Fix**: rewrite the pseudocode to match the GPTQ paper's Algorithm 1 more closely, or reference the AutoGPTQ implementation directly.

---

## What's GOOD about the plan

The good parts shouldn't get lost in the issue list:

1. **Composability with AWQ is correctly identified** as the path to ship. The math holds for AWQ pre-scaling + GPTQ optimization + FWHT + MQ4. No wire-format changes; same runtime.
2. **The phased structure is right.** Hessian collection (1 day) → GPTQ algorithm (3-5 days) → cohort validation (1 day) maps to the actual sequencing of risk.
3. **The risks section (§3) covers the right axes** — calibration domain shift, Cholesky damping, GPTQ-AWQ math interaction, etc. The one I missed in §3 but raised here is FP32 vs FP64 (M2).
4. **The "skip MoE experts for v1" call is right** — the per-expert conditional-on-routing Hessian is meaningfully harder than dense.
5. **The 4-variant cohort design** (Q8 floor + base + AWQ + AWQ+GPTQ) is the right comparison to isolate GPTQ's lift on top of AWQ.

---

## Summary of recommended plan revisions before kicking off Phase 1.1

| Issue | Severity | Plan change |
|---|---|---|
| C1: Hessian basis mismatch in Option α | **CRITICAL** | Adopt Option β (FWHT-then-GPTQ) as v1, not v2. Update §1.2 + §2.2. |
| C2: Per-256 grid breaks unrotated GPTQ | **CRITICAL** | Resolved by C1 fix (Option β operates in rotated basis where the grid IS per-256-block-aligned). |
| C3: WEIGHT-mode actorder missing | **CRITICAL** | Add to v1; two-line change in GPTQ inner loop (sort by `diag(H)` desc before, sort back after). |
| M1: Asymmetric MQ4 vs symmetric GPTQ default | major | Add explicit per-256-block (scale, zero) update inside the GPTQ inner step. |
| M2: FP32 Cholesky on K=12288 | major | Do Cholesky + column-update in FP64; cast to FP32 only when writing quantized weights. |
| M3: Wall-time estimate low | major | Revise §1 Phase 1 wall to 45-60 min (was 30 min). |
| M4: LOC estimate low | major | Revise from ~850 to ~1200-1500. |
| M5: Decision tree thresholds | major | Revise to: < −5% ship, [−5%, +2%] neutral, > +2% bug. Revise predicted 9B KLD to 0.71-0.73 (was 0.62-0.66). |
| N1: Hessian coverage rule explanation | minor | One paragraph in §2.1 explaining the AWQ-whitelist reuse rationale. |
| N2: block_size choice | minor | Default to 256 (= FWHT block), not 128. |
| N3: Stage B → Stage C scaffolding claim | minor | Tone down "reuses scaffolding"; Stage C has structural algorithm differences. |
| N4: pseudocode cosmetics | minor | Rewrite to match GPTQ Algorithm 1 / AutoGPTQ reference. |

## Bottom line

The plan's *concept* is right and the *sequencing* is right. But the v1 design has a math bug (Options α/β as written), and missing one near-free quality lever (actorder=weight). Both should be fixed before any code lands. After those changes, the plan is solid; Phase 1.1 can kick off.

Wall-time impact of the revisions: roughly the same as the original (Option β isn't significantly more code than Option α once you set the rotated-basis math straight), but the LOC and Phase-1-wall-time numbers should be revised upward to match reality.

**Recommendation**: revise gptq.md per the table above, re-circulate for a final read, then start Phase 1.1.
