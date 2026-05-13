# Combined review: 04-pattern-hunt-plan.md

**Reviewers folded:** Claude Opus 4.7 (author/self-review), Gemini CLI, GLM-5.
**Date:** 2026-05-13

After de-duplication: **12 issues require plan revisions**, 4 unique
points from each reviewer worth incorporating, 2 from author's own
audit are reinforced, 0 rejected outright (all reviewer points have
defensible motivation; some are demotions rather than full rejections).

Strongest cross-reviewer agreement: **run the existing
`rmsnorm_f32_f64acc` kernel as a Day-0 prerequisite** — both Gemini
(Section 2.2) and GLM-5 (S3) flag this is a 30-minute test that
falsifies H2 and H5 instantly if it doesn't reduce the 0.0017 rL2.
The original plan missed that this probe kernel already exists in-tree
from the closed investigation.

---

## Critical revisions (must address before Day 1)

### CR1. Day 0 prerequisite — run the existing f64acc probe + derive a real threshold

**Sources:** Gemini (Section 2.2, Section 5), GLM-5 (S3, S2, G1).
**Verdict:** Strong consensus. **Incorporate.**

The plan sets the win threshold at "rL2 ≤ 0.0005" without derivation.
Both reviewers flag this. The correct procedure:

1. Run `rmsnorm_f32_f64acc` (already exists in `kernels/src/rmsnorm.hip`
   from the closed investigation's probe c.3) with `HIPFIRE_RMSNORM_F64=1`.
   Measure stage-1 rL2.
   - If ≤0.0003: reduction precision IS the dominant source. H2/H5 are
     the targets; skip Day 2 hypotheses on other axes.
   - If ~0.0017 (no change): reduction precision is NOT the source.
     H2/H5 are falsified; focus Day 1-3 on H1/H7/H8/H9.

2. Run a CPU fp64 reference rmsnorm on the same bit-exact input.
   Measure (HF fp32 vs fp64) rL2 and (hipfire fp32 vs fp64) rL2.
   - The fp64-vs-fp64 delta is the irreducible fp32-arithmetic floor.
   - Set the Day 5 success threshold to **≤ 2× (HF fp32 - fp64) rL2**,
     not the arbitrary 0.0005.

**Time:** 30 minutes for f64acc probe, ~1 hour for fp64 reference. Done
before any kernel rewrite work begins.

### CR2. H1 (operand grouping) needs 3 variants AND targets the fused production kernel

**Sources:** Gemini (Section 2.1), GLM-5 (C1).
**Verdict:** Both reviewers flag the same issue from different angles.
**Incorporate fully.**

Gemini observes that there are **three** distinct operand groupings,
not two:
- `(x * rms) * weight` — torch-equivalent
- `x * (weight * rms)` — single-rounding shortcut
- `x * weight * rms` — left-to-right, compiler-dependent

GLM-5 observes the production LA path for MQ4 weights goes through
`fused_rmsnorm_mq_rotate.hip:79`, not the plain `rmsnorm.hip:24`. Both
kernels currently use `(x * weight) * rms`. **The Day-1 H1 test must
target the fused kernel as the primary path**, with the plain kernel
as a secondary confirmatory.

Combined: rewrite H1 to test 3 groupings × 2 kernel paths (plain +
fused) = 6 variants. Most are 5-line kernel changes; combined cost
~half a day.

### CR3. Stage-1 measurement conflates rmsnorm-only with rmsnorm+rotation

**Sources:** GLM-5 (C2).
**Verdict:** Unique to GLM-5, but factually correct. **Incorporate.**

The 0.0017 rL2 reading comes from dumping `s.tmp` (post-rmsnorm,
pre-FWHT-rotation). For Q3.5 MQ4 weights, the production path then
applies FWHT rotation to produce `s.x_rot` which feeds the projection
gemv. The combined rmsnorm+rotation drift may differ from 0.0017.

For q8f16 (no FWHT), `s.tmp` is the final output that feeds the next
kernel. So the 0.0017 measurement IS production-relevant for q8f16 —
but NOT for MQ4 paths.

**Fix:** specify two baselines:
- (a) rmsnorm-only via plain kernel (q8f16 production path)
- (b) rmsnorm+rotation combined via fused kernel (MQ4 production path)

Any fix needs to land BOTH. Day 1 baseline measurement collects both.

### CR4. Add FMA contraction hypothesis (H7)

**Sources:** Gemini (Section 3, H7).
**Verdict:** Unique to Gemini. **Incorporate as priority test.**

Hipfire's `sum_sq += v * v` is almost certainly compiled to
`v_fma_f32` (one rounding step). PyTorch's `x.pow(2).mean()` may be
two operations (one MUL, one separate accumulate ADD) with two
rounding steps. For a 1024-element reduction, the systematic divergence
from one missing rounding per step is non-trivial.

**Test:**
- Compile rmsnorm with `-ffp-contract=off` (HIP compiler flag), OR
- Use explicit `__fadd_rn(sum_sq, __fmul_rn(v, v))` to force separate
  rounded MUL + ADD
- Measure stage-1 rL2

This is high-priority because it would manifest uniformly across every
reduction kernel (gemv, attention softmax sum, conv1d) — exactly the
"shared root cause" signature the plan is hunting.

### CR5. Demote H3 (FP16 intermediates) and H5 (wave-32/64) for rmsnorm specifically

**Sources:** GLM-5 (M1, M2).
**Verdict:** Both correct by source inspection. **Demote both.**

- **H3:** rmsnorm.hip uses `float` throughout (verified by reading 26
  lines). Already falsified for rmsnorm without testing. Keep H3 in
  scope for *other* kernels (attention softmax, conv1d) where FP16
  intermediates may exist.
- **H5:** rmsnorm uses shared-memory tree reduce (`sdata[t] += sdata[t+s]`)
  with no warp-size-specific intrinsics. Wave32 vs wave64 doesn't apply.
  Keep H5 in scope for kernels that DO use `__shfl_xor_sync` with
  hardcoded width (gemv kernels, attention).

Removing H3 and H5 from rmsnorm tests frees ~1 day of Day 2 work to
spend on H7 (FMA contraction) instead.

### CR6. H2's framing was wrong — fix to "tree shape A vs B"

**Sources:** GLM-5 (C3).
**Verdict:** Unique to GLM-5, factually correct. **Incorporate.**

The plan claimed HF uses "sequential" accumulation. HF on CUDA/ROCm
uses parallel reduction too (warp-shuffle or shared-memory). The
correct comparison is "different tree shapes" — hipfire's shared-memory
tree vs llama.cpp's warp-shuffle pattern vs PyTorch CUDA's `cub::Reduce`.

The H2 test methodology stays valid (test warp-shuffle variants); only
the motivation text needs correction.

### CR7. Add FTZ/denormal hypothesis (H8) and eps-placement hypothesis (H9)

**Sources:** Gemini (Section 3, H8 + H9).
**Verdict:** Both unique to Gemini. **Incorporate both.**

- **H8 (FTZ):** if hipfire compiles with `-fno-denormal-fp-math` and
  PyTorch CPU doesn't flush denormals, small-magnitude `v*v` products
  flush to zero in hipfire but not in HF. The variance becomes
  systematically smaller in hipfire. Test by checking the compiler
  flags in `kernels/src/build.rs` or by reading clang flags used for
  hipcc compilation.

- **H9 (eps placement):** the formulae must match: `rsqrt(variance + eps)`
  vs `rsqrt(variance) + eps_correction`. Verify the order in both
  hipfire (`rsqrtf(reduce[0] / K + eps)`) and HF
  (`torch.rsqrt(variance + eps)`). Hipfire already matches HF for the
  plain kernel; verify the fused variant too.

Both are ~5-minute tests. Add to Day 1.

### CR8. Add a control experiment: run on FP32 weights

**Sources:** Gemini (Section 4.2, "Deadly coincidence").
**Verdict:** Unique to Gemini. **Incorporate as Day-1 sanity check.**

The plan assumes the uniform ~0.005 rL2 across kernels is a shared
kernel-implementation pattern. Gemini's adversarial alternative: it's
the **quantization SNR floor**. Any kernel reducing a dot product of
Q8 weights × F32 activations will hit a similar SNR limit regardless
of implementation.

**Test:** at Day 0, run rmsnorm against the same input with FP32
weights (no quantization). If 0.0017 rL2 persists, it's a kernel
implementation issue. If it disappears, the "uniform 0.005" was
quantization SNR, and the entire pattern-hunt premise is undercut.

This is a powerful falsifier and should run before the hypothesis
exploration begins. If it falsifies the premise, save 5 days.

---

## Significant revisions (must address by Day 4)

### SR1. FA non-norm kernels not in hypothesis space

**Sources:** GLM-5 (S1).
**Verdict:** Important scope addition. **Incorporate.**

The plan's Day-4 generalization tests only norm kernels (q_norm,
k_norm, l2_norm, gated_norm). FA-specific kernels (`weight_gemv_residual`,
`attention_q8_0_kv`, `sigmoid_mul_f32`) have ~0.005 rL2 drift too. If
a shared root cause exists, it must explain those too.

**Fix:** Day 4 generalization tests one non-norm kernel
(`weight_gemv_residual` is cheapest to instrument — pure gemv with
in-place add) in addition to the norm cluster. If the pattern doesn't
apply to gemv-residual, outcome A's "4-6 weeks" estimate is incomplete.

### SR2. Cover the AWQ-variant fused kernel

**Sources:** GLM-5 (M4).
**Verdict:** Practical correctness concern. **Incorporate.**

When AWQ calibration is active, the LA path uses
`fused_rmsnorm_mq_rotate_awq.hip`, which adds an `awq_scale` divide
before the FWHT rotation. Any rmsnorm fix that targets only
`fused_rmsnorm_mq_rotate.hip` won't apply to AWQ-calibrated models.

**Fix:** when writing the rmsnorm_v2 variants, mirror them in the AWQ
variant too. Tests on Day 1 cover both fused paths.

### SR3. Generalization test on Day 4 is under-specified

**Sources:** GLM-5 (G2).
**Verdict:** Mechanically correct. **Incorporate.**

- `q_norm` / `k_norm` use `rmsnorm_batched` (different thread/block
  geometry), not single-vector `rmsnorm_f32`.
- `l2_norm` in `fused_qk_l2_norm_scale_f32` is L2 normalization
  (`x / sqrt(sum(x²))`), not RMS (no `weight`, different
  output formula).
- `gated_norm_f32` is yet another kernel.

The "same fix" may need per-kernel adaptation. Budget per-kernel time
in Day 4, don't assume mechanical copy-paste.

---

## Minor refinements

### MR1. Build / compile time budget

**Sources:** GLM-5 (M3).
**Verdict:** Practical concern. **Note in plan.**

Each kernel variant requires rebuild. The "5-day" estimate is tight;
either coexist variants as separate kernel files (no rebuild between
A/B) or extend timeline to 6-7 days.

### MR2. Isolated rmsnorm probe — better than full-forward measurement

**Sources:** Gemini (Section 4.1).
**Verdict:** Already in original plan as
`crates/hipfire-runtime/examples/rmsnorm_isolated.rs`. **Promote to
required Day 1 prerequisite.**

Don't try to measure rmsnorm via full-model forward (too noisy). Build
an isolated probe that loads HF stage-0 (input) from the existing
dump, runs ONLY rmsnorm, dumps output. Compare against HF stage-1.
This gives a noise-free per-variant rL2 measurement.

---

## Revised hypothesis ranking (after this review)

| rank | hypothesis | falsification cost | priority |
|---:|---|---|---|
| 1 | **f64acc reduction** (existing kernel) | 30 min | Day 0 |
| 2 | **FP32 weights control** (quantization SNR vs kernel) | 1 hour | Day 0 |
| 3 | **fp64 reference floor** (derive real threshold) | 1 hour | Day 0 |
| 4 | **H1 (operand grouping, 3 variants × 2 kernels)** | 4 hours | Day 1 |
| 5 | **H7 (FMA contraction, `-ffp-contract=off`)** | 1 hour | Day 1 |
| 6 | **H8 (FTZ / denormals)** | 30 min | Day 1 |
| 7 | **H9 (eps placement, source read)** | 15 min | Day 1 |
| 8 | **H4 (rsqrtf vs Newton refinement)** | 1 hour | Day 1 |
| 9 | **H2 (tree reduction shape, warp-shuffle variants)** | 1 day | Day 2 (only if Day 0 f64acc confirms reduction matters) |
| 10 | **H6 (direct llama.cpp port)** | 1 day | Day 3 |
| — | H3 (FP16 intermediates) | falsified by inspection | dropped for rmsnorm |
| — | H5 (wave32/64) | falsified by inspection | dropped for rmsnorm |

**Revised timeline: 5 days remains plausible** if Day 0 + Day 1's
quick tests resolve some hypotheses quickly. Worst case extends to 6-7
days. Either way, the cost/value tradeoff stands.

---

## Rejected proposals (none full-rejects, but qualified scope)

- **None rejected outright.** Every reviewer point has defensible
  motivation. Two are demotions (H3, H5 for rmsnorm specifically) but
  remain in scope for other kernels.

---

## Summary table (every reviewer point, with verdict)

| # | source | proposal | verdict |
|---|---|---|---|
| 1 | Both | Day 0 f64acc + fp64 reference + threshold derivation | **incorporate (CR1)** |
| 2 | Both | H1: 3 groupings × 2 kernel paths | **incorporate (CR2)** |
| 3 | GLM-5 C2 | rmsnorm-only vs rmsnorm+rotation baselines | **incorporate (CR3)** |
| 4 | Gemini H7 | FMA contraction | **incorporate (CR4)** |
| 5 | GLM-5 M1/M2 | Demote H3 and H5 for rmsnorm | **incorporate (CR5)** |
| 6 | GLM-5 C3 | Fix H2 framing (tree A vs B, not sequential vs tree) | **incorporate (CR6)** |
| 7 | Gemini H8/H9 | FTZ + eps placement | **incorporate (CR7)** |
| 8 | Gemini Sec 4.2 | FP32-weight control experiment | **incorporate (CR8)** |
| 9 | GLM-5 S1 | FA non-norm kernels in Day 4 generalization | **incorporate (SR1)** |
| 10 | GLM-5 M4 | AWQ-variant kernel coverage | **incorporate (SR2)** |
| 11 | GLM-5 G2 | Per-kernel adaptation budget on Day 4 | **incorporate (SR3)** |
| 12 | GLM-5 M3 | Build time budget / coexist variants | **note in plan (MR1)** |
| 13 | Gemini Sec 4.1 | Isolated rmsnorm probe | **promote to Day 1 prereq (MR2)** |
| 14 | Gemini Sec 2.1 | All three operand groupings (not two) | **rolled into CR2** |
| 15 | GLM-5 G1 | Control: HF rmsnorm vs fp64 floor | **rolled into CR1** |
| 16 | GLM-5 S3 | Existing f64acc kernel | **rolled into CR1** |

**Plan revisions to apply to `04-pattern-hunt-plan.md`:**

1. Add Day 0 (f64acc probe + fp64 reference + FP32-weight control)
2. Rewrite H1 to enumerate 3 groupings × 2 kernel paths
3. Specify dual baselines (rmsnorm-only + rmsnorm+rotation)
4. Add H7 (FMA contraction) as priority test
5. Add H8 (FTZ) and H9 (eps placement)
6. Demote H3 and H5 with explanations
7. Correct H2's framing
8. Expand Day 4 to include `weight_gemv_residual` + adaptation budget
9. Note AWQ variant coverage requirement
10. Note isolated rmsnorm probe is required, not optional
11. Note build-time budget (coexist variants)
12. Revise threshold to "≤ 2× HF-fp32-vs-fp64 floor" instead of 0.0005
