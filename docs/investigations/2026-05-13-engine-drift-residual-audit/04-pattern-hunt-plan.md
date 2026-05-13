# Pattern-hunt plan — re-opening engine surgery as a shared-root-cause question (2026-05-13)

Sequel to `03-final-verdict.md`. The final verdict closed the audit with
"distributed drift across ~24 kernels, no surgery target". This was the
correct conclusion under the framing the audit used — *"find a single
dominant kernel"* — but missed a third possibility:

**A single root cause expressed in many kernels.**

If every hipfire kernel makes the same implementation choice (e.g., a
particular reduction order, an FP16 intermediate, a particular FMA
grouping) that differs from HF/llama.cpp's choice, every kernel will
diverge by similar magnitude — exactly what we measured. The data
clusters tightly (most kernels contribute ~0.005 rL2 per stage); this
is the signature of common cause, not 24 independent bugs.

This doc plans a 5-day exploration that either:

- **Finds the common pattern on rmsnorm**, generalizes it across ~24
  kernels, and collapses the original "4-8 months graph-level rewrite"
  estimate to ~4-6 weeks of mostly mechanical porting. OR
- **Conclusively falsifies the shared-root-cause hypothesis**,
  confirming the 4-8 month estimate stands, and engine surgery remains
  deferred behind the calibration roadmap.

Either outcome closes a major decision unknown that the audit didn't
resolve.

## Why rmsnorm is the right starting point

The audit data shows every non-trivial kernel contributes per-stage rL2
in a tight ~0.005 cluster:

| stage | kernel | rL2 with bit-exact input |
|---|---|---:|
| 1 | `rmsnorm_f32` (LA-0 input layernorm) | 0.0017 |
| 2 | `gemv_q8_0` (in_proj_qkv) | 0.0037 |
| 8/9 | `conv1d_silu_split_f32` | 0.0048/0.0055 (F16 weight) |
| 11/12 | `fused_qk_l2_norm_scale_f32` | 0.0050/0.0059 (F16 weight) |

**Rmsnorm is the cleanest target** because:

- Single input vector (no GQA, no quantization, no fusion with FWHT)
- Output shape matches input — no reshaping noise
- Math is well-defined and short: `out = x * weight * rsqrt(mean(x²) + eps)`
- Stage 1 measurement at LA-0 with bit-exact input gives 0.0017 rL2 — a
  single number to drive against
- HF reference is `Qwen3_5RMSNorm.forward` — 4 lines of Python, readable
- llama.cpp reference is `ggml-cuda/norm.cu` — short and self-contained

If a common root cause exists, rmsnorm should expose it most clearly
because the math is least obscured by other concerns. Whatever fix gets
rmsnorm from 0.0017 → ≤0.0005 is then tested on q_norm, k_norm, l2norm
(other reduction-based kernels). If those also drop to ≤0.0005 with the
same fix, the pattern is real and we apply it across remaining kernels.

## Candidate root causes (priority ranked)

### H1 — Operation order in the output multiplication

**Hipfire** (`kernels/src/rmsnorm.hip:23`):
```c
out[idx] = x[idx] * weight[i] * rms;     // = (x * weight) * rms
```

**HF reference** (torch RMSNorm equivalent):
```python
x_normalized = x * torch.rsqrt(variance + eps)
return x_normalized * weight                # = (x * rms) * weight
```

Different operand grouping in the multiplication: hipfire does
`(x * weight) * rms`, HF does `(x * rms) * weight`. fp32 multiplication
isn't strictly associative; the per-element rounding differs depending
on which operands are paired first.

**Why this might be the dominant pattern:** every normalization kernel
(rmsnorm, q_norm, k_norm, gated_norm, ffn_norm) does
`x * weight * normfactor` in some order. If hipfire's convention is
`(x * weight) * factor` and HF/llama.cpp use `(x * factor) * weight`,
every normalization kernel inherits the same per-element rounding
divergence.

**Test:** rewrite to `out[idx] = (x[idx] * rms) * weight[i]`. Measure
stage-1 rL2.

### H2 — Reduction order in the variance sum

**Hipfire**: 8-pass tree reduce in `__shared__` memory with
`sdata[t] += sdata[t + s]` at each step. Each cell ends as a sum of 256
fp32 values in a specific binary-tree order.

**HF (eager mode)**: dispatches to CPU `aten::mean` or GPU BLAS sum;
both typically use sequential or different-tree accumulation.

**llama.cpp**: warp-shuffle reduction (`__shfl_xor_sync` butterfly)
followed by cross-warp shared-memory reduce. Different tree shape than
hipfire's pure shared-memory tree.

**Why this might be a pattern:** every kernel with a reduction
(variance for norms, dot product for gemv, softmax denominator for
attention) uses the same tree-reduce primitive. If the tree shape
differs from HF/llama.cpp's, every reduction-based kernel inherits the
same per-element fp32 accumulation drift.

**Test:** rewrite rmsnorm to use a warp-shuffle reduce
(`__shfl_xor_sync`) matching llama.cpp's pattern. Measure stage-1 rL2.

### H3 — FP16 intermediate accumulation somewhere

Some HIP kernels use `__half` (FP16) for intermediates to fit more data
in shared memory or registers. If any of rmsnorm's intermediates use
FP16 where HF uses F32, error compounds.

**Why this might be a pattern:** hipfire's gemv kernels for Q8 weights
do all internal math in F32 per the kernel source — should not be the
issue for rmsnorm. But `attention_q8_0_kv` and conv kernels might use
FP16 accumulators in some paths. If any normalization kernel has an
FP16 intermediate, error per element is ~3 ULP at FP16 = ~3e-4 relative.

**Test:** read all rmsnorm-adjacent intermediate types in the kernel
source. If F32 throughout, this hypothesis is ruled out for rmsnorm
specifically. (Still possible for conv/attention.)

### H4 — `rsqrtf` precision differs from PyTorch's `torch.rsqrt`

`rsqrtf` is an intrinsic with ~1 ULP precision on AMD GPUs.
`torch.rsqrt` calls the same underlying instruction OR uses a Newton
iteration to refine. If hipfire uses `rsqrtf` directly and PyTorch
uses `1.0 / sqrtf(x)` (or vice versa), per-element error is ~1-2 ULP.

**Why this might NOT be a dominant pattern:** rsqrtf is used in
rmsnorm, l2norm, scale operations — but not in projection gemv or
conv. So if rsqrtf were the cause, gemv and conv drift would be
*lower* than norm drift. Data shows opposite (norms are cleaner than
projections). H4 is *probably not* the dominant pattern but worth
testing.

**Test:** replace `rsqrtf(x)` with `1.0f / sqrtf(x)`. Measure stage-1 rL2.

### H5 — Wavefront-32 vs wavefront-64 reduction layout

gfx1151 is wave32 (32 threads per warp). gfx906 / CDNA is wave64.
Hipfire's reductions may have been originally designed for wave64 with
specific shared-memory layouts; on wave32 they still work but produce
different per-thread workload and reduction order.

**Why this might be a pattern:** every kernel that uses warp reductions
inherits wavesize-dependent ordering. If hipfire's kernels were tuned
on gfx906 (wave64) and never re-tuned for wave32, they'd all share the
same wave-mismatch pattern.

**Test:** explicit wave32-specific reduction in rmsnorm_v2 (use
`__shfl_xor_sync` with width=32). Measure stage-1 rL2.

### H6 — Reading and matching llama.cpp's rmsnorm

Llama.cpp's `ggml-cuda/norm.cu` implementation is short. The fastest
way to find the pattern may be to literally port llama.cpp's rmsnorm
to HIP and compare. If the port produces ≤0.0005 rL2, then llama.cpp's
specific implementation choices (whatever they are) are the answer —
and we just enumerate the differences.

**Test:** port llama.cpp's rmsnorm kernel to HIP one-for-one. Measure
stage-1 rL2.

## Methodology — 5-day exploration

### Day 1 — Setup + H1/H4 quick wins

- Build `rmsnorm_v2.hip` with hypothesis-toggleable env-gates
  (`HIPFIRE_RMSNORM_OP_ORDER`, `HIPFIRE_RMSNORM_RSQRT_NEWTON`, etc.)
- Add `dump_la_stage` call right after `fused_rmsnorm_mq_rotate` at LA-0
  to measure stage-1 rL2 in isolation
- Run baseline measurement: current rmsnorm rL2 = 0.0017
- Test H1 (operation order): build variant, measure
- Test H4 (rsqrtf alternatives): build variant, measure
- Both are 5-minute kernel changes. Quick falsification.

### Day 2 — H2/H5 (reduction patterns)

- Build `rmsnorm_v2_warpshfl.hip` using `__shfl_xor_sync` warp reduction
  + shared-memory cross-warp reduce
- Variant A: butterfly tree (xor 1, 2, 4, 8, 16 across 32-thread warp)
- Variant B: linear shfl_down (16, 8, 4, 2, 1)
- Variant C: hybrid matching llama.cpp's specific pattern
- Measure stage-1 rL2 for each

### Day 3 — H6 (direct llama.cpp port)

- Find llama.cpp's `ggml-cuda/norm.cu` (or `rms_norm_f32` variant)
- Translate one-for-one to HIP, keeping their exact thread/block layout,
  reduction order, and operand grouping
- Build, run, measure. If result ≤0.0005 rL2, mark a milestone.

### Day 4 — Generalization test

If any variant from Days 1-3 produces rL2 ≤0.0005 on rmsnorm:

- Apply the **same pattern** to:
  - `q_norm` / `k_norm` (per-head RMSNorm in FA)
  - `l2_norm` in `fused_qk_l2_norm_scale_f32`
  - `gated_norm_f32` (after recurrence in LA)
- Measure those stage rL2s. Target: each drops from ~0.005-0.012 to
  ≤0.0005 with the same fix.
- If all three norm kernels respond uniformly to the same fix, the
  pattern is real and we expect it to generalize across other reduction
  kernels (gemv, softmax) too.

### Day 5 — Decision review

Outcomes ranked:

- **(A) Pattern found and generalizes to all norms:** schedule a
  multi-week graph rewrite. Estimated 4-6 weeks total: 2-3 weeks
  applying pattern across remaining kernels (projection gemv, conv1d,
  attention softmax, sigmoid-mul), 1-2 weeks per-kernel validation,
  1 week AWQ re-calibration.

- **(B) Pattern found on rmsnorm but doesn't generalize:** rmsnorm fix
  ships as a 1-week side project (small per-layer gain, modest model
  KLD impact). Engine surgery scope returns to 4-8 month estimate;
  defer.

- **(C) No variant drops rmsnorm rL2 below 0.001:** strong evidence
  drift is in F32 accumulation precision itself, not in any single
  implementation choice. Confirms graph-level rewrite is the only
  path; defer.

## Decision criteria

| outcome | rmsnorm rL2 reduction | generalization | action |
|---|---|---|---|
| A — pattern + generalizes | 0.0017 → ≤0.0005 | ≥2 other norm kernels drop similarly | GO — graph rewrite, 4-6 weeks |
| B — pattern, no generalization | 0.0017 → ≤0.0005 | only rmsnorm drops | ship rmsnorm fix, defer rest |
| C — no significant reduction | 0.0017 → ≥0.001 | n/a | NO-GO — confirms graph-level scope |

## What we learn from "no pattern"

A negative result (outcome C) is also valuable. It would mean:

- Hipfire's per-kernel implementations DON'T share a fixable pattern
- Each kernel's drift is independent — driven by per-kernel reduction
  shape, per-kernel data layout, per-kernel accumulator choice
- The ~0.005 rL2 uniformity is a *coincidence* of fp32 arithmetic with
  similar-sized inputs, not a shared cause
- Closing the floor requires 24 independent kernel rewrites — the
  4-8 month estimate stands

This conclusion is *useful*: it tells us calibration is genuinely the
only short-horizon answer, and the calibration roadmap is the right
priority. We've spent 5 days to definitively settle the question.

## Why this is the right next step

The audit's blind spot was framing engine drift as "single dominant
kernel OR distributed = closed". A third path exists: **distributed
manifestation of single shared cause.** The data signature (tight
clustering of per-kernel drift at ~0.005 rL2) is exactly what shared
cause would produce.

The cost (5 days, isolated to kernel rewrites, no model retraining) is
small relative to the decision value (collapses or confirms the
graph-rewrite estimate from 4-8 months to 4-6 weeks).

Even if no pattern is found, the work produces a hardened rmsnorm
kernel with documented design choices — useful documentation for
future rewrites.

## Sequencing

- **Independent of calibration work.** rmsnorm rewrite doesn't conflict
  with AWQ/GPTQ calibration on gfx1100. Both can proceed in parallel.
- **Single env-gate**: each variant lives behind a HIPFIRE_RMSNORM_V2
  env-gate so we can A/B compare without touching the default path.
- **Per-arch validation**: if the pattern works on gfx1151, verify it
  also helps gfx1100 before landing as default (cross-arch consistency
  check).

## Exit conditions for the pattern hunt

- Day 5 decision review fires regardless of outcome
- If outcome A or B at Day 5, write Phase 4 results doc (`05-`)
  documenting which hypothesis won and what the cross-kernel propagation
  looks like
- If outcome C at Day 5, write Phase 4 results doc documenting which
  hypotheses were falsified — provides forward roadmap for any future
  re-exploration

## Files to create during exploration

- `kernels/src/rmsnorm_v2.hip` — env-gated variant matrix
- `crates/rdna-compute/src/dispatch.rs` — env-gated dispatch
- `crates/hipfire-runtime/examples/rmsnorm_isolated.rs` — isolated
  rmsnorm probe with bit-exact input + per-variant rL2 measurement
  against an in-tree fp64 reference
- `docs/investigations/2026-05-13-engine-drift-residual-audit/05-pattern-hunt-results.md`

## What this supersedes

`03-final-verdict.md` closed the audit pending re-investigation of the
shared-root-cause hypothesis. This plan opens that investigation.
03's verdict remains accurate for the framing it used (single dominant
kernel); 04 explores the alternative framing.
