# Pattern-hunt plan — re-opening engine surgery as a shared-root-cause question (2026-05-13, rev 2)

Sequel to `03-final-verdict.md`. The final verdict closed the audit with
"distributed drift across ~24 kernels, no surgery target". This was the
correct conclusion under the framing the audit used — *"find a single
dominant kernel"* — but missed a third possibility:

**A single root cause expressed in many kernels.**

If every hipfire kernel makes the same implementation choice (e.g., a
particular reduction order, an FMA grouping, a denormal flush behavior)
that differs from HF/llama.cpp's choice, every kernel will diverge by
similar magnitude — exactly what we measured. The data clusters tightly
(most kernels contribute ~0.005 rL2 per stage); this is the signature
of common cause, not 24 independent bugs.

This doc plans a 5-day exploration. Either outcome closes a major
decision unknown:

- **Finds the common pattern**, generalizes across ~24 kernels, and
  collapses the original "4-8 months graph-level rewrite" estimate to
  ~4-6 weeks of mostly mechanical porting. OR
- **Conclusively falsifies the shared-root-cause hypothesis**,
  confirming the 4-8 month estimate stands, and engine surgery remains
  deferred behind the calibration roadmap.

**Plan revisions from combined adversarial review** (see
`04-pattern-hunt_rev_combined.md`): added Day 0 prerequisites,
incorporated 3 new hypotheses (H7/H8/H9), demoted 2 (H3/H5 for
rmsnorm), corrected H2 framing, expanded Day 4 to include
non-norm kernels.

## Why rmsnorm is the right starting point

The audit data shows every non-trivial kernel contributes per-stage rL2
in a tight ~0.005 cluster:

| stage | kernel | rL2 with bit-exact input | path |
|---|---|---:|---|
| 1 | `rmsnorm_f32` (LA-0 input layernorm) | 0.0017 | q8f16 path |
| 1 | `fused_rmsnorm_mq_rotate` Phase 1c | (TBD Day 0) | MQ4 path (production) |
| 2 | `gemv_q8_0` (in_proj_qkv) | 0.0037 | both |
| 8/9 | `conv1d_silu_split_f32` | 0.0048/0.0055 (F16 weight) | both |
| 11/12 | `fused_qk_l2_norm_scale_f32` | 0.0050/0.0059 (F16 weight) | both |

**Rmsnorm is the cleanest target** because:

- Single input vector (no GQA, no quantization, simplest formula)
- Math is well-defined: `out = x * weight * rsqrt(mean(x²) + eps)`
- Stage 1 measurement at LA-0 with bit-exact input gives 0.0017 rL2
- HF reference is `Qwen3_5RMSNorm.forward` — 4 lines of Python, readable
- llama.cpp reference is `ggml-cuda/norm.cu` — short and self-contained

**Two production paths for rmsnorm** (revision from combined review):

- **q8f16 / plain path**: input → `rmsnorm_f32` → `s.tmp` → next kernel.
  The 0.0017 measurement is this path.
- **MQ4 / fused path**: input → `fused_rmsnorm_mq_rotate` (rmsnorm +
  FWHT rotation in one kernel) → `s.x_rot` → next kernel.

Any rmsnorm fix must address BOTH paths. Day 1 measurements collect
baselines for both.

## Day 0 — prerequisites (1-2 hours, before any kernel rewrite)

Three quick falsifications and a derived threshold. Each can rule out
a hypothesis or set the budget for subsequent days.

### D0.1 — Run existing `rmsnorm_f32_f64acc` probe (30 min)

This kernel already exists in `kernels/src/rmsnorm.hip:33-56` from the
closed investigation's probe c.3. It computes sum-of-squares and rms
in fp64, casts back to fp32 for output. Triggered by env-gate
`HIPFIRE_RMSNORM_F64=1`.

**Test:** measure stage-1 rL2 with `HIPFIRE_RMSNORM_F64=1` on
Q3.5-0.8B q8f16 LA-0 chunk 0.

**Outcomes:**
- If stage-1 rL2 drops 0.0017 → ≤0.0003: **reduction precision is the
  dominant source**. H2/H5 are the targets. Skip H1/H4/H7/H8/H9 hypotheses
  about operation order — those are in the noise.
- If stage-1 rL2 stays ~0.0017: **reduction precision is NOT the source**.
  H2/H5 are falsified. Focus Day 1-3 on H1/H7/H8/H9 (operation-order
  hypotheses).

### D0.2 — FP32-weight control experiment (1 hour)

Adversarial premise: maybe the uniform ~0.005 rL2 isn't a kernel bug
at all but the **quantization SNR floor** of Q8 weights × F32
activations. Any kernel doing a reduction over Q8-dequantized weights
would hit this limit regardless of kernel implementation.

**Test:** rmsnorm operates on a `[hidden_dim]` vector with no
quantization (the rmsnorm weight is F16 already, not Q8). So the
direct test would target a different kernel like `in_proj_qkv` gemv.
For Day 0 we run a simpler version: re-quantize with FP32 weights
storage on rmsnorm's `attn_norm` tensor and measure stage 1.

If stage-1 rL2 disappears (drops to ~fp64 noise floor): rmsnorm drift
was driven by F16-vs-BF16 storage of the norm weight, not kernel
implementation. Pattern hunt is unnecessary; just upgrade norm-weight
storage to BF16.

If stage-1 rL2 persists: rmsnorm drift is in the kernel math, not the
weight storage. Pattern hunt proceeds.

### D0.3 — Derive the real win threshold (1 hour)

Run a CPU fp64 reference rmsnorm on the same bit-exact input as HF's
forward used. Specifically:

1. Read the chunk-0 stage-0 (input residual) from
   `/data/cache/hipfire/audit-2026-05-13/hf_layer0_chunk0.bin`
2. Compute rmsnorm in fp64 with the exact HF formula
3. Compare:
   - `(HF stage 1 fp32) - (fp64 reference)` rL2 → HF's own fp32 noise floor
   - `(hipfire stage 1 fp32) - (fp64 reference)` rL2 → hipfire's full drift

The first number is the irreducible fp32-arithmetic floor for ANY
implementation. The plan's success threshold becomes:

**Day 5 success target: hipfire's rL2 vs fp64 reference ≤ 2× HF's rL2 vs fp64 reference.**

This replaces the arbitrary "≤0.0005". If HF-vs-fp64 is 0.0002, target
is 0.0004. If HF-vs-fp64 is 0.0008, target is 0.0016 (and ≤0.0005 was
unachievable from the start).

### D0.4 — Build the isolated rmsnorm probe (2 hours)

Write `crates/hipfire-runtime/examples/rmsnorm_isolated.rs`. Inputs:
HF stage-0 (input residual from the existing dump) + the norm weight
tensor from the .hfq file. Output: a HIPFIRE_DUMP_LA_STAGES-format
file with stage 1 only.

Why isolated rather than full-model dump:
- One kernel under test, no upstream noise
- Each hypothesis variant tests in seconds, not minutes
- Direct A/B comparison without full-forward overhead

This is the workhorse for Day 1-3.

---

## Day 1 — operand-grouping + low-cost falsifications (~1 day)

Run multiple hypothesis tests in parallel, all quick. Each is at most
a few lines of kernel code; together they test 4 hypotheses and
generate per-variant rL2 measurements via the isolated probe.

### H1 — Operand grouping (rewritten per CR2)

**Three groupings to test**, each on BOTH the plain kernel
(`rmsnorm.hip:24`) AND the fused kernel (`fused_rmsnorm_mq_rotate.hip:79`):

| variant | expression | provenance |
|---|---|---|
| H1a | `(x[i] * rms) * weight[i]` | torch reference order |
| H1b | `x[i] * (weight[i] * rms)` | minimum-rounding shortcut |
| H1c | `x[i] * weight[i] * rms` | left-to-right (current hipfire) |

3 variants × 2 paths = 6 builds. Each is a 5-line change. The fused
kernel's modification is more invasive due to its 153-line structure,
but the change is local to the line that does the multiplication.

**Coverage requirement** (per SR2): mirror variants in
`fused_rmsnorm_mq_rotate_awq.hip` so AWQ-calibrated models are also
covered. Same change pattern, different file.

### H7 — FMA contraction (added per CR4)

Hipfire's `sum_sq += v * v` is almost certainly compiled to
`v_fma_f32` (one rounding step). PyTorch's `x.pow(2).mean()` may be
two operations (one MUL, one ADD) with two rounding steps. For a
1024-element reduction, missing one rounding per step is non-trivial.

**Tests:**
- **H7a:** Add `#pragma clang fp contract(off)` around the reduction
  loop. Compile, measure rL2.
- **H7b:** Use explicit `__fadd_rn(sum_sq, __fmul_rn(v, v))` — forces
  two separately-rounded operations.
- **H7c:** Build with `-ffp-contract=off` HIP compiler flag globally.

Why this is high-priority: FMA contraction would manifest uniformly
across every reduction kernel (gemv, attention softmax sum, conv1d,
norms). This is the strongest single candidate for the shared root
cause.

### H8 — FTZ / denormal handling (added per CR4)

Many GPU kernels are compiled with flush-to-zero on denormals (FTZ).
If hipfire flushes small `v*v` products to zero but HF (especially on
CPU eager) does not, the variance accumulates differently for inputs
with very small magnitude entries.

**Tests:**
- Check the build flags for hipcc in `kernels/src/build.rs` /
  `crates/rdna-compute/Cargo.toml`.
- If FTZ is on, rebuild with `-fno-denormal-fp-math` and measure.
- Conversely, force FTZ in CPU reference and measure HF's drift change.

Probably <30 min of investigation, mostly reading build configuration.

### H9 — Eps placement (added per CR4)

Verify the formula matches exactly between hipfire and HF:

**Hipfire (`rmsnorm.hip:21`):**
```c
float rms = rsqrtf(sdata[0] / (float)n + eps);
```

**HF (`Qwen3_5RMSNorm.forward`):**
```python
variance = x.pow(2).mean(-1, keepdim=True)
x = x * torch.rsqrt(variance + eps)
```

Both apply eps to mean-of-squares before rsqrt. Match. **H9 is already
verified for the plain kernel.** Check the fused kernel too — should
also match. ~5 min source read.

**Verdict on H9:** falsifiable by source inspection if it matches.
Record as confirmed and move on.

### H4 — rsqrtf precision (kept, lower priority)

`rsqrtf` is the hardware `v_rsq_f32` intrinsic with ~1 ULP precision.
Some BLAS reductions use Newton-Raphson refinement to get to ~0.5 ULP.

**Test:** replace `rsqrtf(x)` with `1.0f / sqrtf(x)`. Measure.

Lower priority because: rsqrt is used in norms but NOT in projection
gemv or conv. If rsqrt were the dominant pattern, norms would have
worse drift than gemv. Data shows the opposite. Probably not it, but
cheap to test.

---

## Day 2 — reduction tree shape (only if Day 0 confirms reduction matters)

Conditional on D0.1 outcome. **If `rmsnorm_f32_f64acc` significantly
reduces stage-1 rL2, Day 2 proceeds; otherwise skip and use this day
for additional Day 1 variants.**

### H2 — Reduction tree shape (corrected per CR6)

**NOT** sequential vs tree (the original framing was wrong — HF on
CUDA/ROCm uses parallel reduction too, just a different tree shape).
The actual comparison is **hipfire's shared-memory tree vs other
parallel-reduction trees**.

Hipfire's current pattern (`rmsnorm.hip:17-19`):
```c
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

Variants to test:

| variant | tree shape |
|---|---|
| H2a | Current shared-memory binary tree (baseline) |
| H2b | Warp-shuffle butterfly: `__shfl_xor_sync` with xor 1,2,4,8,16 (wave32) |
| H2c | Warp-shuffle linear: `__shfl_down_sync` with 16,8,4,2,1 |
| H2d | Hybrid: warp-shuffle within warp + shared-memory across warps |
| H2e | Direct llama.cpp port (matches their `ggml-cuda/norm.cu` exactly) |

Each variant is a self-contained kernel file. Coexist for A/B testing
(no rebuild penalty per MR1).

**H5 (wave32 vs wave64) was demoted** (per CR5): rmsnorm uses no
warp-size-specific intrinsics, so wave-size doesn't apply. H5 stays in
scope for kernels that DO use `__shfl_xor_sync` with hardcoded
width — gemv kernels, attention.

**H3 (FP16 intermediates) was demoted** (per CR5): rmsnorm.hip uses
`float` throughout (verified by source inspection). H3 stays in scope
for attention softmax and conv1d kernels where FP16 accumulators are
possible.

---

## Day 3 — H6 (direct llama.cpp port) or fallback

If Days 0-2 haven't produced a variant that hits the threshold (D0.3),
port llama.cpp's rmsnorm kernel one-for-one to HIP. Keep their exact
thread/block layout, reduction order, and operand grouping. Build,
run, measure.

If llama.cpp's port hits ≤ target rL2, we know the *exact* combination
of choices that works. Then enumerate the differences from hipfire's
current implementation — that diff is the fix recipe.

If llama.cpp's port ALSO hits ~0.0017: rmsnorm drift is not in any
implementation choice we can find. Either the fp64 reference reveals
the irreducible floor IS ~0.0017 (rarely the case at the math level)
or we missed a hypothesis. Time for a fresh look at HF's exact
reduction primitive.

---

## Day 4 — generalization (expanded per SR1, SR3)

If Days 0-3 found a fix for rmsnorm, test whether it generalizes.

### Generalization to norm cluster (per SR3 — budget per-kernel)

| target kernel | structure | adaptation needed |
|---|---|---|
| `rmsnorm_batched` (q_norm/k_norm in FA) | batched across heads | adjust thread/block geometry |
| `fused_qk_l2_norm_scale_f32` | L2 norm formula `x / sqrt(sum(x²) / n)` | no `weight` term, slightly different output |
| `gated_norm_f32` | rmsnorm+gate fused | adapt to fused form |
| `fused_rmsnorm_mq_rotate.hip` | rmsnorm + FWHT (production) | apply fix; verify FWHT path unchanged |
| `fused_rmsnorm_mq_rotate_awq.hip` | rmsnorm + AWQ divide + FWHT | same fix + AWQ path |

Each is ~half a day of adaptation + measurement. The "same fix" applied
literally won't always compile — each kernel needs targeted edits in
the equivalent operand-grouping / reduction / FMA-contraction spot.

### Generalization to non-norm kernels (added per SR1)

Test one non-norm kernel to verify the pattern isn't norm-specific.
Recommended: **`weight_gemv_residual`** — pure gemv with in-place
residual add. It's the FA-block's wo + residual fusion. If the same
pattern reduces its drift (currently part of stages 12→13 in FA), the
pattern is genuinely a shared root cause across all reduction
operations.

If `weight_gemv_residual` does NOT respond to the same fix: the
pattern is norm-specific, and the engine-surgery estimate revises to
"norm-cluster rewrite ~2 weeks, projection/gemv cluster requires
separate investigation."

### Generalization outcome decision

| signal | meaning | action |
|---|---|---|
| Same fix drops all 5 norm kernels' rL2 below target + drops `weight_gemv_residual` too | **outcome A** | Schedule 4-6 week graph rewrite |
| Same fix drops all norms but not gemv-residual | norm-cluster only | Ship norm fix (~2 weeks). Engine surgery for non-norm cluster needs separate plan |
| Same fix drops rmsnorm only | **outcome B** | Ship rmsnorm fix as a side project. Defer broader surgery |
| No variant on Day 3 hit the threshold | **outcome C** | Confirms graph-level scope; defer surgery |

---

## Day 5 — decision review (no work, just calibration of next step)

Outcome A: write a formal "Phase 4 — graph rewrite" proposal doc with
the discovered pattern, the list of all kernels needing the fix, the
4-6 week timeline including AWQ re-calibration, and a kickoff plan.

Outcome B: ship the rmsnorm fix as a standalone PR (test, coherence-gate,
land). Quality eval before+after.

Outcome C: write `05-pattern-hunt-results.md` documenting which
hypotheses were falsified. Audit closes definitively.

---

## Decision criteria (revised per CR1, S2)

| outcome | criterion (vs fp64 reference) | generalization | action |
|---|---|---|---|
| A — pattern + generalizes | hipfire rL2 vs fp64 ≤ 2× (HF fp32 vs fp64) on rmsnorm | ≥3 other norm kernels + 1 non-norm kernel drop similarly | GO — 4-6 week graph rewrite |
| B — pattern, narrow | rmsnorm hits target | only norm kernels | Ship norm fix; defer rest |
| C — no significant rmsnorm reduction | rmsnorm doesn't hit target | n/a | NO-GO — confirms graph-level scope |

---

## What we learn from "no pattern" (outcome C)

A negative result is also valuable. It would mean:

- Hipfire's per-kernel implementations DON'T share a fixable pattern
- Each kernel's drift is independent — driven by per-kernel reduction
  shape, per-kernel data layout, per-kernel accumulator choice
- The ~0.005 rL2 uniformity is a coincidence of fp32 arithmetic with
  similar-sized inputs, not a shared cause
- Closing the floor requires 24 independent kernel rewrites — the
  4-8 month estimate stands

This conclusion is *useful*: it tells us calibration is genuinely the
only short-horizon answer, and the calibration roadmap is the right
priority. 5 days to settle the question.

---

## Sequencing with calibration work

- **Independent of calibration.** rmsnorm rewrite doesn't conflict
  with AWQ/GPTQ work on gfx1100. Both can proceed in parallel.
- **Per-arch validation**: if the pattern works on gfx1151, verify it
  also helps gfx1100 before landing as default.
- **Build-time budget** (per MR1): coexist variants as separate kernel
  files so A/B comparisons don't require rebuilding. Estimated total
  compile time across the 5 days: ~2 hours wasted-time absorbed into
  the existing schedule.

---

## Hypothesis ranking (final, revised)

By falsification cost, cheapest-first:

| rank | hypothesis | cost | day |
|---:|---|---|---:|
| 1 | f64acc reduction (existing kernel) | 30 min | 0 |
| 2 | FP32 norm-weight control | 1 hour | 0 |
| 3 | fp64 reference floor (target derivation) | 1 hour | 0 |
| 4 | Isolated rmsnorm probe (workhorse) | 2 hours | 0 |
| 5 | H9 eps placement (source inspection) | 15 min | 1 |
| 6 | H8 FTZ / denormals | 30 min | 1 |
| 7 | H4 rsqrtf precision | 1 hour | 1 |
| 8 | H7 FMA contraction | 1 hour | 1 |
| 9 | H1 operand grouping (3 variants × 2 paths) | 4 hours | 1 |
| 10 | H2 reduction tree (5 variants) | 1 day | 2 (conditional) |
| 11 | H6 direct llama.cpp port | 1 day | 3 |
| — | H3 FP16 intermediates | falsified by inspection | dropped for rmsnorm |
| — | H5 wave32 vs wave64 | falsified by inspection | dropped for rmsnorm |

---

## Files to create

- `crates/hipfire-runtime/examples/rmsnorm_isolated.rs` — isolated
  rmsnorm probe (required, Day 0)
- `kernels/src/rmsnorm_v2_h1a.hip`, `_h1b.hip`, `_h2b.hip`, etc. —
  per-variant kernels coexisting with the baseline
- `kernels/src/rmsnorm_v2_llama.hip` — direct llama.cpp port (Day 3)
- `crates/rdna-compute/src/dispatch.rs` — env-gated variant selection
- `docs/investigations/2026-05-13-engine-drift-residual-audit/05-pattern-hunt-results.md`

## Exit conditions

- Day 5 decision review fires regardless of outcome
- Outcome A or B at Day 5 → Phase 4 results doc + next step plan
- Outcome C at Day 5 → results doc documenting falsified hypotheses,
  audit closes definitively

## What this supersedes

`03-final-verdict.md` closed the audit pending re-investigation of the
shared-root-cause hypothesis. This plan (rev 2, post combined review)
opens that investigation. 03's verdict remains accurate for the
framing it used; 04 explores the alternative framing.
