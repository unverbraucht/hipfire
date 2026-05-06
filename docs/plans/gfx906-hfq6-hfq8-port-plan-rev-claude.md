# Adversarial Review: gfx906 HFQ6 + HFQ8 kernel-coverage analysis

**Reviewer:** Claude Opus 4.7 (1M ctx, self-review)
**Date:** 2026-05-06
**Reviewed:** `docs/plans/gfx906-hfq6-hfq8-port.md` (419 lines)
**Branch:** `feat/gfx906-hfq6-hfq8-analysis`

This is a self-review of the analysis I just wrote. The methodology
that paid off on the dot8 PRD review (verify every kernel-presence
claim against `dispatch.rs` and `kernels/src/`) caught **three
substantive errors** in my analysis, plus several softer issues
worth flagging.

---

## Verdict

**Reject the plan as written.** The §1 coverage table is wrong on
two cells, the §3.2 "HFQ8 dp4a is no useful lift" conclusion is
**factually wrong** (MQ8 ships exactly that lever and uses it), and
an entire MoE-indexed kernel surface is missing from the analysis.

The Phase A recommendations (mechanical wave64 ports for HFQ6 and
HFQ8) are still defensible. The dp4a / MMQ analysis needs material
revision before it should drive any work.

The PRD remains useful as a *starting point* for an HFQ6/HFQ8 plan,
but anyone implementing from this draft will have to re-derive
half the analysis when they discover the existing MQ8 path that
contradicts §3.2.

---

## Blocking errors (factual, must fix before plan drives any work)

### B1: §1 table incorrectly says HFQ6 has no fused gate_up / qkv / qkvza

The §1 coverage table claims HFQ6's "fused gate_up / qkv / qkvza"
column is **✗** (missing). This is wrong at the **GEMM** level —
`crates/rdna-compute/src/dispatch.rs` has all three fused HFQ6
batched-GEMM dispatch fns, plus their fp16 / dot2 / WMMA variants:

```
8052:   pub fn gemm_qkvza_hfq6g256(...)          (wave32 FP)
8141:   pub fn gemm_qkvza_hfq6g256_fp16(...)
8221:   pub fn gemm_qkvza_hfq6g256_dot2(...)
8286:   pub fn gemm_qkvza_hfq6g256_wmma(...)
8368:   pub fn gemm_qkvza_hfq6g256_wmma_gfx12(...)
8453:   pub fn gemm_qkv_hfq6g256(...)
8535:   pub fn gemm_qkv_hfq6g256_fp16(...)
8608:   pub fn gemm_qkv_hfq6g256_dot2(...)
8667:   pub fn gemm_qkv_hfq6g256_wmma(...)
8742:   pub fn gemm_qkv_hfq6g256_wmma_gfx12(...)
8820:   pub fn gemm_gate_up_hfq6g256(...)
8895:   pub fn gemm_gate_up_hfq6g256_fp16(...)
8961:   pub fn gemm_gate_up_hfq6g256_dot2(...)
9014:   pub fn gemm_gate_up_hfq6g256_wmma(...)
9082:   pub fn gemm_gate_up_hfq6g256_wmma_gfx12(...)
```

What HFQ6 *doesn't* have is the **GEMV-level** fused single-token
variant (`fused_gate_up_hfq4g256_wave64.hip` etc. — the AR-decode
batch=1 path that PR #158 optimized).

**The plan conflates two distinct kernel surfaces:**
- **Batched fused GEMM** (gate_up + qkv + qkvza for prefill + DFlash
  verify, B>1) — HFQ6 has full coverage including dot2 + WMMA.
- **Single-token fused GEMV** (the same logical operation but B=1,
  used in AR decode) — HFQ6 has *none*.

The plan's "wave64 fused GEMVs (~1 session)" line item is real, but
it's adding a new GEMV-level surface, not filling a GEMM-level gap
that already exists.

**Recommended fix:** §1 table needs separate rows for "fused
batched GEMM" vs "fused single-token GEMV." Phase A scope needs
restating: "add the GEMV variant; the GEMM variant exists at FP."

### B2: §3.2 "dp4a-on-HFQ8 has no useful lift" is contradicted by shipped MQ8 code

The plan's §3.2.1 argues:

> Net dp4a lift on HFQ8: probably negative once you account for the
> Q8_1 quantize-x overhead.

This is **factually wrong**. The codebase ships MQ8 (FWHT-rotated
HFQ8) and **uses dp4a on int8 weights via `v_dot4_i32_iu8`**:

```
kernels/src/gemv_mq8g256.hip:
// MagnumQuant MQ8 GEMV: FWHT-rotated symmetric INT8 with dp4a.
// Inner loop uses v_dot4_i32_iu8 for 4x VALU throughput vs FP32.
```

Plus the dispatch:
```
crates/rdna-compute/src/dispatch.rs:2522: pub fn gemv_mq8g256_prerotated(...)
crates/rdna-compute/src/dispatch.rs:2545: pub fn gemv_mq8g256_with_rotate(...)
crates/rdna-compute/src/dispatch.rs:2498: rotate_quantize_x_mq8 (the int8 quantize for x)
```

The MQ8 kernel already does what my plan said wouldn't work: int8
weights × int8 activations × dp4a, with a per-group scale. The
kernel's own comment claims "4x VALU throughput vs FP32" — which is
the standard dp4a-vs-FP-FMA win and exactly the lever I dismissed.

The mistake in my analysis was assuming dp4a's win on HFQ4 came
*purely* from "no need to dequant from 4-bit to int8." It actually
comes from **integer dp4a being faster than FP-FMA on gfx906 even
when both consume the same byte count** — gfx906 has 4× the VALU
throughput on `v_dot4_i32_i8` vs FP-FMA per cycle (the same wiki
that documented dot8 documents this).

**Recommended fix:** §3.2 needs full rewrite. The HFQ8 dp4a
opportunity is real — the implementation pattern is right there in
`gemv_mq8g256.hip`. Rewrite §3.2 around "port the MQ8 dp4a pattern
to the un-rotated HFQ8 weights, plus add wave64 / residual / fused
variants." Estimated lift: probably +30-50% over the wave32 FP path
(matching MQ8's own measured-vs-FP comparison if any exists in the
dev logs).

### B3: Entire MoE-indexed surface is missing from the analysis

`kernels/src/` has 5 MoE-indexed kernel files for HFQ4:

```
gemv_hfq4g256_moe_down.hip
gemv_hfq4g256_moe_down_indexed.hip
gemv_hfq4g256_moe_down_indexed_batched.hip
gemv_hfq4g256_moe_down_indexed_batched_wave64.hip
gemv_hfq4g256_moe_down_indexed_wave64.hip
gemv_hfq4g256_moe_gate_up_indexed.hip (and 3 more)
```

These power Qwen 3.5 MoE / A3B inference. **Zero MoE-indexed
kernels exist for HFQ6 or HFQ8.** The plan says nothing about
this. Currently, anyone trying to run an A3B-class MoE model with
mq6 weights on gfx906 hits the same wave32-FP-fallback path the
predecessor PR #158 work was specifically built to escape.

**Recommended fix:** add a "MoE-indexed kernel coverage" section to
§3.1 (HFQ6) and §3.2 (HFQ8). For HFQ6, this is ~5 new kernels (the
HFQ4 family minus any non-MoE ones). For HFQ8, same plus the
quantize-x-to-int8 step. Cost: ~1 session per quant added to Phase A.

The user's question explicitly asked about "with and without
DFlash" — DFlash on MoE uses these MoE-indexed kernels, so this
gap is *exactly* the kind of gap the question is trying to surface.

---

## Moderate issues

### M1: §3.1.2 "HFQ6 dp4a +15-25%" estimate is too precise to be honest

The plan claims HFQ6 dp4a will deliver "+15-25% over FP" — a tighter
range than the underlying analysis supports. The actual derivation is
"dp4a wins, but the unpack overhead is larger than HFQ4's, so the win
is smaller." That argues for a *qualitative* claim ("smaller than
HFQ4's, possibly memory-bound regression") not a quantitative range.

The PR #158 dot8 PRD review (claude_rev_glm5.md, gfx906_dot8_rev_claude.md)
explicitly criticized v1's optimistic per-call lift estimates that
weren't backed by data. This plan repeats the pattern.

**Recommended fix:** §3.1.2 should state: "expected lift uncertain;
range from -10% (memory-bound regression from extra unpack arithmetic)
to +30% (best-case for kernels still ALU-headroom-positive). PMC pass
on the Phase A wave64 variant before committing to dp4a port."

### M2: §3.1.3 MMQ port at "2-3 sessions" understates the LDS-tuning cost

The PR #158 MMQ redesign took the predecessor 4 days *after* the
dp4a kernel was working, just for the LDS bank-conflict diagnostic
+ per-mmq_x stride tuning. That's clearly documented in
`docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`.

The plan's §3.1.3 says HFQ6 MMQ would need "the same X_STRIDE /
bank-conflict diagnostic from PR #158 will need to be redone." It
acknowledges this happens but budgets the entire Phase C at "2-3
sessions" — which is roughly what PR #158's MMQ work took *after
the kernel was already written and correctness-validated*.

The realistic budget for HFQ6 MMQ from scratch:
- Initial port (mirror HFQ4 body.cuh structure): ~1 session
- Correctness validation against FP wave64 reference (mirror of
  PR #158's `test_gfx906_mmq_correctness` + real-data NRMSE):
  ~0.5 session
- LDS bank-conflict diagnostic + per-mmq_x stride sweep: 4 days
- Real-data validation on Qwen 9B mq6 prefill: ~0.5 session
- mmq_screen_threshold sweep: ~0.5 session

**Total: ~5-6 sessions, not 2-3.** The plan needs to either acknowledge
this or specify a reduced scope (e.g. "only support mmq_x=64; drop
the partial-shape variants" — which trades coverage for speed but
makes the timeline more honest).

### M3: The "DFlash" framing in §3.2.3 is muddled

The plan says HFQ8 + DFlash "isn't a supported configuration on
gfx906" and recommends adding wave64 batched GEMM. But it doesn't
distinguish:

1. **HFQ8 weights with HFQ8 drafter** — production scenario,
   needs wave64 batched + verify-pass support.
2. **HFQ8 *target* with mq4 *drafter*** — currently the standard
   DFlash setup; the drafter is small and uses a different quant.
   The HFQ8 work only affects the target's verify pass.
3. **MQ8 (rotated HFQ8) target** — exists today, should be the
   actual baseline since `gemv_mq8g256_prerotated` ships.

The plan jumps to "FP-fallback" without noting that MQ8 with its
existing dp4a kernel may already be the best-case-supported state.
**HFQ8 specifically (without rotation) may not need new work** if
production users have moved to MQ8.

**Recommended fix:** §3.2.3 should explicitly compare current state
of (HFQ8, MQ8) for DFlash, and check whether anyone uses raw HFQ8
in production.

### M4: §4 "coherence note" is asserting from a different domain

The dot8 PRD's Q4_1 NRMSE failure ("18× Q8_1") was about activation
quantization, not weights. The §4 of this plan extrapolates that
to "the activation format for any HFQ6/HFQ8 dp4a variant must be
Q8_1." That's correct as a conclusion but the reasoning chain skips
a step: HFQ6/HFQ8 weights are 6/8-bit, **activations are
independent**.

The right way to state this: "Q8_1 activations are required for any
of these dp4a variants because Q4_1 was shown infeasible. The weight
quant choice (HFQ6 / HFQ8) is independent." It's a small framing
issue but the current §4 implies the conclusion is somehow stronger
than it is.

### M5: Missing comparison to existing MQ8 perf

The plan analyzes HFQ8 in isolation, but **MQ8 is the working
reference implementation of "int8 weights × int8 activations + dp4a
on gfx906."** Any HFQ8 work plan should benchmark against MQ8's
actual measured tok/s, not against the wave32 FP baseline.

**Recommended fix:** before any kernel work on HFQ8, run a
3-run AR decode bench at:
- Qwen 9B mq8 (existing path)
- Qwen 9B [hypothetical] hf8 (would need a model)
- Establish the gap (or absence of gap) as the work-plan baseline.

If MQ8 is already at parity with FP-equivalent throughput, the
HFQ8 wave64 port loses most of its motivation.

### M6: No production-demand analysis

The plan ends with "do priority 1 and 2 if/when there's measured
production demand for mq6/hf8 on gfx906." That's the right gate, but
the plan doesn't say *how* to measure that demand or *who* is
responsible for measuring.

A realistic production-demand check:
- Does any model in `~/.hipfire/models/` ship in mq6 / hf8 form by
  default? (Check the model registry / catalog.)
- Have any users requested mq6 / hf8 on gfx906 in PR comments,
  Discord, or issue trackers?
- Is there a workload (a particular Qwen / Llama variant) that
  would *specifically* benefit from 6-bit weights on gfx906 that
  doesn't already have an MQ4/MQ8 alternative?

Without this, the plan is "build it and they will come" — same
failure mode as my dot8 v1 PRD which the reviewers correctly
flagged.

---

## Minor issues

### m1: "HFQ4G128" support claim in §1 is unverified

The §1 table doesn't mention HFQ4G128 (smaller group size), but the
codebase has `gemv_hfq4g128` and `gemm_hfq4g128` dispatch fns. The
plan focuses on G256 quants exclusively. Probably fine — G128 is a
rare format — but worth noting that "HFQ6G128" / "HFQ8G128" are
*also* missing surfaces if anyone needs the higher-quality smaller-
group variants.

### m2: §2.2 "+60% lift HFQ4 dp4a" is fuzzy attribution

The plan claims "HFQ4's +60% per-call lift" from dp4a. Looking at
the actual PR #158 results in the dev log:

- `fused_gate_up_dp4a`: +7.1% **end-to-end DFlash** (not per-call)
- LM-head dp4a port: +12% end-to-end DFlash, +7% AR decode
- The MMQ kernel itself: ~5× over wave32 in pp512

Per-call lift on the fused GEMVs was never explicitly measured (we
inferred it from the end-to-end deltas). "+60% per-call" is an
upper-bound guess, not a measured number. The plan should either
cite the specific PMC measurement or downgrade to "+30-50% per-call
estimated."

### m3: §6 doesn't suggest a measurement plan

The "if there's production demand" gate has no operational
definition. A realistic measurement plan would be:
- Add HFQ6 / HFQ8 model variants to the standard coherence-gate
  battery (one row each).
- Run them on gfx906; record decode tok/s.
- Compare to existing MQ4 / MQ8 / Q4_K_M baselines.
- If decode lift from porting would be <5% **and** quality is
  matched by an existing format, defer.

This makes the abort path measurable instead of vibey.

### m4: References section omits two relevant docs

The plan's §7 references list misses:
- `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md` —
  the LDS-bank-conflict diagnostic that §3.1.3 references.
- The merged PR #158 commit `afb84bd` (cited in §1 status header
  but not §7 reference list).

Trivial fix.

---

## What the plan gets right

For balance, the structural shape of the plan is good:

1. **Phase-gated structure with measured demand check at the end.**
   The dot8 PRD's Q1 gate was the right pattern; this plan inherits
   it.
2. **Honest about uncertainty in §3.1.2.** Even the imprecise
   "+15-25%" range acknowledges PMC validation is needed before
   committing to dp4a.
3. **Inherits the Q4_1 lesson from dot8.** §4's "must use Q8_1
   activations" is the right load-bearing constraint.
4. **Correctly identifies Phase A (wave64 FP) as the cheap and
   low-risk path.** This is a defensible recommendation even with
   the dp4a / MoE / MQ8 issues fixed.
5. **Distinguishes "AR-only" from "DFlash" coverage.** The user's
   original question explicitly asked for this split, and the plan
   delivers on it (for the surfaces it analyzed).

---

## Recommendation

**Revise the plan substantively before starting implementation.**
The §1 coverage table and §3.2 HFQ8 analysis are wrong in ways that
would mislead anyone implementing from the doc.

Concrete changes:
1. **B1:** Rewrite §1 coverage table to separate batched-GEMM from
   single-token-GEMV surfaces. Update Phase A scope language.
2. **B2:** Rewrite §3.2 around the MQ8 reference implementation.
   The lever exists; the plan needs to describe porting it to HFQ8.
3. **B3:** Add MoE-indexed kernel coverage analysis to both §3.1
   (HFQ6) and §3.2 (HFQ8). At least one paragraph per quant.
4. **M1, M2, m2:** Soften per-call lift estimates; cite measured
   numbers from PR #158 dev logs where available, mark the rest as
   "estimate, PMC required."
5. **M3, M5:** Add explicit MQ8 baseline comparison for HFQ8 work;
   defer HFQ8 work if MQ8 is already at parity.
6. **M6, m3:** Add operational definition of "production demand"
   gate.
7. **m1, m4:** Fix the small omissions (G128 mention, references).

Estimated revision time: ~2 hours. Once revised, the plan is a
defensible starting point for the priorities 1 and 2 work it
recommends.

**Hold the priorities 3 and 4 (HFQ6 dp4a + MMQ batched) until the
revised plan demonstrates measured workload demand.** The lessons
from PR #158 + the closed dot8 PRD both point toward "don't build
speculative kernel optimizations" — that lesson needs to apply
here too.
