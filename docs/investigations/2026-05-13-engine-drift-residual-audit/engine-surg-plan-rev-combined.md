# Combined adversarial review: 01-engine-surgery-plan.md

**Reviewers folded:** Claude Opus 4.7 (self-review), Gemini CLI, GLM-5.
**Date:** 2026-05-13
**Target doc:** `docs/investigations/2026-05-13-engine-drift-residual-audit/01-engine-surgery-plan.md`

## Synthesis

Three independent reviewers raised 22 distinct issues across the plan. After de-duplication, **eight load-bearing concerns** require plan revisions before Phase 2 starts. Three other issues are nice-to-haves; one is rejected. The strongest single point of cross-reviewer agreement is that **Phase 3a should run before Phase 2, not after** — GLM-5 argued this explicitly; my own review and Gemini's reach the same conclusion via different paths (matched-input probe is the cheapest decisive experiment).

This document keeps reviewer attribution where the framing matters; everything is consolidated and adjudicated, not just concatenated.

---

## Critical issues (must address before Phase 2)

### C1. Phase 3a should run first; Phase 2 is conditional on its outcome

**Sources:** GLM-5 (Recommendation 1, strongest), self-review (#1, methodological), Gemini (Section 1, implicit).

**Argument.** Phase 3a (matched-input recurrence probe) is a **decisive** experiment:
- If hipfire's `gated_delta_net_f32` matches HF's `chunk_gated_delta_rule` to ≤ 0.005 rL2 with bit-exact inputs, the recurrence is faithful. The 0.045 stage-13 rL2 we measured is then *entirely* upstream-amplified, and Phase 1d's "no per-weight intervention closes the floor" verdict generalizes to "no per-kernel intervention closes the floor" — close the branch immediately.
- If hipfire's recurrence diverges (≥ 0.01 rL2 with bit-exact inputs), the recurrence kernel **is** the surgery target. Phase 2 (FA pipeline) becomes a complementary audit — useful but not gating. Phase 4 targets the recurrence kernel directly.

Cost: 1 day for Phase 3a. **Resolves a multi-week scope question.** Phase 2 (1-2 days) is upstream-amplification data that's only informative if the recurrence is the target — running it first is risk-symmetric.

**Plan revision required:** reorder Phase 3a before Phase 2. Make Phase 2 conditional on a positive Phase 3a outcome.

### C2. Phase 1d's "structural drift" conclusion is being partially over-extended

**Sources:** GLM-5 (Critical Problem section, strongest framing), self-review (acknowledged but underweighted).

**Argument.** Phase 1d concluded that **per-weight-precision** interventions don't close the floor. It did NOT test kernel-rewrite interventions. The plan implicitly treats these as the same intervention space; they aren't. Q8→F16 weight storage changes per-element values; kernel rewrites change accumulation order, fusion choices, and reduction patterns. The systematic engine bias measured at Phase 1d's 0.0017 rL2 stage-1 (bit-exact input) is in the *kernels*, not the weights.

But the plan is also right that Phase 1d's outcome is a strong prior: distributed drift across stages 0-12 (each ≤ 0.012) means individual kernels have small drift. The recurrence (4× amplification, stage 13 at 0.045) is the only stage where drift is *concentrated*. So the recurrence is the only viable single-kernel surgery target.

If Phase 3a clears the recurrence, the residual drift is the cumulative effect of many small per-kernel issues — at which point Phase 1d's verdict (re-extended to kernels) holds and surgery is exploration, not engineering.

**Plan revision required:** explicitly acknowledge in the decision matrix that if Phase 3a clears the recurrence, surgery is closed because the per-kernel cumulative-drift hypothesis matches Phase 1d's pattern.

### C3. "Byte-match llama.cpp" is the wrong framing

**Sources:** GLM-5 (Issue 4, with specific evidence), Gemini (Section 2), self-review (#12).

**Argument.** GLM-5 cited specific incompatibilities:
1. llama.cpp's GDN kernel uses 4 warps per column with row-subset ownership; hipfire uses 1 warp per 4-row tile with warp shuffles. Different reduction orders.
2. llama.cpp's FA path uses `ggml_flash_attn_ext` (cuBLAS/cuDNN Flash Attention); hipfire has its own `attention_f32` with shared-memory online softmax. Fundamentally different implementations.
3. Compute graph fusion differs (e.g., scale factor inside vs outside the attention kernel).

"Byte-match" implies bit-equivalence, which requires reproducing the wave/thread layout, accumulator order, AND fusion structure. That's a graph-level rewrite, not a single-kernel port.

Gemini's complementary point: llama.cpp (GGML) has its own quantization-aware accumulation patterns optimized for GGML, not "what is most correct vs HF". Inheriting llama.cpp's implementation imports its biases.

**Plan revision required:** Phase 4 target should be **"match per-stage rL2 within 2× of llama.cpp's measurements"** or **"end-to-end KLD within 0.005 of llama.cpp's Q8_0 on the same model"**. NOT byte-match.

### C4. Missing pre-Phase-2 sanity checks

**Sources:** self-review (#1, #2, #3 in critical section).

Three checks need to happen before any Phase 2 instrumentation, total ~1 day:

1. **llama.cpp Q8_0 KLD on Q3.5-0.8B at matched Q8 KV** (~5 min). The Q4_K_M-beats-q8f16 datum motivates the entire investigation reopening. But Q4_K_M is *imatrix-calibrated*. If llama.cpp Q8_0 (uncalibrated, like hipfire's q8f16) is at ~0.005 KLD, the 0.07 gap is genuinely engine drift. If llama.cpp Q8_0 is at ~0.04 KLD, ~half the gap is calibration, not engine. This radically reshapes the surgery scope.

2. **Verify F16 dispatch isn't buggy** (~30 min). Phase 1d concluded "F16 weights regress KLD". I attributed this to "Q8 noise was load-bearing for cancellation". An alternative explanation: the F16 dispatch path has a bug (loader chunks wrong, gemv_f16 differs in accumulator semantics from gemv_q8_0_dequant_to_f32, etc.). The closed conclusion of Phase 1d hinges on this — if the F16 path is buggy, "structural drift" is premature.

3. **Read and verify the FA branch stage map** (~1 hour). The plan drafts 12 stages without checking the actual code. Hipfire uses fused kernels (fused_qkv, fused_qk_norm) that may collapse multiple stages. Wrong stage map = instrumentation at wrong points = misleading data.

**Plan revision required:** add these three checks as Phase 0 (pre-Phase-2). Phase 2 instrumentation does not start until they complete.

---

## Methodological gaps (must address during Phase 3a)

### M1. Phase 3a state initialization is unspecified

**Sources:** self-review (#4), GLM-5 (Issue 2 implicit), Gemini (Section 1 implicit).

**Argument.** The recurrence has per-position state accumulated from positions 0..t-1. The plan says "feed HF's exact q/k/v/α/β at each position" but doesn't specify state init:

- **Option (a) — clean reset**: reset state to zero at position 0, feed HF inputs sequentially. State at position t is hipfire-computed from HF inputs at 0..t-1. This is a *mixed* test: HF inputs + hipfire state arithmetic. If output diverges at position t > 0, the cause could be hipfire's per-step state-update math OR HF's intrinsic state being different at that point (HF may use a different parallelization).

- **Option (b) — HF state injection**: dump HF's intermediate state at each position (requires extending the HF probe), inject it before each position's hipfire compute. Output at position t depends only on HF's state and HF's inputs at t. Pure test of hipfire's per-step math.

Only Option (b) is the clean kernel-isolation test. The plan should commit to (b) and add the HF state-dump to Phase 3a's prerequisites.

**Cost addition:** ~0.5 day on Phase 3a to add the HF state-dump probe.

### M2. The fp32 ↔ fp64 fp64 equivalence isn't the right test for kernel correctness

**Sources:** GLM-5 (Issue 2, sharpened), Gemini (Section 1, sharpened).

**Argument.** Phase 1's probe c.3 showed hipfire's `gated_delta_net_f32` is byte-identical with fp64 internal state. The plan cites this as evidence the recurrence is "internally precise". But this only tests **accumulation precision**, not **operation ordering**. PyTorch's `chunk_gated_delta_rule` may compute the recurrence in a different order (e.g., chunk-then-update vs update-then-chunk, or different reduction directions across heads) that produces a different bit-pattern in fp32 even with bit-exact inputs.

Phase 3a's matched-input test directly probes this — but only if state-init is Option (b).

Additionally, the plan should:
- Compare hipfire's recurrence math against both HF's `torch_chunk_gated_delta_rule` (reference) AND llama.cpp's GDN kernel (porting target). These may differ from each other.
- Establish an **fp64 reference noise floor**: run the recurrence math in pure fp64 on CPU with HF's inputs, compare to HF's fp32 output. The difference is the fp32 noise floor that any fp32 implementation will share with HF's. Phase 3a's pass/fail threshold should be this floor + a margin.

**Plan revision required:** Phase 3a includes (a) HF reference (with Option-b state injection), (b) llama.cpp reference (port reference), (c) fp64 noise floor (calibration baseline). Three-way comparison, not binary pass/fail.

### M3. Layout verification step is missing

**Sources:** Gemini (Section 7, unique).

**Argument.** PyTorch's `Qwen3_5Attention.forward` produces intermediates in specific layouts (`[B, T, n_heads, head_dim]` per-head). Hipfire's stage dumps assume contiguous `[k_dim]` or `[v_dim]` flat layouts (post repeat-interleave for q/k, pre repeat-interleave possible). Compare across two engines requires matching these *exactly* — a stage 8 dump that captures q before HF's view-reshape vs. after will produce phantom rL2 from layout differences alone.

The existing LA-stage dump (Phase 1) worked because we wrote both sides with matching layouts. For FA stages there are more permute/view sites and they're easier to get wrong.

**Plan revision required:** add a Phase 2a "Layout Verification" sub-step before per-stage rL2 analysis. Concretely: for each stage, write a 5-line sanity check that loads one position from each engine's dump and prints the first 16 elements side-by-side. Confirms layouts match before computing rL2.

---

## Decision-rule weaknesses

### D1. The "single kernel > 50%" threshold has a false-dichotomy problem

**Sources:** GLM-5 (Issue 3, strongest), self-review (D1, #8), Gemini (Section 3).

**Argument.** All three reviewers independently raised this. Three failure modes for the current binary GO/NO-GO:

1. **Distributed but tractable**: 4 kernels at 20% each. Total surgery time 4-8 weeks. Plan classifies as NO-GO; reality is a longer GO.
2. **Layer-varying dominant**: QK-norm dominates FA-3, softmax dominates FA-23. Plan classifies as NO-GO because no single kernel >50% at any one layer; reality is 2-3 kernels with high total contribution.
3. **Coupled kernels**: kernel A's output feeds kernel B's input; B amplifies A's drift. Fixing only B doesn't help if A still drifts. Fixing only A doesn't help if B amplifies the residual. They must be fixed together — plan doesn't model this.

**Plan revision required:** replace the 5-row decision matrix with a sharper triage:

| outcome | criterion | action | effort |
|---|---|---|---|
| Recurrence diverges (Phase 3a) | recurrence rL2 ≥ 0.01 with HF-state-injected inputs | port recurrence kernel | 2-3 weeks |
| FA single kernel dominant | one kernel > 40% of per-FA-layer drift, consistent across 3/7/23 | port that kernel | 1-2 weeks |
| FA cluster localizable | top-2 kernels sum > 70%, independent of each other | port both | 3-6 weeks |
| FA coupled | top-2 kernels sum > 70%, one feeds the other | port both in tandem | 4-6 weeks + integration risk |
| FA distributed | top-2 < 50%, drift spread across 5+ stages | close — switch to calibration |
| Recurrence clean + FA distributed | both | close — Phase 1d verdict extends to kernels |

The 40% threshold (vs the original 50%) is the smallest single-kernel contribution that's worth 1-2 weeks of work to land an end-to-end measurable KLD improvement.

### D2. NO-GO outcomes lack a graceful exit

**Sources:** self-review (#9), implicit in GLM-5's Recommendation 6.

**Argument.** "Close branch, switch to calibration" loses the per-stage audit data. The Phase 2/3a data are valuable beyond the GO/NO-GO decision: they tell us *where* the bias lives, which informs:
- Future quant-aware fine-tuning approaches (target the high-drift activations)
- Calibration techniques that explicitly model the engine's bias pattern
- Long-term kernel-fusion redesigns when the runtime is rewritten for other reasons

**Plan revision required:** if NO-GO fires, write a Phase-2-results doc that summarizes the per-stage breakdown and explicitly tags which kernels contribute most. This becomes input to the calibration roadmap.

---

## Optimistic estimates

### O1. Phase 4 effort estimate is the optimistic case

**Sources:** Gemini (Section 4, strongest), self-review (#6, O1), GLM-5 (implicit in Issue 7).

**Argument.** "1-2 weeks per kernel rewrite" assumes the kernel is small/simple. Realistic:
- **Recurrence (gated_delta_net)**: 200+ lines in llama.cpp, 4-warp parallelism, complex state management. **2-3 weeks** plus 1 week validation.
- **Flash attention** (if FA target): 500+ lines, multi-stage softmax, shared-memory tiling. **3-4 weeks** plus 1-2 weeks tuning.
- **QK-norm + projection**: simpler. **1 week** plus validation.

Gemini specifically flagged RDNA3-specific concerns:
- LDS (Local Data Share) sizing and bank conflicts
- Wave-size differences (32 on RDNA3, 64 on CDNA3, often 32 on NVIDIA)
- Memory tier differences (no XCM, different L2 partitioning)

**Plan revision required:** Phase 4 budget should give a **range** per kernel:
- Optimistic: 1 week
- Realistic: 2-3 weeks
- Pessimistic: 4-6 weeks if blocked on HW-specific issues

Include a "blocked" milestone: if a single kernel rewrite hasn't passed milestone 2 (model-output KLD drop) after 3 weeks, escalate to "this kernel needs a graph-level rethink, not a port".

### O2. KLD-drop target needs a derivation

**Sources:** GLM-5 (Issue 7, with concrete proposal), self-review (#5).

**Argument.** Current floor: 0.08 KLD. Plan's target: ≤ 0.02. GLM-5's pushback: Phase 1d showed a 6-stage F16 upgrade (touching every precision-sensitive tensor) couldn't close the gap; it regressed. Reaching 0.02 requires graph-level rewrite, not single-kernel.

GLM-5's proposed revised target: **KLD ≤ 0.05**, matching llama.cpp's Q4_K_M at matched Q8 KV. This is achievable evidence-wise (llama.cpp shows it's possible) and a strong result (hipfire 8-bit weights matching 4-bit imatrix-calibrated GGUF).

**Plan revision required:**
- Revise Phase 4 milestone 2 target to **KLD ≤ 0.05** on 5-chunk eval.
- Compute the expected drop from per-stage data: stage 13 rL2 drops from 0.045 → 0.005 (10× reduction in recurrence drift). LA-block contribution to residual drops proportionally. Through 18 LA layers, residual-stream drift drops ~2-3× (compounding isn't perfectly multiplicative because of saturation). Model-output KLD expected drop: ~50-60%, i.e., 0.08 → 0.03-0.04.
- A 0.05 target is *conservative*; a 0.04 target is *plausible if the recurrence is the dominant kernel*. The plan should commit to the conservative number as the gate but report the optimistic number as the stretch goal.

### O3. AWQ compose-check is a sweep, not a check

**Sources:** Gemini (Section 5, strongest), self-review (#7), GLM-5 (Issue 5).

**Argument.** All three reviewers agree: AWQ scales were calibrated on the current engine's bias pattern. After kernel rewrite, the bias pattern changes. AWQ scales optimized for the old bias are not optimal for the new bias. Half-day "compose-check" understates the dependency.

**Plan revision required:**
- Rename milestone 3 from "AWQ compose-check" to **"AWQ re-calibration + validation"**.
- Budget: 1 day for re-calibration (Hessian collection + AWQ scale recomputation), 0.5 day for bench, 0.5 day for iteration if numbers shift. Total **2 days, not 0.5**.
- Add explicit decision criterion: if AWQ above-floor delta on the v2 engine is *less than* it was on the v1 engine, investigate why before declaring v2 successful. Possibilities: (a) v2 engine has less systematic bias for AWQ to correct (good — fewer AWQ scales needed); (b) v2 engine has different bias that AWQ's current calibration method doesn't model (need new calibration approach).

---

## Risks not addressed

### R1. Coherence-gate sharp cliff

**Sources:** self-review (#10), unique.

**Argument.** Phase 4 milestone 4 lists coherence-gate as a hard pass/fail. But a correct kernel rewrite may flip tokens (because the output is now closer to HF, which is the "right" answer) and the gate's panic/zero-token/timeout failure modes might fire for engineering reasons (e.g., the rewrite is slightly slower, hits a timeout).

**Plan revision required:** distinguish coherence-gate failure causes:
- Correctness regression (output diverges further from HF) → revert
- Engineering issue (timeout, panic, memory bug) → debug before reverting
- Output divergence in a "better" direction (closer to HF, but coherence-gate flags as different from baseline) → require human review

### R2. Cross-arch implications (gfx1100 vs gfx1151)

**Sources:** self-review (#11), unique.

**Argument.** Engine surgery happens on gfx1151. AWQ work happens on gfx1100. HIP kernels usually run on both with arch-specific perf tuning, but the bias offsets we're fixing are algorithmic (accumulator order, fusion choices) — they should be cross-arch consistent. So engine surgery should help gfx1100 too, but with a wavefront-size caveat: gfx1100 RDNA3 is wavefront-32; gfx1151 RDNA3.5 is wavefront-32 too (both RDNA3 generations). So no wavefront-size issue between them.

**Plan revision required:** Phase 4 milestone 5 explicitly validates on gfx1100 after gfx1151 lands. If the kernel uses arch-specific intrinsics (warp shuffles, etc.), a gfx1100-tuned variant is needed.

### R3. Phase 1c's "4× recurrence amplification" claim is layer-0-specific

**Sources:** self-review (#13), unique.

**Argument.** The 4× amplification factor (input drift 0.012 → output drift 0.045) was measured at LA layer 0. At LA layer 4, input drift is already 0.06+ (carrying 3 prior layers of drift). At that point the recurrence may saturate rather than amplify linearly. The "recurrence amplifies 4×" claim shouldn't generalize without verification.

**Plan revision required:** Phase 2 (which audits FA layers) should also re-audit LA layer 4 with the same instrumentation to verify the amplification factor. If it's <2× at deeper layers, the recurrence kernel's contribution to model-output KLD is even smaller than current estimate, and engine surgery's expected gain shrinks accordingly.

---

## Unique points worth keeping

### U1. Reference target is itself moving

**Sources:** self-review (#12).

llama.cpp's Q3.5 DeltaNet support is recent. Its 0.002 KLD vs HF was measured on chunk 0 of 20 chunks total. On the full slice, llama.cpp's drift could be higher. Run llama.cpp's 20-chunk eval against HF BF16 to verify the reference is stable across the eval slice. **Cost: 30 min.**

### U2. Phase 2 may produce confirmation, not discovery

**Sources:** GLM-5 (Issue 1, sharpened).

If FA layers mirror LA layers' distributed-drift pattern, Phase 2's deliverable is "confirmed distributed". GLM-5 argues this isn't worth 1-2 days. Counter-argument: the FA pipeline has FA-specific kernels (attention softmax, output gate) that LA doesn't share. The audit needs to confirm these don't dominate. **Compromise: scope Phase 2 down to FA-3 only as a fast-path verification (1 day); only extend to FA-7/FA-23 if FA-3 shows a non-distributed pattern.** This matches GLM-5's Recommendation 2.

---

## Rejected points

### Rej-1. Instrumentation overhead (macro-fy the dump call sites)

**Source:** Gemini (Section 6).

**Reasoning.** The existing 16 LA-stage dump call sites are clear, grep-able, and gated. Macro-fying introduces a layer of indirection that obscures what's being dumped where. Standard Rust style here is "explicit is better than implicit" — especially for a diagnostic codepath. The instrumentation is also expected to live in-tree as research scaffolding; macro abstraction is over-engineering for a known-static set of stage IDs.

---

## Revised plan (delta from `01-engine-surgery-plan.md`)

Concrete edits the original doc needs, in order:

1. **Add Phase 0 — pre-Phase-2 sanity checks (1 day)**: llama.cpp Q8_0 eval, F16 dispatch verification, FA stage map verification. Phase 2 doesn't start until all three pass.

2. **Reorder Phase 3a before Phase 2** (was: Phase 2 then Phase 3a). Phase 3a's binary outcome determines whether Phase 2 is needed.

3. **Phase 3a methodology**: commit to Option-b state injection (requires extending HF probe ~0.5 day). Three-way comparison (HF reference, llama.cpp reference, fp64 noise floor).

4. **Phase 2 scope**: FA-3 only initially. FA-7/FA-23 conditional on FA-3 showing a non-distributed pattern. Add layout-verification sub-step before computing rL2.

5. **Re-write decision matrix** (6 outcomes, see D1 table above). Replace 5-row matrix.

6. **Phase 4 timeline ranges**: 1 week (optimistic) / 2-3 weeks (realistic) / 4-6 weeks (pessimistic) per kernel, with an explicit "blocked" gate at 3 weeks.

7. **Phase 4 milestone 2 target**: KLD ≤ 0.05 (conservative gate), ≤ 0.04 stretch goal. Replace ≤ 0.02 target.

8. **Phase 4 milestone 3**: rename "AWQ compose-check" to "AWQ re-calibration + validation". Budget 2 days, not 0.5.

9. **Phase 4 framing**: "match per-stage rL2 within 2× of llama.cpp's measurements" or "end-to-end KLD within 0.005 of llama.cpp's Q8_0". Drop "byte-match" language.

10. **AWQ sequencing**: explicitly recommend sequential, not parallel, on the same model. Either complete surgery first or complete AWQ first; don't run them concurrently with shared benchmarking.

11. **Decision exits**: NO-GO outcomes preserve audit data in a Phase 2/3a results doc that feeds the calibration roadmap. Not just "close branch".

12. **Coherence-gate** failure-mode triage in Phase 4 milestone 4.

13. **gfx1100 validation** as Phase 4 milestone 5.

---

## Headline recommendations (final)

| action | wall | rationale |
|---|---:|---|
| **Phase 0** — sanity checks (llama.cpp Q8, F16 dispatch, FA stage map) | 1 day | Settles whether the audit's premise is sound. |
| **Phase 3a** — matched-input recurrence probe with HF state injection | 1.5 days | Decisive: clean → close branch; diverges → recurrence is the surgery target. |
| **Phase 2** — FA-3 stage audit with layout verification | 1 day | Conditional on Phase 3a outcome. Extend to FA-7/23 only if FA-3 shows non-distributed pattern. |
| **Decision review** | 0.5 day | Six-outcome triage, not binary GO/NO-GO. |
| **Phase 4** (if GO) — kernel rewrite with re-cal | 2-6 weeks | Range, not point estimate. |

**Total wall to decision: ~3 days** (was 4 in original plan, now slightly faster because Phase 3a's binary outcome can short-circuit Phase 2). **Total wall to surgery shipped: 4-8 weeks** (was 3 weeks; revised for realistic kernel-rewrite timelines + AWQ re-calibration).

The plan is sound in its overall arc — per-stage instrumentation, matched-input probe, decision-gated progression — but it needed (1) reordering, (2) sharper decision rules, (3) honest timelines, and (4) a pre-Phase-2 sanity check round before committing to multi-week surgery. After these revisions, it's ready to execute.
