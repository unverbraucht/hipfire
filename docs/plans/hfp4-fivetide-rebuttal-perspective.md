# Perspective on fivetide's HFP4 quality analysis

**Date:** 2026-05-11 (KLD update: 2026-05-12)
**Triggering doc:** `fivetide/hipfire@docs/hfp4-quality-investigation/docs/investigations/2026-05-11-hfp4-quality-analysis.md`
**Status:** Active engagement, now bidirectional. The PPL story and the KLD story diverge sharply — neither alone is sufficient evidence. The "measure, don't assume" line in the original next-actions list turned out to apply equally to fivetide's PPL-only conclusion and to my MSE-only extrapolation.

## 2026-05-12 update — KLD reverses the picture (per fivetide)

Fivetide ran KLD measurement on Qwen3.5-0.8B and reports the picture is **different from PPL** at first look. Specifically: **FWHT helps E2M1 on KLD**, which directly inverts what their PPL data showed and what their kurtosis-driven theoretical argument predicted.

This is significant for three reasons:

1. **Fivetide retracted the "FWHT + E2M1 anti-synergy" framing in real-time** based on KLD data. That's the right way to do this work, and it sharpens rather than weakens their overall investigation — they're now driving the same conclusion I argued in §"On the bigger question: is PPL the right yardstick?" below: PPL is one metric, not the metric, and looking at multiple yardsticks changes the picture.

2. **The kurtosis / Lloyd-Max codebook argument needs reframing, not retraction.** Post-FWHT weight kurtosis really is ~2.82 (sub-Gaussian) and INT4-uniform really does beat E2M1 on per-block reconstruction MSE. Both still measured, both still true. The piece that turned out to be the wrong inference is "therefore E2M1's non-uniformity is a model-quality liability." Distribution shape → optimal codebook is a per-block reconstruction story; KLD measures something downstream that the reconstruction story doesn't capture.

3. **My §"What this means for the format roadmap" downgrade was directionally right but partly for the wrong reason.** I downgraded "commit to HFP4" based on PPL evidence. The actual situation is: PPL is one signal, KLD is a different (and arguably more representative) signal, and they disagree. The honest update is **not** "MFP4 is the wrong default" — it's "MFP4 vs MQ4 is metric-dependent on dense Qwen3.5, and we don't have enough data yet to call it on either side."

### What this changes about everything below

The PPL discussion in §"Where fivetide is right" and §"What this means for the format roadmap" stays as a record of what looked true on 2026-05-11. **The strategic implications are weaker than I framed them** because:

- The "NRMSE paradox" (MFP4 wins per-tensor reconstruction, loses PPL) is real but is now joined by a counter-paradox: **the PPL paradox** — MFP4 loses PPL but apparently doesn't lose KLD. Two metrics, two answers, neither sufficient alone.
- The Lloyd-Max codebook argument continues to be right *about codebook MSE* and underdetermined *about model quality*. The "kurtosis → uniform optimal" framing is true for the reconstruction problem and not necessarily true for the inference problem.
- Phase A engineering (imatrix calibration + weighted LS) is **still** the right work, and the bet that calibrated MFP4 beats UD-Q4_K_XL on KLD now has *empirical support* (at least on 0.8B; replication on 9B + A3B remains owed).
- The "Path A: calibrated MQ4 vs Path B: revised HFP4" framing in §"What this means for the format roadmap" is now better framed as **"calibrate both, measure both KLD and PPL and downstream tasks, decide on evidence."**

### What I'd add to my own §"On the bigger question: is PPL the right yardstick?"

The KLD reversal is a real-time empirical confirmation of that section's argument. I'd extend it now with:

- **PPL and KLD can disagree at the same bpw on the same model with the same format pair.** This isn't a metric-quibble; it's a structural property of how each metric weights different parts of the output distribution. PPL is dominated by the most-confident tokens (where the FP16 reference is sharply peaked); KLD measures distribution-shape preservation across all probabilities. A quant that "blurs" sharp peaks but preserves shape will look worse on PPL and better on KLD.
- **For hipfire's user use-cases, KLD is plausibly closer to "what users care about" than PPL.** Reasoning, instruction-following, and tool-use depend on the model preserving its probability shape across many low-probability tokens (the ones that carry instruction-following signal). A model that confidently produces wikitext-fluent fluff but loses tool-call shape preservation is PPL-good and KLD-bad. The agentic-coding workflow described in CLAUDE.md is much more KLD-sensitive than PPL-sensitive.
- **The right next bench is downstream tasks, not just metric-stacking.** Even KLD is a proxy. HumanEval, MMLU, an agentic harness, or LLM-as-judge on real prompts would be more direct evidence than either PPL or KLD.

### Updated honest meta-lesson

My original meta-lesson said "I was too confident in the per-weight-MSE framing." The updated version: **the empirical investigation correctly humbled the MSE story; the PPL data correctly humbled the MFP4 story; the KLD data correctly humbled the PPL story.** None of these metrics is the final word on quantization quality. The right working stance is: every metric tells you about a *projection* of model quality, and shipping a format requires looking at enough projections to triangulate. We don't have enough projections yet to ship either MQ4 or MFP4 with confidence — but we have enough to know that the question is empirically tractable with the engineering we've already planned.

This is good news, not bad. It means the path forward is well-defined: build the multi-metric bench, run the comparisons, decide on evidence. The strategic update to `qwen35-mq4-quality-gap.md` should reflect "format choice pending multi-metric empirical resolution" rather than either "ship HFP4" or "abandon HFP4."

---

## TL;DR (original 2026-05-11 — partially superseded by the KLD update above)

Fivetide presents the strongest empirical case against MFP4G32 I've seen: **+25–94% PPL regression vs MQ4G256 across three Qwen3.5 model sizes**, with a theoretical root-cause story (post-FWHT kurtosis ≈ 2.82 → sub-Gaussian → uniform-optimal → E2M1 non-uniform spacing is counterproductive) and a Lloyd-Max codebook analysis backing it.

My current `qwen35-mq4-quality-gap.md` predicted MFP4G32 would be a 3–5× per-weight MSE *win* over MQ4G256 and projected calibrated MFP4 beating UD-Q4_K_XL on KLD. **Fivetide's data is consistent with that per-weight MSE win** (their NRMSE table actually confirms E2M1+FP16 beats MQ4 on per-tensor reconstruction error, 0.1011 vs 0.1087) **and simultaneously shows MFP4G32 losing badly on PPL.** That's the load-bearing contradiction.

The reconciliation isn't a single answer. It's:

1. **Fivetide's PPL numbers are likely correct** for the configuration they measured (and that configuration is the one many users care about for "raw" model quality).
2. **My MSE-based extrapolation was wrong** in the specific claim that lower per-weight MSE implies better model quality. Fivetide directly demonstrates the opposite ("the NRMSE paradox"), and the demonstration is convincing.
3. **PPL is also not the only metric and may not be the right yardstick** for what hipfire users actually do (instruction-following, reasoning coherence, tool-use). The same investigation that found MFP4 didn't fix the A3B `<think>` spiral (`qwen35-moe-coherence-investigation.md` Phase 11) found that *MQ4 doesn't fix it either*. PPL is necessary, not sufficient.
4. **The strategic conclusion has to update.** Phase A's projected "calibrated MFP4 beats UD-Q4_K_XL on KLD" is now in doubt — at minimum it needs measurement before being claimed. The default-format question (MQ4 vs MFP4G32 for new models) shifts from "ship MFP4" toward "measure both, ship the winner per-arch."

## Where fivetide is right

### The per-weight MSE → quality assumption is wrong

This is the most important thing in fivetide's doc. Section 4.3 (my reading): E2M1+FP16-block achieves NRMSE 0.1011 vs MQ4's 0.1087 — E2M1 is *better* at per-tensor reconstruction. Yet MQ4 has substantially better PPL.

This invalidates a chain of reasoning in `qwen35-mq4-quality-gap.md`:
- §1.3 "MFP4G32 closes L1 — Estimated MSE win vs MQ4 L1: ~1.3–1.6×"
- §1.3 "L1 × L2 × L3 ≈ 2.6–4.2× — matches the empirical 5.8e-7 vs MQ4's 1.5–2.9e-6"
- §2.4 implied: lower per-weight MSE will translate to better downstream metrics

The chain is wrong because per-tensor MSE doesn't account for **how the error distribution propagates through the transformer stack**. Fivetide's framing — that MQ4's uniform-distributed quantization error propagates more favorably than E2M1's near-zero-concentrated error — is the right framing to investigate.

I should have caught this. PR #225's commit message reports 5.8e-7 MSE for MFP4, and I extrapolated to "therefore better model quality" without checking the PPL. The MSE is a measurement of one thing; PPL is a measurement of a different (downstream) thing.

### The kurtosis argument is empirically grounded

Fivetide measured per-block kurtosis on 615M post-FWHT real weight elements and got **2.82 stable across three model sizes**. That's directly below Gaussian (3.0) — sub-Gaussian — and on a sub-Gaussian distribution, uniform quantization is closer to optimal than E2M1's non-uniform spacing.

This is a real distribution-theory result. Lloyd-Max optimal at g=32 gives MSE 0.00123; INT4 uniform is +33.7%; E2M1 is +58.8%. **INT4 uniform beats E2M1 on the actual post-FWHT distribution.** That's not arguable — it's measured on real weights.

The implication: **the rotation is at war with E2M1's element format**, not synergistic with it. MQ4's "FWHT + uniform INT4" stack is a coherent design; MFP4's "FWHT + E2M1" stack has internal tension that the per-weight MSE measurement doesn't surface.

### The format-design recommendations are sound

If we keep E2M1 codes for hardware reasons (RDNA4 native FP4 decode, MXFP4 interop), fivetide's three concrete improvements are well-targeted:

1. Drop the FWHT for E2M1 variants → unrotated HFP4G32 stays
2. Replace UE8M0 with FP16 block scale → +0.25 bpw cost, −8.76% NRMSE
3. Add per-block FP16 bias (zero-point) → restores asymmetry that helps non-zero-mean blocks

The HFP4 wire format already accommodates #2 and #3 (per `docs/quant-formats/hfp4.md`'s reserved fields).

## Where fivetide's analysis has methodology gaps

I want to push on these because the conclusions are load-bearing for our strategic direction, and the methodology questions are real:

### Gap 1: K-map mode unspecified

The PPL table doesn't say whether the runs used hipfire's K-map mode 1 (alternating promotions — `mlp.gate` to Q8, edge layers to MQ6, every-3rd `ffn_down` to MQ6) or pure uniform format. This matters because:

- **Promoted tensors are not E2M1 in MFP4G32** — they're Q8, identical to MQ4's promotions. The K-map mixes formats by tensor class.
- A pure-format run is a *kernel-correctness* test, not a *production-quality* test. Hipfire ships K-map mode 1 by default; users see the mixed distribution.

If fivetide ran pure-format (no promotions), the +25–94% gap is the worst case. If they ran K-map mode 1, the gap is real-world. I don't know which it is.

### Gap 2: Tensor selection unspecified

"262 tensors, 19.25M blocks" for the scale precision analysis — but the PPL runs don't enumerate. If embeddings, lm_head, and router weights were all E2M1-quantized rather than Q8-promoted, that alone could explain a large fraction of the PPL gap (those tensors are the most sensitive in any quant scheme). The K-map promotion design exists *because* these tensors warrant higher precision.

### Gap 3: Activation-side rotation interaction not addressed

The runtime applies `mq_rotate_x` to activations so that `dot(W_rot, x_rot) == dot(W, x)`. This is the trick that makes FWHT-rotation orthogonal at inference time. The MFP4 quantize path inherits this — but if the activation rotation has subtle precision interactions with the dequantized E2M1 weights (which now span a different magnitude range than the rotated INT4 weights did), there's a different precision-cliff mechanism the analysis doesn't surface.

### Gap 4: PPL gap decomposition is estimated, not measured

Fivetide's "60% codebook / 25% UE8M0 / 10% no-zero-point / 5% FP16-vs-FP32 scale" decomposition is acknowledged in the doc as "estimated...based on component analysis, not ablation." An actual ablation (one format change at a time) might surface different relative magnitudes — possibly the FP16 row scale carries more weight than the decomposition gives it.

### Gap 5: A3B / MoE wasn't tested

The PPL runs are on Qwen3.5 dense (0.8B, 4B, 9B). MoE models (3.6-A3B) have different sensitivity to weight quantization because the routing decision compounds quant noise across 8 experts × 30+ layers. Phase 11 of `qwen35-moe-coherence-investigation.md` shows MFP4 didn't fix the A3B spiral — but neither did MQ4 (the May 6 baseline only worked because of the under-scaled norm regime). The relevant question for MoE isn't "MQ4 vs MFP4 PPL on dense," it's "does either format avoid the coherence cliff," and the answer so far is "no."

## On the bigger question: is PPL the right yardstick?

The user pushed on this and they're right to. PPL has known weaknesses as a quality measure:

### Where PPL is reliable
- Same model architecture, same tokenizer, same calibration corpus — relative comparison of quantization variants is meaningful (this is fivetide's regime, and their numbers are credible there)
- Catches catastrophic quantization failures (PPL doubles or worse) — fivetide's 0.8B +94% is a flashing red light regardless of metric quibbles

### Where PPL is misleading
- **Instruction-following quality**: a model can have 5% lower PPL and 20% lower instruction-following success rate. Different things.
- **Reasoning / chain-of-thought**: per-token cross-entropy on wikitext is a fluency proxy; it doesn't measure whether the model reaches correct conclusions. Phase 11's "MFP4 doesn't fix spiral but MQ4 doesn't either" is invisible to PPL.
- **Long-context behavior**: PPL is averaged over context positions; tail-attention failures don't show until the model is asked to actually use long context.
- **Tool-use / agentic workflows**: PPL says nothing about whether function-call shape is preserved.
- **The hipfire community uses cases**: Qwen3.5-9B for coding, 27B-DFlash for chat, A3B for reasoning. None of these are primarily measured by wikitext PPL.

### Where Unsloth's choice is informative

Unsloth's Dynamic 2.0 calibration corpus (the >1.5M-token Calibration_v3/v5) is **deliberately not just wikitext**. They use hand-curated multi-turn chat + code + reasoning prompts because they observed that wikitext-only calibration overfits a specific token distribution. The imatrix lever (Thread 1 §2 lever L5) is fundamentally about *which tokens you weight as important for quality*, and the right answer depends on the use case.

This connects directly to fivetide's "PPL paradox" — MQ4 has higher NRMSE but lower PPL because of how the errors propagate. Generalizing: **the relationship between quantization noise and downstream quality is task-specific, and PPL captures one specific task's view of it.** Wikitext PPL favors "uniform error distribution" because language fluency is robust to random noise but fragile to structured (near-zero-clustered) noise. A code-generation task might prefer the opposite — fewer extreme errors at the cost of more near-zero clustering.

### Honest conclusion on PPL

Fivetide's PPL numbers are **necessary evidence**: a 94% PPL regression on 0.8B is a real failure regardless of methodology nits. We can't ship MFP4 as a default for 0.8B based on those numbers. But PPL is **not sufficient** evidence for the bigger format question:

- For a 9B model with +25% PPL on wikitext, the right next test is **KLD on a domain-relevant corpus + downstream task metrics** (HumanEval, MMLU, or domain-specific benchmarks the user actually runs), not just "ship the lower PPL."
- For MoE/A3B specifically, neither PPL nor MSE caught the spiral. We need attractor detection (period-N block-repeat) on the same benchmarks.
- For instruction-following / reasoning preservation, the right metric is paired-completion comparison or LLM-as-judge, not PPL.

## What this means for the format roadmap

The honest update to `qwen35-mq4-quality-gap.md`:

### Things that change

1. **§1.3's claim "MFP4G32 closed L1+L2+L3" is misleading at the model-quality level.** MFP4 closed those levers *on per-weight MSE*. It did not deliver a model-quality win, and on Qwen3.5 dense it delivers a model-quality regression at the PPL metric. Reword to reflect the asymmetry.

2. **§2.4's "we sit a couple levers behind Unsloth at equal bpw" is too optimistic.** The lever taxonomy assumed MFP4 was already a wash with Q4_K_M on quality and we only needed L4+L5. Fivetide shows MFP4 may be substantially *behind* MQ4 on quality, which means we have to first close the FWHT+E2M1 incompatibility before stacking calibration on top.

3. **Phase A's projected "calibrated MFP4 beats UD-Q4_K_XL on KLD" is no longer the bet to make.** Phase A is still the right *engineering work* (imatrix calibration, weighted LS, per-tensor bit allocation) — but the baseline format that calibration goes on top of needs to be revisited. Two viable paths:
   - **Path A**: imatrix-calibrated MQ4G256 — sticks with the format that the empirical data favors, adds the missing levers
   - **Path B**: imatrix-calibrated unrotated HFP4G32 (per fivetide's recommendation) + FP16 block scale + zero-point — restructures HFP4 to remove the FWHT+E2M1 tension

4. **Phase B' (gfx906 kernel port) priority drops.** If MFP4 isn't a quality win, porting MFP4 kernels to gfx906 doesn't deliver user value. The dp4a/MMQ/wave64 fixes from the audit are still valuable for MQ4 itself.

### Things that don't change

1. **The wire-format extensibility analysis is still correct.** HFP4's reserved bits accommodate L4+L5 calibration regardless of which element format wins. The L1 decision (E2M1 vs INT4) doesn't change the fact that the format can carry imatrix-derived weighted-LS scales.

2. **The gfx906 acceleration analysis stands.** The dp4a path works for any 4-bit format via LUT decode. Whether we end up shipping MFP4 or MQ4-v2 on gfx906 is a quality question, not an acceleration question.

3. **The arch-portability analysis stands.** Adding Gemma / Qwen2.5-VL is independent of the element-format choice.

4. **The Phase 11 MoE coherence story stands.** MFP4 didn't close the spiral; fivetide's data adds another reason to question whether MFP4 is the right default. But the *runtime-side* fixes (sampler intervention, vLLM FP16 router contract) are still the path forward for the coherence problem.

### Things that need to be measured before claiming

1. **MQ4 vs MFP4G32 PPL on Qwen3.6-A3B** — fivetide tested dense; we don't have MoE numbers
2. **MQ4 vs MFP4G32 instruction-following on humaneval / mmlu** — does the PPL story reproduce on downstream tasks?
3. **MQ4 vs MFP4G32 KLD vs FP16 reference on a real-task corpus** — is fivetide's NRMSE paradox visible at the KLD level too?
4. **Independent ablation of fivetide's PPL gap decomposition** — what fraction is actually E2M1 vs UE8M0 vs no-zero-point vs FP16-vs-FP32 scale?
5. **A FWHT-off variant of MFP4** — pure HFP4G32 (qt=21) on Qwen3.5-9B vs MQ4G256, does the gap close significantly?

## The honest meta-lesson

I was too confident in the format-design framing of `qwen35-mq4-quality-gap.md`. The doc's strategic claim ("HFP4 is the format we should commit to for the next several years") was built on a chain of per-weight-MSE reasoning that fivetide has shown does not predict model quality reliably. The format taxonomy (L1–L5) is still useful for organizing engineering work, but the predicted noise-win multipliers in §2.2 and §1.3 are theoretical extrapolations that the empirical data does not support.

The fivetide doc is a much stronger piece of empirical work than my analysis was. It measures the thing we actually care about (PPL on real models), shows a result that contradicts the per-weight-MSE prediction, and gives a defensible mechanism (kurtosis → distribution shape → optimal codebook shape) for why.

That said, I don't think the conclusion is "MFP4 was a mistake." It's "MFP4 traded per-tensor reconstruction quality for hardware-friendliness without re-validating that the trade preserved model quality." The fix is the same engineering — calibration, per-tensor selection, downstream-task validation — that we'd need to do anyway, just with a more honest baseline.

## Concrete next actions

In rough priority order:

1. **Measure, don't assume.** Re-run fivetide's PPL benchmarks in-tree on Qwen3.5-9B and Qwen3.6-A3B with both MQ4G256 and MFP4G32 quantized identically (same K-map mode, same Q8 promotions, same FWHT seeds). Confirm or refute the +25% gap on our infrastructure.
2. **Ablation:** quantize Qwen3.5-9B as (a) HFP4G32 unrotated, (b) MFP4G32 (current), (c) hypothetical "MFP4G32 + FP16 block scale + zero-point" — see which knobs matter. This either validates fivetide's decomposition or surfaces a different one.
3. **Add KLD measurement to `bench_quant_quality.sh`.** Currently the bench has MSE + smoke test; adding KLD vs FP16 reference would catch the "NRMSE paradox" cases.
4. **Rewrite `qwen35-mq4-quality-gap.md` §1.3 and §2.4** to reflect the empirical PPL data rather than per-weight MSE extrapolation. Note the open question on whether Path A (calibrated MQ4) or Path B (revised HFP4) is the right calibration baseline.
5. **Engage with fivetide directly.** They have data we don't; we have engine internals they may not have inspected. Cross-validating the methodology gaps (K-map mode, tensor selection, A3B/MoE coverage) is straightforward joint work.

The strategic posture shifts from "commit to HFP4 for the next several years" to "we have a serious data point that says the current HFP4 may be the wrong default; investigate before committing further." That's a less confident position than the current design doc, but it's the honest one given the evidence.

## References

- fivetide's analysis: `https://github.com/fivetide/hipfire/blob/docs/hfp4-quality-investigation/docs/investigations/2026-05-11-hfp4-quality-analysis.md`
- `docs/plans/qwen35-mq4-quality-gap.md` — the format roadmap this doc challenges
- `docs/plans/qwen35-moe-coherence-investigation.md` Phase 11 — the prior data point that quantization-side fixes alone don't close the A3B spiral
- `docs/quant-formats/hfp4.md` — wire format spec (reserved fields still extend naturally to FP16 block scale + zero-point)
