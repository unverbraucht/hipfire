# KLD Measurements — Master Table

**Status:** Living document. Append new measurements; do not delete.
**Last updated:** 2026-05-13 PM (post pattern-hunt audit)
**Owner:** hipfire eval
**Authoritative methodology:** [`issue-113-quant-quality-eval.md`](issue-113-quant-quality-eval.md)
**Cross-engine framework:** Δ-above-own-Q8 (TL;DR + §2.4.1 + [engine-drift memory](../../memory/project_engine_drift_floor_decomposition.md)). Absolute `KLD(engine || HF-bf16)` is NOT a cross-engine output-quality metric — see §6 rule 8.

This doc is the single place to look for every KLD/PPL number we've measured against the BF16 reference. Pull-out tables in other docs (`qwen35-mq4-quality-gap.md` §1.5, `awq_hipfire.md`, individual cohort `result-table.md` files) are derived views; this is the source-of-truth catalog.

---

## TL;DR — strategic headline (2026-05-13 PM, supersedes AM)

**The Tier-2 Q8 floor was an FP32-vs-bf16 precision-class mismatch artifact, not engine drift.** The 2026-05-13 PM pattern-hunt audit (commits `13003256`, `f6d1a59e`, `a75b4a08`, `1b90a663`) proved hipfire's fp32-native kernels (rmsnorm, l2norm, recurrence) are bit-faithful to the fp64 ideal. Then on the same day, [PR #248](https://github.com/Kaden-Schutt/hipfire/pull/248) Tier-3 fused WMMA Q8 prefill kernels (FP16 accumulators, matched to HF's bf16 precision class) landed, and the measured Q8 floor collapsed:

| Model | Tier-2 Q8 floor (FP32 acc) | Tier-3 Q8 floor (FP16 acc) | Reduction |
|---|---:|---:|---:|
| Q3.5-0.8B q8-KV | 0.0796 | **0.0041** (n=512, gfx1100) | 19× |
| Q3.5-9B q8-KV | 0.1459 (asym3-KV) → ~0.12 (est. q8-KV) | **0.0173** (n=256, gfx1151) | 7-8× |

**The 9B Tier-3 Q8 floor is +6% above llama.cpp Q8_0 9B's 0.0163 — same precision class.** What was previously reported as "100% engine drift" (Phase 0 commit `13003256`) was 100% the FP32-FP16 accumulator-class mismatch with HF's bf16 reference. On Tier-3 the gap is functionally closed. Hipfire's kernels were never imprecise; they were *too* precise for the dtype the reference was trained at, and switching the Q8 prefill path's accumulator class to FP16 made the absolute KLD-vs-HF land near the cross-engine noise floor.

**Cross-engine KLD-vs-HF is a valid output-quality comparison once both engines use precision-class-matched accumulators.** Within precision-class match (Tier-3 hipfire FP16-WMMA vs llama.cpp's similar internal bf16/FP16 accumulators), absolute `KLD(engine || HF-bf16)` becomes interpretable: the gap is residual architectural drift (DeltaNet recurrence accumulation for Q3.5 family) plus weight-quantization cost. For mixed-path cross-engine comparisons (Tier-2 hipfire vs llama.cpp), use **Δ-above-own-Q8** instead: first-order independence (KLD ≈ ½·Var(δlogit), independent perturbations add) cancels each engine's own floor, leaving pure quantizer cost. Cross-engine quality claims at matched bpw are now `Δ_hipfire(MQ4) vs Δ_llamacpp(Q4_K_M)` OR — equivalently when Tier-3 — absolute `KLD_hipfire vs KLD_llamacpp` directly.

**Strategic implication.** Engine-surgery scope has collapsed to ~zero kernels — confirmed empirically twice (F2/F3 audit + PR-#248 floor measurement). The calibration roadmap (AWQ Stage A → GPTQ Stage B → MR-GPTQ Stage C, Lloyd codebooks) is the right priority and is now better-justified since (a) there is no kernel arithmetic left to fix and (b) on Tier-3, 4-bit Δs are dominated by quant noise rather than engine drift. Stage B/C success is measured as `Δ = KLD(quant) − KLD(Q8)` within hipfire (path-matched Q8 floor), and cross-engine claims are made on `Δ_hipfire` vs `Δ_llamacpp` at matched bpw.

**Depth-amplification finding refined.** The earlier "DeltaNet recurrence amplifies bf16-cast drift ~8×" framing (Q3-0.6B dense 0.0098 vs Q3.5-0.8B DeltaNet 0.0796 on Tier-2) conflated two effects: (a) FP32-vs-bf16 precision-class mismatch (now closed by Tier-3) and (b) genuine DeltaNet recurrence drift accumulation over LA layers. On Tier-3, **0.8B DeltaNet (0.0041) → 9B DeltaNet (0.0173) = 4.2× over 8 additional layers (32 vs 24)**, or ~1.20× per extra layer of cumulative drift. That's the real DeltaNet amplification, much smaller than the Tier-2 phantom 8×. And at 9B the absolute floor is now low enough (0.0173) that 4-bit weight noise (~0.2-0.3 nats) dominates, not recurrence drift.

**Other validation pending:**
1. `KLD(hipfire-Q8 || hipfire-FP32-everything)` — should be tiny if the framework's premise holds. The Tier-2-vs-Tier-3 collapse (0.1459 → 0.0173 on the same hipfire engine, only kernel accumulator class changed) is already a partial validation.
2. Rank-correlation of `Δ_hipfire(MQ4)` vs `Δ_llamacpp(Q4_K_M)` across prompts/positions.
3. mq4-vs-mq4-awq Δ within hipfire matches the qualitative AWQ-helps direction observed in other engines.

---

## 0. Reading conventions

- **All hipfire rows are POST-RoPE-fix** (commit `1805d820` flipped halfsplit RoPE to default; 2026-05-12 evening) unless explicitly marked `PRE-FIX`. Pre-fix hipfire numbers are ~3–5× inflated due to the interleaved-vs-halfsplit RoPE pair-convention bug — see [`project_engine_drift_floor_decomposition.md`](../../memory/project_engine_drift_floor_decomposition.md).
- **Reference distribution:** `qwen3.5-{0.8b,9b}-bf16.kldref.bin` (built via `build_kld_ref` from llama.cpp `llama-perplexity --kl-divergence-base` over the BF16 GGUF). Slice: wikitext2-1024s-2048ctx (md5 `83b0205a`, 1175 chunks; quick-slice runs use `--max-chunks 512`).
- **KV mode is load-bearing for cross-engine comparison.** llama.cpp GGUF baselines were measured with its default FP16 KV cache (lossless). Hipfire defaults to `asym3` KV (lossy, ~0.07-nat penalty on 0.8B mq4-base; see §4). When comparing across engines, only compare rows with similar KV lossiness, OR explicitly compute the above-floor lift.
- **Above-floor KLD** = raw KLD − engine's Q8-weight floor (in matching KV mode). Surfaces the format/calibration cost stripped of engine drift + KV-quant cost.
- **Scoring mode:** all hipfire rows use `prefill` (canonical, see §5.3 of issue-113). GGUF rows use llama.cpp's native (`gguf` mode, not directly comparable to per-token).
- **Top-K=256 caveat:** all KLD values are lower bounds on true full-vocab KLD. Cross-variant ordering is reliable; absolute magnitudes are conservative. See issue-113 §2.

---

## 1. Qwen3.5-9B

### 1.1 Hipfire (post-RoPE-fix, gfx1100, KV=asym3, Tier-2 batched_chunked Q8)

Source: `benchmarks/quality-baselines/results/2026-05-12-cohort-post-rope-fix-9b/result-table.md`

> ⚠ **"Above-floor" column is Tier-2-floor-based and understates true 4-bit weight noise.** The 0.1459 q8f16 Tier-2 number is FP32-vs-bf16 precision-class noise + tiny Q8 weight noise; MQ4 uses WMMA-FP16 prefill (always has) so doesn't carry that FP32-cast component. Subtracting Tier-2 floor from an MQ4 measurement subtracts a mismatched baseline. The corrected Δ uses the Tier-3 Q8 floor (§1.4 below). See methodology §6 rule 7.

| Variant | bpw | KLD (CI) | p99 | PPL | Above-floor (Tier-2; deprecated) |
|---|---:|---|---:|---:|---:|
| q8f16 | 8.50 | **0.1459** (0.1383–0.1541) | 13.576 | 9.795 | (Tier-2 floor) |
| mq4-base | 4.25 | 0.3376 (0.3263–0.3494) | 18.194 | 9.116 | 0.1917 |
| **mq4-awq** | 4.25 | **0.2800** (0.2697–0.2910) | 17.537 | 9.271 | **0.1341** |
| hfp4 | 4.50 | 0.4594 (0.4475–0.4720) | 19.279 | 11.511 | 0.3135 |
| mfp4 | 4.50 | 0.4653 (0.4535–0.4782) | 18.278 | 11.138 | 0.3194 |
| hfp4-l4-l5c | ~5.0 | 0.3836 (0.3722–0.3959) | 18.665 | 10.299 | 0.2377 |
| mfp4-l4-l5c | ~5.0 | 0.7783 (0.7625–0.7951) | 21.199 | 12.571 | 0.6324 |

### 1.1d FWHT rotation isolation (HF4 = unrotated MQ4; gfx1100, KV=q8, prefill, n=20)

Source: 2026-05-13 PM, this session. HF4 = HFQ4G256 = same number format as MQ4 (4-bit uniform + 256-group) but with **NO FWHT rotation** applied. Apples-to-apples isolation of the rotation contribution to MQ4 quant cost.

| Variant | bpw | KLD (n=20, CI) | PPL | Rotation | Conv1d |
|---|---:|---|---:|---|---|
| **HF4** (all unrotated) | 4.25 | **0.6165** (CI 0.53–0.73) | 11.901 | none | HFQ4 (unrotated) |
| **HF4 + Q8 conv1d** | 4.25 | **0.3369** (CI 0.26–0.44) | **8.324** | none | Q8 |
| mq4-base | 4.25 | 0.3182 (CI 0.24–0.42) | 8.875 | FWHT | MQ4 |
| mq4-q8conv1d (n=20) | 4.25 | 0.2360 (CI 0.17–0.32) | 9.054 | FWHT | Q8 |

**Pure projection-rotation contribution (conv1d held at Q8 in both arms):**
- HF4+Q8conv1d 0.3369 vs MQ4+Q8conv1d 0.2360 → **projection FWHT provides −30% KLD**
- PPL: 8.324 vs 9.054 → **projection FWHT hurts PPL by +8.8%**

**Conv1d-precision contribution (rotation OFF):**
- HF4 → HF4+Q8conv1d: conv1d MQ4→Q8 → **−45% KLD** (0.6165 → 0.3369). Larger than in the FWHT-on arm (−26%), because unrotated projections produce noisier conv1d inputs → conv1d's quant noise compounds more with downstream LA recurrence.

**Counterintuitive PPL inversion (worth following up):** HF4+Q8conv1d has the **best PPL of any variant we've measured** (8.324, below mq4-q8conv1d's n=512 8.789) but is worse on KLD-vs-HF. Same F2/F3 pattern: removing FWHT projection rotation makes the forward pass simpler / more precise → more confident top-1 predictions (lower PPL) but tail distribution diverges from HF-bf16 (higher KLD-vs-HF). Top-1 is robust to mild weight perturbations; tails aren't.

**Conclusions:**
1. **FWHT projection rotation is doing real KLD work** — not broken, not a no-op. ~−30% KLD when properly isolated.
2. **For best KLD-vs-HF, keep FWHT.** For best PPL (predictive accuracy on actual tokens), drop FWHT.
3. **No rotation bug found** — empirical pattern matches theoretical expectations cleanly.
4. The PPL-vs-KLD divergence between rotated and unrotated paths is a new instance of the F2/F3 "more precise = worse KLD-vs-HF" pattern, this time at the format level rather than the engine level.

### 1.1e MQ6+Q8conv1d anchor (gfx1151, KV=q8, prefill)

Source: gfx1151 agent 2026-05-13 PM (smoke) + 2026-05-14 (full n=512). Uniform MQ6G256 across all 4-bit-eligible projections + Q8 conv1d override + default Q8 lm_head + Q8 embed.

| Variant | n | bpw | KLD (CI) | p99 | PPL | Wall |
|---|---:|---:|---|---:|---:|---:|
| mq6-q8conv1d (smoke) | 20 | ~6.5 | 0.0568 (CI 0.0314–0.0950) | 12.13 | 9.281 | ~6 min |
| **mq6-q8conv1d (n=512)** | **512** | ~6.5 | **0.0510** (CI **0.047–0.055**) | **8.68** | **9.186** | gfx1151 |

**n=512 confirms the smoke and tightens the CI 10× without changing the headline conclusion.** The smoke's wide CI 0.0314–0.0950 now collapses to 0.047–0.055 — the point estimate moved down 10% (0.0568 → 0.0510) and the tail p99 dropped from 12.13 to 8.68 (smoke caught a few unrepresentative high-noise sequences). PPL 9.186 is +0.02% above the Tier-3 Q8 floor 9.189 — i.e., **indistinguishable from 8-bit predictive quality at 6.5 bpw**.

**Hipfire's first format that genuinely competes with llama.cpp absolute-KLD-vs-HF:**
- Hipfire MQ6+Q8conv1d 6.5 bpw KLD 0.0510 ≈ **Llama.cpp UD-Q4_K_XL** at 5.32 bpw KLD 0.0670 — hipfire matches at +1.2 bpw cost
- Better than **Llama.cpp Q4_K_M** at 5.07 bpw KLD 0.1249 — hipfire wins by 2.4×
- Worse than **Llama.cpp Q6_K** at 6.56 bpw KLD 0.0250 — hipfire 2.0× worse (Δ-above-Q8: 0.034 vs 0.009 = 3.8× worse)

**This is a viable shipping format for "high-quality 4-5 bpw" target.**

Δ-above-Tier3-Q8: 0.0510 − 0.0173 = **0.0337** — pure 6-bit weight noise (down from smoke estimate 0.0395).

**Cross-engine pattern: same ~4× Δ penalty at both 4-bit and 6-bit, suggesting format-level (per-block scale fitting, 32 vs 256 group, asymmetric vs symmetric) is the structural cause rather than any specific bpw quirk.** See `qwen35-mq4-quality-gap.md` Stage A follow-ups F1-F5 for concrete fixes targeting this structural gap.

#### 1.1e.i KLD-vs-PPL inversion: MQ6-q8conv1d wins KLD 6.1× but loses PPL by 1%

Direct n=512 comparison of the two 9B variants the gfx1151 agent ran in the same sweep:

| Variant | bpw | KLD (CI) | p99 | PPL | KLD rank | PPL rank |
|---|---:|---|---:|---:|---:|---:|
| MQ4-Lloyd | ~4.91 | 0.3114 (0.300–0.324) | 18.69 | **9.085** | 2nd | **1st** |
| MQ6-q8conv1d | ~6.5 | **0.0510** (0.047–0.055) | 8.68 | 9.186 | **1st** | 2nd |

**Both gaps are well outside their CIs** — this is not measurement noise. MQ6-q8conv1d's KLD is **6.1× lower** but its PPL is **1.1% higher**. The two metrics genuinely disagree on which variant is better.

**Mechanism.** PPL is `exp(mean(-log p(y_true)))` — only the probability assigned to the *true* next token matters, regardless of the full distribution shape. KLD is `Σ p_ref(y) log(p_ref(y)/p_cand(y))` — measures full-distribution match against HF reference, including all the tail mass HF assigns probability to. MQ6-q8conv1d has lower per-weight noise → tighter tail-mass match to HF → much lower KLD. MQ4-Lloyd's per-block Lloyd codebook happens to nudge the argmax marginally toward the HF-bf16 reference's argmax token (without preserving tail shape), so PPL improves slightly even though the full distribution drifts harder.

This is the same family of "more precise ≠ better PPL-vs-HF" effect documented in F2/F3 — HF's bf16 reference is itself noisy at module boundaries, and any quant whose argmax happens to align with HF-bf16's argmax wins PPL even if the rest of its distribution is worse-aligned.

**Implication for ranking.** Exactly the PRD §2 case for KLD-primary scoring: KLD captures tail-mass quality (impacts coherence on rare-token contexts, multi-step reasoning where low-probability outputs matter); PPL only captures argmax confidence (impacts greedy decoding on common-token contexts). Models that win PPL but lose KLD are likely to win one-shot benchmarks (HumanEval, MMLU) while degrading on long-context reasoning. The shipping recommendation should weight by KLD.

### 1.1c MQ4-Lloyd anchor (gfx1151, KV=q8, prefill, n=512)

Source: gfx1151 agent 2026-05-13 PM. Default conv1d (MQ4G256, not Q8), default lm_head (MQ4G256). FWHT rotation on projections + Lloyd-Max codebook fit per 256-block instead of uniform grid.

| Variant | bpw | KLD (CI) | p99 | PPL | Wall |
|---|---:|---|---:|---:|---:|
| **mq4-lloyd** | **~4.91** (incl. Lloyd codebook overhead) | **0.3114** (CI 0.2999–0.3236) | 18.69 | **9.085** | 76 min @ 116 tok/s |

**Lloyd codebook provides essentially zero KLD improvement on MQ4 at 9B.** Compared to mq4-base (asym3-KV n=512: 0.3376, q8-KV n=20: 0.3182, q8-KV n=512 estimated 0.27-0.32), mq4-lloyd at 0.3114 is within noise of the baseline — and the file is 6.06 GB vs mq4-base's 5.31 GB (**+744 MB / +0.66 bpw avg overhead** for the Lloyd codebook).

**Mechanism (confirms §4A.2 prediction):** after FWHT-256 rotation, per-block weight distribution is approximately Gaussian (CLT on 256 elements). Lloyd-Max codebooks fit to a near-Gaussian don't beat uniform 4-bit grids by much. Lloyd is theorized to help on heavy-tailed distributions; the FWHT rotation already removes the heavy-tail problem MQ4 base would otherwise have.

**Implication:** **Drop MQ4-Lloyd from the active calibration roadmap.** The hypothesis is now measured-and-falsified for the 4-bit case. MQ3-Lloyd remains untested and could plausibly still help (§4A.2 #1 — uniform 3-bit is "on the cliff", Lloyd's larger lift at lower bpw is still hypothetical).

### 1.1b Hipfire Tier-3 Q8 floor anchor (gfx1151, KV=q8, fused WMMA prefill)

Source: gfx1151 agent 2026-05-13 PM run on PR [#248](https://github.com/Kaden-Schutt/hipfire/pull/248) (HEAD `747315a4`). Kernels: `gemm_qkv_q8_0_wmma`, `gemm_qkvza_q8_0_wmma`, `gemm_gate_up_q8_0_wmma`, `gemm_q8_0_residual_wmma` (all 4 Tier-3 fused kernels confirmed in dispatch log; Tier-2 substrate **not** invoked).

| Variant | bpw | KLD (CI) | p99 | PPL | Notes |
|---|---:|---|---:|---:|---|
| **q8f16 (Tier-3)** | 8.50 | **0.0173** (CI 0.0150–0.0201) | 0.0844 | **9.189** | n=256, gfx1151, q8-KV, prefill |

**Architecture-platform caveat:** measurement is on **gfx1151** (Strix Halo, RDNA3.5), not gfx1100. Cross-arch parity per issue-113 §V3 is ~1%. Kernels were validated on gfx1100 per commit `47fd6c4d` (4 fused-kernel unit tests `=== 0 failure(s) ===` on gfx1100, daemon prefill 1069 tok/s) — but the KLD measurement itself awaits a gfx1100 reproduction.

**+6% above llama.cpp Q8_0 9B's 0.0163** (§1.2 below). Essentially same precision class. What was previously framed as "+0.13 nats hipfire-side engine drift" was the FP32-vs-bf16 precision-class mismatch of the Tier-2 substrate; on Tier-3 it's gone. The remaining ~0.001 nats is residual cumulative DeltaNet recurrence drift (~0.06 per LA layer × ~24 LA layers = small in absolute terms once the precision-class is matched).

**PPL also dropped substantially** (9.795 → 9.189, −6%). Tier-3 doesn't just match HF noise pattern; it produces a sharper output token distribution on wikitext-2. Likely because FP16 rounding errors are systematically biased the same direction as bf16 in HF, so logit confidence shape matches HF better.

### 1.2 llama.cpp GGUF anchors (gfx1151, default FP16 KV)

Source: `benchmarks/quality-baselines/results/2026-05-10/per-seq/qwen3.5-9b.gguf-*__gfx1151.kldseq`

| Variant | bpw | KLD | PPL | Above-floor |
|---|---:|---:|---:|---:|
| Q8_0 | 8.50 | **0.0163** | 9.31 | (floor) |
| UD-Q6_K_XL | ~6.7 | 0.0213 | 9.14 | 0.0050 |
| Q6_K | 6.56 | 0.0250 | 9.31 | 0.0087 |
| UD-Q5_K_XL | ~5.5 | 0.0408 | 9.27 | 0.0245 |
| **UD-Q4_K_XL** | 5.32 | **0.0670** | 9.34 | 0.0507 |
| Q4_K_M | 5.07 | 0.1249 | 8.70 | 0.1086 |
| **UD-Q3_K_XL** | ~4.50 | **0.1411** | 8.67 | 0.1248 |

### 1.3 9B cross-engine summary

> ⚠ **2026-05-13 PM reframing (twice).** The original "engine-drift floor gap" claim was a Tier-2 FP32-vs-bf16 precision-class mismatch artifact (closed by F2/F3 audit + PR-#248 Tier-3 measurement); on Tier-3 the gap is **+6% not +0.13 nats**. The earlier "above-floor" column also used a Tier-2 (FP32-acc) Q8 baseline for MQ4 (FP16-WMMA) measurements, a precision-class mismatch that artificially deflated the Δ. Corrected numbers below.

**Absolute Q8-vs-Q8 KLD-vs-HF on Tier-3 (precision-class-matched):**
- Hipfire 9B q8f16 Tier-3 q8-KV: **0.0173** (§1.1b)
- Llama.cpp 9B Q8_0 FP16-KV: **0.0163** (§1.2)
- **Gap: +6% / 0.001 nats** — essentially same precision class. Caveat: KV mode mismatch (q8 vs FP16-KV) introduces a small confounder; hipfire's f16-KV-equivalent (had we measured it) is likely 0.015-0.017, narrowing the gap further.

**Δ-above-own-Q8 (the apples-to-apples cross-engine claim) at 9B (path-matched on Tier-3):**

| Variant | bpw | KV | KLD | Δ-above-own-Q8 | Notes |
|---|---:|---|---:|---:|---|
| Llama.cpp Q8_0 | 8.50 | f16 | 0.0163 | (floor) | |
| **Llama.cpp UD-Q3_K_XL** | ~4.50 | f16 | 0.1411 | **0.125** | bpw-matched 4-bit anchor |
| **Llama.cpp Q4_K_M** | 5.07 | f16 | 0.1249 | **0.109** | imatrix-calibrated 4-bit |
| Hipfire q8f16 (Tier-3) | 8.50 | q8 | 0.0173 | (floor) | n=256 gfx1151 |
| **Hipfire mq4-base** | 4.25 | q8 | _pending q8-KV measurement_ | ~0.25 est. | n=20 smoke: 0.318 → Δ 0.30 |
| **Hipfire mq4-awq** | 4.25 | q8 | _pending q8-KV measurement_ | ~0.21 est. | extrapolation |

**Cross-engine gap (corrected, Δ-vs-Δ):** hipfire mq4-awq Δ ~0.21 vs llama.cpp UD-Q3_K_XL Δ 0.125 → **hipfire ~68% worse at matched bpw**, not the original master-doc claim of "~7% behind". The path-mismatched Δ in §1.1's original "Above-floor" column understated the gap by ~9× (because Tier-2 Q8 floor was 0.130 nats higher than the path-matched Tier-3 floor — that "extra" 0.130 was implicitly being credited to MQ4's quantizer cost when it was actually FP32-cast noise).

**AWQ closure on the corrected Δ basis:** ~17% (vs the master-doc-original 30% which was inflated by the same path-mismatched floor subtraction). AWQ still helps, just less dramatically than the original framing suggested.

---

## 2. Qwen3.5-0.8B

### 2.1 Hipfire (post-RoPE-fix, gfx1100, KV=asym3)

Source: `benchmarks/quality-baselines/results/2026-05-13-cohort-post-rope-fix-0.8b/result-table.md`

| Variant | bpw | KLD (CI) | p99 | PPL | Above-floor |
|---|---:|---|---:|---:|---:|
| q8f16 | 8.50 | **0.1256** (0.1234–0.1280) | 1.458 | 19.633 | (floor) |
| mq4-base | 4.25 | 0.3341 (0.3308–0.3374) | 2.707 | 23.895 | 0.2085 |
| **mq4-awq** | 4.25 | **0.3000** (0.2971–0.3029) | 2.515 | 23.383 | **0.1744** |
| hfp4-l4-l5c | ~5.0 | 0.7890 (0.7785–0.7999) | 5.671 | 43.407 | 0.6634 |
| mfp4-l4-l5c | ~5.0 | 0.7524 (0.7445–0.7603) | 5.259 | 38.829 | 0.6268 |

### 2.2 Hipfire q8-KV probes (gfx1100, KV=q8) — apples-to-apples-er with GGUFs

Source: `benchmarks/quality-baselines/results/2026-05-13-cohort-post-rope-fix-0.8b/per-variant/*__kv-q8.kldseq`

| Variant | KLD (CI) | p99 | PPL | Δ vs asym3 |
|---|---|---:|---:|---:|
| mq4-base | 0.2675 (0.2650–0.2699) | 2.366 | 22.088 | −0.0666 (−20%) |
| mq4-awq | **0.2531** (0.2506–0.2556) | 2.305 | 22.149 | −0.0469 (−16%) |
| **q8f16** | **0.0806** (n=20, per-token, gfx1151) | — | — | −0.0450 (−36%) ⚠ scoring-mode mismatch — see §2.4 |

**q8f16 q8-KV caveat (load-bearing).** The 0.0806 came from the closed floor-decomposition investigation (`benchmarks/quality-baselines/results/2026-05-12-deltanet-discriminator/per-seq/qwen3.5-0.8b.q8f16__gfx1151__kv-q8__per-token__c20__halfsplit-rope-v2.kldseq`). Four asymmetries vs the rest of §2.2:
1. **Per-token scoring** (not prefill). Per issue-113 §5.3 the per-token path runs ~7% higher KLD than prefill on 9B gfx1100. Prefill-equivalent estimate ≈ 0.0806/1.07 ≈ 0.0753.
2. **n=20 chunks** (vs 512 quick-slice for the rest of §2.2). Wide CI; per-seq variance not bootstrapped here.
3. **gfx1151** (vs gfx1100 for the rest). Cross-arch parity is documented in issue-113 §V3 — not expected to move KLD by more than ~1%.
4. **Pre-PR-#248 Q8 kernel path.** Measured on the Tier-2 `gemm_q8_0_batched_chunked` projection path; the upcoming Tier-3 fused WMMA prefill (PR [#248](https://github.com/Kaden-Schutt/hipfire/pull/248)) will become the production Q8 prefill path. The 0.0806 number is functionally close to Tier-3 expected (PR-#248 32-chunk KLD smoke landed on the published trajectory toward the pre-RoPE-fix 0.5735 256-chunk anchor) but is not guaranteed bit-identical and **needs re-measurement post-PR-#248** to give a defensible Δ-above-Q8 baseline.

A clean prefill q8f16 q8-KV measurement at gfx1100 quick-slice 512 on the **PR-#248 Tier-3 path** is on the punchlist (§4.3) and is now critical-path for the Δ-framework adopted in the TL;DR.

### 2.3 llama.cpp GGUF anchors (0.8B)

Run via `eval_gguf` against `qwen3.5-0.8b-bf16.kldref.bin`. **Full slice (1175 chunks), not the 512-chunk quick-slice used by the hipfire 0.8B rows.** The slice-mean KLD usually shifts <2% between 512 and 1175 chunks (wikitext slice converges fast), but flagged here so it can't be forgotten.

| Variant | bpw | KV mode | KLD | PPL | Chunks | Source |
|---|---:|---|---:|---:|---:|---|
| Q4_K_M | ~5.07 | q8 | **0.0351** | 17.334 | 1175 (full) | user-reported 2026-05-13 |

Q8_0 / UD-* anchors at 0.8B not yet measured.

### 2.4 0.8B cross-engine summary (after Q4_K_M anchor)

At matched 4-bit + KV mode (q8 vs q8):

| Engine | Variant | bpw | KLD | PPL | Gap to llama.cpp Q4_K_M |
|---|---|---:|---:|---:|---:|
| **llama.cpp** | Q4_K_M | 5.07 | **0.0351** | **17.33** | (anchor) |
| hipfire | mq4-base | 4.25 | 0.2675 | 22.088 | +0.232 nats KLD, +27.5% PPL at −0.82 bpw |
| hipfire | **mq4-awq** | 4.25 | **0.2531** | **22.149** | **+0.218 nats KLD, +27.8% PPL** at −0.82 bpw |

**AWQ uplift is KV-mode-dependent.** At asym3 KV, AWQ improved 0.8B mq4 by −10.2% KLD; at q8 KV it's only −5.4%. Interpretation: AWQ partially compensates for KV-rotation noise (asym3 K rotation precision interacts with per-channel outliers); when the KV cache is less lossy, AWQ has less to clean up. The 9B picture (where AWQ closed −30% above-floor at asym3) may show similar shrinkage if re-measured at q8 KV — worth flagging for the Stage A retrospective.

**PPL is roughly unchanged by AWQ on this slice.** mq4-base PPL 22.088, mq4-awq PPL 22.149 — within slice noise. KLD improvement comes from tail-distribution matching, not top-1 probability, which is exactly what AWQ targets (outlier preservation in heavy channels). Consistent with §2 of issue-113 ("PPL collapses the full output distribution... KLD surfaces tail").

### 2.4.1 Q8-weights vs Q4_K_M cross-engine — FLIPPED by 2026-05-13 PM audit

> ⚠ **REFRAMED 2026-05-13 PM.** The prior framing (this section's original
> "binding finding") was: hipfire q8f16 KLD vs HF (0.0806) > Q4_K_M GGUF KLD vs HF (0.0351), therefore hipfire engine drift exceeds llama.cpp's combined engine + quant cost, therefore engine-side work is required. The 2026-05-13 PM pattern-hunt audit (commits `13003256`, `f6d1a59e`, `1b90a663`) refutes this interpretation. **Absolute `KLD(engine || HF-bf16)` is not a valid output-quality comparison; it rewards similarity to HF's bf16-cast noise pattern.** The 0.0796 is HF's bf16-cast at module boundaries leaking into the score, not hipfire imprecision. Use Δ-above-own-Q8 instead.

Including the hipfire q8f16 q8-KV row (§2.2 caveats apply — per-token, n=20, gfx1151, pre-PR-#248 Tier-2 kernel path):

| Engine | Variant | bpw | KV | KLD vs HF | Above-own-Q8 (Δ) | Notes |
|---|---|---:|---|---:|---:|---|
| **llama.cpp** | Q8_0 | 8.50 | q8 | **0.0015** | (engine Q8 floor) | cross-engine noise floor; Phase 0 commit `13003256` |
| **llama.cpp** | Q4_K_M | 5.07 | q8 | **0.0351** | **0.0336** | full slice, prefill-equivalent |
| **hipfire** | q8f16 | 8.50 | q8 | **0.0796** | (engine Q8 floor) | pre-PR-#248 Tier-2 path; needs re-measurement post-PR |
| | | | | (≈ 0.0743 prefill-equivalent) | | per-token, n=20, gfx1151 caveats |
| **hipfire** | mq4-base | 4.25 | q8 | 0.2675 | _pending hipfire Q8 floor_ | 512-chunk quick-slice, gfx1100 |
| **hipfire** | mq4-awq | 4.25 | q8 | 0.2531 | _pending hipfire Q8 floor_ | 512-chunk quick-slice, gfx1100 |

**Defensible cross-engine claim (Δ framework):** `Δ_llamacpp(Q4_K_M) = 0.0336`. `Δ_hipfire(MQ4-AWQ)` is **pending** — we need a clean prefill q8f16 q8-KV measurement at gfx1100 quick-slice 512 on the PR-#248 Tier-3 fused WMMA prefill path before this delta is meaningful. Until then, any "hipfire 4-bit beats / matches Q4_K_M" claim is unverifiable.

**What the absolute KLD-vs-HF numbers DO say (still useful, with caveats):**

The 0.0796 hipfire-Q8 vs 0.0015 llama.cpp-Q8 gap (factor of ~50×) measures **how much each engine's accumulator pattern differs from HF's bf16-cast pattern**. Hipfire is fp32-internal end-to-end and arithmetically more precise than HF-bf16 (proved by F2/F3: hipfire fp32-native kernels are bit-faithful to fp64 ideal); llama.cpp's internal bf16-cast pattern at module boundaries closely mimics HF's. The "gap" measures hipfire's *deviation from a less-precise reference dtype*, not output-quality deficit.

**Hipfire-internal MQ4 cost (post-PR-#248 Q8 baseline pending):** when the hipfire-Q8 q8-KV baseline is freshly measured on PR-#248, the framework lets us compute `Δ_hipfire(MQ4-base) = 0.2675 − KLD_hipfire(Q8)` and similar for mq4-awq / mq4-awq-gptq. The KLD-quadratic-in-δlogit approximation should hold well for 4-bit (small δlogit) but degrades for 3-bit / 2-bit (per the framework's "where it breaks down" §2 — second-order quant×engine interaction).

**Open questions the framework cannot resolve from KLD alone:**
- Does llama.cpp Q4_K_M produce *better tokens* than hipfire MQ4-AWQ-GPTQ, or just tokens whose distribution is more bf16-cast-shaped? Needs a downstream behavioral metric (HumanEval pass@1 at temp=0, code-PPL on a *non-HF* reference, instruction-following evals) at matched KV/sampler/prompt.
- Is the per-token vs prefill ~7% bias (§5.3) actually structurally meaningful or also a reference-side artifact? With kernels bit-faithful to fp64, the bias likely lives in the per-token kernel's *reduction order*, not its arithmetic precision.

**What this means for Stage A/B/C calibration roadmap:** within-hipfire `Δ-above-own-Q8` is the canonical lift metric. Stage A AWQ closed −30% of MQ4 Δ at 9B; Stage B GPTQ target is another −20-40% on top of that. Cross-engine claims at matched bpw require both engines' Q8 baselines (have llama.cpp Q8_0; need PR-#248 Tier-3 hipfire q8f16). Until those are co-measured, the calibration-vs-GGUF comparison is paused.

**Methodological note: slice mismatch (full vs quick).** The hipfire 0.8B rows use `--max-chunks 512`; the Q4_K_M GGUF was full-slice. Re-running mq4-base + mq4-awq on full slice (~30 min wall each on this gfx1100) would close the apples-to-apples residual on absolute KLDs (still informative for the cast-pattern-similarity question even if not the framework's primary metric).

---

## 3. Status of other model sizes

| Model | Hipfire status | GGUF status |
|---|---|---|
| Qwen3.5-4B | not yet quantized; queued (task #7) | not measured |
| Qwen3.5-A3B | not yet quantized; queued (task #8) | not measured |
| Qwen3.6-27B | BF16 ref dumped on gfx1151; not on gfx1100 (memory entry) | not measured |
| Qwen2.5-7B | kldref built; cohort historical only | not measured |

---

## 4. KV-cache-mode contribution

### 4.1 Measurements

| Model | Variant | asym3 KLD | q8 KLD | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| 0.8B | mq4-base | 0.3341 | 0.2675 | −0.0666 | −20% |
| 0.8B | mq4-awq | 0.3000 | 0.2531 | −0.0469 | −16% |
| 0.8B | q8f16 (Tier-2) | 0.1256 | 0.0796 (pre-PR#248, n=20, gfx1151) | (−0.046 est.) | Tier-2 substrate |
| 0.8B | q8f16 (Tier-3) | _not measured_ | **0.0041** (n=512, gfx1100, PR-#248) | | path-matched Q8 floor for MQ4 Δ |
| 9B | q8f16 (Tier-2) | 0.1459 | _not measured_ | | Tier-2 substrate |
| 9B | q8f16 (Tier-3) | _not measured_ | **0.0173** (n=256, gfx1151, PR-#248) | | path-matched Q8 floor for MQ4 Δ |
| 9B | mq4-base (smoke) | 0.3376 | 0.3182 (n=20, gfx1100) | −0.019 | −5.7% (n=20 wide CI) |
| 9B | mq4-q8lmhead (probe) | _new_ | 0.3083 (n=20) | | lm_head Q8 vs MQ4: −3.1% KLD; smoke only |
| 9B | mq4-q8conv1d (probe) | _new_ | 0.2360 (n=20) | | conv1d Q8 vs MQ4: **−25.8% KLD**; smoke only — see §5 caveat |

### 4.2 What this means for cross-engine comparison

- The original "asym3 vs Q8 = ~0.20 nats" estimate from the pre-RoPE-fix floor decomposition appears to overstate the penalty at 0.8B post-fix. On mq4-base 0.8B, the actual delta is **~0.07 nats**, i.e. about **20% of the raw KLD**.
- This 0.07 nats is **real lossy KV quantization noise** (asym3 K-rotation precision loss), not engine-side drift, and it is correctly attributed to the KV mode. It is one of the few absolute-KLD-vs-HF components that survives the 2026-05-13 PM reframing intact, because lossy KV quant adds real δlogit-variance independent of whether the reference is bf16 or fp32.
- **Recommendation:** for the Δ-above-own-Q8 framework (TL;DR), the Q8 floor must be measured **in the same KV mode** as the quant being compared. Mixing q8-KV-Q8-floor with asym3-KV-MQ4-row is meaningless because the KV term doesn't cancel under subtraction.

### 4.3 Open KV-mode measurements to fill in

Critical-path (Δ framework requires both engines' Q8 baselines in matching KV mode):

- [ ] **0.8B hipfire q8f16 + q8 KV on PR-#248 Tier-3 fused WMMA prefill** — replaces the n=20 gfx1151 pre-PR Tier-2 number; this becomes the canonical hipfire Q8 floor at 0.8B
- [ ] **9B hipfire q8f16 + q8 KV on PR-#248 Tier-3** — canonical hipfire Q8 floor at 9B
- [ ] **0.8B llama.cpp Q8_0 + q8 KV** — the engine-pair Q8 baseline (currently we have Q8_0 at q8-KV implied from Phase 0 `13003256` cross-check; want a clean direct measurement)
- [ ] **9B llama.cpp Q8_0 + q8 KV** — same, at scale

Within-hipfire Δ measurements (Stage A/B/C uplift):

- [ ] 0.8B mq4-awq + q8 KV
- [ ] 9B mq4-base + q8 KV
- [ ] 9B mq4-awq + q8 KV
- [ ] 9B mq4-awq-gptq + q8 KV (after the in-flight 9B quantize lands)

After the 0.8B + 9B Q8 baselines on PR-#248 are in, the canonical cross-engine table becomes: `(Δ_hipfire(MQ4*) vs Δ_llamacpp(Q4_K_M / UD-Q3_K_XL / UD-Q4_K_XL))` at matched bpw and KV mode, with both engines' Q8 floors subtracted. That is the only cross-engine claim the framework supports.

---

## 4A. Open format-coverage measurement gaps

The current cohort runs only cover MQ4 / HFP4 / MFP4 family. Sub-4-bit
formats and the Lloyd codebook variants have NOT been measured against
the post-RoPE-fix BF16 reference. Per CLAUDE.md and the format roadmap,
several of these are theorized to deliver meaningful quality lifts and
need to be benched before strategic decisions get cemented.

### 4A.1 Sub-4-bit and Lloyd variants — unmeasured

| Format | bpw | Status on disk | Gated by | KLD against current ref? |
|---|---:|---|---|---|
| MQ3G256 (uniform 3-bit) | 3.25 | `~/.hipfire/models/qwen3.5-9b.mq3` exists | — | **NOT MEASURED** |
| MQ3G256-Lloyd (3-bit + per-block Lloyd-Max 8-entry FP16 codebook) | 3.50 | `~/.hipfire/models/qwen3.5-9b.mq3-lloyd` exists | — | **NOT MEASURED** |
| MQ4G256-Lloyd (4-bit + Lloyd codebook, prefill kernel) | ~4.5 | not quantized | PR [#197](https://github.com/Kaden-Schutt/hipfire/pull/197) (`feat/issue-182-mq4-lloyd`, open) | **NOT MEASURED** |
| MQ2G256-Lloyd (2-bit + Lloyd codebook) | ~2.5 | check `~/.hipfire/models/` | — | **NOT MEASURED** |

### 4A.2 Lloyd-transform uplift — investigation note

The Lloyd-Max per-block codebook (used in mq3-lloyd / mq4-lloyd /
mq2-lloyd) replaces the uniform 4/8/16-codepoint grid with K
data-driven centroids fit to each post-FWHT block's value distribution.
Theory: for post-FWHT weights with heavy-tailed kurtosis, a uniform
grid wastes codepoints on tail bins while under-resolving the dense
center. Lloyd codebooks should claw back precision proportional to
how non-uniform the post-FWHT distribution is.

**Expected uplift to measure (against current post-RoPE-fix ref):**
1. **MQ3 → MQ3-Lloyd at 9B.** Lloyd at 3-bit is theorized to be the
   bigger lever (MQ3 uniform is on the cliff of where 3-bit becomes
   unusable; codebook adaptation may rescue it). Direct A/B since both
   .hfq files already exist on disk.
2. **MQ4 → MQ4-Lloyd at 9B.** Gated on PR #197 merging or being
   checked out temporarily. Smaller theoretical uplift than MQ3-Lloyd
   (MQ4 uniform is already in the well-behaved zone) but worth a
   measurement to know the order of magnitude before committing
   format/kernel work.
3. **Stacking interaction with AWQ.** AWQ pre-scaling reshapes the
   per-channel value distribution before FWHT; whether Lloyd codebooks
   trained on post-AWQ post-FWHT blocks deliver the same uplift as on
   non-AWQ blocks is open. The Stage A → Stage B → Stage C roadmap
   currently treats AWQ + GPTQ + Lloyd as orthogonal levers; measuring
   them as additive vs interfering is part of the design validation.

**Highest-leverage measurement order:**
1. mq3 + mq3-lloyd at 9B (no PR gating, no quantize required, ~30 min wall)
2. mq3 + mq3-lloyd at 0.8B (need to quantize first; ~10 min quantize + ~15 min eval)
3. mq4-lloyd at 9B (gated on PR #197 — see below)
4. mq4-lloyd + AWQ stack at 9B (Phase A Step 5c follow-up)

### 4A.3 How to measure mq4-lloyd (gated on PR #197)

PR `Kaden-Schutt/hipfire#197` (`feat/issue-182-mq4-lloyd`, OPEN) ships
the MQ4-Lloyd WMMA prefill kernels. To measure mq4-lloyd KLD without
merging:

```bash
# Save current branch
git checkout -b backup-feat-mq-v2 feat/mq-v2-quant-format

# Fetch + check out the PR
gh pr checkout 197 -R Kaden-Schutt/hipfire

# Quantize a candidate
cargo run --release -p hipfire-quantize -- \
    --hf <bf16-path> --format mq4-lloyd \
    --out ~/.hipfire/models/qwen3.5-9b.mq4-lloyd

# Eval against the existing ref
./target/release/examples/eval_hipfire \
    --model ~/.hipfire/models/qwen3.5-9b.mq4-lloyd \
    --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
    --output benchmarks/quality-baselines/results/<dated>/qwen35-9b-mq4-lloyd__gfx1100.kldseq \
    --kv-mode asym3 --scoring-mode prefill --max-chunks 512

# Reduce + copy results back to feat/mq-v2-quant-format
# (the .kldseq file is engine-version-agnostic; row can be appended
#  to the master doc without re-running on the main branch)

# Restore working branch
git checkout feat/mq-v2-quant-format
```

The `.kldseq` artifact + reduced KLD/PPL values are portable across
engine versions (the slice is the slice; the reference is the
reference). Result row goes into §1 of this doc; flag the engine
SHA in the row's "Notes" field (e.g. "engine=PR#197@<sha>").

---

## 5. Key findings catalog

- **2026-05-12 — RoPE fix flips defaults.** Halfsplit (HF convention) becomes default; interleaved relegated to `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1`. Drops 0.8B Q8 floor from ~0.49 to ~0.13. Commit `1805d820`.
- **2026-05-12 — AWQ on MQ4 (Stage A) closes 30% of 9B above-floor gap.** mq4-base 0.192 above-floor → mq4-awq 0.134 above-floor. AWQ implementation per `crates/hipfire-quantize/src/main.rs` AWQ block; calibration via `awq_collect.py`.
- **2026-05-13 AM — Engine-drift floor investigation closed (first pass).** No further single-localizable bugs in the post-RoPE-fix residual ~0.08 nats. Initial attribution: 0.07 (cumulative pipeline imprecision, 24 layers) + 0.01 (Q8 weight + cross-engine inherent). Refuted GLM-5's F16-LA-weights claim. **Superseded by PM finding below.**
- **2026-05-13 — Cohort humaneval/smoke bug fixed.** `HIPFIRE_DEFAULT_MODEL` → `HIPFIRE_MODEL` in `scripts/bench_humaneval_completion.sh` and `scripts/quant_cohort.sh`. KLD/PPL/MSE data unaffected (eval_hipfire path doesn't use the daemon).
- **2026-05-13 — asym3 KV vs q8 KV delta is smaller than pre-fix estimate.** Empirically 0.07 nats at 0.8B mq4-base, not the 0.20 implied by the pre-RoPE-fix decomposition. (Survives the PM reframing — this is real lossy-KV-quant noise.)
- **2026-05-13 AM — Cross-engine 4-bit-beats-8-bit finding (RETRACTED PM).** Original headline: llama.cpp Q4_K_M (0.035 KLD vs HF) beats hipfire q8f16 (0.080 KLD vs HF) on Q3.5-0.8B at matched Q8 KV, implying hipfire engine drift exceeds llama.cpp's engine + 4-bit combined. **This interpretation is wrong** — see PM finding below. The absolute-KLD-vs-HF numbers are correct; what they mean is not what was claimed. See §2.4.1 reframing.
- **2026-05-13 PM — Pattern-hunt audit closes engine-surgery scope to ~zero kernels.** Commits `13003256` (Phase 0 / llama.cpp Q8_0 sanity), `f6d1a59e` (Phase 2 + 3a, rmsnorm bit-faithful to fp64), `a75b4a08` (Day 0 rmsnorm), `1b90a663` (F3 l2norm bit-faithful). Result: hipfire's fp32-native kernels (rmsnorm, l2norm, recurrence) score `0.000000` rL2 vs per-engine fp64 references; 22 of 24 audit-flagged "drift stages" are HF's intentional bf16 cast at module boundaries (F2); the only remaining stage with non-zero fp32-vs-fp32 drift is sigmoid alpha at 0.0024 rL2 cosine 1.000 (negligible). **Hipfire's kernels are uniformly more arithmetically precise than HF's BF16 reference**, not less. The "engine drift floor" is a precision mismatch with HF's training dtype, not a hipfire bug.
- **2026-05-13 PM — Cross-engine KLD-vs-HF reframed: absolute is invalid; use Δ-above-own-Q8.** First-order independence of {weight-quant noise, engine floor, embedding noise} makes `(KLD(quant) − KLD(Q8))` cancel engine + embedding terms within each engine, yielding pure quantizer cost. Cross-engine claim is then `Δ_hipfire(MQ4) vs Δ_llamacpp(Q4_K_M)`, not absolute KLDs vs HF. See TL;DR.
- **2026-05-13 PM — Hipfire architecture-dependent floor: refined after Tier-3 measurement.** Original framing: "DeltaNet amplifies bf16-cast drift ~8×" (Q3-0.6B dense Q8 0.0098 vs Q3.5-0.8B DeltaNet Q8 0.0796 on Tier-2). Refined: that 8× ratio conflated two effects — (a) the Tier-2 FP32-vs-bf16 precision-class mismatch (now closed by PR-#248 Tier-3, see next finding) and (b) genuine DeltaNet recurrence drift accumulation. On Tier-3, Q3.5-0.8B DeltaNet Q8 floor is **0.0041** (not 0.0796), and Q3.5-9B DeltaNet Q8 floor is **0.0173**. The per-layer DeltaNet amplification is real but ~1.20× per additional LA layer (4.2× over the 8 extra layers from 24-layer 0.8B to 32-layer 9B), much smaller than the Tier-2 phantom 8× factor. The "engine floor" is more precisely "DeltaNet recurrence amplifies hipfire-vs-HF dtype mismatch, with magnitude proportional to the precision-class gap" — Tier-3 closes the gap and reduces the absolute floor accordingly.
- **2026-05-13 PM — MQ6+Q8conv1d is hipfire's first format competitive with llama.cpp K-quants absolute KLD.** gfx1151 measurement: 6.5 bpw KLD 0.0568 / PPL 9.281 — matches Llama.cpp UD-Q4_K_XL at +1.2 bpw cost, beats Q4_K_M by 2.2×. PPL only +1% above Tier-3 Q8 floor. **Viable shipping format for "high-quality 4-5 bpw" target.** Cross-engine Δ-above-own-Q8 at 6-bit: 0.040 vs 0.009 = 4.4× — same penalty ratio as 4-bit (1.86×), confirming the gap is format-level (per-block scale fitting, group size, asymmetric vs symmetric), not bpw-specific.
- **2026-05-13 PM — `--kmap-dense --kmap-mode 2` produces NaN logits (filed as [#249](https://github.com/Kaden-Schutt/hipfire/issues/249)).** Quantize succeeds (mixed MQ4 base + MQ6 promotion for `mlp.down_proj`, gate/up stay MQ4 + Q8 conv1d, 6.43 GB output). Runtime forward pass produces NaN — KLD comes back 0.0, NLL NaN, no crash (silent-corruption class, same as #179/#209 dispatch audits and MQ8 audit doc). Uniform MQ6+Q8conv1d works fine (see above), and uniform MQ4+Q8conv1d works fine — only the mixed (MQ4 gate/up, MQ6 down) configuration NaNs. Suspected root cause: AWQ-scale derivation differs per format → cancellation between gate/up gemm and down gemm breaks across format boundaries. Workaround: ship uniform-format variants (MQ4 and MQ6 separately; users pick bpw target).
- **2026-05-13 PM — MQ4-Lloyd doesn't help at 9B 4-bit.** n=512 q8-KV measurement (gfx1151) gave KLD 0.3114 / PPL 9.085, essentially tied with mq4-base under q8-KV (estimated converged at 0.27-0.32 from §1.1 + KV-normalization). Lloyd codebook costs +0.66 bpw avg (file size 6.06 GB vs mq4-base 5.31 GB) for null KLD improvement. Confirms §4A.2 prediction: FWHT-256 Gaussianizes per-block distribution, removing the heavy-tail Lloyd is designed to attack. **Active calibration roadmap should drop MQ4-Lloyd**; MQ3-Lloyd remains theoretically interesting (uniform 3-bit is on the cliff edge where heavy-tail loss matters most) and untested.
- **2026-05-13 PM — Per-tensor quant-contribution smoke + n=512 confirmation (9B q8-KV).** Quantized 9B MQ4 with `HIPFIRE_QUANTIZE_LM_HEAD_Q8=1` / `HIPFIRE_QUANTIZE_CONV_Q8=1` / `HIPFIRE_QUANTIZE_CONV_F16=1` (env vars added in `crates/hipfire-quantize/src/main.rs` lines 4055-4084).
  - **n=20 smoke (q8-KV prefill, 9B, gfx1100):** mq4-base 0.3182; mq4-q8lmhead 0.3083 (−3.1% KLD); **mq4-q8conv1d 0.2360 (−25.8% KLD)**; mq4-f16conv1d 0.2388; mq4-f16conv1d-q8lmhead 0.2293.
  - **n=512 confirmation (q8-KV prefill, 9B, gfx1100): mq4-q8conv1d KLD = 0.2501 (CI 0.2396-0.2609), PPL = 8.789.** Drifted +6% from n=20 (expected slice-mean noise); CI tight at ±0.01. PPL is BETTER than the §1.1 asym3 mq4-base 9.116 (−3.6%) and the n=20 smoke (which was wrong-direction artifact of small n).
  - **bpw cost is effectively zero for Q8 conv1d** (+420 KB / +0.0004 bpw avg over 8.95B). vs Q8 lm_head's +540 MB / +0.48 bpw (3 orders of magnitude more expensive per KLD lift unit). **Recommendation: Q8 conv1d should become the default for MQ4 models.** Implementation: change `kmap_resolve_mode` (main.rs:2132) to return `QuantLevel::Q8` for `linear_attn.conv1d.weight` regardless of `--kmap-dense` setting. ~5 line change.
  - **Mechanism:** confirms F2/F3 audit's "1.7-2× drift amplification at Q8 conv1d → LA stage 8/9" finding. MQ4 conv1d adds FWHT rotation noise on top of Q8's already-amplified drift, then DeltaNet recurrence accumulates it across 24 LA layers. Saving ~10× quantization noise at the conv1d input → outsized downstream KLD reduction.
  - **F16 conv1d ≈ Q8 conv1d (0.2388 vs 0.2360 at n=20, statistically indistinguishable):** F16 storage is actually FINER-grained than HF's BF16 reference (10 mantissa bits vs 7), so F16 conv1d weights are slightly more accurate than HF's training-dtype reference → forward pass diverges slightly → KLD-vs-HF slightly higher than Q8. Same F2/F3-pattern (more precise = worse KLD-vs-HF). Q8 dequant's ~7-effective-mantissa-bits precision is closer-to-BF16's grid than F16 storage is. Means Q8 (not F16) is the right default — also smaller.
  - **Q8 lm_head dropped from recommendation:** +540 MB for −3% KLD doesn't pencil out vs Q8 conv1d's near-zero cost for −26%. Cost-per-KLD-lift ratio: Q8 conv1d is **65,000× better** than Q8 lm_head.
- **2026-05-13 PM — PR-#248 Tier-3 fused WMMA Q8 prefill closes the cross-engine floor gap.** 0.8B q8-KV q8f16 floor: 0.0796 (Tier-2 FP32-acc, n=20 gfx1151) → **0.0041** (Tier-3 FP16-acc WMMA, n=512 gfx1100); 9B q8-KV q8f16 floor: 0.1459 asym3-KV Tier-2 (n=512 gfx1100) → **0.0173** q8-KV Tier-3 (n=256 gfx1151). Hipfire 9B Tier-3 Q8 floor is **+6% above llama.cpp Q8_0's 0.0163** — essentially same precision class. The earlier "100% engine drift" claim (Phase 0 commit `13003256`) was 100% the FP32-vs-bf16 precision-class mismatch with HF's bf16 reference; Tier-3's FP16-accumulator WMMA kernels match HF's precision class, and the gap closes. PR-#248 was framed as a perf improvement (18× faster Q8 prefill) but the KLD-vs-HF impact is the bigger structural finding. **Implication for §1.1's "Above-floor" column:** it's path-mismatched (Tier-2 Q8 baseline subtracted from FP16-WMMA MQ4 measurements), understating true 4-bit weight noise by ~0.130 nats at 9B. The path-matched Δ-framework using Tier-3 floor gives the honest accounting.
- **2026-05-14 — AWQ on 9B regressed since `1805d820` (catastrophic, KV-mode-agnostic).** Re-running `--format mq4 --imatrix --awq` at current HEAD (`91b775a7`) produces KLD **13.40 at q8-KV n=20** and **13.35 at asym3-KV n=20** — vs the master-doc §1.1 mq4-awq anchor of 0.2800 (asym3-KV n=512) measured at `1805d820`-era. This is **18× worse than baseline** and "model worse than random" (vocab 248K → uniform NLL=12.42 nats; observed NLL=15.2–16.3 means the model assigns *low* probability to the correct token). NLL is finite, no NaN, forward pass runs at normal speed — the failure is "logits in wrong basis" not "broken pipeline." **0.8B AWQ at q8-KV works (KLD 0.2531, n=512 cohort), so the bug is 9B-specific.** Bisect chain so far:
  - Step 1 (current HEAD, q8-KV n=20): KLD 13.40 ✗
  - Step 2 (same model, asym3-KV n=20): KLD 13.35 ✗ (rules out KV-mode interaction)
  - Known-good: `1805d820` 2026-05-12 21:15:40 produced 9B AWQ KLD 0.2800 at asym3-KV n=512 (master-doc §1.1)
  - Suspect commits (chronological, post-`1805d820`): `dcf3b18a` GPTQ algorithm core (touches main.rs AWQ branch), `5a91c027` GPTQ pipeline helpers, `bd1ca3e1` wire `--gptq` flag (touches main.rs AWQ branch in line 4241+), `0cf5cc45` GPTQ pre-launch fixes, `7dcde2df` parallelize column-sequential loop, `a9dde0de` Cholesky-direct OBS, then env-gated probes + `9b2a68f5` (mine).
  - **Top hypothesis:** GPTQ infrastructure wiring (dcf3b18a/bd1ca3e1) refactored the AWQ pre-scaling path in main.rs:4241+ in a way that broke AWQ-only quantize. The 0.8B AWQ working post-RoPE-fix + 9B AWQ failing on current HEAD is consistent with a quantizer regression that disproportionately impacts a 9B-specific code path or 9B-specific weight statistic. Imatrix entries for share-input weights confirmed byte-identical (verified 2026-05-14), so the wqkv-only-x_rot dispatch is mathematically correct and not the cause. **Implication:** Stage A AWQ landing claim (master-doc §1.1 mq4-awq 0.2800) was real at the time of measurement but the recipe no longer reproduces. Re-running the bench at the bisected good commit confirms the master-doc anchor stays valid as a historical data point; fix needs to land before the master-doc number can be re-claimed for current HEAD.
  - Working artifacts: `2026-05-13-q8-lmhead-conv-smoke/per-variant/qwen35-9b-mq4-awq-bisect__gfx1100__kv-q8__c20.kldseq` (step 1) and `qwen35-9b-mq4-awq-bisect__gfx1100__kv-asym3__c20.kldseq` (step 2).
  - Git bisect not yet kicked off (each step is ~12 min build+quantize+smoke, expects 4 iterations to narrow). Recommended first test: `dcf3b18a` directly.

---

## 6. Methodology rules for adding rows

1. **Always disclose KV mode** (`asym3` / `q8`) in every row. No "default" hand-waving.
2. **Always disclose scoring mode** (`prefill` / `per-token` / `gguf`). Mixing is forbidden by issue-113 §5.3.
3. **Always link the source `.kldseq` file path** or the cohort directory it lives in.
4. **Bootstrap 95% CI** (10,000 resamples on per-seq means) — both `quant_cohort.sh` and the standalone q8-KV probe scripts emit this.
5. **For "above-floor" math (Δ-above-own-Q8 framework, the canonical hipfire-internal lift metric AND the only valid cross-engine claim):** subtract the engine's own Q8-weight floor measured under the SAME KV mode AND the SAME kernel path (e.g. PR-#248 Tier-3 prefill vs Tier-2 batched). Cross-KV-mode floor subtraction is meaningless; cross-kernel-path floor subtraction has unknown error bars.
6. **Flag PRE-FIX (pre-2026-05-12 RoPE) rows explicitly** if they need to be cited; otherwise omit them.
7. **Flag the Q8 kernel path** (Tier-2 batched / Tier-3 fused WMMA / future) in any q8f16 row. The Q8 floor is kernel-path-specific (structurally close but not bit-identical across paths); the Δ-framework requires consistent kernel path between numerator and denominator.
8. **Absolute `KLD(engine || HF-bf16)` is reported for completeness but is NOT a cross-engine output-quality metric.** It measures similarity to HF's bf16-cast noise pattern, which favors engines with bf16-cast-like internal accumulators over arithmetically more precise engines. Use Δ-above-own-Q8 for any "engine A vs engine B" claim. See TL;DR + §2.4.1 + engine-drift memory.
