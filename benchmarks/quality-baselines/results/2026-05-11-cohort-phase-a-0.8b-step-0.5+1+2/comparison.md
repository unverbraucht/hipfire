# Phase A 0.8B Step 0.5 + Step 1+2 — small-model cross-check findings

**Cohort:** `2026-05-11-cohort-phase-a-0.8b-step-0.5+1+2` (this directory)
**Date:** 2026-05-11
**Host arch:** gfx1100 (7900 XTX)
**Model:** Qwen3.5-0.8B (752 M params)
**BF16 kldref source:** produced on gfx1151 (parallel work, commit 1792959c)
**Slice:** wikitext2-test, ctx=2048, slice md5 `83b0205a304bf4e52172ecdb05f2e895`, max-chunks=256
**Scoring mode:** prefill
**KV mode:** asym3

## 1. Headline result (5 variants)

| variant | MSE (4-bit qts) | KLD vs BF16 | KLD-p99 | PPL | Δ vs MQ4 (KLD / PPL) | Δ vs same-format baseline |
|---|---:|---:|---:|---:|:---|:---|
| MQ4G256 | 1.52e-5 | 0.6721 | 5.21 | 37.30 | — | — |
| HFP4G32 | 3.37e-6 | 1.6419 | 9.49 | 117.14 | **+144% / +214%** | — |
| MFP4G32 | 3.22e-6 | 1.3012 | 7.75 | 76.19 | **+94% / +104%** | — |
| HFP4-L4 | 3.00e-6 (-11%) | 1.6248 | 9.64 | 113.82 | +142% / +205% | -1.0% / -2.8% |
| MFP4-L4 | 2.92e-6 (-9%) | 1.3870 | 8.59 | 85.91 | +106% / +130% | **+6.6% / +12.7%** |

## 2. Three findings that update the strategic picture

### 2.1 The format gap is much more severe at 0.8B than at 9B

The Step 0.5 9B cohort reported MFP4 +37.5% KLD vs MQ4. On 0.8B, **MFP4 is +94%** — more than double the relative gap. HFP4 is even worse at +144%.

Fivetide's published 0.8B MFP4 PPL gap was +93.8%. Our **+104% PPL** is consistent within methodology noise (different KV mode + scoring mode shift the absolute numbers, but the direction + magnitude track each other across this two-data-point cross-check).

This is consistent with the general "small models punish quant noise harder" rule (less weight redundancy → individual block-level errors have larger downstream effect). The §1.5 finding that "L1+L2 combined is a per-tensor regression on dominant-mass weights" is *amplified* at small scale because the small model can't average out the noise across as many independent channels.

### 2.2 L4 rotation-conditional verdict REVERSES at 0.8B

The 9B cohort yielded:
- HFP4-L4: KLD **+5.8% (worse)** — pure-MSE L4 hurts unrotated format
- MFP4-L4: KLD **-4.0% (better)** — pure-MSE L4 helps rotated format

My §1.5 framing inferred "L4 is rotation-conditional: helps rotated, hurts unrotated" from this. The 0.8B cohort flips both signs:
- HFP4-L4: KLD **-1.0% (slight improvement)** — pure-MSE L4 slightly helps unrotated on small models
- MFP4-L4: KLD **+6.6% (worse)** — pure-MSE L4 hurts rotated on small models

**The single-data-point inference was over-confident.** The actual mechanism is more complex than just "rotation". Plausible additional factors:

- **Model size**: small models have less channel redundancy, so a few clipped outlier weights per block disproportionately affect critical channels. L4's tighter-scale choice clips outliers; at 9B, outliers in MFP4's FWHT-rotated distribution are mostly noise (helps), but at 0.8B even rotated outliers might carry irreplaceable per-channel signal (hurts).

- **The MFP4 / MQ4 starting gap**: at 9B, MFP4-uncalibrated is +37.5% KLD over MQ4. At 0.8B it's +94%. The "headroom" for L4 to help is structurally different — possibly L4 in pure-MSE form is fighting against a much steeper degradation curve at small scale.

- **Embedding-table confound**: at 0.8B, the embedding table is ~50% of weight params (vs ~12% on 9B per fivetide §1.2). HFP4/MFP4 keep `model.embed_tokens` as Q8_0 (per the K-map default), so the per-tensor MSE *delta* between L4 variants is concentrated in the non-embedding tensors. At 9B those non-embedding tensors dominate; at 0.8B they're a minority. The L4 effect we're measuring on small models may be specific to the residual non-embedding bulk.

**Implication for §1.5**: the "rotation-conditional L4" hypothesis needs to drop to a weaker claim — L4 effect is **model-size-and-rotation-dependent**, with the specific dependence pending more data points (4B + A3B cohorts would help triangulate).

### 2.3 MSE is directionally misleading at both scales

On 9B: MFP4-L4 -9% MSE, -4% KLD (consistent direction). HFP4-L4 -11% MSE, +5.8% KLD (opposite direction).
On 0.8B: HFP4-L4 -11% MSE, -1% KLD (consistent direction, tiny magnitude). MFP4-L4 -9% MSE, **+6.6% KLD** (opposite direction).

Four data points across two model sizes × two formats: **two consistent + two inverted.** Pure-MSE minimization is not just noisy — it's a coin-flip whether MSE wins translate to KLD wins.

This sharpens the Phase A Step 5 case: **L5c (activation-weighted LS) is the right L4-replacement**, not pure-MSE L4. Activation weighting puts the optimization objective on the channels that matter for forward-pass output, which is approximately what KLD/PPL care about.

## 3. Cross-cohort comparison: 9B vs 0.8B

| variant | 9B KLD | 0.8B KLD | 9B PPL | 0.8B PPL |
|---|---:|---:|---:|---:|
| MQ4 | 0.808 | 0.672 | 15.16 | 37.30 |
| HFP4 | 0.976 (+21%) | 1.642 (+144%) | 18.68 (+23%) | 117.14 (+214%) |
| MFP4 | 1.112 (+38%) | 1.301 (+94%) | 21.02 (+39%) | 76.19 (+104%) |
| HFP4-L4 | 1.033 (+28%) | 1.625 (+142%) | 20.23 (+33%) | 113.82 (+205%) |
| MFP4-L4 | 1.067 (+32%) | 1.387 (+106%) | 20.50 (+35%) | 85.91 (+130%) |

The MQ4 baseline KLD is *lower* on 0.8B (0.672) than on 9B (0.808) — the smaller model with the rotated INT4 format actually tracks BF16 closer in absolute terms, even though PPL is much higher (37.3 vs 15.2). KLD vs BF16 is an information-theoretic distance metric while PPL measures token-prediction confidence; smaller models have less confidence overall, so PPL goes up, but the relative-distribution-to-BF16 distance is what KLD measures and that's more stable across scale.

For the format-vs-format DELTAS, the trend is clear: **the gap widens dramatically as model size shrinks.** Fivetide's published 9B +25% / 4B +32% / 0.8B +94% PPL trend is reproduced here on KLD-vs-BF16 (the canonical hipfire metric) too.

## 4. Caveats

| caveat | notes |
|---|---|
| 0.8B is closer to a developer-bench tool than a deployment target | Hipfire's deployment shapes are ~4B (small users), 9B (medium), 27B (large), A3B (MoE). 0.8B is in the "research-grade" tier — useful for iteration speed but not what users actually run. |
| Smoke + HE columns broken (same false-failure modes as Step 0.5 cohort) | Don't use those columns for decisions. Same fixes pending. |
| K-map = OFF (default for dense) | Apples-to-apples format comparison. K-map ON would change the calibration story. |
| 256-chunk quick-slice | Same methodology shortcut as the 9B cohort. Full-slice cross-check not yet run for 0.8B. |
| Embedding-table confound at 0.8B | 0.8B's embed_tokens is ~50% of weight params (vs ~12% at 9B). All variants Q8-promote the embedding by default, so this isn't a direct cross-variant confound, but it means the format effect on the non-embedding bulk is *less diluted* at 0.8B and more diluted at 9B. |

## 5. Pointers

- Per-variant artifacts: `per-variant/` (kldseq, mse.txt)
- 9B cohort for comparison: `../2026-05-11-cohort-phase-a-step-0.5/` + `../2026-05-11-cohort-phase-a-step-1+2/`
- 0.8B BF16 kldref: `../../refs/qwen3.5-0.8b-bf16.kldref.bin` (gfx1151-produced)
- §1.5 framing to update: `docs/plans/qwen35-mq4-quality-gap.md` — the "L4 is rotation-conditional" claim should be downgraded; the cohort surfaces model-size-and-rotation interaction, not pure rotation dependence
