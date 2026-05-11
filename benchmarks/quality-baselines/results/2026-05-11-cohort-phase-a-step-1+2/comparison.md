# Phase A Step 1+2 — L4 weighted-LS bench findings

**Cohort:** `2026-05-11-cohort-phase-a-step-1+2` (this directory)
**Date:** 2026-05-11
**Branch:** `feat/mq-v2-quant-format`
**Host arch:** gfx1100 (7900 XTX)
**Slice:** wikitext2-test, ctx=2048, slice md5 `83b0205a304bf4e52172ecdb05f2e895`, max-chunks=256 (quick-slice)
**Scoring mode:** prefill (canonical post-#113)
**KV mode:** asym3

## 1. Baselines vs Step 1+2 (L4) result

Combining this cohort with the Step 0.5 baselines (`../2026-05-11-cohort-phase-a-step-0.5/`):

| variant | MSE mean (4-bit qts) | KLD vs BF16 | KLD-p99 | PPL | Δ vs same-format baseline |
|---|---:|---:|---:|---:|:---|
| MQ4G256 (baseline) | 6.62e-6 | 0.8084 | 19.86 | 15.16 | — |
| HFP4G32 (baseline) | 3.15e-6 | 0.9763 | 19.54 | 18.68 | — |
| MFP4G32 (baseline) | 2.98e-6 | 1.1116 | 19.04 | 21.02 | — |
| **HFP4G32 + L4** | **2.80e-6 (-11.1%)** | **1.0327 (+5.8%)** | 18.96 | 20.23 (+8.3%) | **L4 hurts unrotated HFP4** |
| **MFP4G32 + L4** | **2.71e-6 (-9.2%)** | **1.0669 (-4.0%)** | 20.74 | 20.50 (-2.5%) | **L4 helps rotated MFP4** |

## 2. Headline — rotation-conditional L4 effect

**L4 (weighted-LS UE8M0 + FP16-row chooser) is rotation-conditional.** It closes 15% of MFP4's KLD gap to MQ4 (0.30 → 0.26 absolute distance to MQ4), and regresses HFP4 KLD by 6% (0.98 → 1.03). Both formats see ~10% per-tensor MSE drop — the MSE→model-quality relationship is *split* across the two formats.

This is informative beyond the Phase A roadmap framing. The gap doc §5 wrote *"MSE is a leading indicator only"* — the cohort shows MSE is not just noisy but **directionally inconsistent across formats**: a 10% MSE drop maps to +6% KLD loss on HFP4 and -4% KLD win on MFP4.

## 3. Mechanism — why rotation flips L4's verdict

The L4A chooser (per-block UE8M0 weighted-LS) considers candidates `{e_ideal-1, e_ideal, e_ideal+1}`. The "win" of `e_ideal-1` is *tighter spacing* — better precision for weights below max, at the cost of *clipping* weights at or near the block max. The MSE-minimization rule picks `e_ideal-1` when the precision gain on bulk weights outweighs the clip cost on outliers.

The two formats see qualitatively different outlier distributions:

| format | block-level weight distribution | L4A's preferred choice | model-quality effect |
|---|---|---|---|
| **HFP4G32 (unrotated)** | Heavy-tailed / log-normal-ish. Block-level outliers carry semantic content (concentrated attention/MLP channel weights). | Often `e_ideal-1` (large precision gain on tail) | **Hurts** — outlier clipping destroys semantic information faster than bulk precision recovers it |
| **MFP4G32 (FWHT-rotated)** | Sub-Gaussian (kurtosis ≈ 2.82 per fivetide §2.1). Block-level "outliers" are mostly rotation-noise — the semantic outliers got diffused across the block by the FWHT. | Often `e_ideal-1` (large precision gain) | **Helps** — outlier clipping clips noise; precision gain captures real signal |

The lever taxonomy in `docs/plans/qwen35-mq4-quality-gap.md` §1.1 treats L4 (scale fitting) and L3 (rotation) as independent levers that multiply. The cohort shows they **interact**: L3 changes the importance landscape of block-level outliers, which in turn changes whether L4's per-block tightening helps or hurts.

## 4. Implications for the Phase A roadmap

### 4.1 L4 should be gated on rotated formats

Implementation suggestion: rather than `--l4` enabling L4 universally, only enable it when `format == mfp4 || format == mq4_rotated` (any FWHT-baked format). For unrotated formats (HFP4G32, HFQ4G256, Q4_K), L4 in its pure-MSE form is a regression.

Even cleaner: skip the pure-MSE L4 step entirely from the Phase A roadmap and go straight to **activation-weighted L4 (= L4 combined with L5b)**. The mechanism above suggests activation-weighted LS (which weights block reconstruction error by activation magnitudes per channel) would correctly *not* clip semantic-channel outliers regardless of rotation status — the activation weights down-weight blocks that would be clipped on important channels. This unifies the L4+L5b work into a single quantizer-side change.

### 4.2 The Path A bet sharpens

The Step 0.5 cohort established calibrated-MQ4 (Path A) is favored over calibrated-MFP4 (Path B). The L4 result narrows this further:

- Calibrated-MQ4: needs L4 with MQ4-style (FP32 scale + ZP) candidate search. Per the gap doc §5 there's no L4 implementation for MQ4 yet — Step 1c/2c work pending. **The L4 result on MFP4 (+4% KLD win) suggests MQ4's L4 should also help** because MQ4 is FWHT-rotated; same mechanism.
- Calibrated-MFP4: L4 + L5 stacked. L4 alone closed 15% of the MFP4→MQ4 gap (0.30 → 0.26). L5 needs to close the remaining 0.26 absolute KLD distance — a tall order for activation calibration alone. The remaining gap is likely a real format-level deficit (sub-Gaussian + E2M1 codebook mismatch from fivetide's analysis) that calibration can't fully overcome.

The cohort doesn't kill Path B but makes Path A more clearly the lower-risk Phase A bet.

### 4.3 The Step 1+2 ship recommendation

| recommendation | rationale |
|---|---|
| **Keep `--l4` as opt-in, default off** | Default-on for HFP4 would regress shipped quants. Default-on for MFP4 is reasonable but should be measured at deployment scale (9B + 27B + A3B) before flipping. |
| **Don't ship --l4 as the new default for any current cohort variant** | The 9B Step 1+2 cohort is the most thorough measurement so far — five variants. The HFP4-L4 regression rules out unconditional default. |
| **Phase A Step 3 (L5a tensor-type CLI) is unblocked** | Step 3 is wire-format-independent and works on any quantization scheme. No bench dependency. |
| **Phase A Step 4+5 (imatrix + activation-weighted LS) is the next high-value work** | L4 on its own only closes 15% of the MFP4 gap to MQ4. L5 is the dominant lever per the gap doc §2.3 and the cohort doesn't refute that — it just confirms that L4 alone won't be enough. |
| **Add Step 1c/2c: L4 for MQ4G256** | The L4 lever for MQ4 is FP32 scale + FP32 ZP weighted-LS. Quantizer-only patch. If MFP4-L4's +4% KLD win on rotated weights transfers to MQ4 (FWHT-rotated like MFP4), we'd see a similar lift on the format we actually want to ship. ~1 week dev. |

## 5. Caveats — what this cohort does NOT measure

| caveat | why it matters |
|---|---|
| Quick-slice (256 chunks) only | Full-slice (1175 chunks) confirmation needed before committing format-level conclusions. MQ4 step-0.5 cross-check showed quick-slice reproduces full-slice within bench noise — but verifying on the L4 variants is worth one ~3-hour cohort run. |
| 9B Qwen3.5 dense only | 0.8B + 4B + A3B cohorts not yet run (BF16 kldref dependency). 0.8B in flight as of this writing (parallel gfx1151 ref-build + local cohort, see ../2026-05-11-cohort-phase-a-0.8b-step-0.5+1+2/). The L4 effect may scale differently — fivetide's data showed MFP4 vs MQ4 PPL gap goes from +25.4% on 9B to +93.8% on 0.8B; L4's relative impact may differ. |
| K-map = OFF (dense default) | Pure format-vs-format comparison. K-map ON would promote some tensors to MQ6/Q8, changing the L4 effect on the unpromoted residual. Worth a follow-up cohort. |
| Smoke + HumanEval columns still unreliable | Same false-failure modes as Step 0.5 cohort — see that cohort's comparison.md §4. Bonus columns; don't drive decisions. |

## 6. Pointers

- L4 implementation: `crates/hipfire-quantize/src/main.rs` — search for `L4A_ENABLED` / `L4B_ENABLED` (static AtomicBools) and `hfp4_choose_block_e_l4a` / `pack_hfp4g32_row_with_scale`
- CLI flags: `--l4a` (block UE8M0 only), `--l4b` (row scale only), `--l4` (both)
- Default path bit-equivalent to pre-L4 (verified via 31/31 unit tests in `cargo test -p hipfire-quantize`)
- Strategic framing this updates: `docs/plans/qwen35-mq4-quality-gap.md` §1.5 ("the L1+L2 lever analysis was wrong by sign on dominant-mass weights" finding is sharpened by this cohort's L4-is-rotation-conditional result)
- The rebuttal doc anchoring the discussion: `docs/plans/hfp4-fivetide-rebuttal-perspective.md`
