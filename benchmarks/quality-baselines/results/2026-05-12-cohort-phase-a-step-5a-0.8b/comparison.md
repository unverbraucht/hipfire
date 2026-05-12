# Phase A Step 5a — 0.8B activation-weighted LS first cohort

**Cohort:** `2026-05-12-cohort-phase-a-step-5a-0.8b` (this directory)
**Date:** 2026-05-12
**Branch:** `feat/mq-v2-quant-format`
**Host arch:** gfx1100 (7900 XTX)
**Slice:** wikitext2-test, ctx=2048, slice md5 `83b0205a304bf4e52172ecdb05f2e895`, max-chunks=256
**Scoring mode:** prefill · **KV mode:** asym3
**Imatrix:** `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf` (gfx1151-produced)

## 1. Full 0.8B picture (Steps 0.5 + 1+2 + 5a)

| variant | MSE | KLD | PPL | Δ KLD vs same-format baseline | Δ PPL same |
|---|---:|---:|---:|:---|:---|
| MQ4 (reference) | 1.52e-5 | 0.6721 | 37.30 | — | — |
| HFP4 baseline | 3.37e-6 | 1.6419 | 117.14 | — | — |
| HFP4-L4 | 3.00e-6 | 1.6248 | 113.82 | -1.0% | -2.8% |
| **HFP4-L4-L5c** | **3.13e-6** | **1.4926** | **99.04** | **-9.1%** | **-15.4%** |
| MFP4 baseline | 3.22e-6 | 1.3012 | 76.19 | — | — |
| MFP4-L4 | 2.92e-6 | 1.3870 | 85.91 | +6.6% | +12.7% |
| **MFP4-L4-L5c** | **3.02e-6** | **1.4670** | **94.18** | **+12.8%** | **+23.6%** |

## 2. Headline — L5c is rotation-conditional in the OPPOSITE direction from L4

**HFP4 (unrotated) gets a substantial L5c win.** Closes 22 percentage points of HFP4's KLD gap to MQ4 (from +144% to +122%) and 49 percentage points of the PPL gap (from +214% to +165%). The activation-weighted LS objective is doing exactly the right thing on the unrotated weights.

**MFP4 (FWHT-rotated) gets a substantial L5c REGRESSION.** KLD +13% vs MFP4 baseline, PPL +24%. The same lever that helps HFP4 hurts MFP4.

## 3. Root cause — imatrix is captured on unrotated activations

The implementation comment in `crates/hipfire-quantize/src/main.rs::quantize_mfp4g32_2d` already flagged this risk:

> "Imatrix consideration: the per-channel act² values are captured against the BF16 (unrotated) activation. When the runtime applies the same FWHT to activations at inference time so the dot product cancels, the EFFECTIVE per-channel importance in the ROTATED basis is the FWHT'd imatrix vector. For MFP4 in Step 5a we use the unrotated imatrix weights as-is — this is approximate but the FWHT is orthogonal, so the *total* importance is preserved; only the per-channel attribution gets smeared across the 256-element FWHT block."

The 0.8B cohort shows the prediction "only smearing" was wrong. **The smearing is destructive**, not neutral.

**Mechanism:** the imatrix tells us "channel `i` of the unrotated activation has act² magnitude w[i]". For MFP4, the weights have already been FWHT-rotated at quantize-time, so channel `i` of the rotated weight tensor corresponds to a *linear combination* of unrotated activation channels (the FWHT basis). Using `w[i]` (unrotated importance) as the weight for rotated channel `i` is a per-channel mismatch — high-importance unrotated channels can end up paired with low-importance rotated dimensions and vice versa. The L4 candidate search then *minimizes the wrong quantity*, producing scale/exponent choices that are pessimal for the actual rotated-weight reconstruction error.

## 4. Fix — Step 5a-prime (queued as task #15)

In `quantize_mfp4g32_2d`, apply the same 256-element FWHT to the `imatrix_weights` vector before threading it through `quantize_hfp4g32_row`. The rotated imatrix vector then aligns with the rotated weight basis.

Implementation outline (~half-day dev):

```rust
let imatrix_rotated = imatrix_weights.map(|w| {
    let mut rotated = w.to_vec();
    for seg in 0..(k / 256) {
        cpu_fwht_256(&mut rotated[seg * 256..(seg + 1) * 256], signs1, signs2);
    }
    rotated
});
```

Then pass `imatrix_rotated.as_deref()` to `quantize_hfp4g32_row`.

**Theoretical expectation post-fix:** MFP4-L4-L5c should swing from +13% KLD (regression) toward parity-or-better with MFP4 baseline. Whether it beats HFP4-L4-L5c (1.49 KLD) depends on the format's intrinsic quality — MFP4 baseline is already 1.30 on 0.8B vs HFP4's 1.64, so calibrated-MFP4 has more headroom to drop.

**Skip-the-Fix alternative:** since HFP4-L4-L5c (1.49) is still WORSE than MFP4-baseline (1.30) on 0.8B, the practical decision could be "ship MFP4 uncalibrated, ignore calibration for MFP4". But the §1.5 framing assumed calibration would close most of the format-quality gap — this cohort says it might not for MFP4 specifically. Path A (calibrated-MQ4) becomes the strategically clear winner regardless of whether the MFP4 fix lands.

## 5. Step 5a verdict on 0.8B

**HFP4 + calibration works well.** -9% KLD, -15% PPL vs uncalibrated HFP4. The activation-weighted LS lever is real and substantial.

**MFP4 + calibration is broken until Step 5a-prime.** The rotation interaction destroys the per-channel attribution. Easy fix, queued.

**Calibrated-HFP4 still doesn't beat uncalibrated-MQ4.** At 0.8B, MQ4 KLD 0.67 vs HFP4-L4-L5c KLD 1.49 — a 122% gap. Calibration closes part of the format-quality cost but the underlying L1+L2 mismatch (E2M1 + UE8M0/FP16-row on small models) is real and large. 9B cohort should show whether calibration closes more of the gap at deployment scale.

## 6. Caveats

- 256-chunk quick-slice methodology; full-slice cross-check pending
- Smoke + HE columns broken (same false-failure modes as prior cohorts)
- Imatrix is computed on BF16 model with llama.cpp tokenizer; ~46% tokenizer disagreement with hipfire (per `issue-113-quant-quality-eval.md:126`) introduces small systematic noise — small effect on aggregate statistics, but a Tier 1 native imatrix collector would close this gap
- L5c uses unrotated imatrix for MFP4 (the bug above); fix queued as Step 5a-prime

## 7. Pointers

- Code: `crates/hipfire-quantize/src/main.rs` — search for `IMATRIX`, `safetensors_to_ggml_name`, `load_imatrix`, the `Option<&[f32]>` weight chain through `hfp4_pack_block_at_e` → `hfp4_choose_block_e_l4a` → `pack_hfp4g32_row_with_scale` → `quantize_hfp4g32_row`
- CLI: `hipfire-quantize --imatrix <path-to-imatrix.gguf>` triggers L5c
- Imatrix file: `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf` (1.1 MB, gitignored — regenerate via `imatrix_collect` example per the §3.4 producer_cmd in `harness/manifest.json`)
- 9B cohort: queued in background after 0.8B finishes; will live at `../2026-05-12-cohort-phase-a-step-5a-9b/`
