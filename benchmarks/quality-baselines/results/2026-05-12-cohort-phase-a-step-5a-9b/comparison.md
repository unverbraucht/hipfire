# Phase A Step 5a — 9B calibrated cohort + GGUF anchor reality check

**Cohort:** `2026-05-12-cohort-phase-a-step-5a-9b` (this directory)
**Date:** 2026-05-12
**Branch:** `feat/mq-v2-quant-format`
**Host arch:** gfx1100
**Imatrix:** locally produced via Step 4 Tier 2 (`imatrix_collect`) on the same wikitext slice + pinned llama.cpp commit as the 0.8B run
**Slice:** 256-chunk quick-slice, prefill mode, asym3 KV

## 1. Headline 9B cohort numbers

| variant | MSE | KLD | PPL | Δ KLD vs same-format baseline | Δ PPL same |
|---|---:|---:|---:|:---|:---|
| MQ4 (reference) | 6.62e-6 | 0.8084 | 15.16 | — | — |
| HFP4 baseline | 3.15e-6 | 0.9763 | 18.68 | — | — |
| HFP4-L4 | 2.80e-6 | 1.0327 | 20.23 | +5.8% | +8.3% |
| **HFP4-L4-L5c** | 2.92e-6 | **1.0074** | 19.09 | **+3.2% (recovered but still WORSE)** | +2.2% |
| MFP4 baseline | 2.98e-6 | 1.1116 | 21.02 | — | — |
| MFP4-L4 | 2.71e-6 | 1.0669 | 20.50 | -4.0% | -2.5% |
| **MFP4-L4-L5c (FWHT-fix BUG)** | **7.30e-6** | **1.2014** | 21.33 | **+8.1% (BROKEN)** | +1.5% |

## 2. Three findings, ranked by strategic impact

### 2.1 The most important finding — GGUF anchors expose how much further hipfire has to go

Cross-referencing earlier GGUF anchor work from `benchmarks/quality-baselines/results/2026-05-10/per-seq/` (commit `6c00a558`, gfx1151 per-token mode against the same llama.cpp BF16 reference):

| variant | KLD | PPL | bpw | engine |
|---|---:|---:|---|---|
| GGUF Q8_0 (anchor) | **0.0163** | 9.31 | 8.5 | llama.cpp |
| GGUF UD-Q6_K_XL (Unsloth) | 0.0213 | 9.14 | ~6.7 | llama.cpp |
| GGUF Q6_K | 0.0250 | 9.31 | 6.56 | llama.cpp |
| GGUF UD-Q5_K_XL | 0.0408 | 9.27 | ~5.5 | llama.cpp |
| **GGUF UD-Q4_K_XL (Unsloth)** | **0.0670** | **9.34** | ~5.3 | llama.cpp |
| GGUF Q4_K_M | 0.1249 | 8.70 | 5.07 | llama.cpp |
| GGUF UD-Q3_K_XL | 0.1411 | 8.67 | ~4.5 | llama.cpp |
| **hipfire MQ4 (our best)** | **0.8084** | **15.16** | 4.25 | hipfire |
| hipfire HFP4-L4-L5c | 1.0074 | 19.09 | 4.5 | hipfire |
| hipfire MFP4 | 1.1116 | 21.02 | 4.5 | hipfire |

**The KLD numbers aren't apples-to-apples** because:

- GGUF anchor KLD: `KLD(llama.cpp_quant_X || llama.cpp_BF16)` — same engine, same tokenizer on both sides. Pure intrinsic format quality.
- Hipfire KLD: `KLD(hipfire_quant_X || llama.cpp_BF16)` — **cross-engine, with ~46% tokenizer disagreement (per `issue-113-quant-quality-eval.md:126`) baked into the signal.**

Most of the 12× gap (MQ4 0.81 vs UD-Q4_K_XL 0.07) is likely engine + tokenizer drift, NOT pure format quality.

**PPL is the more comparable signal** — it's less engine-sensitive than KLD — **but the comparison must be bpw-matched.** Aggregate bpw (file size × 8 / total params):

| variant | file size | params | aggregate bpw | PPL |
|---|---:|---:|---:|---:|
| hipfire MQ4 | 5.07 GB | 8.95B | **~4.87 bpw** | 15.16 |
| hipfire HFP4 | 5.34 GB | 8.95B | ~5.00 bpw | 18.68 |
| GGUF UD-Q3_K_XL | 5.05 GB | 8.95B | **~4.50 bpw** | 8.67 |
| GGUF Q4_K_M | 5.67 GB | 8.95B | 5.07 bpw | 8.70 |
| GGUF UD-Q4_K_XL | 5.97 GB | 8.95B | 5.32 bpw | 9.34 |

**The bpw-matched anchor for hipfire MQ4 (4.87 bpw) is GGUF UD-Q3_K_XL (~4.50 bpw)** — *not* UD-Q4_K_XL. Unsloth's UD-Q4_K_XL spends ~0.45 more bits/weight than hipfire MQ4. UD-Q3_K_XL ("Q3 modal with imatrix-driven promotions to Q4/Q5/Q6/Q8") is the actual peer in hipfire's bpw zone.

At matched bpw: **hipfire MQ4 PPL 15.16 vs UD-Q3_K_XL PPL 8.67 → +75% gap, despite hipfire using +0.37 MORE bits per weight.** The gap doc's earlier "+62% vs UD-Q4_K_XL" framing was cross-bpw and made the gap look smaller than it actually is at hipfire's deployment bpw.

A 75% PPL gap at matched-or-favored bpw is the actual calibration target. That's substantial — Phase A levers need to close most of it.

### 2.1.b Structural / scheme differences (size + format levers)

The bpw is only half the comparison. The wire-format scheme also differs:

| lever | hipfire MQ4G256 | llama.cpp Q4_K_M / UD-Q*_K_XL |
|---|---|---|
| Codebook | INT4 uniform | INT4 uniform (same) |
| Per-256 scale | 1 FP32 scale + 1 FP32 ZP | 1 super-FP16 + 1 super-FP16 ZP + 8×6-bit sub-scales + 8×6-bit sub-ZPs |
| Per-32 sub-scaling | none | **yes** (the "Q4_K" innovation) |
| Rotation | FWHT-256 (offline) | none |
| Scale fitting | min-max (or 3-cand L4) | weighted-LS (`make_qkx2_quants`, 20-candidate) |
| Imatrix LS | none (until Step 5b ships) | yes (UD variants) |
| Per-tensor allocation | hardcoded K-map (OFF on dense) | imatrix-driven (UD) |

Hipfire trades **2 scale-grain levers** (per-32 sub-scaling + super-FP16) for **1 rotation lever** (FWHT-256). The gap doc §1.3 predicted rotation would compensate. The bpw-matched cohort says it doesn't compensate enough.

**Structural implication for Phase A:** the calibrated-MQ4 path (Step 5b) has to make up for the missing scale-granularity lever via better candidate-search exploration. If hipfire MQ4 had per-32 sub-scales (like Q4_K), it would already be much closer to UD-Q3_K_XL territory at the same bpw. Adopting Q4_K-style sub-scales to hipfire's MQ4 wire format ("MQ4K") is a structural Phase B option worth costing — possibly higher leverage than further calibration on the current MQ4 wire format.

**Methodology gap to fix:** a hipfire Q8 baseline (`KLD(hipfire_Q8 || llama.cpp_BF16)`) is needed to calibrate the engine + tokenizer drift floor. Without it we can't separate "format quality" from "engine/tokenizer drift" in any of these hipfire numbers. Cheap measurement — ~30 min cohort wall.

### 2.2 L5c results are partial / mixed at 9B (compared to clean win at 0.8B)

On 0.8B HFP4, L5c gave a clean -9% KLD win. On 9B HFP4:

- L4 alone: regression of +5.8% KLD vs baseline
- L4+L5c: still +3.2% KLD vs baseline (better than L4 alone, worse than uncalibrated)

L5c is partially correcting L4's regression but not fully reversing it on 9B. Possible reasons:

- **Per-channel importance is flatter at 9B than 0.8B.** Bigger models have more channel redundancy, so the per-channel importance distribution is more uniform. L5c's lever (preferentially weight important channels) has less to work with on a flatter distribution.
- **Bench noise.** 256-chunk is fast but noisy; the 9B run has only 256 chunks of signal. Full-slice 1175-chunk re-bench would tighten the CIs.
- **Hipfire's L4 framework underexplores the search space.** Hipfire's L4 is a 3-candidate search over `{e_ideal-1, e_ideal, e_ideal+1}`. Unsloth's `make_qkx2_quants` is a 20-candidate weighted-LS search. L5c on top of an under-explored search may not find the right minimum.

### 2.3 Step 5a-prime FWHT-imatrix fix was wrong — corrected to "skip L5c for rotated formats"

My initial Step 5a-prime fix (committed in `a46a90b3`) applied FWHT to the imatrix vector before threading it through the candidate search. The 9B MFP4-L4-L5c run measured this approach at **7.30e-6 MSE (2.7× regression vs MFP4-L4 alone at 2.71e-6).**

Root cause: FWHT mixes ±1 so the rotated imatrix vector has *negative* entries. The L4 candidate scoring `Σ w[i] · err²` with mixed-sign weights REWARDS errors on negatively-weighted channels — inverted optimization.

**The mathematically correct treatment**: per the variance-of-rotated-vector argument, for rotated weights the per-channel importance is `Var[x_rot[i]] = Σ_j H[i,j]² Var[x[j]] = Σ_j Var[x[j]]` — **constant across channels within a 256-segment.** The rotation literally flattens per-channel importance. L5c's lever loses its purchase on rotated formats.

Fixed in this cohort's accompanying commit: when format is rotated (MFP4), pass `None` for imatrix → pure-MSE L4 path. This matches the math.

**Implication for MFP4 calibration**: the current per-block-LS L5c framework cannot calibrate rotated formats meaningfully. To get a calibration win on MFP4, we'd need a different lever (e.g., per-segment row-scale weighting where `Σ Var` varies across 256-segments). Not addressed by Step 5a.

## 3. Strategic implications

The §1.5 framing of "calibration is the dominant lever, will close most of the format-quality gap" is **not validated** by the 9B cohort + GGUF anchor reality check:

- L5c moves HFP4 a couple percentage points on 9B; doesn't reverse L4's regression
- L5c can't help MFP4 (rotation math)
- Even the cleanest hipfire variant (MQ4) is 62% PPL behind UD-Q4_K_XL — calibration of MQ4 (Step 5b — not yet implemented) is the remaining open lever, but the gap is structural enough that calibration alone might not close it

**Re-prioritized Phase A backlog:**

1. **Hipfire Q8 baseline cohort** (~30 min wall) — separates engine-drift floor from format-quality signal. Required before any further calibration claims.
2. **Step 5b — MQ4 weighted-LS** (~1-1.5 days dev) — the actual Path A bet. MQ4 uses (FP32 scale + FP32 ZP) per 256 with no L4 yet; needs new candidate-search code path. THIS is the load-bearing measurement.
3. **Step 6b — apply UD decompile kmap to hipfire quantize** (~1-2 days dev) — Unsloth's per-tensor bit allocation might be doing more lift than per-block calibration. The UD-Q4_K_XL kmap artifact (commit `679bff46`) is already in tree; we need to apply it via a `--kmap-file` CLI option to the quantizer.
4. **Full-slice cross-check on the best calibrated variants** (~3 hours per cohort) — quick-slice is noisy; before drawing strategic conclusions, full-slice the best 2-3 calibrated variants.
5. **Tokenizer parity fix** — separate ~46% disagreement effect from format quality. Either fix hipfire's tokenizer to match llama.cpp's behavior for the bench, or compute an in-engine KLD reference for hipfire.

## 4. Caveats

- 256-chunk quick-slice — full-slice cross-check would tighten CIs
- Cross-engine KLD measurement noise dominates the absolute numbers (PPL is the more honest signal)
- Smoke + HE columns still broken
- 9B imatrix took ~93 min wall (4.79s/pass × 1175 chunks); model is mixed GPU/CPU due to 20 GB VRAM limit
- L5c uses calibration from wikitext slice — not the broader corpus Unsloth uses (Calibration_v3/v5 mixed code + chat + reasoning). Better corpus might yield better signal.

## 5. Pointers

- Step 5a-prime correction: `crates/hipfire-quantize/src/main.rs` — search for "L5c imatrix handling for rotated formats — SKIP"
- 0.8B Step 5a baseline cohort + analysis: `../2026-05-12-cohort-phase-a-step-5a-0.8b/comparison.md`
- 9B Step 0.5 baseline + Step 1+2 L4 cohort: `../2026-05-11-cohort-phase-a-step-0.5/` and `../2026-05-11-cohort-phase-a-step-1+2/`
- GGUF anchor data: `../2026-05-10/per-seq/qwen3.5-9b.gguf-*__gfx1151.kldseq`
- UD-Q4_K_XL kmap (for Step 6b): `../../external/ud-decompile/qwen3.5-9b-ud-q4_k_xl.kmap.json`
- §1.5 framing to update: `docs/plans/qwen35-mq4-quality-gap.md` — calibration-as-dominant-lever needs revision after this cohort
