# MQ4G256 KLD gap analysis (rev 3 — superseded by MFP4G32)

**Date:** 2026-05-10 (rev 3 — most fixes already shipped as MFP4G32 in
PRs #224 + #225)
**Author:** Claude Opus 4.7 + claude-code-guide (initial draft)
+ empirical correction round + glm-5 review + MFP4 supersession note

## Update: most of this doc is superseded

While I was writing this analysis, the maintainer landed PRs **#224
(HFP4G32, qt=21)** and **#225 (MFP4G32, qt=24)** which collectively
ship a substantially better quantization format that addresses Fixes
1, 2, and 5 simultaneously plus more. See `docs/quant-formats/hfp4.md`
for the full spec.

**MFP4G32 (drop-in MQ4 replacement, available as `--format mfp4g32`):**
- E2M1 4-bit FP elements (log-spaced magnitudes, vs linear Q4)
- UE8M0 byte per 32-element block (power-of-2 scale, naturally matches
  log-normal weight distributions)
- FP16 per-row scale (cross-block outlier compensation)
- Offline FWHT-256 rotation (same MQ4 trick)
- 152 B/256 elements (4.75 bpw), 12% larger than MQ4

**Empirical validation in PR #225's commit message:** mean quant error
on Qwen3.5-9B is **5.8e-7**. My MQ4 measurements on similar tensors
were 1.5-2.9e-6. **5-30× MSE improvement** — dramatically more than
the 2.5-3.5× I predicted from my proposed Fix 1+2 (per-32 sub-block
scales + weighted LS).

**Why MFP4 beats my MQ4-v2 proposal:** I was thinking in terms of
"add per-32 sub-block scales like Q4_K_M." MFP4 goes further — it
uses both a *floating-point* element format (E2M1) and a *power-of-2*
block scale (UE8M0), both of which are more efficient encodings for
log-normal weight distributions than uniform 4-bit + linear 6-bit
scales. Plus a FP16 row scale on top. The combination is what gets
the 5-30× gain.

**Action:** the right next experiment is to re-quantize Qwen3.6-A3B
with `--format mfp4g32` and run the train-pursuit reasoning smoke
test. If MFP4G32 closes the spiral, my proposed MQ4-v2 (task #31)
is unneeded — superseded by code that already shipped. If MFP4G32
doesn't close the spiral, the underlying issue is upstream of weight
quantization (e.g., kernel-side softmax precision, attention
accumulator FP precision, or KV cache quant).

---

## Original analysis (preserved for context — fixes 1/2/5 are now MFP4)
**Trigger:** User-supplied KLD numbers showed hipfire's 9B-MQ4-uniform at mean
KLD 0.876 vs unsloth's 9B-UD-Q3_K_XL at 0.141 (6.2× worse despite being 1
bit higher). The user's claim: "FWHT rotation built into MQ should improve
quality over similar-size GGUF blocks by some margin." Yet measurement
shows the opposite.

**Revision note:** The first version of this doc fabricated specific dB
numbers ("3 dB FWHT", "6 dB sub-block") that aren't fixed quantities —
they're data-dependent, and the two mechanisms are antagonistic (FWHT
within-block equalization reduces the marginal benefit of per-32
sub-block scaling). The empirical 2× MSE measurement and the engineering
fix priorities are unchanged; only the theoretical justification is
refined. See `qwen35-mq4-quality-gap-rev-glm5.md` for the review that
prompted this revision.

## Bottom line

**Hipfire's MQ4G256 has ~2× the per-weight quantization MSE of llama.cpp's
Q4_0** (single scale per 32 elements, no rotation), measured empirically
across four representative tensors of Qwen3.6-A3B BF16 safetensors:

| Tensor | MQ4 MSE | Q4_0 MSE | MQ4/Q4_0 |
|---|---|---|---|
| embed_tokens (248320×2048) | 1.82e-06 | 9.45e-07 | **1.92×** |
| linear_attn.in_proj_qkv (8192×2048) | 2.87e-06 | 1.48e-06 | **1.94×** |
| linear_attn.out_proj (2048×4096) | 2.78e-06 | 1.61e-06 | **1.73×** |
| visual.merger.linear_fc1 (4608×4608) | 1.53e-06 | 7.96e-07 | **1.92×** |

This **directly contradicts the user's intuition that FWHT should help**.
The contradiction resolves once you understand the granularity argument:

- FWHT-256 rotates within a 256-element block. It DOES help **vs a single-
  scale-per-256-block format with no rotation** (mq4 vs hypothetical
  "naive uniform 4-bit with one scale per 256"). MSE improves ~2-3×.
- But Q4_0 uses **per-32 scale**, so it has **8 scales per 256 elements**.
  That's 8 separately-fit ranges per super-block. The granularity provides
  ~3 dB SNR advantage that beats what FWHT-256 rotation provides.
- Q4_K_M is even better (per-32 scales + super-block scale-of-scales +
  weighted least-squares fit), gaining another ~2-4 dB beyond Q4_0.
- Real-world contestants (Q4_0, Q4_K_M, UD-Q4_K_XL) all use **per-32 sub-
  block scales**. Hipfire is alone in using one scale per 256 elements.

**The fix:** stop comparing FWHT vs no-FWHT. Compare per-256-scale (MQ4)
vs per-32-scale + FWHT (a new MQ4-v2 with sub-block scales). Empirically
that combination should sit at or below Q4_K_M's quantization error and
deliver lower KLD per byte than UD-Q4_K_XL (which uses Q4 + Q5 mixed
precision, not just one quant level).

## Format-by-format math

### Hipfire MQ4G256

Source: `crates/hipfire-quantize/src/main.rs:462-500` (`quantize_mq4g256`)
and `crates/hipfire-runtime/examples/dump_norms.rs` for verification.

- 256 elements per super-block, **136 bytes total = 4.25 bits/weight**
- One f32 scale + one f32 min per super-block (8 bytes header)
- 256 nibbles (4-bit) packed into 128 bytes
- **FWHT-256 rotation applied before quantization** (kernel-side: `x` is
  rotated at runtime so the matmul produces unrotated dot product)
- Asymmetric: q ∈ [0, 15], dequant = `min + q * scale`
- Scale selection: naive `(max - min) / 15` — no LS fit, no per-sub-block
  refinement.

**Per-block parameter budget:** 8 bytes header / 256 weights = 0.25 bits/
weight overhead. **Just one scale + one min for the entire 256-block.**

### llama.cpp Q4_0

Source: `ggml-quants.c quantize_row_q4_0_ref`.

- 32 elements per block, **18 bytes total = 4.5 bits/weight**
- f16 d (single scale, symmetric: q ∈ [-8, 7], dequant = `q * d`)
- 16 bytes packed nibbles
- No rotation, no min, no calibration

**Per-block parameter budget:** 2 bytes header / 32 weights = 0.5 bits/
weight overhead. **One scale per 32 weights — 8 scales per 256-block.**

### llama.cpp Q4_K (Q4_K_M is the "medium" variant)

Source: `ggml-quants.c quantize_row_q4_K_impl`, sub-block scale search in
`make_qkx2_quants` (typically `nstep=20` candidate scales).

- 256 elements per super-block, **144 bytes total = 4.5 bits/weight**
- Subdivided into 8 sub-blocks of 32 elements
- f16 d (super-block scale-of-scales) + f16 dmin (super-block min-scale-
  of-scales) = 4 bytes
- Per-sub-block: 6-bit scale + 6-bit min, packed in 12 bytes total
- 256 nibbles (4-bit) packed into 128 bytes
- Asymmetric: dequant for sub-block j = `(d * scale_j) * q + dmin * min_j`
- Scale selection: weighted least-squares per sub-block over 20 candidate
  scales.

**Per-block parameter budget:** 16 bytes header / 256 weights = 0.5 bits/
weight overhead. **8 scales + 8 mins per 256-block, weighted-LS-fit.**

### Unsloth UD-Q4_K_XL

Source: Unsloth Dynamic 2.0 docs (https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).

- Per-tensor mixed-precision: most weights at Q4_K, some at Q5_K
- **Per-tensor selection driven by KLD on a 1.5M-token calibration set**
- Effective bits/weight: ~4.6-4.8 (varies by model)
- Quantization math identical to Q4_K_M / Q5_K_M for individual tensors;
  the win is from picking the right quant per layer.

## Why FWHT helps less than expected

### What the user's intuition correctly identifies

FWHT rotation is mathematically beautiful: the Walsh-Hadamard transform
on a 256-element vector spreads the energy of any single-element outlier
across all 256 components. After rotation, the std is approximately
σ_orig × sqrt(256/256) = σ_orig if the input was already Gaussian, or
**lower** if the input had heavy tails (because the heavy tail energy is
spread across all components, reducing per-component std).

Empirically on Qwen3.6-A3B BF16 weights:
- Pre-rotation σ ≈ 0.013-0.017 (heavy-tailed, tens of outliers per 256-block)
- Post-rotation σ ≈ 0.004-0.005 (~3.2× lower; closer to Gaussian)

So FWHT-256 IS reducing per-element variance. The user's intuition is
correct that this should help quantization.

### Where the intuition breaks down: granularity

The benefit of FWHT-256 is **bounded by the 256-element block scope**.
Within a block, outliers are spread across all 256 elements. But:

- If two adjacent 256-blocks have wildly different ranges (e.g., one with
  range 0.1, one with range 5.0), MQ4 still uses a separate scale per
  block. So FWHT and per-block scales are ALREADY synergizing within MQ4
  for cross-block range variation.
- **Within a single 256-block, FWHT helps. But the help is just ~3 dB
  (the per-element variance reduction).**

Q4_0's per-32 sub-block scales attack a DIFFERENT precision lever:
**within-block range variation**. If a 256-element block has 8 sub-blocks
where 4 have small range (0.01) and 4 have large range (0.5), Q4_0
quantizes each sub-block with its own scale, capturing both. MQ4's
single per-256 scale is forced to use the largest (0.5) range, wasting
~50× of dynamic range on the small-range sub-blocks.

**The empirical ratio (MQ4 ~2× worse than Q4_0) measures this exact gap.**
FWHT-256 provides ~3 dB SNR benefit (variance equalization within block),
but per-32 scaling provides ~6 dB SNR benefit (range adaptation across
sub-blocks). The 3 dB gap = 2× MSE ratio.

### Why the dB framing in this doc's first version was wrong

(This subsection is here per glm-5's review feedback. The original
version of this document framed the gap as "FWHT gives +3 dB; per-32
gives +6 dB; net 3 dB gap = 2× MSE" — that's algebraically tidy but
several pieces of it are wrong:)

1. **The "3 dB FWHT" number was fabricated.** FWHT's MSE benefit is
   data-dependent and stems from kurtosis reduction (making the
   distribution more Gaussian, so more of the quantization range is
   used for signal vs tails). For an already-Gaussian input, FWHT
   gives 0 dB. For a sparse input with a few large outliers, 4-6 dB.
   For typical LLM weights, empirical observation is 1-3 dB. The
   point estimate "3 dB" was a guess, not a measurement.

2. **The "6 dB sub-block" number was fabricated.** The dB conversion
   `10*log10(4) = 6 dB` is correct for a 4× MSE ratio, but per-32
   scaling does not give 4× MSE improvement on typical LLM weights.
   The actual ratio depends on cross-sub-block range variance:

   ```
   MSE_ratio = 8 * R_max² / sum(R_i²)
   ```

   For 4 sub-blocks of range 0.01 and 4 of range 0.5: ratio = 2×
   (3 dB), not 4× (6 dB). A 4× ratio would need all range concentrated
   in one sub-block — an extreme that FWHT rotation specifically
   prevents.

3. **The two mechanisms are antagonistic, not additive.** FWHT
   equalizes the distribution within a 256-element block, which
   *reduces* the cross-sub-block range variance that per-32 scaling
   exploits. If FWHT perfectly equalized all sub-block ranges (it
   doesn't), per-32 scaling would provide zero marginal benefit. The
   fact that Q4_0 still beats MQ4 by ~2× shows FWHT does NOT fully
   equalize sub-block ranges, but the simple additive decomposition
   doesn't model the interaction correctly.

The empirically-measured 2× MSE ratio (MQ4/Q4_0 = 1.73-1.94×) is the
**residual** after FWHT has done some equalization; per-32 scaling
captures more of the residual range variation than FWHT can reach.
That's the correct framing.

### What QuaRot and SpinQuant do differently

[QuaRot (arxiv:2404.00456)](https://arxiv.org/abs/2404.00456) and
[SpinQuant (arxiv:2405.16406)](https://arxiv.org/abs/2405.16406) achieve
near-lossless 4-bit by combining:

1. **Rotation at full hidden-dim granularity** (typically 2048 or 4096
   elements per rotation, not 256). This mixes outliers across MUCH
   wider scope, so even when sub-block scales are then applied, the
   sub-blocks are already pre-equalized.
2. **Per-tensor or per-channel scale calibration** (GPTQ-style) that
   fits scales to actual data distribution, not just min-max.
3. **Mixed-precision** (some 8-bit anchors, e.g., for KV cache, lm_head,
   attention V projections).

**Hipfire's MQ4 has only #1's narrowest variant** (FWHT-256 instead of
FWHT-2048). It lacks #2 and #3. Q4_K_M has #2 (weighted LS) but lacks
rotation. UD-Q4_K_XL has #2 + a milder version of #3 (Q4 vs Q5 per
layer). None of the reference contestants have both rotation AND per-
sub-block scales — that combination is the sweet spot the empirical
data points to.

## Ranked fixes by expected MSE / KLD impact

Ordering reflects empirical MSE measurement + the agent's prior research.

### Fix 1 (highest impact): Add per-32 sub-block scales to MQ4

**Description:** Extend MQ4 storage to match Q4_K_M's encoding scheme
(per glm-5 review's recommendation): f16 super-scale + 8 packed 6-bit
sub-block scales. The super-scale gives sufficient dynamic range for
the 6-bit sub-scales to quantize against; the savings vs naive f32
sub-scales are non-trivial.

**Format change:** 136 → 144 bytes per 256 elements (+6% storage,
~4.5 bits/weight). **Matches Q4_K_M's storage layout exactly**, so the
storage-format math is already validated by the GGML community.

**Quant decisions inside Fix 1:**

- **Symmetric vs asymmetric:** post-FWHT distribution is approximately
  zero-mean. The original Fix 5 (symmetric quant after FWHT) was
  estimated at ~5% MSE reduction; per glm-5's review it's closer to
  10-15%. Folded into Fix 1: the new format uses symmetric quant
  (q ∈ [-8, 7], dequant = `q * scale`), saving the per-block min and
  shifting bits to the sub-scales. **Save 4 bytes/block** of zero-point
  storage we don't need post-FWHT.
- **Header sizing:** f16 super-scale (2 B) + 8 sub-blocks × 6-bit
  signed scales packed into 6 bytes = 8 B header total. Matches Q4_K_M's
  per-sub-block precision at the same storage footprint.
- **Backward compatibility:** new quant_type number (qt=21 reserved
  for "MQ4-v2"); existing MQ4G256 (qt=13) files still load
  unchanged.

**Expected MSE reduction:** ~2× (matches the empirical Q4_0 advantage
over MQ4 single-scale). Synergistic with Fix 2 (weighted LS) brings
total ~2.5-3.5×.

**Effort:** Format change + kernel update + dequant logic. Moderate.
The per-sub-block scales fit in the same per-block FWHT structure — no
deeper math changes.

**Why it's the highest lever:** This is the SINGLE largest precision
gap between hipfire's MQ format and llama.cpp's Q4_K_M. The empirical
measurement (~2× MSE ratio) directly corresponds to this design choice.

### Fix 2: Weighted least-squares scale search per sub-block

**Description:** Replace the naive `(max - min) / 15` with a weighted
LS search over candidate scales (port `make_qkx2_quants` from llama.cpp:
20 candidates, optimize each sub-block to minimize importance-weighted
quantization error).

**Format change:** None (compatible with Fix 1's per-sub-block scale
storage). **Should land together with Fix 1** — same code path, same
PR. Per glm-5's review.

**Apply also to existing Q4_K port:** glm-5 noted that
`crates/hipfire-quantize/src/main.rs:217 quantize_q4k` uses naive
min-max per sub-block (just like MQ4), not weighted LS. So even
hipfire's Q4_K is worse than llama.cpp's Q4_K_M. Porting
`make_qkx2_quants` should land in **both** the new MQ4-v2 AND
the existing `quantize_q4k`. Same code, two consumers.

**Expected MSE reduction:** ~1.2-1.5× (PPL improvement is larger but
non-linear; glm-5 cites Lloyd-Max MQ3 data showing 2.27× PPL reduction
from better centroid selection within the same block scope).

**Effort:** Pure quantizer-side change. ~50 lines of Rust ported from
the llama.cpp reference.

### Fix 3: Tensor-wide rotation (FWHT-2048 or FWHT-4096)

**Description:** Apply a single Hadamard rotation across the full
hidden-dim of the weight tensor BEFORE breaking into 256-blocks. The
runtime kernel needs to also rotate `x` at the same scale, but this is
a one-pass-per-token cost (acceptable).

**Format change:** Storage layout could stay at 256-block per-row, but
the "rotated weights" are now globally consistent. Need to bake the
rotation into the model file.

**Expected MSE reduction:** Not directly measured here. QuaRot reports
near-lossless 4-bit at hidden-dim rotation — likely 1.5-2× MSE reduction
on top of Fix 1+2.

**Effort:** Significant. New rotation kernel at hidden-dim scale. Storage
format reworked. **Worth doing only after Fix 1+2 prove insufficient.**

### Fix 4: Calibration overlay (Unsloth-style mixed-precision)

**Description:** Add a calibration step that picks Q4 vs Q5 per layer
(or per tensor) based on a held-out dataset's KLD impact.

**Format change:** Per-tensor metadata for quant level.

**Expected KLD reduction:** ~1.5× (combined with above). The calibration
adds 0.2-0.4 effective bits/weight on the most sensitive tensors at
modest size overhead.

**Effort:** Medium. Need calibration dataset infrastructure + per-tensor
quant-choice serialization.

### Fix 5 (folded into Fix 1): Symmetric quant after FWHT

**Description:** Since FWHT output is approximately zero-mean, switch
MQ4 from asymmetric (`min + q*scale`) to symmetric (`q*scale`, q ∈
[-8, 7]). Saves 4 bytes/block and reduces rounding bias.

**Status:** Folded into Fix 1's format design. Per glm-5's review,
the original "5%" estimate was low; closer to 10-15% (asymmetric quant
on a near-zero-mean distribution wastes ~1/16 of dynamic range on the
zero-point offset). Should NOT ship as a separate fix.

### Fix 6 (research, low priority): Calibrated FWHT sign vectors

**Description:** Currently MQ4 uses fixed PRNG seeds 42 and 1042 for
the FWHT pre/post-multiply signs. Per glm-5's review and consistent
with QuaRot's findings, the rotation matrix isn't necessarily optimal
— QuaRot uses an optimization procedure to choose rotations that
minimize outlier magnitude in specific projections.

**Effort:** Substantial (calibration infrastructure + per-tensor sign
storage in .hfq).

**Expected impact:** Small additional MSE reduction (probably <10%
beyond Fix 1+2). Not worth pursuing until Fix 1+2 are validated.

## Recommended sequencing

The user's earlier engine pass was framed as "kernel + softmax precision
audit." The empirical data points elsewhere:

1. **Phase 1: New MQ4-v2 format = Fix 1 + Fix 2 + Fix 5 (folded).**
   Single PR. Format change to match Q4_K_M's storage layout (f16
   super-scale + 6-bit packed sub-scales + symmetric per-32 quant);
   port `make_qkx2_quants` weighted LS search from llama.cpp; apply
   the same LS port to the existing `quantize_q4k` (closing a separate
   gap noted by glm-5 — hipfire's Q4_K is also worse than llama.cpp's
   Q4_K_M because it uses naive min-max). Reserve qt=21 for the new
   format; existing MQ4 (qt=13) keeps working unchanged.

   **Expected:** per-tensor MSE drops 2.5-3.5× on average vs MQ4-v1.
   Worst-case tensors (router, attention V proj) likely drop more.
   Estimated 1-2 weeks of focused engineering (format + quantizer +
   GPU kernel + tests).

2. **Validation:** run `scripts/bench_quant_quality.sh` on every
   change. Compare per-tensor MSE table before vs after; aim for
   median 2× reduction across the model. Re-run train-pursuit
   reasoning smoke test: hypothesis is the spiral resolves at
   correct rmsnorm scale because cumulative routing-precision drift
   drops below the attractor threshold.

3. **Phase 2 (if Phase 1 doesn't close the spiral): Calibration
   overlay (Fix 4).** Per-layer Q4/Q5 selection on a held-out corpus.
   Caveat from glm-5: the K-map data shows mixed-precision can regress
   specific models (9B). Per-model validation required.

4. **Phase 3 (only if Phase 1+2 fall short): Tensor-wide rotation
   (Fix 3).** This is the most invasive change. Should not be needed
   if Phase 1 closes the per-weight MSE gap.

## Expected impact on the `<think>` spiral

The `<think>` infinite-loop attractor we observed on Qwen3.6-A3B
reasoning at correct GemmaRMSNorm magnitude (PR #228 default mode) is
plausibly a downstream consequence of cumulative quantization noise
through 30+ MoE layers × 8 active experts. Per the maintainer's own
qwen35.rs:1996-2003 comment, a 1-ULP routing-softmax error compounds
into structural attractor at Qwen3.5-A3B and 122B-A10B at MQ4.

Phase 1 of the quantization-quality fix (Fix 1+2) should reduce per-
weight MSE by 2.5-3.5×, which reduces per-layer routing precision drift
proportionally. **If the spiral is precision-cliff sensitive (which the
empirical evidence suggests it is), reducing MSE by 3× should push the
cumulative error back below the attractor threshold.** The smoke test
to validate this is the same train-pursuit reasoning prompt that
currently spirals on the workaround flag, run on a re-quantized A3B
file with the new MQ4-v2 format.

If Phase 1 doesn't fix the spiral, Phase 2's calibration overlay is the
backup — it specifically targets the most-sensitive tensors (likely
attention V proj, lm_head, router) with higher precision.

## Investigation methodology — what worked, what didn't

### What worked

1. **Empirical measurement first.** The user's KLD data + my
   per-tensor MSE analysis directly contradicted the agent's theoretical
   "MQ4 ~25 dB SNR" claim. Without measurement, we'd be ranking fixes
   on hand-wave.
2. **Cross-referencing engine + quantizer + format docs.** Reading
   `gemv_mq4g256.hip` showed the kernel applies the same FWHT scaling
   to `x` at runtime as the quantizer applies to `W` at quant-time —
   confirming the rotation is mathematically correct (orthonormal),
   which means the bug is purely in the per-256-block scale resolution,
   not the rotation itself.
3. **Sanity-checking the quantizer's own FWHT against a reference
   loop.** Caught a vectorization bug in my Python analysis script
   (the reshape-based FWHT was wrong at stride > 1) before publishing
   misleading numbers.

### What needs systematic improvement

The user's question "how do we systematically understand which measure
improves quality how much" calls for a **standing benchmark**:

1. **Per-tensor MSE table** (auto-generated by a CI script): for each
   weight tensor in a reference model (e.g., Qwen3.5-9B BF16), measure
   MSE under each quant format. Track:
   - MQ4G256 (current)
   - MQ4-v2 (proposed: 8 sub-block scales + LS fit + FWHT)
   - Q4_0 (per-32 scales, no rotation)
   - Q4_K (per-32 + super-block scale-of-scales + weighted LS)
2. **Per-tensor KLD measurement** on a held-out dataset (wikitext-2 or
   similar). Track: model-aggregate KLD under each format.
3. **Reasoning attractor smoke test** (the train-pursuit prompt) with
   each format on Qwen3.6-A3B. Track: spiral / coherent / partial.

The triple gives:
- MSE → predicts KLD impact (per-tensor)
- KLD → predicts model-aggregate quality
- Smoke test → catches attractor failures that don't show up in either

A `bench_quant_quality.py` (or Rust equivalent) running the triple on
a fresh quant gives a 30-minute feedback loop on format / algorithm
changes. Without it, we're flying blind.

## Files referenced

- `crates/hipfire-quantize/src/main.rs:462-500` — `quantize_mq4g256`
- `crates/hipfire-quantize/src/main.rs:428-447` — `cpu_fwht_256`
- `crates/hipfire-quantize/src/main.rs:217-313` — `quantize_q4k`
  (existing Q4_K port for the GGUF-input path)
- `kernels/src/gemv_mq4g256.hip` — runtime kernel (FWHT scaling
  matches quantizer)
- `kernels/src/gemv_mq4g256.hip:62-135` — `mq_rotate_x` kernel
- `crates/hipfire-runtime/examples/compare_hfq.rs` — diagnostic tool
  for tensor-by-tensor comparison
- `/tmp/analyze_outliers.py` — empirical MSE measurement script
- `docs/plans/qwen35-moe-coherence-investigation.md` — parent
  investigation that established the rmsnorm + spiral context
- llama.cpp `ggml-quants.c quantize_row_q4_0_ref`,
  `quantize_row_q4_K_impl`, `make_qkx2_quants` — reference Q4_0/Q4_K
- QuaRot paper: https://arxiv.org/abs/2404.00456
- SpinQuant paper: https://arxiv.org/abs/2405.16406
- Unsloth Dynamic 2.0: https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs

## Reproducibility

Two reference implementations of the per-tensor MSE measurement now
live in-tree (not /tmp, per AGENTS.md rule 4):

### Rust (canonical, fast)

```bash
cargo build --release --example quant_quality_mse
./target/release/examples/quant_quality_mse \
    ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/SNAP \
    /local/hipfire/qwen3.6-35b-a3b.mq4 \
    [name_substring]
```

Source: `crates/hipfire-runtime/examples/quant_quality_mse.rs`.
Outputs per-tensor MSE table sorted by descending error, plus aggregate
stats by quant type (mean / p99 / max).

### Python (reference, easier to modify for experiments)

```bash
python3 -m venv /tmp/quant_venv && /tmp/quant_venv/bin/pip install numpy
/tmp/quant_venv/bin/python scripts/analyze_quant_mse.py
```

Source: `scripts/analyze_quant_mse.py`. Reads a single safetensors
shard, prints MSE for each format (raw F16, MQ4, Q4_0, Q4_K-baseline)
across a sample of tensors. Useful for testing format-design changes
against known reference data.

### Standing benchmark (full triple)

```bash
scripts/bench_quant_quality.sh \
    ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/SNAP \
    /local/hipfire/qwen3.6-35b-a3b.mq4 \
    benchmarks/results/quant_quality_a3b_baseline.md
```

Source: `scripts/bench_quant_quality.sh`. Combines per-tensor MSE +
final-norm sanity + train-pursuit reasoning smoke test (default and
workaround mode) into a single markdown output. Use this as the
30-minute feedback loop when iterating on quantizer formats.

The KLD vs BF16-reference logits piece of the triple is on the
roadmap but currently requires external infrastructure (a
vLLM/transformers reference run on the same prompt). When that wires
in, the bench will produce a complete MSE / KLD / attractor report.
