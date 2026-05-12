# MQ4K — proposal to absorb Q4_K's per-32 sub-scaling onto MQ4's rotation lever

**Status:** Proposal, awaiting Stage 1 validation
**Date:** 2026-05-12
**Author:** synthesized during Phase A Step 5a + GGUF-anchor reality check
**Related:** `qwen35-mq4-quality-gap.md` §1.5 (cohort-driven re-framing of the lever
analysis), `hfp4-fivetide-rebuttal-perspective.md` (the original "calibration is
the dominant lever" framing this doc revises)

## 0. TL;DR

The Phase A Step 5a + 9B GGUF-anchor cross-reference cohort
(`benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-step-5a-9b/comparison.md`)
revealed that at bpw-matched comparison hipfire MQ4 trails Unsloth UD-Q3_K_XL
by **+75% PPL despite hipfire using +0.37 more bits per weight.** The §1.5
framing that "calibration is the dominant lever, will close most of the gap"
isn't supported by the empirical L5c cohort — calibration moves HFP4 a few
percentage points on 9B and can't apply to MFP4 at all (rotation math).

Hipfire's wire format trades **two scale-granularity levers**
(per-32 sub-block scales + per-32 sub-block ZPs) for **one rotation lever**
(FWHT-256). The trade is currently losing on the bpw-matched PPL signal.
This proposal adds the missing scale-granularity levers without giving up
the rotation, and reserves wire-format extensibility for future format
work (online MXFP4-style rotation, per-tensor metadata sidecars, bit-width
flexibility for sibling formats MQ5K/MQ6K).

**Staged approach**:
- Stage 1 (~1.5-2 weeks): quality-only prototype via dequant-to-F16 at load.
  Validates the lever ROI before committing to kernel work. Bench against
  hipfire MQ4 baseline + UD-Q3_K_XL anchor.
- Stage 2 (~2-3 weeks): GPU kernel family, **GATED** on Stage 1 quality
  validation.
- Stage 3 (~1 week): production deployment.

Estimated total: **5-7 weeks for full deployment.** Stage 1 alone is enough
to make a Phase A-grade format-choice decision.

## 1. Strategic context

### 1.1 Where we are after Step 5a

The 9B Step 5a cohort (committed `1c3cd639`) measured calibration's
empirical effect against the existing hipfire wire format family:

- HFP4-L4-L5c on 9B: still +3.2% KLD vs HFP4 baseline (calibration partially
  recovered L4's regression but didn't reverse it)
- MFP4-L4-L5c on 9B: rotation math forbids L5c lever — pure-MSE L4 is the
  ceiling for rotated formats in the per-block-LS framework
- L5c absolute improvement on 9B HFP4 (vs uncalibrated baseline): ~0%
  (no net win)

This is far less impact than the gap doc §2.3 projected ("Phase A delivers
KLD-mean 0.20-0.30 reduction" → translates to ~10-15× KLD improvement on
hipfire's current ~0.8 KLD baseline). Calibration on hipfire's existing
wire format family doesn't have the leverage we assumed.

### 1.2 What the GGUF anchors reveal

The 2026-05-10 GGUF anchor cohort (commit `6c00a558`, gfx1151 per-token
mode) puts hipfire's quality in a different light:

| variant | bpw | KLD | PPL | engine |
|---|---:|---:|---:|---|
| GGUF Q8_0 (anchor) | 8.5 | 0.016 | 9.31 | llama.cpp |
| GGUF UD-Q6_K_XL | ~6.7 | 0.021 | 9.14 | llama.cpp |
| GGUF Q6_K | 6.56 | 0.025 | 9.31 | llama.cpp |
| GGUF UD-Q5_K_XL | ~5.5 | 0.041 | 9.27 | llama.cpp |
| GGUF UD-Q4_K_XL | 5.32 | 0.067 | 9.34 | llama.cpp |
| GGUF Q4_K_M | 5.07 | 0.125 | 8.70 | llama.cpp |
| GGUF UD-Q3_K_XL | ~4.5 | 0.141 | 8.67 | llama.cpp |
| **hipfire MQ4** | **~4.87** | **0.808** | **15.16** | hipfire |
| hipfire HFP4-L4-L5c | ~5.0 | 1.007 | 19.09 | hipfire |
| hipfire MFP4 | ~5.0 | 1.112 | 21.02 | hipfire |

KLD numbers aren't apples-to-apples (cross-engine measurement inflates
hipfire numbers via ~46% tokenizer disagreement). **PPL is the more honest
signal** — and crucially must be **bpw-matched**.

The bpw-matched anchor for hipfire MQ4 (4.87 bpw) is **UD-Q3_K_XL (~4.50 bpw)**,
not UD-Q4_K_XL. Unsloth's UD-Q4_K_XL spends ~0.45 more bits/weight than
hipfire MQ4.

**At matched bpw: hipfire MQ4 PPL 15.16 vs UD-Q3_K_XL PPL 8.67 → +75% gap,
DESPITE hipfire using +0.37 more bits per weight.**

The earlier "+62% vs UD-Q4_K_XL" framing in `qwen35-mq4-quality-gap.md` §2.4
was cross-bpw and made the gap look smaller than it actually is at hipfire's
deployment-bpw zone.

### 1.3 Lever-by-lever structural diff

| lever | hipfire MQ4G256 | llama.cpp Q4_K_M / UD-Q*_K_XL |
|---|---|---|
| Codebook | INT4 uniform | INT4 uniform (same) |
| Per-256 scale | 1 FP32 scale + 1 FP32 ZP | 1 super-FP16 + 1 super-FP16 ZP + 8×6-bit sub-scales + 8×6-bit sub-ZPs |
| Per-32 sub-scaling | none | yes (the "Q4_K" innovation) |
| Rotation | FWHT-256 (offline) | none |
| Scale fitting | min-max (or 3-cand L4) | weighted-LS (`make_qkx2_quants`, 20-candidate) |
| Imatrix LS | none on MQ4 (Step 5b not implemented); partial on HFP4 | yes (UD variants) |
| Per-tensor allocation | hardcoded K-map (default OFF on dense) | imatrix-driven (UD) |

Hipfire trades **2 scale-granularity levers** for **1 rotation lever**. The
gap doc §1.3 predicted rotation would compensate; the cohort says it
doesn't compensate enough at hipfire's bpw zone.

## 2. Theoretical gain from MQ4K = MQ4 + Q4_K levers

What does each lever buy if we apply it to a Q4_K-style wire format with
hipfire's rotation kept?

| lever | estimated KLD improvement | mechanism | data source |
|---|---|---|---|
| **L2 sub-block scales** (per-32 sub-scales + sub-ZPs vs hipfire's per-256-only) | ~30-50% | finer-grain per-block range adaptation. The per-256 single (scale, ZP) loses ~2× MSE vs per-32 finer-grained per gap doc §1.2 measurement | Q4_0 (g=32) vs MQ4G256 MSE gap: ~2× — most of this is L2 |
| **L4 weighted-LS scale fit** (20-candidate `make_qkx2_quants` vs min-max) | ~5-10% | finds local minima the min-max heuristic misses, esp. for skewed blocks | gap doc §2.2 estimate; loosely consistent with our HFP4 L4 cohort (-11% MSE) |
| **L5c imatrix** (activation-weighted LS) | ~30-50% | Q4_K_M → UD-Q4_K_XL gives 1.86× KLD improvement (0.125 → 0.067) at +0.25 bpw | direct anchor measurement, gfx1151 |
| **L5d per-tensor allocation** (UD-style imatrix-driven) | ~20-30% | promotes critical tensors to Q5/Q6/Q8 within budget; UD-Q4_K_XL embeds 24 Q8_0 tensors and 67 Q5_K tensors per our ud_decompile output | Step 6b artifact + UD-Q3_K_XL vs UD-Q4_K_XL delta |
| **L3 rotation** (FWHT-256, KEPT — hipfire's unique lever) | -10% to +30% | helps INT4 on rotated weights, neutral or hurts on E2M1 (per fivetide analysis + 0.8B cohort) | empirical from §1.5 cohort |

**Stacked theoretical end state**: calibrated MQ4K should land **at or below
UD-Q4_K_XL KLD (0.067)** at matched bpw, with rotation as the differentiator.
The current MQ4 sits at ~5-12× this (depending on engine-drift floor — see
§6), so the lift is substantial.

**Concrete target**: cut the 75% bpw-matched PPL gap to UD-Q3_K_XL by
**80-90%** — i.e., calibrated MQ4K PPL should land within 10-15% of
UD-Q3_K_XL on a fair comparison. That's a competitive 4.5 bpw quant.

The gain estimates above are theoretical / lever-by-lever. The Stage 1
prototype (§5) measures actual ROI before kernel investment.

## 3. Forward-thinking wire format

Borrowing the HFP4 design ethos (reserved bits + extensibility) — concrete
layout:

```
MQ4K row layout (per row of K elements)
═══════════════════════════════════════════════════════════════════════

Header (16 bytes):
  magic_short[2]:    "MK"            ← distinct from HFP4 "HF"
  version:           u8              = 1
  format_flags:      u8
    bit 0-1:         rotation_kind
                     00 = none
                     01 = FWHT-256 (offline)
                     10 = FWHT-128 (offline; reserved for future
                                    per-128 rotation variant)
                     11 = MXFP4-online (reserved; runtime applies
                                        block-diag rotation per AMD
                                        ROCm "Advanced MXFP4 with
                                        Online Rotation")
    bit 2-4:         sub_scale_bits - 3
                     001 = 4         ← MQ2K future
                     010 = 5         ← MQ3K future
                     011 = 6         ← MQ4K (this proposal)
                     100 = 7         ← MQ5K future
    bit 5:           has_sidecar_meta
                     1 = per-tensor metadata sidecar exists at a
                         well-known path (.hfq.kmeta.json); runtime
                         loads at model open
    bit 6-7:         reserved
  super_scale:       fp16
  super_zp:          fp16
  n_blocks:          u16
  reserved[6]:       zeros
                     # future fields candidate list:
                     #   - per-tensor calibration recipe ID (1 byte)
                     #   - ZD score quantized to 1 byte
                     #   - per-tensor importance class (4 bits)
                     #   - rotation seed override (2 bytes — supports
                     #     per-tensor randomized rotation experiments)

Per 256-element block (144 bytes — equivalent to Q4_K_M's 4.5 bpw):
  packed_sub_scales: 8 × 6-bit = 6 bytes
                     # at sub_scale_bits=6 (default). Widens for other
                     # sub_scale_bits values:
                     #   bits=4 → 4 bytes, bits=5 → 5 bytes, bits=7 → 7 bytes
  packed_sub_zps:    8 × 6-bit = 6 bytes
                     # same widening as sub_scales
  nibbles:           128 bytes        # 256 elements × 4 bits

Total per row at K=4096:
  16 + 144 × (4096/256) = 16 + 144 × 16 = 2320 bytes
  Aggregate bpw: 2320 × 8 / 4096 = 4.53 bpw

For a 9B model with K=4096 (most tensors):
  ≈ 4.53 bpw per weight (matches Q4_K_M exactly at 4.5)
```

### 3.1 Extension story

The format above is designed so future variants drop into the same wire
format with only the relevant flag bits changing:

| variant | rotation_kind | sub_scale_bits | source bpw | use case |
|---|---|---|---|---|
| MQ4K (this proposal) | 01 (FWHT-256) | 6 | 4.5 | Phase A target, 9B/27B/A3B production |
| MQ4K unrotated | 00 | 6 | 4.5 | Direct Q4_K_M wire-compat (post-dequant); for tooling interop |
| MQ4K with online rotation | 11 | 6 | 4.5 + online cost | Future RDNA-native rotation (no offline FWHT bake) |
| MQ5K | 01 | 7 | 5.5 | Sibling variant — 7-bit sub-scales, more precision |
| MQ3K | 01 | 5 | 3.5 | Sub-4-bit variant for memory-constrained deployment |
| MQ4K + per-tensor metadata | 01 | 6 | 4.5 + tiny sidecar | Runtime carries imatrix + ZD score + calibration recipe ID |

### 3.2 Versioning

- `version: 1` ships with this proposal
- Future major changes that aren't backward-compatible bump `version` →
  loaders reject unknown versions with a clear error
- Backward-compatible additions go in `reserved[6]` (no version bump
  needed; old loaders ignore the bytes)

## 4. Engineering cost breakdown

| component | dev cost | notes |
|---|---|---|
| Wire format design + spec doc | 1 day | This doc + a hipfire `docs/quant-formats/mq4k.md` mirror of `hfp4.md` |
| Quantizer encoder (`quantize_mq4k_2d`) | 3-4 days | Port `make_qkx2_quants` weighted-LS from llama.cpp; add FWHT pre-rotation; reuse the HFP4 row-loop scaffold from §1.3 of `crates/hipfire-quantize/src/main.rs` |
| CPU reference dequant | 1 day | For round-trip tests + the staged ROI-validation path (§5 Stage 1) |
| Quantize-time L5c + L5d hooks | 1-2 days | Reuse imatrix loader from Step 5a (commit `2d050152`); UD-kmap ingestion from Step 6b (commit `679bff46`) |
| Unit + integration tests | 1-2 days | Round-trip, FWHT cancellation, K-map preservation, version handling |
| **Subtotal: quantizer-side** | **~1.5-2 weeks** | Stage 1 ends here for the ROI-validation prototype |
| **GPU kernel family** (gemv + gemm prefill for gfx11 + gfx12) | **2-3 weeks** | THE LONG POLE. Q4_K's sub-scale unpack pattern is well-documented in llama.cpp (`ggml-cuda/q4_K.cu` etc); pattern transfers to HIP but every fused-kernel variant (qkv, qkvza, gate_up, residual) needs porting. ~12-15 HIP kernel files. |
| Runtime loader + arch gate | 2-3 days | DType enum + `is_batchable_la` flip + dispatcher branches (mirrors the PR #235 pattern from CLAUDE.md) |
| Bench cohort + tuning | 3-5 days | 9B + 0.8B cross-check, calibrated variants, full-slice validation |
| **Total** | **5-7 weeks** for full production deployment | |

## 5. Staging proposal

The kernel work is 2-3 weeks. Before committing to that, validate quality
lever ROI cheaply via the Stage 1 prototype.

### Stage 1 — quality-only prototype (~1.5-2 weeks)

**Goal**: measure MQ4K quality at hipfire's bpw without any GPU kernel
work, to decide whether the lever ROI justifies kernel investment.

1. Wire format design + spec doc (§3)
2. Quantizer encoder (port `make_qkx2_quants` + add FWHT pre-rotation)
3. CPU reference dequant
4. **Runtime path: dequant MQ4K → F16 at load time, run through hipfire's
   existing F16 inference kernels.** Slow (~10 min load on 9B; CPU dequant
   of ~4.5 bpw → F16 requires ~36 GB intermediate), but lets us measure
   quality via `eval_hipfire` without any GPU kernel work.
5. Bench cohort:
   - hipfire MQ4 (existing baseline)
   - hipfire MQ4K (no calibration)
   - hipfire MQ4K + L4 weighted-LS
   - hipfire MQ4K + L4 + L5c imatrix
   - hipfire MQ4K + L4 + L5c + L5d (UD-derived kmap from Step 6b)
   - UD-Q3_K_XL anchor (already measured, commit `6c00a558`)

### Stage 1 decision point

- **MQ4K closes ≥ 60% of the bpw-matched PPL gap** → green light Stage 2
  (GPU kernels). The lever ROI is real; kernel investment is justified.
- **MQ4K closes < 30%** → reconsider. Most of the gap might be engine /
  tokenizer drift, not format quality. Better return on fixing tokenizer
  parity (issue-113 §"tokenizer parity") than on building new kernels.
- **MQ4K closes 30-60%** → strategic call. Possibly worth narrowing scope
  to gemv-only first (decode-path kernel), gemm later (prefill-path).
  Decode-only is ~1 week vs full ~2-3 weeks, lets users opt-in to the
  format for inference even before prefill perf parity.

### Stage 2 — GPU kernel family (~2-3 weeks)

**Gated on Stage 1.** Port Q4_K_M's sub-scale unpack pattern from
llama.cpp into hipfire kernels:

- `gemv_mq4k_g256.gfx11.hip` + `gemv_mq4k_g256.gfx12.hip`
- `gemm_qkv_mq4k_wmma.gfx11.hip` + `gfx12` variant
- `gemm_qkvza_mq4k_wmma.gfx11.hip` + `gfx12` variant
- `gemm_gate_up_mq4k_wmma.gfx11.hip` + `gfx12` variant
- `gemm_mq4k_residual_wmma.gfx11.hip` + `gfx12` variant
- gfx906 wave64 dp4a variants (for Phase B' coverage)

### Stage 3 — production deployment (~1 week)

- K-map presets (UD-derived, from `ud_decompile` artifact + tuning)
- Runtime arch gate (`is_batchable_la`)
- Cohort regression suite extension
- CLI registry entries (`qwen3.5:9b-mq4k` etc.)
- Documentation (`docs/quant-formats/mq4k.md`)

## 6. The engine-drift floor — must measure first

A meaningful Stage 1 decision requires knowing the engine-drift floor of
hipfire's KLD measurement. The 9B hipfire MQ4 KLD of 0.808 vs llama.cpp
BF16 includes:

1. **Format quality drift** (what MQ4K aims to reduce)
2. **Engine drift** (hipfire's forward kernels vs llama.cpp's; different
   reduction-tree precision, different rounding, etc.)
3. **Tokenizer drift** (~46% disagreement per `issue-113-quant-quality-eval.md:126`
   — different per-token logit comparison alignment between the two
   tokenizers)

Components 2+3 are **format-independent** — they affect every hipfire
quant equally. The right way to measure them is a **hipfire Q8 baseline
cohort**: quantize 9B as `--format q8f16` (near-lossless), run the standard
cohort. Whatever KLD that produces is the floor — anything format-quality
can only IMPROVE down to that floor.

**Action**: run hipfire Q8 baseline cohort as Task #18 (`~30 min wall, single
variant`) BEFORE Stage 1. Without it, "MQ4K closes X% of the gap" can't be
interpreted (X% of what, exactly?).

Once we know the floor, Stage 1's decision is:

- engine-drift floor F (say F = 0.4 KLD)
- target: UD-Q3_K_XL-equivalent at matched bpw (KLD 0.141 in llama.cpp engine; we don't have hipfire-equivalent yet)
- MQ4K bench result M
- closure: (0.808 - M) / (0.808 - F) → fraction of closable gap MQ4K achieved

## 7. Alternative considered — sidecar approach

Instead of a new wire format, ship per-32 sub-scales as a sidecar file:

- `.hfq.subscales` adjacent to `.hfq`
- Existing MQ4 wire format unchanged
- Runtime loads sidecar, applies per-32 sub-scales at dequant time
- ~3-5 days dev + a kernel modification to use the sidecar values

**Pros**:
- No wire-format proliferation
- Faster to ship (~1 week vs ~5-7)
- Lets us test the L2 hypothesis in isolation

**Cons**:
- Hacky; doesn't compose with other future levers (L4 weighted-LS, L5d
  per-tensor allocation, etc. — each would need its own sidecar)
- Separate file is operationally annoying (downloads, hash checks, etc.)
- Doesn't generalize to sibling formats (MQ3K, MQ5K)

**Verdict**: skip the sidecar. The MQ4K wire format is the right
long-term answer. Stage 1 prototype lets us validate the lever ROI
cheaply without sidecar gymnastics.

## 8. Concrete next actions

Tracked in TaskList:

1. **Task #18: hipfire Q8 baseline cohort** (~30 min wall) — engine-drift
   floor. MUST happen before Stage 1 decision is interpretable.
2. **Task #16: Stage 1 MQ4K quality prototype** (~1.5-2 weeks dev) — wire
   format + quantizer + CPU dequant + dequant-to-F16-at-load runtime path
   + bench cohort. Decision point for Stage 2 GPU kernels.
3. **Task #17: Stage 2 MQ4K GPU kernel family** (~2-3 weeks dev) — gated
   on Stage 1.
4. Stage 3 — production deployment (~1 week) — gated on Stage 2.

Total Phase B-equivalent timeline: **5-7 weeks for full deployment**, with
clear decision points at Stage 1 to abort if the lever ROI doesn't
materialize.

## 9. What this does NOT solve

The MQ4K proposal closes the per-block scale-granularity gap to Q4_K. It
does NOT solve:

- **Tokenizer parity** — the ~46% disagreement between hipfire's tokenizer
  and llama.cpp's affects every hipfire KLD measurement. Independent
  lever; should be tackled separately.
- **Engine reduction-tree drift** — different fp16/fp32 accumulator
  patterns between hipfire kernels and llama.cpp kernels produce small
  systematic logit differences. Probably 0.01-0.05 KLD of the floor.
  Improving requires aligning kernel-path numerics with llama.cpp.
- **Format-independent runtime work** — multi-GPU scaling, KV-cache
  optimization, spec-decode, etc. all orthogonal.
- **Sub-3-bit formats** — MQ4K's spec lets us define MQ3K later, but
  designing it well needs additional research (Lloyd-Max-style codebooks
  outperform uniform INT3, per issue #116).

## References

### In-tree

- `docs/plans/qwen35-mq4-quality-gap.md` §1.5 — the cohort that revised
  the calibration-as-dominant-lever framing
- `docs/plans/hfp4-fivetide-rebuttal-perspective.md` — original framing
- `docs/quant-formats/hfp4.md` — wire format spec that this proposal mirrors
- `crates/hipfire-quantize/src/main.rs` — quantizer (where MQ4K encoder lands)
- `crates/hipfire-quantize/src/bin/ud_decompile.rs` — for Step 6b
  UD-derived kmap ingestion
- `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-step-5a-9b/comparison.md`
  — the 9B cohort that triggered this proposal
- `benchmarks/quality-baselines/results/2026-05-10/per-seq/` — GGUF
  anchor data
- `benchmarks/quality-baselines/external/ud-decompile/qwen3.5-9b-ud-q4_k_xl.kmap.json`
  — Unsloth's per-tensor allocation choices for Step 6b reference

### External

- llama.cpp `ggml-cuda/q4_K.cu` — Q4_K_M kernel reference (port target)
- llama.cpp `tools/quantize/ggml-quants.c` — `make_qkx2_quants` weighted-LS
  implementation (port target)
- AMD ROCm Blog "Advanced MXFP4 with Online Rotation" — reference for the
  reserved `rotation_kind=11` slot
