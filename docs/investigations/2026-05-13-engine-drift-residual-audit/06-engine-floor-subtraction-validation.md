# Engine-floor subtraction framework — 10-minute validation (2026-05-13)

> **REVISION 2 (2026-05-13, late):** The original tables in this doc
> used `hipfire Q3.5-0.8B Q8 = 0.0796` as the *shared* engine floor,
> implicitly assuming all hipfire quants ran on the same matmul kernel
> as Q8. They don't. MQ4/MFP4/HFP4 have always used FP16 WMMA in
> prefill, while Q8 was on Tier-2 FP32-accumulator
> `gemm_q8_0_batched_chunked` until PR-#248. The 0.0796 was a Q8-kernel
> artifact, not a DeltaNet-engine floor.
>
> PR-#248's Tier-3 WMMA Q8 swap drops the Q8 measurement to **0.0041
> nats** (gfx1100 prefill, n=512, CI 0.004009–0.004162) — within 2.7×
> of llama.cpp Q8_0's 0.0015 cross-engine floor instead of 53× away.
>
> The right shared floor for hipfire WMMA-prefill quants is **~0.0041
> nats**, not 0.0796. Tables below show OLD (using inflated Tier-2
> floor) and NEW (using Tier-3-corrected floor) numbers side-by-side.
> An F1 follow-up experiment (explicit bf16 round-trip at module
> boundaries) was empirically falsified at the same time — see the
> "F1 falsified" section at the end.

Companion to `05-pattern-hunt-results.md`. After D0.3+F3 showed
hipfire's kernels are at the fp64 ideal and the audit's "engine drift
floor" is HF's bf16-cast pattern + DeltaNet recurrence amplification,
the natural follow-up question:

> Can we use `KLD(any hipfire quant) − KLD(hipfire Q8 floor)` as a
> "weight-quantization-only cost" that's comparable to llama.cpp's
> analogous `KLD(QX) − KLD(Q8_0)` subtraction?

Short answer: **yes within a fixed (engine, architecture) pair**,
**yes across engines with one refinement** — the engine floor turns
out to be architecture-dependent, not engine-dependent.

## Method

For small KLD perturbations, KL is approximately quadratic in logit
perturbation. The variances of independent zero-mean noise sources
(embedding storage, engine arithmetic precision, weight quantization)
approximately add. So if engine and embedding noise are constant
across quant choices, `KLD(QX) − KLD(Q8_baseline)` cancels them out,
leaving the incremental weight-quantization cost of QX over Q8.

Tools used (no new GPU work needed for validation — all input data
already on disk):

- `scripts/kld_engine_floor_subtraction.py` — analysis script
- existing per-seq HFKSEQ kldseqs from prior eval runs
- audit baseline kldseq at `/data/cache/hipfire/audit-2026-05-13/eval/`

## Results

### Engine floor is architecture-dependent (key finding)

| model + arch | n | hipfire Q8 KLD vs HF |
|---|---:|---:|
| Q3-0.6B dense (no DeltaNet) | 20 | **0.0098** |
| Q3.5-0.8B DeltaNet | 5 | **0.0796** |
| (reference: llama.cpp Q3.5-9B Q8_0) | 1175 | 0.0163 |

Hipfire's dense-model floor (Q3-0.6B Q8 = 0.0098) **matches
llama.cpp's Q8_0 floor** at 9B (0.0163). The 8× larger floor on
Q3.5-0.8B Q8 is the DeltaNet recurrence amplifier — bf16-cast drift at
upstream module boundaries gets amplified ~4× per DeltaNet block (per
audit obs #4 in `00-summary.md`). Dense transformers have no such
amplifier.

This refines the engine-floor framework: it's not "hipfire vs
llama.cpp" — it's "DeltaNet-architecture-with-fp32-intermediates vs
anything-else". The same hipfire kernels on a dense Q3 model produce a
llama.cpp-equivalent floor.

### Hipfire-internal subtraction at Q3.5-0.8B (clean ordering)

Q8 floor = 0.0796 (audit baseline, n=5).

| variant | KLD vs HF | − Q8 floor | rank |
|---|---:|---:|---:|
| MQ4 | 0.6721 | **0.5925** | best |
| MFP4 | 1.3012 | 1.2216 | 2 |
| MFP4-L4 | 1.3870 | 1.3074 | 3 |
| HFP4-L4 | 1.6248 | 1.5452 | 4 |
| HFP4 | 1.6419 | 1.5623 | worst |

Ordering is stable and matches the qualitative finding in the
2026-05-11 Pivot (MQ format is structurally noisier than community
K-quants — confirmed here, but MQ4 still beats hipfire's other 4-bit
options at 0.8B).

### Cross-engine Q3.5-9B (incomplete — Q8 floor not measured)

Hipfire 9B Q8 was attempted but didn't complete in budget:
`lm_head dtype=Q8_0; F16 batched fast path=false` makes prefill
mode fall back to ~7 tok/s scoring speed on gfx1151, ~25 min for
5 chunks. Will measure on gfx1100 in a follow-up.

In the interim, a conservative bracket (lo = 0.08 = 0.8B Q8 value,
hi = 0.50 = generous linear-scaling estimate) is used. Even under the
**high** floor estimate (most charitable to hipfire), the cross-engine
ratios are:

| bit width | hipfire raw KLD | hipfire Δ (− 0.5) | llama.cpp Δ | ratio |
|---|---:|---:|---:|---:|
| 6-bit (MQ6 vs Q6_K) | 0.6254 | 0.125 | 0.009 | **14.5×** |
| 4-bit (MQ4 vs Q4_K_M) | 0.8171 | 0.317 | 0.109 | **2.9×** |
| 4-bit (MQ4 vs UD-Q4_K_XL) | 0.8171 | 0.317 | 0.051 | **6.3×** |
| 3-bit (MQ3 vs UD-Q3_K_XL) | 2.6221 | 2.122 | 0.125 | **17×** |
| 3-bit (MQ3-Lloyd vs UD-Q3_K_XL) | 1.6913 | 1.191 | 0.125 | **9.5×** |

MQ-family quants carry **3-17× more weight-quantization noise than
equivalent-bit-width K-quants**, even after the floor subtraction. This
is real weight-quantization quality gap, not an engine-precision
artifact.

## Interpretation

1. **The subtraction framework works** for principled quant ranking
   within a fixed (engine, architecture) pair. Hipfire-internal MQ4 vs
   MFP4 vs HFP4 ranking at 0.8B is preserved after floor subtraction
   (it has to be, since the floor is constant across quants), and the
   ABSOLUTE incremental cost (0.59 nats for MQ4) is now interpretable
   as "what MQ4 adds over Q8" rather than confounded with "what
   hipfire+DeltaNet adds over llama.cpp+anything".

2. **Cross-engine comparison is sound at first order**, *if* you
   subtract each engine's own architecture-matched floor. For Q3.5,
   that means using hipfire's DeltaNet Q8 floor for hipfire numbers
   and llama.cpp's matching Q8_0 floor for llama.cpp numbers.

3. **The Q4_K_M vs MQ4 gap is real**, not an engine artifact. The
   2.9-6.3× ratio (after floor subtraction) confirms the 2026-05-11
   Pivot's qualitative finding: MQ-family quants need calibration
   (AWQ Stage A→B/C) to close on K-quant quality.

4. **A dense-model port of the same kernels has no floor problem.**
   Q3-0.6B at 0.0098 nats is essentially llama.cpp-equivalent. This
   means the entire "engine drift" story is a DeltaNet-specific
   phenomenon, not a general hipfire weakness.

## Limitations

- **Hipfire 9B Q8 floor not measured.** Used a conservative bracket
  (0.08-0.5) instead. On gfx1100 with the same code, this would take
  ~10-15 minutes of wall time to measure cleanly. Pending follow-up.
- **First-order approximation.** Breaks down at very coarse quants
  (MQ2/MQ3) where the noise is large enough that second-order
  interaction with the engine floor becomes non-negligible. Use with
  caution for sub-3-bit quants.
- **Different sample sizes across rows.** The 0.8B Q8 floor is n=5
  chunks; the 0.8B quants are n=256; the 9B quants are n=1175. CI
  bounds aren't shown but would be wider on the n=5 floor.
- **gfx1100 vs gfx1151.** The 9B hipfire MQ rows are gfx1100; the
  llama.cpp anchors are gfx1151. Both archs should produce
  bit-equivalent KLDs (same algorithms) but a strict comparison should
  control for this.

## Reproducer

```bash
# Aggregate existing data into a working dir
mkdir -p /tmp/kld_v2/{08b_quants,9b_hip,9b_gguf}
cp /data/cache/hipfire/audit-2026-05-13/eval/baseline_q8f16.kldseq /tmp/kld_v2/q8_floor.kldseq
cp benchmarks/quality-baselines/results/2026-05-11-cohort-phase-a-0.8b-step-0.5+1+2/per-variant/*.kldseq \
   /tmp/kld_v2/08b_quants/
cp benchmarks/quality-baselines/results/2026-05-08/per-seq/qwen3.5-9b.mq*.kldseq \
   benchmarks/quality-baselines/results/2026-05-11/per-seq/qwen3.5-9b.mq*.kldseq \
   /tmp/kld_v2/9b_hip/
cp benchmarks/quality-baselines/results/2026-05-10/per-seq/qwen3.5-9b.gguf-*.kldseq \
   /tmp/kld_v2/9b_gguf/

# Run analysis
.venv/bin/python scripts/kld_engine_floor_subtraction.py \
    --hipfire-q8 /tmp/kld_v2/q8_floor.kldseq \
    --hipfire-08b-dir /tmp/kld_v2/08b_quants \
    --hipfire-9b-dir /tmp/kld_v2/9b_hip \
    --gguf-9b-dir /tmp/kld_v2/9b_gguf \
    --dense-control benchmarks/quality-baselines/results/2026-05-12-deltanet-discriminator/per-seq/qwen3-0.6b.q8f16__gfx1151__kv-q8__per-token__c20.kldseq
```

Once the 9B hipfire Q8 measurement lands on gfx1100, add
`--hipfire-q8-9b <path>` to drop the bracket and produce a single
measured-floor cross-engine column.

## Open follow-ups

- **Run hipfire Q3.5-9B Q8 on gfx1100** (`eval_hipfire --max-chunks 20
  --scoring-mode prefill --kv-mode q8`). Estimated ~10-15 min wall
  time. Removes the floor bracket and gives the tightest cross-engine
  numbers.
- **Fix the Q8 lm_head prefill fast-path.** Currently
  `F16 batched fast path=false` for Q8_0 lm_head means Q8 prefill is
  ~7 tok/s on gfx1151 (vs ~50+ tok/s for MQ4 prefill). Out of scope
  for this validation but a real perf gap.
- **F1 hypothesis test:** falsified. See section below.

## Engine-floor correction — Tier-2 Q8 was the odd one out

The original framing in this doc treated `hipfire Q3.5-0.8B Q8 = 0.0796`
as a shared DeltaNet engine floor and subtracted it across MQ4/MFP4/
HFP4. That was wrong. MQ-family quants have always used FP16 WMMA in
prefill, putting them in the precision-matched-to-HF regime. Q8 was
the only quant on FP32-accumulator Tier-2 kernels, which over-preserved
precision relative to HF's bf16 reference. Hipfire MQ4 was running with
its *own* engine floor (~0.0041) all along; doc 06 rev-1 was just
subtracting a Q8-specific artifact and labeling it "engine floor".

### Corrected hipfire-internal table for Q3.5-0.8B

Tier-3 Q8 measurement (PR-#248, gfx1100 prefill, n=512): **0.0041 nats**.

| variant | raw KLD | OLD: − 0.0796 | NEW: − 0.0041 | shift |
|---|---:|---:|---:|---:|
| MQ4 | 0.6721 | 0.5925 | **0.6680** | +0.075 |
| MFP4 | 1.3012 | 1.2216 | **1.2971** | +0.075 |
| MFP4-L4 | 1.3870 | 1.3074 | **1.3829** | +0.075 |
| HFP4-L4 | 1.6248 | 1.5452 | **1.6207** | +0.075 |
| HFP4 | 1.6419 | 1.5623 | **1.6378** | +0.075 |

Every variant's "above-floor" cost increased by the same 0.075 nats
constant — the value that was being incorrectly attributed to the
shared engine. Rank ordering and inter-quant ratios are unchanged.
**MQ4's true incremental weight-quantization cost is 0.668 nats, not
0.59 nats.**

### Corrected cross-engine ratios at Q3.5-9B

Under OLD rev-1, the doc used a conservative 0.08–0.50 bracket for
hipfire 9B Q8 because no measurement existed. PR-#248's measurement
shows the real floor is ~0.0041, far below even the bracket's lower
bound. The cross-engine gap is *larger* than rev-1 reported:

| comparison | hipfire raw | OLD Δ (− 0.5 bracket) | NEW Δ (− 0.005 est.) | gguf Δ | OLD ratio | NEW ratio |
|---|---:|---:|---:|---:|---:|---:|
| MQ6 vs Q6_K | 0.6254 | 0.125 | **0.620** | 0.009 | 14.5× | **70×** |
| MQ4 vs Q4_K_M | 0.8171 | 0.317 | **0.812** | 0.109 | 2.9× | **7.4×** |
| MQ4 vs UD-Q4_K_XL | 0.8171 | 0.317 | **0.812** | 0.051 | 6.3× | **16×** |
| MQ3 vs UD-Q3_K_XL | 2.6221 | 2.122 | **2.617** | 0.125 | 17× | **21×** |
| MQ3-Lloyd vs UD-Q3_K_XL | 1.6913 | 1.191 | **1.686** | 0.125 | 9.5× | **13×** |

The rev-1 "most charitable" 0.5-nat floor bracket was so generous it
materially understated the gap. With the correct floor, hipfire's
MQ-family quants carry **7–70× more weight-quantization noise** than
equivalent-bit-width K-quants — confirming the 2026-05-11 Pivot's
qualitative finding and quantifying it more precisely.

(The 9B hipfire Q8 measurement on gfx1100 is still nominally pending
to remove the "~0.005 est." extrapolation, but PR-#248's 0.8B
measurement is tight enough that the conclusion is robust to the
exact 9B value.)

## What "engine floor" means in the framework

A clarification that should have been in rev-1: the engine floor isn't
the Q8 measurement specifically — it's the irreducible
matmul-precision-class noise shared by all quants using the same
matmul kernel family. If different quants use different matmul kernels,
they have different floors and the subtraction isn't apples-to-apples.

The framework works *within a matmul-precision class*. With PR-#248,
all hipfire prefill quants on Q3.5 now share the FP16 WMMA precision
class → shared floor ~0.004 nats → subtraction is meaningful and
comparable to llama.cpp's Q8_0-subtracted deltas (llama.cpp's prefill
also uses FP16-class kernels throughout).

## F1 falsified — boundary casts can't match HF's matmul precision

The original doc 06 closed with an F1 proposal: insert explicit BF16
round-trip casts at every HF module boundary in hipfire's forward to
match HF's bf16 precision pattern. Implementation lives in (now
reverted) commits-not-landed:
`kernels/src/bf16_roundtrip.hip` + `Gpu::bf16_rt_if` helper +
`f1_bf16_rt_enabled()` OnceLock + ~16 `gpu.bf16_rt_if(...)` call sites
in `forward_scratch_layers` (LA + FA paths, both pre- and post-residual).
Behind env-gate `HIPFIRE_F1_BF16_RT=1`.

### F1 result (Q3.5-0.8B Q8 per-token, Tier-2 build, n=5)

| variant | mean KLD | 95% CI |
|---|---:|---|
| baseline (no F1) | 0.0796 | [0.0695, 0.0899] |
| **F1 (HIPFIRE_F1_BF16_RT=1)** | **0.0865** | [0.0809, 0.0931] |
| per-chunk paired Δ (F1 − base) | **+0.0069** | [+0.0017, +0.0125] |

Paired CI excludes zero → F1 is a real (small) regression, not noise.

### Why F1 failed

The cast-at-boundary mechanism is structurally different from HF's
actual precision profile:

| step | HF (bf16 matmul) | F1 (fp32 matmul + bf16 round-trip at output) |
|---|---|---|
| input precision into matmul | bf16 | fp32 (cast back from prior bf16-rt) |
| per-multiply precision | bf16 × bf16 | fp32 × fp32 of bf16-valued operands |
| accumulator | fp32 | fp32 |
| output cast | fp32 → bf16 | fp32 → bf16 |

The per-multiply step is where the precision difference lives. F1's
fp32 × fp32 of bf16-valued operands produces *exact* fp32 products
(no rounding), then sums them in fp32. HF's bf16 × bf16 rounds each
product to bf16 before summing. The per-product rounding errors
compound; F1's pattern doesn't reproduce that error distribution. The
boundary cast at the end is irrelevant to closing the gap because the
gap originates inside the matmul.

PR-#248's Tier-3 swap to FP16 WMMA accumulators *does* match the
per-multiply precision (FP16 multiply, FP16 partial sums). That's why
Tier-3 drops the floor 19× while F1 cannot — it's a matmul-kernel
problem, not a glue-code problem.

### What F1 could not have been (variants considered)

- **F1b — cast inputs to bf16 before each matmul.** Algebraically
  equivalent to F1a since the bf16-rt at output of op N becomes the
  bf16 input of op N+1. Same regression expected.
- **F1c — force matmul accumulator to FP16 in software.** Truncate
  partial sums after every N multiplies. High kernel-launch cost,
  partial match, vastly inferior to doing it in hardware (Tier-3).
- **F1d — replace FP32 matmul with BF16-accumulator matmul.** This is
  the *right* fix in principle, but it's functionally identical to
  PR-#248 with BF16 instead of FP16. PR-#248 already shipped this for
  Q8; adding a BF16 variant would be redundant.

**None of the F1 variants beat PR-#248 in either correctness or
perf.** Anything that genuinely closes the floor requires touching the
matmul kernel itself, not the surrounding glue.

### F1 disposition

Reverted. The empirical falsification + the structural argument
together make F1 a dead end. PR-#248 is the canonical fix for this
class of problem.

## Open follow-ups (revised)

- **9B hipfire Q8 on gfx1100 prefill** still useful for tightening the
  cross-engine numbers — but the conclusion is already robust to the
  estimated floor since the OLD (− 0.5) bracket already excluded
  parity. Lower priority now.
- **Re-measure 0.8B MQ4/MFP4/HFP4 with PR-#248 Tier-3 build.** Their
  Q8 floor is already at WMMA precision (no change), but if any
  fused-prefill optimizations in PR-#248 touched MQ-prefill paths
  too, their numbers may shift. Quick check.
- **Calibration roadmap (AWQ Stage A→B/C).** Now better-justified than
  ever: the remaining 0.67-nat MQ4 incremental cost is pure weight-
  quantization noise, ~7× worse than Q4_K_M at the same bit width.
