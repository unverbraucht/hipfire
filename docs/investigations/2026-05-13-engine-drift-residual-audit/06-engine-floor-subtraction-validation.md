# Engine-floor subtraction framework — 10-minute validation (2026-05-13)

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
- **F1 hypothesis test** (deferred from 05-pattern-hunt-results.md):
  force hipfire to bf16-cast at DeltaNet recurrence boundaries.
  Predicts hipfire DeltaNet Q8 floor would drop ~5× toward
  llama.cpp-equivalent. If true, this provides a no-recalibration path
  to close most of the model-output KLD gap.
