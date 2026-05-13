# Final verdict — engine-drift residual audit (2026-05-13)

Consolidates Phase 0 + Phase 1 + Phase 2 + Phase 3a findings. **The audit
is closed. No engine surgery action recommended; calibration roadmap
(AWQ Stage A on 0.8B → GPTQ/MR-GPTQ) is the path forward.**

## Evidence summary (5 independent confirmations)

| evidence | finding | implies |
|---|---|---|
| **Phase 0 Check 1** (llama.cpp Q8_0 eval) | KLD 0.0015 vs BF16 ref | 100% of hipfire's 0.08 gap is engine drift |
| **Phase 0 Check 2** (F16 dispatch verify) | Bit-exact F16 storage works correctly | Phase 1d's regression is real engine bias, not loader bug |
| **Phase 1c** (LA layer 0 per-stage) | All stages 0-12 ≤ 0.012 rL2; recurrence at 0.045 | Distributed drift, no single dominant LA stage |
| **Phase 1d** (3 F16-precision probes) | All three regress KLD (+6.8% to +12.4%) | Random Q8 noise was load-bearing for cancellation |
| **Phase 3a** (matched-input recurrence) | Recurrence matches HF at 0.0025 rL2 with bit-exact inputs | Recurrence kernel is faithful |
| **Phase 2** (FA-3 per-stage) | All stages 0-12 within 0.04-0.10 of input bias; no FA-specific dominant kernel | Distributed pattern matches LA |

All five point in the same direction: the 0.08-nat floor is structural,
distributed across ~24 kernels (rmsnorm, projections, conv1d, q_norm,
k_norm, l2norm, alpha-gate, deinterleave, attention compute, sigmoid-mul,
o_proj), each contributing ~0.005 rL2 per stage on uncalibrated Q8
weights. **There is no single localizable kernel to rewrite that closes
a meaningful fraction of the gap.**

## Quantitative breakdown

Achievable target floor: **0.0015 KLD** (llama.cpp Q8_0 measurement).

Current hipfire floor: **0.0796 KLD** (q8f16 5-chunk eval).

Engine drift contribution: ~0.078 nats. Per-kernel per-stage budget:
0.078 spread across ~24 kernels × ~18 LA layers + 6 FA layers stacking
through residual stream + recurrence amplification ≈ each kernel
needs to drop from ~0.005 → ~0.0005 rL2.

To close the gap by surgery would require rewriting essentially every
kernel in the LA and FA pipelines to match llama.cpp's accumulator
order, fusion patterns, and reduction semantics. Estimated effort
(per the combined adversarial review): **4-8 weeks per kernel × ~20
kernels = many months**.

## What we did NOT find (and was hypothesized)

- **F16 embedding fix** would help: did not (Phase 1c+1d).
- **F16 conv weights** would help: actually regressed model output
  KLD by 12.4% (Phase 1d) despite reducing per-stage drift 50% (Phase 1c).
- **Recurrence kernel bug**: kernel is faithful with bit-exact inputs
  (Phase 3a).
- **FA-specific dominant kernel** (attention softmax / q_norm / o_proj
  / sigmoid-mul / etc.): every FA stage 0-12 is within 0.04-0.10 of input
  bias — no localized contributor.
- **Single-kernel surgery target**: combined review's 6-outcome
  decision matrix has no GO outcome match in the data.

## What we DID find (side findings, not floor-closing)

1. **Flash-vs-non-flash precision mismatch at pos 2047** (Phase 2
   outlier analysis): hipfire dispatches `attention_q8_0_kv` for
   pos < 2047 and `attention_flash_q8_0` at pos 2047 (because
   `pos + 1 >= 2048`). The kernels produce 0.75 rL2 different output at
   this boundary position. Long-context impact may be substantial; for
   short-context (≤ 2048) eval the impact is negligible (~0.0002 nats).
   **Filed as a separate quality concern, out of scope for floor audit.**

2. **Q-projection 2× wide averages drift down**: stage 2 (post-q_proj)
   drift is 0.040 vs input 0.094 — the wider projection averages out
   per-channel bias variance. Numerical curiosity; not actionable.

3. **Gate channel has lower drift than Q channel** (stage 6 vs stage 5):
   different per-channel statistics in the q_proj output. Minor
   curiosity; not actionable.

## Path forward (recommended)

### Immediate

1. **Calibration roadmap continues** — AWQ Stage A on Qwen3.5-0.8B
   (already validated −32.6% above-floor KLD on 9B at 2026-05-12).
   Stage B (GPTQ) and Stage C (MR-GPTQ) extend the pattern.

2. **Q3.5 cohort comparisons report above-floor delta**, not absolute
   KLD vs external BF16 references. The 0.08-nat hipfire engine pedestal
   is not apples-to-apples with llama.cpp's Q8_0 0.0015 floor.

3. **Long-context investigation** for the flash-vs-non-flash precision
   mismatch (separate ticket, not engine-drift-floor scope).

### Long term

If the calibration roadmap (AWQ + GPTQ + MR-GPTQ) leaves a meaningful
quality gap to llama.cpp's calibrated-and-clean-engine baseline, **re-open
this question as a graph-level rewrite proposal** (estimated 4-8 months
for full per-kernel reimplementation against llama.cpp's reference graph).
Not currently justified by available data.

## Audit artifacts preserved

In-tree:
- `crates/hipfire-arch-qwen35/src/qwen35.rs`: `dump_la_stage` + 16 LA
  stage call sites; `dump_fa_stage` + 14 FA stage call sites
- `crates/hipfire-quantize/src/main.rs`: `HIPFIRE_QUANTIZE_EMBED_F16`,
  `HIPFIRE_QUANTIZE_CONV_F16` env-gates
- `crates/hipfire-runtime/examples/dump_qwen35_hidden_states.rs`
- `crates/hipfire-runtime/examples/recurrence_matched_input.rs`
- `scripts/dump_hf_la_stages.py`, `dump_hf_fa_stages.py`,
  `compare_la_stages.py`

Out-of-tree dumps (preserved at `/data/cache/hipfire/audit-2026-05-13/`):
- LA layer 0 per-stage hipfire + HF
- LA layer 0 embed-F16 / conv-F16 / full-LA-F16 hipfire variants
- FA-3 per-stage hipfire + HF
- Recurrence matched-input output

Eval results (committed):
- `benchmarks/quality-baselines/results/2026-05-13-unsloth-0.8b-q8kv/per-seq/`:
  Q4_K_M + Q8_0 llama.cpp baselines

## Status

**CLOSED 2026-05-13** with definitive verdict: per-weight-precision and
single-kernel surgery do not close the engine-drift floor. Floor is
structural (~24 kernels each at ~0.005 rL2). Calibration roadmap is the
shipped answer; engine surgery is deferred pending future re-prioritization
of a multi-month graph-level rewrite.
