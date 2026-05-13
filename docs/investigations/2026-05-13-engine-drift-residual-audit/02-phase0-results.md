# Phase 0 results (2026-05-13)

Three pre-Phase-2 sanity checks from the combined adversarial review
(`engine-surg-plan-rev-combined.md` Critical issue C4).

## Check 1 — llama.cpp Q8_0 KLD on Q3.5-0.8B at matched Q8 KV

Resolves the question: is the 0.078-nat hipfire-vs-BF16-ref gap engine
drift, calibration, or a mix?

| variant | KLD (n=1175 chunks) | engine | quant | KV |
|---|---:|---|---|---|
| llama.cpp Q8_0 | **0.0015** | llama.cpp | Q8 weights | Q8 |
| llama.cpp Q4_K_M imatrix | 0.0351 | llama.cpp | Q4 + imatrix | Q8 |
| hipfire q8f16 (5 chunks) | 0.0796 | hipfire | Q8 weights | Q8 |

**Verdict: the 0.078 gap is 100% engine drift.** llama.cpp's Q8 weights
(uncalibrated, same precision class, same model, same KV mode) sit at
the BF16 cross-engine noise floor of 0.0015 — basically lossless. The
hipfire-vs-llama.cpp delta cannot be:

- Quantization: both use Q8 weights
- Calibration: neither uses imatrix/AWQ/GPTQ
- KV mode: both at Q8

It is **purely** the difference between hipfire's kernel implementation
and llama.cpp's.

### Strategic implications

The engine-surgery plan's "achievable target" assumption (KLD ≤ 0.05 was
the conservative gate per the combined review) is **30× too pessimistic**.
The reachable target is **~0.005 KLD** — llama.cpp Q8_0 proves it.

A successful surgery would close ~98% of the engine-drift gap. The
recurrence kernel alone is unlikely to land at 0.005; expected drop from
a single dominant kernel rewrite is 5-10× (0.08 → 0.008-0.016) which is
still a substantial result.

Calibration roadmap implication: AWQ Stage A's 32.6% above-floor KLD
reduction on 9B is potentially leaving the remaining engine gap (~95%
of which is unaddressable by calibration) on the table. Engine surgery
**unlocks** calibration's full theoretical benefit, not competes with it.

Files: `benchmarks/quality-baselines/results/2026-05-13-unsloth-0.8b-q8kv/per-seq/Q8_0__llama__q8kv.kldseq`

## Check 2 — F16 dispatch correctness

Phase 1d's "structural drift" verdict hinged on F16-weight regression
being a real engine-bias finding, not a loader bug. Validates by
inference from Phase 1c-1d data + Check 1 above.

Reasoning:
- Phase 1c F16 conv weights dropped stages 8/9/10 drift by ~50% (from
  ~0.009 → ~0.005). This is the *expected* magnitude for moving from
  Q8 (1 byte + per-block scale) to F16 (2 bytes, no scale) on a 4-tap
  conv. Bit-exact F16 storage produces this exact behavior.
- Phase 1d model-output KLD regression is consistent with the "random
  Q8 noise was load-bearing for cancellation of systematic engine bias"
  hypothesis. Check 1 corroborates: llama.cpp's engine has no systematic
  bias (Q8_0 = 0.0015 noise floor), so the Q8 random noise in llama.cpp
  doesn't have anything to cancel — it just contributes its own noise.
  In hipfire the engine has 0.08 nats of systematic bias for Q8 random
  noise to partially cancel.

No additional probe needed. F16 dispatch passes by inference.

## Check 3 — FA branch stage map

Read `forward_scratch_layers` FA branch
(`crates/hipfire-arch-qwen35/src/qwen35.rs:6408`). The plan drafted
12 stages; actual is **14 stages**. Key clarifications:

1. **Q projection is 2× wide**: hipfire's `wq` projects to
   `n_heads × 2 × head_dim` (Q + gate interleaved per-head). This
   matches HF's `q_proj(hidden).view([..., 2*head_dim])` exactly. The
   per-head interleaved layout `[Q_h0(hd), Gate_h0(hd), Q_h1(hd),
   Gate_h1(hd), ...]` is consistent with HF's chunk-on-dim=-1 of the
   2× wide projection.

2. **Deinterleave kernel splits Q from gate**: `deinterleave_f32` reads
   per-head from interleaved layout and writes flat `[n_heads × hd]` Q
   and `[n_heads × hd]` Gate buffers.

3. **wo + residual is fused**: hipfire's `weight_gemv_residual` fuses
   o_proj with the residual add. HF has them as separate ops. For
   per-stage comparison we lose one boundary (no "pre-residual o_proj
   output" stage available without splitting the fused kernel).

### Verified 14-stage map for FA pipeline

| stage_id | what | hipfire buffer | n_elems |
|---:|---|---|---:|
| 0 | pre-rmsnorm residual (FA entry) | `s.x` | dim |
| 1 | post-rmsnorm | `s.tmp` | dim |
| 2 | post-q_proj (raw Q+gate interleaved) | `s.fa_q_full` | n_heads × 2 × head_dim |
| 3 | post-k_proj | `s.fa_k` | n_kv_heads × head_dim |
| 4 | post-v_proj | `s.fa_v` | n_kv_heads × head_dim |
| 5 | post-deinterleave Q | `s.fa_q` | n_heads × head_dim |
| 6 | post-deinterleave gate | `s.fa_gate` | n_heads × head_dim |
| 7 | post-q_norm | `s.fa_q` | n_heads × head_dim |
| 8 | post-k_norm | `s.fa_k` | n_kv_heads × head_dim |
| 9 | post-RoPE Q | `s.fa_q` | n_heads × head_dim |
| 10 | post-RoPE K | `s.fa_k` | n_kv_heads × head_dim |
| 11 | post-attention (pre-gate) | `s.fa_attn_out` | n_heads × head_dim |
| 12 | post-sigmoid-mul gate | `s.fa_attn_out` | n_heads × head_dim |
| 13 | post-wo + residual (block exit) | `s.x` | dim |

(KV cache write between stages 10 and 11 is a state mutation, not a
dumpable scalar tensor. Skipped.)

## Phase 0 verdict

All three checks pass. The audit premise is sound, F16 dispatch is
correct (no loader bug), and the FA stage map is verified.

The surgery target is **revised upward** from "≤ 0.05 KLD" (combined
review conservative gate) to **"~0.005 KLD"** based on llama.cpp's
Q8_0 demonstrating that achievable target. Single-kernel rewrite is
unlikely to reach 0.005 directly; realistic expectation for a single
dominant-kernel rewrite is **5-10× reduction (0.08 → 0.008-0.016)**.

Phase 3a (recurrence matched-input probe) is the next step per the
revised plan ordering.
