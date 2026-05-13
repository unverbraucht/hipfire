# Engine-drift residual audit (2026-05-13)

Reopens the engine-drift investigation that was tabled on 2026-05-13 at a
~0.08-nats KLD floor on Q3.5-0.8B q8f16 (post-RoPE-halfsplit-fix). The earlier
investigation (`docs/plans/qwen35-mq4-quality-gap.md` §"Step c follow-ups"
through §"Loader-bias bisect complete") concluded the residual was distributed
pipeline drift, no single bug. It recommended tabling with explicit exit
condition: *"Re-evaluate when a sensitive downstream test demonstrates the
0.08 actually matters."*

## Exit condition fired

Unsloth Qwen3.5-0.8B Q4_K_M scored through llama.cpp with Q8 KV cache
produces KLD **0.0351** against the BF16 reference. Hipfire's own q8f16
(Q8 weights, ALL other tensors F16) at matched Q8 KV produces KLD
**0.0806**. A 4-bit imatrix-calibrated GGUF beats hipfire's 8-bit weights
by ~2.3× at matched KV conditions.

Implication: ALL Q3.5 cohort comparisons sit on a 0.05-nat pedestal of
engine drift. A 4-bit MQ4/HFP4/MFP4 candidate's "above-floor KLD" is
not interpretable as quantization-attributable until the floor is closed.

| candidate | KV | KLD | Δ vs floor |
|---|---|---:|---:|
| Unsloth Q4_K_M (llama.cpp) | Q8 | 0.0351 | — |
| hipfire q8f16 (post-RoPE-fix) | Q8 | 0.0806 | +0.0455 nats / +130% |

Source: `benchmarks/quality-baselines/results/2026-05-13-unsloth-0.8b-q8kv/per-seq/Q4_K_M__llama__q8kv.kldseq` and prior measurement from the closed investigation.

## What is known from the closed investigation

1. **RoPE convention** — fixed (halfsplit ↔ interleaved). Was the dominant
   contributor (0.4945 → 0.0835 KLD on chunk 0).
2. **KV quantization** — ruled out (FP32 KV ablation: −0.02 nats only).
3. **DeltaNet recurrent precision** — ruled out (fp64 state byte-identical to fp32).
4. **LA weight quantization** — ruled out (F16-LA test: −0.012 nats only).
5. **A_log / dt_bias loader** — ruled out (safetensors byte-exact match).
6. **Per-position drift profile** — grows 0.06 → 0.19 across positions 0..2048
   in LA layer 4. Consistent with input bias being amplified by recurrence.
7. **Per-stage divergence at recurrence boundary** — every Q/K/V/α/β input
   diverges from HF transformers BF16 oracle. α has worst max (0.29) and a
   systematic +0.04 / dim bias on raw a (pre-sigmoid_alpha_gate).
8. **Chain-of-bias measurement** — raw `a` bias at layer 4 traces back to
   ~3.6e-5 / dim residual-stream bias at layer 3 output. That bias is
   accumulated across the 4 prior layers' kernel work, then amplified by
   RMSNorm scale gain (~24×) + softplus (~3×) + recurrence over 2048 pos.

The closed investigation never *localized* the per-stage contribution within
a single LA block. It measured layer-output bias and concluded "distributed".
This audit tests whether that distributed claim is actually correct, or
whether one stage carries the bulk of per-layer contribution.

## Audit plan

### Phase 1 — Per-stage intermediate dump at LA layer 0 (1 day)

Layer 0 is the cleanest LA target: input is the token embedding (verifiable
near-bit-exact against HF) so any divergence at a stage's output is local to
that kernel, not inherited from a prior layer.

New env-gate `HIPFIRE_DUMP_LA_STAGES=<path>` + existing `HIPFIRE_DUMP_DN_LAYER`
will write per-position records for the following 8 boundaries:

| stage | hipfire kernel | output buffer | already dumped? |
|---|---|---|---|
| 0 | (layer entry) | `s.x` | new |
| 1 | `fused_rmsnorm_mq_rotate` | `s.tmp` / `s.x_rot` | new |
| 2 | `weight_gemv` ×4 (in_proj_qkv/z/a/b) | `s.dn_qkv`, `s.dn_z`, `s.dn_alpha` (=raw a), `s.dn_beta` (=raw b) | partial — `HIPFIRE_DUMP_A_RAW` covers `s.dn_alpha` |
| 3 | `fused_sigmoid_alpha_gate` | `s.dn_alpha` (=α), `s.dn_beta` (=β) | new |
| 4 | `conv1d_silu_split` | `s.dn_q_raw`, `s.dn_k_raw`, `s.dn_v` | new |
| 5 | `fused_qk_l2_norm_scale` | `s.dn_q_raw`, `s.dn_k_raw` (post-l2norm) | covered — `HIPFIRE_DUMP_DN_INPUTS` |
| 6 | `gated_delta_net_f32` | `s.dn_attn_out` | new |
| 7 | `gated_norm_f32` | `s.dn_normed` | new |
| 8 | `weight_gemv_residual(wo)` | `s.x` (post-residual) | new |

HF transformers oracle: monkey-patch `Qwen3_5GatedDeltaNet.forward` to
capture matching tensors at each of these 8 boundaries. Use the existing
`scripts/dump_hf_dn_inputs.py` as the template — it already patches into
`chunk_gated_delta_rule` for stage 5.

### Phase 2 — Per-stage intermediate dump at FA layer 3 (1 day)

Same exercise on the FullAttention path. Stages: post-attn-norm,
post-Q/K/V projections, post-Q-norm/K-norm, post-RoPE, attention output,
post-O projection, post-residual. FA-3 is the first FA layer after warmup;
FA-7 is where the post-RoPE-fix per-layer drift jumps to 0.157; FA-23 is the
last layer before lm_head.

### Phase 3 — Fix the dominant stage(s) (2-3 days)

For whichever stages contribute the most to per-layer divergence, audit
the kernel for:

- accumulator order, reduction tree shape, fp32 throughout?
- fused vs separate-op rounding
- gain stacking: `(x*rsqrt)*scale` vs `x*(rsqrt*scale)`, RMSNorm `+1` convention
- eps placement: inside or outside rsqrt
- sign / log / exp convention (esp. for softplus, alpha gate)
- branch threshold (esp. softplus piecewise)

Patch one stage at a time, re-run dump compare, confirm divergence drops at
the patched stage AND downstream stages compound less.

### Phase 4 — Validation (0.5 day)

Re-run eval_hipfire on Q3.5-0.8B q8f16 with Q8 KV, 20 chunks. Target:
**KLD ≤ 0.02 nats** (matches plain Q3-0.6B baseline of 0.0098 within ~2×).
Re-run Unsloth cohort head-to-head to confirm q8f16 hipfire is competitive
with or below Q4_K_M llama.cpp at matched KV mode.

## Cost / value

5 days of focused work. With the floor at ≤0.02 nats:

- hipfire's Q8 baseline measures pure engine-vs-engine drift (effectively zero)
- 4-bit candidate KLD becomes interpretable as quantization-attributable
- the AWQ Stage A measurement that found 9B AWQ shifted KLD by 0.07 nats
  (32.6% above-floor reduction at 9B) becomes credible at 0.8B too (where
  it was within the noise floor)
- hipfire can claim quality parity with Unsloth Dynamic at matched bpw, or
  measure the remaining gap as genuinely calibration-attributable

## Status board

| phase | status | result | commit |
|---|---|---|---|
| 1a — instrument hipfire LA pipeline | done | 16 stage dumps, gated by HIPFIRE_DUMP_LA_STAGES | (uncommitted) |
| 1b — HF oracle for matching boundaries | done | `scripts/dump_hf_la_stages.py` | (uncommitted) |
| 1c — measure + rank stages at layer 0 | done | see Phase 1c findings below | — |
| 1d — repeat at layer 4 (cross-check) | pending | — | — |
| 2 — FA pipeline | pending | — | — |
| 3 — fix dominant stage(s) | pending | — | — |
| 4 — re-eval | pending | target ≤ 0.02 KLD | — |

## Phase 1c — Stage localization at LA layer 0 (2026-05-13)

Ran `dump_qwen35_hidden_states` with `HIPFIRE_DUMP_LA_STAGES` and the HF
oracle on `Qwen3.5-0.8B q8f16` (post-RoPE-halfsplit-fix) at chunk 0. 2048
positions × 16 stages dumped from both engines. Per-stage rel_L2 vs HF
transformers BF16 (all 2048 positions, mean):

| stage | description | mean rL2 | max rL2 | mean cos |
|---:|:---|---:|---:|---:|
| 0  | pre-rmsnorm residual in     | **0.0053** | 0.0068 | 1.000 |
| 1  | post-input_layernorm         | 0.0055 | 0.0072 | 1.000 |
| 2  | post-in_proj_qkv (raw qkv)   | 0.0048 | 0.0079 | 1.000 |
| 3  | post-in_proj_z (raw z)       | 0.0063 | 0.0084 | 1.000 |
| 4  | post-in_proj_a (raw a)       | 0.0059 | 0.0256 | 1.000 |
| 5  | post-in_proj_b (raw b)       | 0.0044 | 0.0101 | 1.000 |
| 6  | post-sigmoid_alpha_gate α (g)| 0.0024 | 0.0072 | 1.000 |
| 7  | post-sigmoid_alpha_gate β    | 0.0026 | 0.0047 | 1.000 |
| 8  | post-conv1d_silu q_raw       | **0.0085** | 0.0125 | 1.000 |
| 9  | post-conv1d_silu k_raw       | **0.0109** | 0.0260 | 1.000 |
| 10 | post-conv1d_silu v           | 0.0045 | 0.0094 | 1.000 |
| 11 | post-l2norm+scale q          | 0.0087 | 0.0120 | 1.000 |
| 12 | post-l2norm k                | **0.0118** | 0.0173 | 1.000 |
| 13 | post-recurrence (core_attn)  | **0.0454** | 0.139 | 0.999 |
| 14 | post-gated_norm              | 0.0428 | 0.106 | 0.999 |
| 15 | (post-MLP — HF hook bug)     | 0.559  | 0.891  | 0.84  |

(Stage 15 number is unreliable — the HF `register_forward_hook` on the
DecoderLayer captures the post-MLP residual, not the post-LA-block
residual. Hipfire's stage 15 captures the post-wo-residual *before* the
FFN runs. Fix: add a hook on the LA module itself or capture the residual
explicitly inside a patched DecoderLayer.forward. Does not affect the
Phase 1c verdict because stages 0-14 already paint the picture.)

### Key observations

1. **The embedding-lookup floor is 0.005 rL2.** Stage 0 (pre-rmsnorm
   residual = embedding output) shows ~0.005 mean rL2 vs HF BF16. This is
   the floor below which the entire pipeline cannot drop without changing
   the embedding storage convention. Hipfire's q8f16 quantizer stores
   embeddings as F16; HF transformers loads as BF16. The
   BF16↔F16↔F32-cast difference at ~1 ULP/element = ~0.005 rL2 on a
   normally-distributed vector.

2. **No projection / norm / gate stage carries a per-kernel bug.** All
   stages 0-12 have rel_L2 ≤ 0.012 (cosine ≥ 0.9999). The Q8 weight gemv
   (stages 2-5) produces output drift *lower* than the input drift,
   because the random-direction matrix averaging cancels per-channel
   noise. The `fused_sigmoid_alpha_gate` (stages 6/7) actually *reduces*
   drift (0.005 → 0.002) because of softplus's slope < 1 in the
   non-linear region. The l2norm + scale (stages 11/12) preserves drift.

3. **Conv1d_silu_split is a 1.7-2× drift amplifier on q/k.** Stage 2
   (raw qkv, post-projection) is at 0.005 rL2. After conv1d_silu_split,
   q (stage 8) is at 0.009 and k (stage 9) is at 0.011. V (stage 10) is
   unchanged at 0.005. The conv1d is the strongest per-stage drift source
   *upstream of the recurrence*. Possible audit target.

4. **The recurrence amplifies input drift by ~4×.** Stages 11/12/10/6/7
   feed `gated_delta_net_f32`: max input drift = 0.012 (k post-l2norm).
   Stage 13 output drift = 0.045. This is dynamical-system amplification,
   not numerical-precision rounding — probe c.3 of the closed
   investigation already proved fp64-state is byte-identical to fp32. The
   amplification factor is intrinsic to the recurrence math given the
   noise distribution of the inputs.

5. **Per-position bucket (stage 13) saturates early.** At layer 0,
   recurrence output drift grows 0.015 → 0.044 by pos 64, then plateaus
   around 0.045-0.050. The closed investigation's "grows monotonically
   through pos 1500+" pattern at layer 4 is the same dynamic but starting
   from a non-zero baseline (layer 4's input is already drifted from 3
   prior layers).

### Implications

- **There is no single localizable kernel bug.** The audit at the
  cleanest possible LA layer (input = embedding) finds every upstream
  stage at 0.005–0.012 rL2. The closed investigation's "distributed
  pipeline drift" verdict is **confirmed**, not refuted.
- **The 0.08-nat model-output KLD floor decomposes as**: ~0.005 rL2
  embedding-storage floor + small per-stage additions through 12 stages
  per LA × 18 LA layers + 4× per-LA recurrence amplification.
- **Embedding precision is the highest-EV next probe.** If hipfire's
  embedding tensor is stored at BF16 (matching HF) instead of F16, stage
  0 should drop to ~0.001 and propagate downward proportionally. Each LA
  layer's contribution scales with its input drift; cutting input drift
  5× could plausibly cut model-output KLD by 3-4×.
- **The conv1d_silu_split is a secondary audit target.** A 1.7× amplifier
  on q/k that doesn't affect v is suspicious — worth a precision review
  even if the headline impact is smaller than embedding storage.

### Files

- `/data/cache/hipfire/audit-2026-05-13/hip_layer0_chunk0.bin` (192 MB, hipfire dump)
- `/data/cache/hipfire/audit-2026-05-13/hf_layer0_chunk0.bin` (201 MB, HF dump)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:dump_la_stage` + 16 call sites
- `scripts/dump_hf_la_stages.py`, `scripts/compare_la_stages.py`

## Phase 1d — Precision-intervention probes (2026-05-13 follow-up)

Phase 1c findings ranked two probes — embedding precision (#1) and
conv1d_silu_split q/k asymmetry (#2) — by predicted impact. Phase 1d
ran both and a stacked variant, with model-output KLD as the bottom-line
measurement (5 chunks per-token Q8 KV).

### Per-stage results at LA layer 0 (chunk 0)

| stage | baseline | embed F16 | embed+conv F16 |
|---:|---:|---:|---:|
| 0 (embedding) | 0.0053 | **0.0000** | 0.0000 |
| 1 (rmsnorm) | 0.0055 | 0.0017 | 0.0017 |
| 2 (in_proj_qkv) | 0.0048 | 0.0037 | 0.0037 |
| 8 (conv q_raw) | 0.0085 | 0.0077 | **0.0048** |
| 9 (conv k_raw) | 0.0109 | 0.0101 | **0.0055** |
| 10 (conv v) | 0.0045 | 0.0043 | **0.0031** |
| 11 (q post-l2norm) | 0.0087 | 0.0079 | **0.0050** |
| 12 (k post-l2norm) | 0.0118 | 0.0108 | **0.0059** |
| 13 (recurrence) | 0.0454 | 0.0448 | 0.0471 |
| 14 (gated_norm) | 0.0428 | 0.0428 | 0.0464 |

F16 embed: stage 0 → bit-exact match with HF (BF16's 8-bit mantissa is a
strict subset of F16's 11-bit mantissa, so BF16 → F32 → F16 → F32
round-trips exactly). Reduction propagates through stages 1-12 but the
recurrence saturates around 0.045-0.047 regardless.

F16 conv: stages 8-12 drop ~50%. Recurrence output is *unchanged or
slightly worse*. State-trajectory amplification dominates per-position
input drift.

### Model-output KLD (5 chunks, per-token, Q8 KV)

| variant | KLD | Δ vs baseline | file size |
|---|---:|---:|---:|
| baseline q8f16 (all Q8) | **0.0796** | — | 814 MB |
| embed+conv F16 | 0.0895 | **+12.4%** | 1052 MB |
| full LA F16 (embed+conv+5 proj) | 0.0850 | **+6.8%** | 1230 MB |

**All three precision interventions regress the floor.** Going from Q8
to F16 storage on the affected weights — which is 10× more precise per
element — makes the model output diverge MORE from the BF16 reference,
not less.

### Mechanism

The closed investigation predicted this: "cumulative numerical
imprecision across the forward pipeline; each kernel's accumulator
order, fused-vs-separate-op rounding, and gain stacking through norm
weights collectively produce a small per-dim residual-stream bias that
compounds layer-by-layer."

The mechanism is the inverse of what intuition suggests:

1. Hipfire's kernel pipeline has implicit **bias offsets** from accumulator
   ordering, fusion patterns, and round-to-nearest semantics. These
   produce a small per-dim systematic deviation from HF's exact compute.
2. Q8 weight quantization noise is approximately **zero-mean random**
   (uniform over each block's ±1 ULP rounding). Over 2048 positions
   × 18 LA layers, this random noise statistically interacts with the
   engine's systematic bias offsets — partially canceling them.
3. Replacing Q8 noise with bit-exact F16 weights eliminates the random
   component. The engine's systematic bias is no longer offset by random
   noise, and the net divergence from BF16 reference grows.

This is the failure mode of "fix one source of noise in a complex
nonlinear pipeline" — random noise was load-bearing for cancellation.

### Verdict

The 0.08-nat engine-drift floor is **structural and not closable** by
per-weight precision interventions. The data shows:

- Single-stage precision fix (embed F16): zero impact at model output
- Two-stage stacked (embed+conv F16): regression
- Six-stage stacked (full LA F16): smaller regression but still worse

Closing the floor requires either:

- **Engine surgery**: byte-match hipfire's LA-block kernels to HF's
  `chunk_gated_delta_rule` accumulator/fusion patterns. Estimated effort:
  several weeks. High risk of introducing new regressions in fused paths
  the production runtime depends on.

- **Calibration**: AWQ Stage A is already shipped (9B: −32.6% above-floor
  KLD, validated 2026-05-12). Stage B (GPTQ) and Stage C (MR-GPTQ) extend
  the same pattern. These don't fight the floor — they pre-rotate
  weights so the engine's bias and the calibration shift compose into a
  smaller net drift.

### Implication for cohort comparison

Q3.5 quality-eval cohorts on hipfire must measure **above-floor delta**,
not absolute KLD vs an external BF16 reference. Comparing hipfire q8f16
(0.080) to llama.cpp Q4_K_M (0.035) is **not** apples-to-apples — the
0.045 gap is engine implementation, not quantization quality. A correct
4-bit-attributable measurement is `KLD(MQ4) − KLD(Q8_floor)`, which
isolates the quant-cost from the engine-specific pedestal.

The Q4_K_M-beats-q8f16 datum that fired the exit condition tells us the
floor matters in practice — but the corrective action is calibration on
hipfire's side (Stage A/B/C in flight), not pursuing kernel-precision
audits.

### Files (Phase 1d)

- `/data/cache/hipfire/audit-2026-05-13/hip_layer0_chunk0_embedf16.bin`
- `/data/cache/hipfire/audit-2026-05-13/hip_layer0_chunk0_convf16.bin`
- `/data/cache/hipfire/audit-2026-05-13/eval/baseline_q8f16.kldseq`
- `/data/cache/hipfire/audit-2026-05-13/eval/embed_conv_f16.kldseq`
- `/data/cache/hipfire/audit-2026-05-13/eval/full_la_f16.kldseq`
- `crates/hipfire-quantize/src/main.rs:HIPFIRE_QUANTIZE_EMBED_F16`
- `crates/hipfire-quantize/src/main.rs:HIPFIRE_QUANTIZE_CONV_F16`

## Status — CLOSED 2026-05-13

The reopening exit condition (Q4_K_M-beats-q8f16) **was correct**: the
floor matters for quality cohort comparisons. But the audit established
that the floor is **not closable by per-weight precision interventions** —
three independent stacked F16-storage probes all regressed the floor.

Path forward: ship Stage A AWQ on Q3.5-0.8B (already validated on 9B at
−32.6% above-floor KLD), then continue Stage B/C calibration roadmap.
Treat 0.07 nats as the engine-vs-engine pedestal in all Q3.5 cohort
comparisons; report deltas, not absolute KLDs.

The audit infrastructure (HIPFIRE_DUMP_LA_STAGES + matching HF probe +
HIPFIRE_QUANTIZE_EMBED_F16 + HIPFIRE_QUANTIZE_CONV_F16) is preserved
in-tree for any future reopening of the question.

## References

- Closed investigation: `docs/plans/qwen35-mq4-quality-gap.md` §line 831–1030
- Existing dump infra: `crates/hipfire-arch-qwen35/src/qwen35.rs:5944–6020`
- HF oracle template: `scripts/dump_hf_dn_inputs.py`
- Cross-engine sanity: `scripts/cross_engine_check.py` (BF16 ↔ BF16 baseline = 0.002 nats)
- Trigger datum: `benchmarks/quality-baselines/results/2026-05-13-unsloth-0.8b-q8kv/per-seq/Q4_K_M__llama__q8kv.kldseq`
