# Phase 3 Stage A AWQ cohort findings — 0.8B (2026-05-12)

**Cohort dir:** `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-stage-a-awq-0.8b/`
**Git:** `feat/mq-v2-quant-format @ eafb7578`
**Setup:** Qwen3.5-0.8B, gfx1100, 512 chunks, asym3 KV, prefill scoring, kldref = `qwen3.5-0.8b-bf16.kldref.bin`

## Headline

**Stage A AWQ runtime is broken.** AWQ-quantized model produces effectively random logits at inference. NOT a "AWQ didn't help" finding — a "AWQ runtime path is wrong" finding.

| Variant | KLD mean ± CI | PPL | Above-floor |
|---|---:|---:|---:|
| `qwen35-0.8b-q8f16` (engine floor) | 0.4598 (CI 0.4519–0.4678) | 30.996 | — |
| `qwen35-0.8b-mq4-base` | 0.6721 (CI 0.6641–0.6803) | 36.594 | +0.2124 KLD |
| `qwen35-0.8b-mq4-awq` | **13.489** | **10,395,252** | **+13.029 KLD** |

For reference, `log(248320) ≈ 12.4`. A KLD of 13.5 means hipfire's AWQ output distribution is *further from* the BF16 reference than uniform-over-the-vocabulary would be. Pure noise.

## Pre-flight verification — Phase 1 (quantizer) looks correct

- AWQ-quantized `.hfq` file contains **248 `awq_scale` tensors** stored at the expected sites (matches 32 layers × ~8 linears/layer); baseline has 0.
- Sidecar names match the convention (`model.language_model.layers.N.{linear_attn,mlp}.<weight_name>.awq_scale.weight`, 1D F16, lengths matching the corresponding K dim).
- Quantizer log emitted `AWQ pre-scaling: ENABLED (alpha=0.5, formula: s[j]=(RMS_act[j])^alpha, geo-mean normalized to 1)`.

## What this rules out

- Not a metric artifact — eval completed all 512 chunks cleanly with no crashes or NaNs in the progress stream.
- Not a quantizer bug producing garbage weights — baseline MQ4 produced from the same quantizer at the same time-of-day runs fine (mq4-base KLD 0.67 is in expected range).
- Not the daemon-race fix — eval_hipfire path doesn't use the daemon; result is from direct path.
- Not a registry / file-naming issue — eval_hipfire loads model directly from path; smoke/HE failures are orthogonal.

## Top hypotheses (ranked by likelihood)

### H1 — Shared-input dispatch in Phase 2b is wrong

The `fused_rmsnorm_rotate_mq_batched_for` helper added in Phase 2b (commit `a4265ce4`) takes the **next-linear's** WeightTensor and dispatches the AWQ kernel iff that next-linear carries `awq_scale`. Several call sites consume the same `x_rot` for multiple downstream linears (fused QKV, fused gate/up, linear-attn's in_proj_{qkv,z,a,b}).

If only ONE of those downstream linears was used for the dispatch decision but the others were quantized with their own (potentially different) `awq_scale`, the math breaks: `x_rot = FWHT(x / s_A)` gets matmul'd against `W_B' = FWHT(W_B · s_B)` where `s_A ≠ s_B`. Result: `(1/s_A) · s_B · W_B · x` — per-channel-scaled garbage.

If the shared-input linears truly share imatrix (same in_sum2 → same scales by construction), this should not cause divergence — BUT only if the quantizer actually computed identical scales. Worth confirming.

### H2 — Scale direction (multiply vs divide) inverted somewhere

Quantizer: `W'[i,j] = W[i,j] * s[j]` (multiply).
Kernel `fused_rmsnorm_mq_rotate_awq.hip:78`: `x_shared[i] = x_shared[i] * weight[i] * rms / awq_scale[i]` (divide).

If somewhere a sign got flipped (e.g., quantizer divides, kernel divides → both apply `1/s`), output is `(1/s²) · W · x` per channel — could produce the observed garbage.

### H3 — Sidecar shape / channel-axis mismatch at load time

The runtime loader (`hfq.rs::load_awq_scale`) converts F16→F32 host-side and uploads. If the upload uses the wrong K dimension (e.g., reads sidecar as `[M]` for an `[M, K]` linear when it should be `[K]`), the per-channel divide is misaligned and scrambles activations.

### H4 — FWHT-AWQ interaction inside the kernel

The kernel applies the divide AFTER RMSNorm and BEFORE FWHT (Phase 1c). The math identity `<FWHT(W·s), FWHT(x/s)> = <W, x>` requires Parseval applies block-wise (per 256-element group). If AWQ scale boundaries don't align with FWHT block boundaries (they should — both are per input channel), bug.

## Discriminating tests (cheapest first)

1. **Inspect actual `awq_scale` values** for one MLP linear: are they all ≈ 1.0 (geo-mean normalized correctly)? Variance across channels? Any NaN/Inf?
2. **A/B at runtime**: build a `HIPFIRE_DISABLE_AWQ=1` env var that makes the dispatcher always pick the non-AWQ kernel. If AWQ-quantized model + `HIPFIRE_DISABLE_AWQ=1` produces *similar quality to mq4-base*, then quantizer's pre-scaled weights `W' = W·s` alone are mathematically fine but the runtime divide is what's broken. If even DISABLE_AWQ doesn't recover (still garbage), then the quantizer is the problem.
3. **Cross-projection scale check**: dump `awq_scale` for all 4 of layer 0's `linear_attn.in_proj_*` projections. Math predicts they should be byte-identical (same input → same imatrix → same scales). If they differ, the quantizer's per-tensor in_sum2 has drifted from "same input" → different scales → H1 confirmed.

## Decision

- **9B follow-up: NOT warranted.** Bug-debugging first.
- **Stage B (GPTQ) blocked.** GPTQ would reuse the same Phase 2 runtime infra; until the AWQ dispatch is verified correct, GPTQ would inherit the same bug.
- **Next step:** discriminating tests above. Cheapest is (1) — inspect scale values for one tensor — single-file Python read on the .hfq, no GPU. Then (2) — runtime A/B with HIPFIRE_DISABLE_AWQ env var (one-line dispatcher change).

## Side notes

- **Daemon-race fix validated** in this cohort: all three variants returned clean `ERR_DAEMON_NOT_READY` instead of the previous `ERR_'choices'` corruption.
- **Registry pattern is stricter than expected**: dated filenames (`*-2026-05-12`) don't auto-register in `/v1/models`. Future cohorts should symlink to bare-extension names before launch.
- **MQ4 vs Q8 inference speed**: MQ4 ran at 572 tok/s vs Q8 at 113 tok/s — 5× difference. Q8's slow path likely contributes some of the engine-drift floor's 0.35 unaccounted nats (see `project_engine_drift_floor_decomposition.md`).
