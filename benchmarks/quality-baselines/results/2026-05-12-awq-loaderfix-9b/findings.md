# Stage A AWQ — 9B validation (2026-05-12 evening)

**Setup:** Qwen3.5-9B (32 layers: 8 full_attn + 24 linear_attn) on gfx1100, asym3 KV, prefill scoring, 512 chunks, kldref = `qwen3.5-9b-bf16.kldref.bin`.
**Quantizer:** commit `0aa58185` (whitelist fix + loader fix both landed).
**Eval binary:** rebuilt 16:50 with both fixes; `fused_rmsnorm_mq_rotate_awq.hsaco` JIT-compiled on first AWQ invocation.

## Headline

**AWQ on MQ4 delivers a real quality lift at 9B:**

| Variant | KLD | Above Q8 floor | PPL |
|---|---:|---:|---:|
| q8f16 (engine floor, 256ch) | 0.5735 | — | 13.383 |
| mq4-base (512ch) | 0.8165 | +0.2430 | 15.063 |
| **mq4-awq (α=0.5, 512ch)** | **0.7373** | **+0.1638** | **14.303** |

**AWQ delta:**
- KLD: −0.0792 nats absolute, **−32.6% reduction in quantization-attributable noise above the Q8 floor**
- PPL: −0.76 absolute, **−5.0% relative**
- Eval wall: 24 min (vs the ~5h I'd projected for Q8's slow path — MQ4 hits the FWHT-fused fast kernel)

## Sidecar coverage on the 9B AWQ file

| Bucket | Count | AWQ status |
|---|---:|---|
| `.q_proj` / `.k_proj` / `.v_proj` | 8 each | ✓ pre-scaled (full_attn input) |
| `.gate_proj` / `.up_proj` | 32 each | ✓ pre-scaled (MLP input) |
| `.in_proj_{qkv,z,a,b}` | 24 each | ✓ pre-scaled (linear_attn input) |
| `.o_proj` / `.out_proj` / `.down_proj` | 0 / 0 / 0 | ✗ correctly skipped (no runtime AWQ inverse) |
| **Total** | **184** | All on whitelist; 0 corrupting sidecars |

## Comparison to 0.8B

| Model | mq4-base KLD | mq4-awq KLD | AWQ Δ above-floor |
|---|---:|---:|---:|
| 0.8B (dense, 24 layers) | 0.6721 | 0.6707 | −0.7% (noise) |
| **9B (hybrid, 32 layers)** | **0.8165** | **0.7373** | **−32.6%** |

The 0.8B → 9B scaling matches AWQ paper predictions: outlier severity grows with parameter count, so AWQ's outlier-preservation lever scales up too. **0.8B is too small to surface AWQ benefit on Qwen3.5; 9B clearly clears the noise floor.** For the future Qwen3.6-A3B (~10B effective) MoE bench, expect AWQ to also help.

## Strategic implications

1. **Stage A is shipped + validated.** AWQ as a calibration lever on hipfire's MQ4G256 + FWHT-256 wire format works as the paper predicts; the −32.6% reduction at 9B is at the upper end of literature's 15-25% range.
2. **Stage B (GPTQ on MQ4) is unblocked.** The same Phase 2a/2b dispatch infrastructure that serves AWQ will serve GPTQ — only the *calibration algorithm* differs. Plan in `docs/plans/gptq.md` (next).
3. **Stage C (MR-GPTQ on MFP4) is unblocked.** GPTQ's per-tensor Hessian collection extends naturally to MR-GPTQ's E8M0 range mapping.
4. **Open follow-up (Option B from `awq_bug_hunt_glm5.md`):** AWQ-aware variants of `rotate_x_mq` / `fused_silu_mul_rotate_mq` so `o_proj` / `out_proj` / `down_proj` also benefit. Cost: 4 new HIP kernels + dispatcher wiring. Defer until Stage B/C measure-up — only revisit if the calibration stack still leaves a meaningful residual gap to UD-Q3_K_XL.

## Files

- `awq-loaderfix.kldseq` — per-sequence KLD output (~10 KB binary)
- `awq-loaderfix.eval.log` — full eval log including JIT compile + chunk progress
- `mq4-base.kldseq` / `mq4-base.eval.log` — baseline mq4 comparison (this run only, not the morning's killed cohort)
