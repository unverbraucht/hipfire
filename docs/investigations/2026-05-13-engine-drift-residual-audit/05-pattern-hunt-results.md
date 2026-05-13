# Pattern-hunt results (running log) — 2026-05-13

Companion to `04-pattern-hunt-plan.md`. Records per-hypothesis measurements
as Day-0/1/2/3 work progresses.

Model under test: `Qwen3.5-0.8B` q8f16 (HF .hfq at
`~/.hipfire/models/qwen3.5-0.8b.q8f16-2026-05-12`).
GPU: gfx1151 (Radeon 8060S, Strix Halo APU).
Bit-exact input: HF stage-0 dumps at
`/data/cache/hipfire/audit-2026-05-13/hf_layer0_chunk0.bin`.
Measurement window: layer 0, all 2048 positions of chunk 0, stage 1
(post-input_layernorm).

## D0.4 — isolated rmsnorm probe (built)

`crates/hipfire-runtime/examples/rmsnorm_isolated.rs`. Reads stage-0 from
an HF LA-stages dump, launches a single rmsnorm kernel variant against
the model's `input_layernorm.weight` (loaded with the Qwen3.5 +1.0
offset), writes stage-1 records. Variant selection via `--kernel <entry>`
into `kernels::RMSNORM_SRC`. Shared-mem-per-thread auto-detects 8 bytes
for entries ending `_f64acc`, otherwise 4.

**Baseline validation** (8 positions, then 2048 positions):

| variant | n_pos | mean rL2 vs HF | max rL2 vs HF | min cos vs HF |
|---|---:|---:|---:|---:|
| `rmsnorm_f32` | 8 | 0.001621 | 0.001704 | 0.999999 |
| `rmsnorm_f32` | 2048 | 0.001653 | 0.001823 | 0.999998 |

Matches the plan's expected ~0.0017 baseline. Probe is faithful to the
production plain-rmsnorm dispatch.

## D0.1 — reduction-precision probe (`rmsnorm_f32_f64acc`)

`kernels/src/rmsnorm.hip:33-56`. Same kernel as baseline except the
sum-of-squares accumulator and the rsqrt are fp64; output cast back to
fp32.

| variant | n_pos | mean rL2 vs HF | max rL2 vs HF | bit-identical to baseline? |
|---|---:|---:|---:|:---:|
| `rmsnorm_f32` | 2048 | **0.001653** | 0.001823 | (baseline) |
| `rmsnorm_f32_f64acc` | 2048 | **0.001653** | 0.001823 | NO (`cmp` differs from byte 53) |

**Result: H2/H5 falsified for rmsnorm.**

The two kernels produce different fp32 output bits but **identical
divergence magnitude vs HF**. The residual drift is not in the
sum-of-squares accumulator precision and not in the rsqrt precision.
Whatever drives the 0.0017 floor lives elsewhere.

Per the plan's D0.1 decision rule, Day 1 focuses on the
operation-order hypotheses:

- **H1** — operand grouping (`(x*rms)*w` vs `x*(w*rms)` vs `x*w*rms`)
- **H7** — FMA contraction (`v*v` accumulator likely compiled to one
  `v_fma_f32`; HF eager may do MUL+ADD with two roundings)
- **H8** — FTZ / denormal handling (build flag inspection)
- **H9** — eps placement (verify formula match by source inspection)

H4 (rsqrtf vs `1/sqrt`) is now also lower priority since rsqrt is part
of the same reduction-output stage that was just falsified.

## D0.2 — FP32 norm-weight control (mooted by D0.3)

The hypothesis "F16-vs-BF16 weight storage is the drift source" is
already falsified by D0.3: hipfire's F16-stored weight is bit-identical
to HF's BF16 weight on this layer (`weight rL2 = 0.000e+00`), and
hipfire's stage-1 output matches the fp64 reference exactly. Weight
storage is not contributing to any drift here. Skipping D0.2.

## D0.3 — fp64 reference threshold (**framing-breaker**)

`scripts/rmsnorm_fp64_reference.py`. CPU-fp64 reference computed with
HF's exact formula (`out = (1 + weight) * x * rsqrt(mean(x²) + eps)`)
using HF's BF16 weight cast to fp32 + cast to fp64. Compared
per-position against HF's stage-1 fp32 dump and hipfire's stage-1
fp32 dump.

### Result

| comparison | mean rL2 | max rL2 |
|---|---:|---:|
| HF fp32 vs fp64 ref (HF weight) | **0.001653** | 0.001823 |
| hipfire fp32 vs fp64 ref (HF weight) | **0.000000** | 0.000000 |
| hipfire fp32 vs fp64 ref (hipfire weight, bit-identical to HF) | 0.000000 | 0.000000 |
| HF fp32 vs (fp64 ref → bf16 cast → f32) | 0.000000 | 1.99e-4 |
| HF fp32 vs (HF-exact fp32 path → bf16 cast → f32) | 0.000000 | 0.99e-4 |
| hipfire fp32 vs (fp64 ref → bf16 cast → f32) | 0.001653 | 0.001823 |

### Interpretation

**The 0.001653 stage-1 "drift" is HF's intentional BF16 output cast,
not a hipfire kernel bug.** HF's `Qwen3_5RMSNorm.forward` does:

```python
input_dtype = hidden_states.dtype       # bfloat16
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + eps)
return (1.0 + self.weight.float()) * hidden_states.to(input_dtype)   # ← bf16
```

The dump script casts that bf16 result back to fp32 for the file
format. The fp64 reference matches HF's dump to ~zero when the
reference includes a bf16-cast step.

Hipfire's path keeps the output in fp32 throughout. So **hipfire's
rmsnorm output IS the fp64 reference**, modulo fp32 storage precision.
HF's output is the fp64 reference *cast through bf16*. The 0.001653 is
the bf16-cast loss, attributed entirely to HF.

### Implications

1. **The rmsnorm kernel has zero arithmetic-precision drift.** It is
   already at the fp64 ideal. There is nothing to fix in the kernel.
2. **Day 1-3 H1/H7/H8/H9 variants all produce ~0 rL2 from the fp64
   reference** because they all operate in fp32. None would reduce the
   gap vs HF, since the gap isn't in fp32 arithmetic — it's in the
   bf16 cast that hipfire *correctly skips*.
3. **The audit's ~0.005 floor across stages 0-12 is most plausibly
   HF's bf16 output cast at each `nn.Module` boundary**, captured by
   the dump script as fp32. The cast happens at the *output* of
   embed_lookup, in_proj_qkv (Linear), conv1d, l2norm, etc. — every
   module exit cycles through `input_dtype = bfloat16`.
4. **Hipfire is structurally MORE precise than HF's BF16 reference.**
   The KLD-floor gap vs llama.cpp Q4_K_M is most plausibly because
   llama.cpp matches HF's bf16 convention (intermediates effectively
   quantized into the trained distribution), while hipfire's fp32
   precision drifts *away* from the dtype the model was trained at.
5. **Day-5 success target ≤2× HF fp32 floor is unreachable without
   intentionally casting hipfire's intermediates to bf16** — a
   regression in arithmetic precision, but a possibly desirable
   regression for matching the trained distribution.

### What's actually worth testing next

The original engine-surgery framing ("fix kernel implementation drift")
is mooted. The new framing is:

- **F1** — does intentionally casting key intermediates (stage-1
  rmsnorm output, stage-2 raw qkv, stage-8/9/10 conv1d outputs) to
  bf16-then-back-to-fp32 reduce hipfire's KLD-vs-HF floor? If yes,
  the audit's "drift" was the right symptom but the wrong direction
  — match HF's BF16 cast convention instead of fighting it.
- **F2** — what's the empirical floor for hipfire's KLD vs llama.cpp
  Q4_K_M (which we know matches HF's bf16 better)? If F1 doesn't
  close it, calibration (AWQ Stage A→B) remains the right path.

H1/H7/H8/H9 variants are scratched. Day 1-3 work as planned would burn
time on hypotheses that D0.3 already falsifies.

## F2 (extended bf16-cast survey across all stages) — **pattern is real but heterogeneous**

Quick CPU-only test: for each HF stage dump record at layer 0, check
the fraction of fp32 elements that round-trip through bf16 unchanged.
1.0 = pure bf16 cast; 0.0 = fp32-native (some real arithmetic precision
that bf16 would round off).

| stage | description | bf16-clean | audit rL2 | meaning |
|---:|---|---:|---:|---|
| 0 | residual in (embedding) | **100%** | 0.0053 | bf16-cast artifact |
| 1 | post-rmsnorm | **100%** | 0.0055 | bf16-cast artifact (confirmed D0.3) |
| 2 | post-in_proj_qkv | **100%** | 0.0048 | bf16-cast artifact |
| 3-5 | post-in_proj_z/a/b | **100%** | 0.004-0.006 | bf16-cast artifact |
| **6** | **sigmoid α gate** | **0%** | 0.0024 | **fp32 native, real drift** (low) |
| 7 | sigmoid β gate | **100%** | 0.0026 | bf16-cast artifact |
| 8-10 | post-conv1d_silu q/k/v | **100%** | 0.0045-0.0109 | bf16-cast artifact (NOT a kernel bug) |
| **11** | **post-l2norm+scale q** | **0.34%** | **0.0087** | **fp32 native, REAL drift** |
| **12** | **post-l2norm k** | **0.34%** | **0.0118** | **fp32 native, REAL drift — HIGHEST upstream** |
| 13 | post-recurrence | **100%** | 0.0454 | bf16-cast (+ recurrence amplification of upstream) |
| 14 | post-gated_norm | **100%** | 0.0428 | bf16-cast |

### Final interpretation

The audit's "~0.005 rL2 across ~24 kernels" was **22 of those kernels
being bf16-cast artifacts and 2-3 being real implementation drift.**
Specifically:

**Real fp32-vs-fp32 kernel drift candidates** (the only stages where
hipfire's actual kernel implementation matters for the audit's
measurement):

- **Stage 6** — `fused_sigmoid_alpha_gate` α path. 0.0024 rL2.
  Low-magnitude.
- **Stage 11** — `fused_qk_l2_norm_scale_f32` q path. 0.0087 rL2.
- **Stage 12** — `fused_qk_l2_norm_scale_f32` k path. 0.0118 rL2.
  **HIGHEST upstream drift**, and the one feeding the recurrence which
  amplifies by ~4× (audit observation #4 in `00-summary.md`).

**Stage 12 is the real target.** If l2norm-k drift drops from 0.0118 →
0.003 with bit-exact input, the recurrence input gets cleaner and
stage-13 may drop proportionally (it currently amplifies 0.012 → 0.045).

The "conv1d 1.7-2× drift amplifier" finding (audit observation #3) was
HF's bf16 cast at conv1d output, not a hipfire conv1d kernel issue.
Reframing: conv1d output is mathematically the same; HF just intentionally
rounds it to bf16 at the module boundary while hipfire keeps fp32.

### Reproducer

```python
# 2-line check per stage
import numpy as np, torch
def bf16_clean_frac(arr_f32):
    rt = torch.from_numpy(arr_f32).to(torch.bfloat16).to(torch.float32).numpy()
    return float(np.sum(arr_f32 == rt) / arr_f32.size)
```

### Implications for next steps

The "engine surgery" scope shrinks from "rewrite ~24 kernels (4-8
months)" to **"investigate l2norm and sigmoid_alpha_gate kernels (~1
week)"**. Even better: stage 12 alone may explain most of the
recurrence-amplified KLD floor, so a single kernel deep-dive on
`fused_qk_l2_norm_scale_f32` could unlock the whole problem.

The H1/H7/H8/H9 hypotheses scratched for rmsnorm may still be live for
l2norm. The l2norm formula is structurally identical to rmsnorm
(`x * rsqrt(sum(x²) + eps)`) but applied per-head rather than per-vector,
and **without** the bf16 cast at HF's output. So variants like operand
grouping, FMA contraction, etc., may produce visible rL2 changes there
where they couldn't for rmsnorm.

## F3 — l2norm fp64 reference check (**rules out l2norm too**)

`scripts/l2norm_fp64_reference.py`. Same approach as D0.3 but for stages
11/12, computing fp64-ref per-engine on each engine's own stage 8/9
input.

### Result

| measurement | mean rL2 |
|---|---:|
| HF stage 11 vs fp64-ref(HF stage 8) — HF's q fp32 floor | **0.000000** |
| hipfire stage 11 vs fp64-ref(hip stage 8) — hipfire's q fp32 floor | **0.000000** |
| HF stage 12 vs fp64-ref(HF stage 9) — HF's k fp32 floor | **0.000000** |
| hipfire stage 12 vs fp64-ref(hip stage 9) — hipfire's k fp32 floor | **0.000000** |
| hipfire stage 11 vs HF stage 11 (audit cross-engine number) | 0.008737 |
| hipfire stage 12 vs HF stage 12 (audit cross-engine number) | 0.011835 |
| hipfire stage 8 vs HF stage 8 (upstream q_raw input drift) | 0.008452 |
| hipfire stage 9 vs HF stage 9 (upstream k_raw input drift) | 0.010947 |

### Interpretation

**Both engines' l2norm kernels are at the fp64 ideal.** The audit's
0.0087/0.0118 "drift" is just stage 8/9 upstream input drift passing
through l2norm with near-isometric ratios (1.03×/1.08×) — per-head
normalization mostly preserves rL2. Stage 9's input drift in turn comes
from HF's bf16 cast at conv1d_silu output (per F2: 100% bf16-clean).

**l2norm-q (stage 11) and l2norm-k (stage 12) are no longer drift
candidates.**

## Final scope after Day-0 + F2 + F3

The only remaining real-fp32-vs-fp32 kernel comparison anywhere in the
audited LA path:

| stage | kernel | rL2 | scope |
|---:|---|---:|---|
| 6 | `fused_sigmoid_alpha_gate` α | 0.0024 | low-magnitude; cosine 1.000 |

Every other "drift" in the audit's 0-12 stages is either:
1. HF's intentional bf16 cast at a module boundary (stages 0-5, 7, 8-10, 13-14), OR
2. Upstream bf16-cast input propagated through a bit-faithful hipfire
   kernel (stages 11, 12).

**The engine-surgery scope has collapsed from "rewrite 24 kernels (4-8
months)" to "essentially zero kernels".** Hipfire's kernels are
structurally **more arithmetically precise** than HF's BF16 reference
dump. The KLD-floor gap vs HF is the consequence of that precision
mismatch with HF's intentional dtype convention, not a hipfire bug.

### What this means for the original audit verdict

`03-final-verdict.md` correctly closed the audit with "no single-kernel
surgery target, calibration is the path forward". The pattern-hunt plan
re-opened it as "maybe shared root cause across kernels". The Day-0 +
F2 + F3 results now close the pattern hunt with a **stronger** verdict
than the original audit: not just "no single bug", but **"no
arithmetic-precision drift in any audited fp32-native kernel"**. The
"distributed pipeline drift" the audit observed is HF's bf16 cast
pattern, captured at fp32 by the dump probe.

### What's actually worth doing now

- **Calibration path (AWQ Stage A→B/C)** remains the right priority,
  and is now better-justified: there is no kernel arithmetic to fix,
  so all available leverage is in calibrating the weight quantization
  to push hipfire's output distribution closer to HF's bf16-trained
  distribution.
- **F1 (test bf16-cast hypothesis)** is a clean follow-up experiment
  for AFTER the calibration roadmap clears. Force hipfire to cast
  rmsnorm/conv1d/in_proj outputs to bf16-then-fp32, re-measure KLD
  vs HF. If F1 closes a meaningful chunk of the floor without quality
  regression, hipfire could ship a "match HF training dtype" mode that
  doesn't require re-quantization. If F1 doesn't help, that's also
  informative — it means the floor's source is the weight quantization
  itself, which calibration addresses directly.
- **Stage 6 (sigmoid_alpha_gate) is low priority.** 0.0024 rL2 with
  cosine 1.000 is unlikely to move any model-quality metric.

The pattern-hunt plan's Day 1-5 work as written is **fully obsolete**.
Day 0 took ~1 hour and closed the entire investigation with a stronger
result than the plan was even designed to produce.

## Artifacts

- `/data/cache/hipfire/audit-2026-05-13/pattern-hunt/baseline_full.bin`
  — `rmsnorm_f32` stage-1 output, 2048 positions.
- `/data/cache/hipfire/audit-2026-05-13/pattern-hunt/f64acc_full.bin`
  — `rmsnorm_f32_f64acc` stage-1 output, 2048 positions.

Reproduce:
```
LD_LIBRARY_PATH=/opt/rocm-7.12/lib:$LD_LIBRARY_PATH \
  ./target/release/examples/rmsnorm_isolated \
  --model ~/.hipfire/models/qwen3.5-0.8b.q8f16-2026-05-12 \
  --hf-dump /data/cache/hipfire/audit-2026-05-13/hf_layer0_chunk0.bin \
  --out /tmp/stage1.bin --n-positions 2048 \
  --kernel rmsnorm_f32             # or rmsnorm_f32_f64acc
.venv/bin/python scripts/compare_la_stages.py \
  --hip /tmp/stage1.bin \
  --hf /data/cache/hipfire/audit-2026-05-13/hf_layer0_chunk0.bin \
  --stages 1 --layer 0
```
