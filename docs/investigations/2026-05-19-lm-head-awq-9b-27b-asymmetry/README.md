# lm_head MQ4-AWQ — 9B/27B asymmetry investigation

**Status:** open. Diagnosis hypotheses framed, no resolving experiment run yet.
**Date:** 2026-05-19.
**Author basis:** synthesis of gfx1100 measurements + gfx1151 modeling
input + master-doc archive search.

## The observation

Adding lm_head MQ4-AWQ-GPTQ to the existing AWQ+GPTQ-F2 body recipe
helps **dramatically** on 9B but is **statistically a no-op** on 27B.
All numbers are gfx1100, KV=q8, n=256 prefill — same eval bench, same
kldref family, **clean within-model A/B**.

| Model | Recipe | KLD | CI | p99 | PPL | Δ vs Q8-head | Notes |
|---|---|---:|---|---:|---:|---:|---|
| 9B | `mq4-awq-gptq-f2-q8head` (§1.1j) | 0.1727 | 0.1600–0.1877 | 16.166 | 8.417 | (baseline) | α=0.55 |
| 9B | `mq4-awq-gptq-f2-lmhead-a100` | **0.0863** | (per gfx1100) | — | — | **−50%** | α=0.55 + lm_head MQ4-AWQ |
| 27B | `mq4-awq-gptq-f2-q8head-v100` (§3.2) | 0.1257 | 0.1126–0.1398 | 16.666 | 8.697 | (baseline) | α=0.55 |
| 27B | `mq4-awq-gptq-f2-lmhead-a100` | 0.1307 | 0.118–0.144 | 15.946 | 8.663 | **+4% mean (within CI), −4% p99, −0.4% PPL** | α=0.55 + lm_head MQ4-AWQ |

Both lmhead-a100 quants are **coherent** (gfx1100 eyeball check
2026-05-19) — the numbers are honest distributional-quality measures,
not signs of model breakdown.

## What gfx1151's parameter-fraction model predicts

Per their analysis:

| Model | lm_head params | Total params | lm_head fraction |
|---|---:|---:|---:|
| 9B | 152K × 4096 ≈ 622M | 9.5B | **6.6%** |
| 27B | 152K × 5120 ≈ 778M | 26.9B | **2.9%** |

Ratio 6.6 / 2.9 ≈ **2.3×**. If the lm_head benefit scales linearly with
parameter fraction (naive null hypothesis), 27B should see roughly:

```
27B expected improvement = 9B improvement / 2.3
                         = 50% / 2.3
                         ≈ 22%
```

We observe **0%** mean-KLD improvement on 27B (technically +4%, within
CI). The **22 percentage points** between expected and observed are
the anomaly that needs explaining.

There's also one weaker but suggestive signal: **p99 dropped 4%** on 27B
(16.666 → 15.946). lm_head AWQ on 27B does measurably improve the tail
of the divergence distribution, while leaving the mean alone. That
hints that lm_head AWQ helps a *subset* of tokens (probably high-
probability ones whose logit precision dominates the worst-case
divergence) without moving the bulk.

## bpw saving — consistent with lm_head shrinking (per gfx1151)

The bpw delta (4.50 → 4.458) is within rounding of the predicted
1.5%-bpw saving from a Q8→MQ4 swap of 2.9%-of-params, confirming the
quantizer did what we asked it to do. **The asymmetry is in the
quality outcome, not in what got quantized.**

## Hypotheses ranked by testability

### H1 — calibration α=0.55 is not 27B's U-curve minimum *(gfx1151's strongest read)*

The α-sensitivity curve was characterized on **9B** (Kaden's
investigation, `docs/investigations/2026-05-18-awq-gptq-sub-0.10-kld/results.md`):

| α | 9B KLD (c512, gfx1201) | Δ from min |
|---:|---:|---:|
| 0.30 | 0.1514 | +20% |
| **0.50** | **0.1257** | **min** |
| 0.55 | (not measured; interpolates ~0.13–0.14) | ~+5–10% |
| 0.70 | 0.1663 | +32% |
| 1.00 | 0.2032 | +62% |

The U-curve is **convex on 9B**, with a clear minimum at α=0.50. We
use α=0.55 for both 9B and 27B — slightly off-minimum but cheap on 9B.
**No analogous sweep exists for 27B.** If the 27B U-curve minimum
is at a different α (say 0.4 or 0.65), running at 0.55 would leave
significant KLD on the table — and the lm_head AWQ improvement could
be **masked** by α-misalignment that doesn't show on 9B.

**Test cost:** ~3 hours GPU on A100 (Stage C × 2 = 2 extra α points).
Reuses the existing 27B Hessian + imatrix. Two points (e.g. α=0.4
and α=0.7) bracketing 0.55 are enough to show whether the U-curve
shape is meaningfully different.

**Discriminator:** if 27B at α∈{0.4, 0.7} drops KLD below 0.10, H1
is the answer. If KLD stays near 0.1307, H1 is ruled out.

### H2 — 27B's 64-layer depth saturates lm_head's per-token benefit *(gfx1100's depth hypothesis)*

27B has 64 transformer layers vs 9B's 32. Each layer's body-AWQ
correction propagates into the hidden state; by the time the post-
RMSNorm activation reaches lm_head, the body's calibrated approximation
already captures most of the addressable error. lm_head AWQ then has
less marginal improvement available.

Mechanism: information-theoretic — the **error budget** that lm_head's
calibration could absorb has already been absorbed by the deeper body
calibration. The benefit *survives in the tail* (a small fraction of
tokens whose logit ordering is on a knife-edge), which is why p99
moved while mean didn't.

**Test cost:** hard. Would need to ablate body AWQ at 27B (re-quantize
with body GPTQ only, no AWQ, + lm_head MQ4-AWQ) and compare against
plain-q8head. ~3 hours GPU. Or compare layer-by-layer per-tensor MSE
(no extra quant cost, just analysis script over existing manifests).

**Discriminator:** if 27B body-AWQ-disabled + lm_head-AWQ improves
KLD more than current body-AWQ + lm_head-AWQ, H2 is supported.

### H3 — the 9B 0.1727 baseline isn't a like-for-like comparison *(gfx1100's recipe-not-host hypothesis)*

Commit `e4bae703` (the 9B 0.1727 measurement) was produced 2026-05-18
on `feat/mq-v2-quant-format-cuda` at α=0.55. The 27B 0.1257 measurement
(commit `ebcabb56`) was produced 2026-05-18 on the same branch family.
They *should* be apples-to-apples — same eval bench (gfx1100, n=256,
KV=q8, prefill), same code, same α.

But: **9B 0.1727 is genuinely worse than 27B 0.1257** despite same
recipe — larger models usually KLD-equal-or-better at same bpw because
more redundancy absorbs more error. The 9B baseline might be sitting
above its own optimum (per Kaden's α=0.50 9B = 0.1257 datapoint, so
9B at the right α reaches 27B's number).

If that's true, the "50% improvement from lm_head AWQ" on 9B is
actually moving 9B from an α-suboptimal baseline closer to where 27B
already sat, masquerading as lm_head benefit.

**Test cost:** ~80 min GPU. Re-quantize 9B at α=0.50 (Kaden's optimum)
**without** lm_head AWQ. If that result is ≈ 0.10–0.11 (not 0.1727),
then a meaningful chunk of the "9B 50%" was really α-correction, and
the true lm_head delta on 9B is smaller — pulling 27B's expected
improvement down toward what we observed.

**Discriminator:** if 9B at α=0.50 (no lm_head AWQ) lands at
≤ 0.13, then H3 explains part of the asymmetry. If it lands at
≥ 0.17, then α isn't the dominant factor on 9B either.

### H4 — Hessian quality difference (multi-pass + memmap on 27B vs single-pass in-RAM on 9B)

27B's Hessian was collected with `--n-passes 4 --accumulator-dir
~/.hipfire/hessian-acc-27b/` to fit 32 GB RAM. 9B's Hessian was
single-pass in-RAM. The math is the same (FP32 accumulation), but the
multi-pass approach exposes the calibration to within-pass-only batch
variance — each layer subset sees a different effective dataset slice.

Mechanism: stochastic. Each pass's 32 sequences × 2048 tokens = 65k
tokens; per-pass the sample is enough for stable RMS estimation per
column. But the joint statistics across layers (correlations between
layer-N's input H and layer-N+1's) are lost when N and N+1 fall in
different passes.

This is a subtler effect. Could show up as: lm_head's pre-norm
activation distribution that AWQ pre-scaled against may not match
what runtime actually sees, because the calibration "lm_head input"
came from a Hessian pass that saw a non-representative cross-layer
correlation structure.

**Test cost:** ~6 hours GPU + needs a 64GB+ RAM box. Re-collect 27B
Hessian with `--n-passes 1` (single-pass) on the A100 box (96 GB
system RAM, enough for the 126 GB → wait, doesn't fit). Would need a
host with > 130 GB system RAM. **Not testable on current cloud
recipes.** Could test the reverse: re-collect 9B with `--n-passes 4`
(artificially fragment) and see if KLD degrades.

**Discriminator:** if 9B with `--n-passes 4` gives noticeably worse
KLD than current single-pass, H4 has support and 27B is paying the
fragmentation tax.

### H5 — Eval methodology drift: 27B kldref vs 9B kldref *(low-prior)*

The two kldref files were produced by different runs on different
calibration corpora. They're each "valid" within-model references but
not directly comparable in absolute scale. **However**, the asymmetry
is within-model in both cases, so this shouldn't affect the 50%-vs-0%
gap. Mentioned for completeness only.

**Test cost:** trivial — read each kldref's metadata. **Almost
certainly not the cause** because both 9B and 27B kldref were produced
by the same `build_kld_ref` tool (b9d4d347 series) with identical
flags.

## Recommended next step

**H1 — α-sweep on 27B — is the cheapest decisive test.** Three reasons:

1. The 27B Hessian + imatrix + manifest are still preserved on the
   A100 box; only Stage C needs re-running per new α. ~80 min wall
   per α point on A100 80GB.
2. Two extra α points (e.g. 0.4 and 0.7) is enough to characterize
   the local U-curve shape and confirm whether 0.55 is near 27B's
   optimum.
3. If H1 lands (27B improves significantly at a different α), the
   lm_head AWQ recipe needs an α-per-model-size rule — useful
   independent of resolving the headline asymmetry.

**Cost:** ~3 hours GPU + ~1 hour Stage D + ~20 min eval × 2 = ~5 hours
total. Single overnight batch.

Defer H2/H3/H4 unless H1 comes back null. H2 and H4 are expensive;
H3 is moderate but only meaningful if H1 + H2 both miss.

## What we are NOT doing (preserves time for the K-map sweep)

- A full 5-point α sweep on 27B (~10h GPU). Two points bracketing
  0.55 is enough to confirm or rule out H1.
- Re-collecting 27B Hessian single-pass — requires hardware we don't
  have.
- Re-measuring the 9B 0.1727 baseline at α=0.50 (H3) — only valuable
  if H1 misses.

## Implementation note for the α sweep

The 27B Hessian sidecar `/workspace/qwen3.6-27b-bf16.hessian.bin`
(118 GB, preserved 2026-05-19) and imatrix
`/workspace/qwen3.6-27b-bf16.imatrix.gguf` (with `--process-output`
coverage, 13.66 MB) are both reusable. Stage C re-runs are just:

```bash
for ALPHA in 0.40 0.70; do
  python scripts/gptq_gpu.py \
    --input /workspace/.hf_home/hub/Qwen3.6-27B \
    --hessian /workspace/qwen3.6-27b-bf16.hessian.bin \
    --imatrix /workspace/qwen3.6-27b-bf16.imatrix.gguf \
    --alpha $ALPHA --bits 4 --lm-head-format mq4-awq \
    --output /workspace/gptq-precomputed/qwen3.6-27b-mq4-awq-gptq-f2-lmhead-a$ALPHA \
    --devices cuda:0 -v
done
```

Then one Stage D per manifest with the existing CLI on master
(`hipfire-quantize --lm-head-format mq4-awq --precomputed-gptq-path …`).
Eval each .hfq at n=256 on gfx1100 — record into a new §3.2.x row.

## References

- `docs/plans/kld-measurements-master.md` §1.1j (9B Q8-head baseline) and §3.2 (27B baseline)
- `docs/investigations/2026-05-18-awq-gptq-sub-0.10-kld/results.md` (9B α-sweep, U-curve shape)
- `docs/plans/gptq_lm_head_awq.md` §6 (results table)
- Commit `e4bae703` — 9B §1.1j measurement provenance
- Commit `ebcabb56` — 27B §3.2 first-light measurement provenance
- Commit `58045409` — regenerated imatrices with `--process-output`
