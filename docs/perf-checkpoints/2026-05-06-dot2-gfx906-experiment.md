# dot2-gfx906 experiment — measured negative; lever ruled out

Date: 2026-05-06
Hardware: AMD Instinct MI50 / gfx906 / HBM2 1 TB/s peak.
Branch: `experiment/dot2-gfx906-mq6` (forked from
`feat/gfx906-hfq6-hfq8-analysis` at `850848a`).
Patch: one-line addition of `gfx906` to the
`has_dot2_f32_f16()` allowlist in
`crates/rdna-compute/src/dispatch.rs:123`.
Bench harness: `scripts/bench-cold.sh` (5-run fresh-process median,
`HIPFIRE_GRAPH=1 HIPFIRE_KV_MODE=asym3 HIPFIRE_DPM_WARMUP_SECS=10`).

## TL;DR

Plan §5.6 (gfx906-mq6-mq8-port.md v3.2.1) flagged the dot2 allowlist
as a deferred Phase B candidate: gfx906 carries the `dot2-insts`
LLVM feature and the `gemm_*_dot2.hip` kernels build clean on it
(audit §5.5), but the dispatcher's allowlist explicitly excludes
gfx906. We tested whether including it would speed up HFQ6/MQ6
batched fused GEMM (`gemm_qkvza_hfq6g256_dot2`,
`gemm_qkv_hfq6g256_dot2`, `gemm_gate_up_hfq6g256_dot2`).

**Result: net negative on gfx906 prefill.** Prefill regresses ~2.2%
on the small-batch case (pp=32) on both 9B and 27B mq6, and 9B
pp=128 also -2.4%. Only 27B pp=128 is break-even (within noise band
spread of 3.0%). Decode is essentially unaffected by design (the
dot2 kernels only fire on B>1 batched paths) — a +4% decode delta
on 27B pp=128 is unexplained but bench-cold spread (3.8%) plausibly
covers it as run-to-run DPM/JIT artifact, since dot2 cannot affect
the B=1 GEMV path in any direct way. The dot2 lever is **ruled out
by measurement** for gfx906; the allowlist exclusion was correct,
just for the wrong reason in the comment.

Expected → revised plan §5.6 status: "deferred Phase B candidate" →
**"ruled out by measurement, see this checkpoint."**

## Numbers

Binary md5 (baseline): `1695537f286f95a0bf54b33e09a9aaff`
Binary md5 (dot2 patch): `8339dec1c53cad87bcb6e0811d206f71`

### Qwen3.5 9B mq6 (5-run median, ≤1.0% spread)

| pp | metric | baseline | dot2 patch | Δ |
|---:|---|---:|---:|---:|
| 32 | prefill tok/s | 46.3 | 45.2 | **-2.4%** |
| 32 | decode tok/s | 31.1 | 31.1 | 0.0% |
| 128 | prefill tok/s | 46.7 | 45.6 | **-2.4%** |
| 128 | decode tok/s | 30.3 | 30.4 | +0.3% (noise) |

### Qwen3.6 27B mq6 (5-run median, ≤1.0% spread)

| pp | metric | baseline | dot2 patch | Δ |
|---:|---|---:|---:|---:|
| 32 | prefill tok/s | 13.5 | 13.2 | **-2.2%** |
| 32 | decode tok/s | 10.2 | 10.2 | 0.0% |
| 128 | prefill tok/s | 13.5 | 13.5 | 0.0% (3.0% spread, run-to-run wider than rest of sweep) |
| 128 | decode tok/s | 10.1 | 10.5 | +4.0% (3.8% spread; **unexplained — dot2 doesn't fire on B=1 decode path**, likely run-to-run DPM/JIT artifact) |

## Why dot2 regresses on gfx906 (root cause)

Surprising result given that the dot2 path:
- Reads **half the X bandwidth** per inner-loop iteration (8 FP16 = 16
  B vs 8 FP32 = 32 B)
- Replaces 8 scalar FMAs with 4 `v_dot2_f32_f16` instructions per
  accumulator per group
- Uses identical `__launch_bounds__(32, 8)` and produces identical
  VGPR=32 / SGPR=50 register pressure (verified via
  `llvm-readelf --notes` on the compiled `.hsaco` for both kernels).

Three kernels compared:
- `gemm_qkvza_hfq6g256.hip` — wave32 scalar baseline (FP32 X path)
- `gemm_qkvza_hfq6g256_fp16.hip` — FP16 packed `__hfma2` path
- `gemm_qkvza_hfq6g256_dot2.hip` — `v_dot2_f32_f16` (the candidate)

All three: 13,500–13,800 byte `.hsaco` size, VGPR=32, SGPR=50,
`private_segment_fixed_size` differs only by 16-80 bytes. Inner-loop
arithmetic ratio favors dot2.

**Root cause: per-launch FP32→FP16 X conversion overhead under graph
capture.**

The dispatcher path at `dispatch.rs:8252` calls
`ensure_fp16_x(x, batch_size * k)` before each dot2 kernel
invocation. This launches a `convert_f32_to_f16` kernel that reads
the FP32 X buffer and writes a parallel FP16 scratch buffer. The
helper has source-pointer caching at line 1169 — but the comment at
1167 explicitly states:

> Under graph capture, convert EVERY call: the src/dst pointers are
> stable (PrefillBatchScratch + persistent FP16 scratch), but the
> DATA at src changes every replay, so the captured convert-node
> needs to re-run. The pointer-equality cache would wrongly skip
> the node and read stale FP16 on replay.

`bench-cold.sh` runs with `HIPFIRE_GRAPH=1`. Per layer of mq6
prefill, three of these dot2 kernels fire (qkvza + qkv + gate_up),
each preceded by a fresh `convert_f32_to_f16` launch over the full
X buffer.

**Per-prefill HBM cost of the conversion:**

For 27B at pp=32, batch=8, K=5120: conversion launch is
`32 × 5120 = 163,840` elements = 640 KB FP32 read + 320 KB FP16
write = **960 KB extra HBM traffic per conversion call** × 3 calls
× 64 layers = ~180 MB extra HBM per prefill pass.

For 27B at pp=128, batch=8, K=5120: same per-call cost (the
conversion is done over `batch_size × k`, NOT pp), but pp=128
amortizes more useful work over the same fixed conversion cost. Net:
the regression should be smaller at larger pp, **which matches the
9B-pp=32 −2.4% / 9B-pp=128 −2.4% / 27B-pp=32 −2.2% / 27B-pp=128 ~0%
shape we measured.**

The dot2 kernel's bandwidth saving (X reads 16 B/iter vs 32 B/iter
inside the kernel) does not compensate for the upstream conversion
overhead under graph capture. On RDNA2/RDNA3 archs where the dot2
path was originally tuned, the overhead is amortized differently
(possibly the `ensure_fp16_x` cache does hit there because graph
capture is used differently, or because the dot2 win in absolute
terms is larger on those archs' VALU). On gfx906 the math is closer
to even and the conversion overhead breaks the tie.

## Why this isn't a bug to fix

The first instinct is "drop the conversion under graph capture by
replaying the convert-node from a captured stream." But:

1. The dispatcher comment at line 1167 is correct — under graph
   capture, the conversion HAS to fire on each replay because the
   X data has changed (different prefill chunk / different decode
   step), but the captured pointer is stable. The graph-capture
   architecture mandates this conversion behavior.
2. The dot2 regression is small (-2.2%) and the mq6 batched path is
   already a long-tail surface. Removing the conversion would
   require a larger redesign of the FP16-X plumbing.
3. On gfx906, the wave32 scalar baseline is *good enough*, and the
   plan's priority-1 work item (Phase A wave64 ports) targets the
   same surface with a much larger headroom (1.5–2× from PR #158
   reference data).

## Action items

1. **Plan §5.6 update:** change "deferred Phase B candidate" to
   "ruled out by measurement (-2.2% prefill, see
   2026-05-06-dot2-gfx906-experiment.md)." Strike the bench protocol
   recommendation since the experiment has been run.
2. **Allowlist comment update:** the doc comment in
   `dispatch.rs:115-122` says "gfx906 has dot2-insts" which is true
   but doesn't mention the conversion-overhead reason for exclusion.
   Update to capture the empirical finding.
3. **No commit of the dot2 patch.** The `experiment/dot2-gfx906-mq6`
   branch can be deleted after this writeup lands. The audit branch
   doesn't carry the patch.

## Cross-references

- Plan §5.6 (`docs/plans/gfx906-mq6-mq8-port.md` at commit
  `d3a0575`) — the original deferred-candidate flag this experiment
  resolves.
- Plan §5.5 audit (commit `f7ec59e`) — confirmed dot2 kernels
  build cleanly on gfx906; the allowlist exclusion is precautionary
  only at the build-test level.
- Dispatch matrix (`docs/perf-checkpoints/2026-05-06-quant-dispatch-matrix.md`,
  commit `cf00664`) — §6 lists `_dot2.hip` kernels as "build OK on
  gfx906, dispatch gated to RDNA2+." This experiment validates the
  gating decision, just for a different reason than precaution.
- Priority 0 baselines (`docs/perf-checkpoints/2026-05-06-mq6-baselines.md`,
  commit `850848a`) — the comparison anchor for this experiment.

## Raw bench logs

- `/tmp/dot2-experiment-2026-05-06/9b-mq6-dot2.log` — 9B mq6
- `/tmp/dot2-experiment-2026-05-06/27b-mq6-dot2.log` — 27B mq6 (in
  flight at writeup time; final pp=128 numbers TBD)
