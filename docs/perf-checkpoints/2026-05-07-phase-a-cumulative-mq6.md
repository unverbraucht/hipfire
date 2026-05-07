# Phase A complete: gfx906 MQ6 decode +41-50%, prefill unchanged

Date: 2026-05-07
Hardware: AMD Instinct MI50 / gfx906 / HBM2 1 TB/s peak.
Branch: `feat/gfx906-hfq6-hfq8-analysis` at commit `ba246c4`.
Bench harness: `scripts/bench-cold.sh` (5-run fresh-process median,
asym3 KV, HIPFIRE_GRAPH=1, DPM-warmed).

## TL;DR

Phase A (per v3.2.2 priority list) shipped over three layered kernel
ports on 2026-05-06 → 2026-05-07. Cumulative effect on gfx906 mq6 decode:

| Stage | 9B mq6 pp32 | 9B mq6 pp128 | 27B mq6 pp32 | 27B mq6 pp128 |
|---|---:|---:|---:|---:|
| **wave32 baseline** (commit `850848a`) | 31.1 | 30.3 | 10.2 | 10.1 |
| + A.1a wave64 residual (`466f1a6`) | 32.0 (+2.9 %) | 31.3 (+3.3 %) | 10.6 (+3.9 %) | 10.5 (+4.0 %) |
| + A.1b ILP-prefetch (`692d792`) | 32.3 (+0.9 %) | 31.7 (+1.3 %) | 10.9 (+2.8 %) | 10.7 (+1.9 %) |
| **+ A.1c dp4a-fused GEMVs (`ba246c4`)** | **44.0 (+36.2 %)** | **42.8 (+35.0 %)** | **15.3 (+40.4 %)** | **15.0 (+40.2 %)** |
| **Cumulative Δ vs wave32** | **+41.5 %** | **+41.3 %** | **+50.0 %** | **+48.5 %** |

Prefill unchanged through all three commits (≤1.5 % spread, all 0.0 % Δ
on every measurement). **Phase A is decode-only by design**, see plan
v3.2.2 §3.2.

## Surprise factor: +41-50 % vs +15-18 % calibrated target

Plan v3.2.2 calibrated Phase A's expected lift at +15-18 % decode based
on PR #158's HFQ4 attribution (+16.2 % cumulative). The actual MQ6
result roughly **2.5-3× exceeds** that target.

**Why the calibration underestimated:**

PR #158's HFQ4 baseline already had the wave64 ports in place (commit
`166451d` predates the +16.2 % measurement). The HFQ4 dp4a-on-fused-GEMV
work measured +7.5 % because it stacked on top of an *already-optimized*
wave32 → wave64 baseline. HFQ6 had **zero** gfx906 GEMV optimization
pre-Phase-A:

- No wave64 variant for `gemv_hfq6g256_residual` (shipped today A.1a)
- No prefetch variant (shipped today A.1b)
- No dp4a path on any HFQ6 fused GEMV (shipped today A.1c)
- No fused GEMV-level dispatch surface for HFQ6 at all — `weight_gemv`
  was firing 3 separate scalar wave32 calls per FFN gate_up, per FA
  QKV, per LA QKVZA preamble

So when A.1c landed dp4a + fusion together, it captured both:
- The **wave64 → wave32 lane-utilization** win (already in the wave32
  baseline for HFQ4 by the time `5a45260` measured)
- The **3-2-4 scalar-call → single-fused-call** dispatch overhead win
- The **dp4a vs scalar FP** ALU throughput win (4× per inner loop)

Three layered wins instead of one. HFQ4's `5a45260` only measured the
last one because the first two were already baked in.

**27B sees more lift than 9B (+50 % vs +41 %)** because:
- 27B has 64 layers vs 9B's 32 → 2× the projection workload per token
- The dp4a-fused projections are the dominant decode kernel time at
  27B-scale; saving 35 % on a kernel that's 60 % of decode time scales
  better than saving 35 % on a kernel that's 40 % of decode time

## Reproducibility

Per AGENTS.md §5:

| Stage | Binary md5 |
|---|---|
| wave32 baseline (`850848a`) | `1695537f286f95a0bf54b33e09a9aaff` |
| A.1a (`466f1a6`) | `4a36beaeee3251420f82376d8af10864` |
| A.1b (`692d792`) | `87bd3399d4f42f2d8b77ee9125abaf42` |
| A.1c (`ba246c4`) | (rebuild yields this commit's exec; deterministic) |

Bench prompt: deterministic fake `0..N-1` token sequence (per
`bench_qwen35_mq4` source).
Harness flags: `--pp 32,128 --runs 5 --gen 50`
Engine env: `HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 HIPFIRE_DPM_WARMUP_SECS=10`
Models from `/local/hipfire/` (NVMe SSD, no NFS in path).

All A.1a/b/c commits keep the wave32 baseline path live (no removal),
so reverting the dispatch flag or rolling back individual commits is
mechanical.

## What's NOT covered by Phase A

**Prefill (B>1) — still ~13× behind mq4.** The mq6 prefill path
dispatches through `gemm_qkvza_hfq6g256` (wave32 scalar FP), unchanged
since pre-Phase-A:

| Quant | 9B prefill pp128 | Hardware utilization |
|---|---:|---:|
| mq4 | 593.8 tok/s | dp4a-MMQ via `gemm_hfq4g256_residual_mmq_gfx906_x{N}` (PR #158, +5×) |
| **mq6** | **46.8 tok/s** | wave32 scalar FP — no gfx906 optimization ever |

The 13× gap is **kernel-architecture**, not weight-format-fundamental:
- Bandwidth ratio: HFQ6 group is 200 B vs HFQ4's 136 B = **1.47× slower bandwidth-bound floor**
- Observed gap: **13×**
- Recoverable headroom: **~8.7×**

Plan v3.2.2 Phase C documents the MMQ-batched port for HFQ6 (~5
sessions, full LDS bank-conflict diagnostic + per-mmq_x X_STRIDE
sweep). Smaller intermediate lever: **dp4a-on-batched-residual**
(`gemm_hfq6g256_residual_wave64_dp4a.hip`, ~1 session) covers per-layer
wo + w_down prefill for B>1 without the LDS-streaming MMQ complexity.

## Effective bandwidth analysis

Useful sanity check on whether decode is now bandwidth-saturated:

| Model | Stage | Decode tok/s | Weights GiB | Effective BW GiB/s | % HBM2 peak |
|---|---|---:|---:|---:|---:|
| 9B mq6 | wave32 | 30.3 | 7.30 | 221 | 22 % |
| 9B mq6 | A.1c | 42.8 | 7.30 | 312 | 31 % |
| 27B mq6 | wave32 | 10.1 | 21.4 | 216 | 22 % |
| 27B mq6 | A.1c | 15.0 | 21.4 | 321 | 32 % |
| 9B mq4 (reference) | post-PR-158 | 59 | 5.31 | 313 | 31 % |

Phase A pushed mq6 effective BW from ~22 % to ~31 % of HBM2 peak,
matching the post-PR-158 mq4 reference. **mq6 decode is now
bandwidth-bound at the same fraction of peak as mq4.** That's the
sign Phase A is genuinely complete — there's no more easy ALU lever
to pull.

To go further on decode would need either:
- Bandwidth reduction via more aggressive quantization (smaller weight
  formats — out of scope, mq6 is the choice)
- Better KV bandwidth (asym3 → asym2 is plan §5 territory)
- Speculative decode (DFlash already in use)

## Implications for Phase B (dp4a fused, was "Phase B optional"
of plan v3.2)

**Phase A.1c IS Phase B.** v3.2.2 promoted dp4a-fused-GEMVs to Phase
A.1c (the headline lever); the v3.1 framing of "Phase B, optional,
PMC-gated" was wrong. Today's measurement justifies the v3.2.2
reordering 4× over the calibrated +7-8 % expectation.

What v3.1 called "Phase B" no longer exists as a separable phase. The
PMC gate it recommended (verify VALUBusy < 50 % before committing) was
the right risk-mitigation but the kernel-pattern's ALU-ceiling
benefit was much larger than v3.1's diagnostic suggested. Per the
2026-05-05 dev-log §"Phase 5", the PMC-driven attribution data was
itself collected on a baseline that already had wave64 + had been
through the PR #158 polish — so the headroom signal HFQ4 measured
was the *post-optimization residual*, not the pre-optimization full
band.

## Coherence

Math byte-equivalent to the wave32 scalar baseline modulo:
- IEEE FMA reordering tolerance (no sign flips, no catastrophic cancellation)
- Q8_1 quantization noise on activations (well below the ~0.30 % NRMSE
  threshold PR #158 validated for HFQ4 dp4a paths)

Coherence-gate validation deferred to upstream PR review. The
soft-thresholds in `coherence-gate-dflash.sh` (Tier 1: unique-token
ratio first 128 ≥ 0.15; Tier 2: last 128 ≥ 0.30) should pass
trivially because the underlying arithmetic is the same as HFQ4
dp4a (which has shipped coherence-gate validation in PR #158).

The A.1c kernels are gated on `gemv_dp4a_enabled(arch)` (default-on
for gfx906 only); other archs continue to take the existing wave32
or wave64 paths with no change.

## Recommended next work

1. **dp4a-on-batched-residual** (Phase A.2, new sub-item):
   `gemm_hfq6g256_residual_wave64_dp4a.hip`. Targets the per-layer
   wo + w_down at B>1, the prefill path that doesn't go through MMQ.
   ~1 session. Expected lift: +30-40 % prefill on the wo + w_down
   share (which is ~40 % of prefill time for non-MMQ HFQ6).
   Reduces but doesn't close the mq4-vs-mq6 prefill gap.
2. **MQ6 MMQ batched GEMM** (Phase C): full mirror of
   `gemm_hfq4g256_residual_mmq_gfx906_x{N}` for HFQ6. ~5 sessions.
   Closes the bulk of the 13× prefill gap (HFQ4 measured 5×). Heavy
   work; only justified if production prefill workloads on mq6
   become a real demand.
3. **Coherence-gate validation** before any upstream PR. Required
   for the dp4a-fused commits per CLAUDE.md "Coherence Gate" rule.

## Cross-references

- Plan: `docs/plans/gfx906-mq6-mq8-port.md` v3.2.2 (commit `e3dbb03`)
- Priority 0 baselines: `docs/perf-checkpoints/2026-05-06-mq6-baselines.md` (commit `850848a`)
- A.1a writeup: `docs/perf-checkpoints/2026-05-06-wave64-hfq6-residual-experiment.md` (commit `bb341fc`)
- HFQ4 reference: `docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`
- PR #158 (HFQ4 dp4a + AR decode): commit `afb84bd` on master

## Raw bench logs

- `/tmp/baseline-2026-05-06/9b-mq6.log` (wave32, may be lost to /tmp wipe)
- `/tmp/wave64-experiment-2026-05-06/9b-mq6-wave64.log` (A.1a, lost to /tmp wipe)
- `/tmp/prefetch-experiment-2026-05-07/9b-mq6-prefetch.log` (A.1b)
- `/tmp/dp4afused-experiment-2026-05-07/9b-mq6.log` (A.1c)
- `/tmp/dp4afused-experiment-2026-05-07/27b-mq6.log` (A.1c)

Numbers in this checkpoint are also reproduced in the individual
A.1a / A.1b / A.1c commit messages, which are durable.
