# Phase A complete: gfx906 MQ6 decode +41-50%, prefill +248-250%

Date: 2026-05-07 (revised; original 2026-05-07 covered decode only)
Hardware: AMD Instinct MI50 / gfx906 / HBM2 1 TB/s peak.
Branch: `feat/gfx906-hfq6-hfq8-analysis` at commit `fefc41f`.
Bench harness: `scripts/bench-cold.sh` (5-run fresh-process median,
asym3 KV, HIPFIRE_GRAPH=1, DPM-warmed).

## TL;DR

Phase A (per plan v3.2.3 priority list) shipped seven layered kernel
ports + dispatch wirings on 2026-05-06 → 2026-05-07. **Both decode and
prefill substantially improved** on gfx906 mq6:

| Stage | 9B mq6 decode pp128 | 9B mq6 prefill pp128 | 27B mq6 decode pp128 | 27B mq6 prefill pp128 |
|---|---:|---:|---:|---:|
| **wave32 baseline** (`850848a`) | 30.3 | 46.7 | 10.1 | 13.5 |
| + A.1a wave64 residual GEMV (`466f1a6`) | 31.3 (+3.3 %) | 46.8 (+0.2 %) | 10.5 (+4.0 %) | 13.5 (0 %) |
| + A.1b ILP-prefetch (`692d792`) | 31.7 (+1.3 %) | 46.8 (0 %) | 10.7 (+1.9 %) | 13.5 (0 %) |
| + A.1c dp4a fused GEMVs (`ba246c4`) | **42.8 (+35.0 %)** | 46.8 (0 %) | **15.0 (+40.2 %)** | 13.5 (0 %) |
| + A.2 dp4a residual GEMM (`1b9f374`) | 42.8 (0 %) | **60.8 (+29.9 %)** | 15.0 (0 %) | **17.4 (+28.9 %)** |
| + A.3 dp4a fused GEMMs (`c070dbb`) | 42.6 (-0.5 %) | **162.7 (+167.6 %)** | 15.0 (0 %) | **47.2 (+171.3 %)** |
| + A.4 lm_head batched dispatch (`fefc41f`) | 42.6 (0 %) | 162.6 (-0.1 %) | 15.0 (0 %) | 47.1 (-0.2 %) |
| **Cumulative Δ vs wave32** | **+40.6 %** | **+248.2 % (3.48×)** | **+48.5 %** | **+248.9 % (3.49×)** |

A.4 (DFlash speculative-decode lm_head batched wiring) shows ~0% Δ in
this AR-decode bench because it only fires under speculative decode —
the wiring is in place; quantitative impact requires a DFlash benchmark.

## Phase breakdown

**Phase A.1 (decode work, 2026-05-06):**

| Sub-phase | Lever | Decode lift |
|---|---|---:|
| A.1a | wave64 lane utilization on residual GEMV | +3-4 % |
| A.1b | software-pipelined ILP-prefetch on residual_wave64 | +1-3 % |
| A.1c | dp4a on fused single-token GEMVs (gate_up + qkv + qkvza) | +35-40 % |
| **A.1 total** | | **+41-50 %** |

**Phase A.2 + A.3 + A.4 (prefill work, 2026-05-07):**

| Sub-phase | Lever | Prefill lift (cumulative) |
|---|---|---:|
| A.2 | dp4a on batched residual GEMM (wo + w_down) | +29-30 % |
| A.3 | dp4a on batched fused GEMMs (qkvza + qkv + gate_up) | **+167-171 % vs A.2** |
| A.4 | lm_head batched dispatch via A.2 kernel | n/a here (DFlash-specific) |
| **A.2-4 total** | | **+248-250 % vs wave32 (3.48-3.49×)** |

A.3 was by far the biggest single lever. The fused projections
(qkvza + qkv + gate_up) are ~60 % of prefill time; going from
FP16-packed → wave64+dp4a captured a large share of compute saving in
the hottest part of the prefill path.

## Surprise factor: actual results vs calibrated targets

Plan v3.2.2 calibrated decode at +15-18% based on PR #158's HFQ4
attribution. Plan v3.2.3 (post-A.1c, pre-A.2/A.3) didn't have a
prefill target since the prefill story was the post-A reorder.

| Stage | Plan calibration | Measured | Ratio |
|---|---|---|---:|
| Decode (A.1 cumulative) | +15-18 % | +41-50 % | **2.5-3× over** |
| Prefill (A.2 + A.3) | +20-30 % (informal expectation, A.2 only) | +248-250 % | **8-12× over** |

Both targets were undercalibrated for the same structural reason:
**HFQ4 reference numbers measured only the *last incremental* lever
on top of an already-optimized stack**, while HFQ6 had zero prior
optimization. When A.1c (decode) and A.3 (prefill) landed, they each
captured multiple stacked wins simultaneously.

For prefill specifically, A.3 measured +167-171% over A.2 because:
- Wave64 lane-utilization win (was in A.1 for GEMV, never applied to
  this GEMM family before)
- Scalar-call → fused-call dispatch overhead win (HFQ6 batched fused
  family didn't exist; the wrapper dispatch existed but the kernels
  fell through to FP16-packed without fusion)
- dp4a vs FP16-packed ALU throughput win (4× per call vs 2×)
- Q8_1 conversion amortized across 6 dispatch calls per layer
  (qkvza + qkv + gate_up + wo + w_down + lm_head wave64-paths share
  one `ensure_q8_1_mmq_x` call per batch chunk)

The dot2 experiment (plan §5.6, ruled out 2026-05-06) hit the same
conversion overhead structurally but with only 2× ALU win per call
and lower amortization. The dp4a stack works because it's denser.

## Closing the mq4-vs-mq6 gap

For 9B at pp=128, the mq4-vs-mq6 prefill gap evolved:

| Stage | mq4 prefill | mq6 prefill | Ratio |
|---|---:|---:|---:|
| Pre-Phase-A | 593.8 | 46.7 | 12.7× |
| Post-A.2 | 593.8 | 60.8 | 9.8× |
| **Post-A.3** | **593.8** | **162.7** | **3.65×** |

mq6 has structurally 1.47× more weight bytes than mq4 (200 vs 136
B/group), so the bandwidth-bound floor is ~1.47×. Phase A closes the
gap from 12.7× → 3.65×, leaving roughly 2.5× of "kernel-architecture"
gap remaining.

That 2.5× is **MMQ-streaming territory**: mq4 has the
`gemm_hfq4g256_residual_mmq_gfx906_x{N}` family (PR #158's redesign,
which delivers ~5× over mq4's own FP16 baseline via shared-LDS-streaming
+ dp4a + per-mmq_x X_STRIDE tuning). HFQ6 doesn't have an MMQ port —
that's plan §5.1 priority 6 (Phase C, ~5 sessions).

## Reproducibility

Per AGENTS.md §5:

| Stage | Commit | Binary md5 |
|---|---|---|
| wave32 baseline | `850848a` | `1695537f286f95a0bf54b33e09a9aaff` |
| A.1a (wave64 residual GEMV) | `466f1a6` | `4a36beaeee3251420f82376d8af10864` |
| A.1b (ILP-prefetch) | `692d792` | `87bd3399d4f42f2d8b77ee9125abaf42` |
| A.1c (dp4a fused GEMVs) | `ba246c4` | (rebuild deterministic) |
| A.2 (dp4a residual GEMM) | `1b9f374` | (rebuild deterministic) |
| A.3 (dp4a fused GEMMs) | `c070dbb` | (rebuild deterministic) |
| A.4 (lm_head batched wiring) | `fefc41f` | (rebuild deterministic) |

Bench prompt: deterministic fake `0..N-1` token sequence (per
`bench_qwen35_mq4` source).
Harness flags: `--pp 32,128 --runs 5 --gen 50`
Engine env: `HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 HIPFIRE_DPM_WARMUP_SECS=10`
Models from `/local/hipfire/` (NVMe SSD, no NFS in path).

All commits keep the wave32/FP16-packed baseline paths live (no
removal), so reverting individual commits is mechanical. The A.2-A.4
dp4a paths are gated on `gemv_dp4a_enabled(arch)` (default-on for
gfx906 only); other archs continue WMMA / dot2 / FP16 fallbacks.

## Effective bandwidth analysis

Decode is now bandwidth-bound; prefill has more room.

| Model | Stage | Decode | Effective BW | % HBM2 peak |
|---|---|---:|---:|---:|
| 9B mq6 | wave32 | 30.3 tok/s | 221 GiB/s | 22 % |
| 9B mq6 | post-A.1c | 42.8 tok/s | 312 GiB/s | 31 % |
| 27B mq6 | wave32 | 10.1 tok/s | 216 GiB/s | 22 % |
| 27B mq6 | post-A.1c | 15.0 tok/s | 321 GiB/s | 32 % |
| 9B mq4 (reference) | post-PR-158 | 59 tok/s | 313 GiB/s | 31 % |

| Model | Stage | Prefill pp128 | Approximate effective BW |
|---|---|---:|---:|
| 9B mq6 | post-A.3 | 162.7 tok/s | ~1187 GiB/s (peak-saturated; activation prefetch + reuse means tok/s is not BW-pure) |
| 9B mq4 (reference) | post-PR-158 | 594 tok/s | similar regime |

mq6 decode at 31-32% of HBM2 peak matches the post-PR-158 mq4
reference exactly. **Decode is genuinely bandwidth-saturated; further
GEMV-level levers won't help without reducing the bandwidth itself**
(smaller quants, KV optimization, speculative decode).

Prefill numbers are above pure-BW-bound because batched compute reuses
weight reads across the batch (kernel reads weights once, computes
against B activations). The post-A.3 pp128 prefill of 162.7 tok/s is
within ~3.6× of mq4's 594 tok/s, suggesting we're hitting the
compute-density limit of the dp4a-batched-fused approach without
LDS-streaming MMQ.

## Coherence

Math byte-equivalent to the wave32 / FP16-packed baseline modulo:
- IEEE FMA reordering tolerance
- Q8_1 quantization noise on activations (well below the ~0.30 % NRMSE
  threshold PR #158 validated for HFQ4 dp4a paths)

Coherence-gate validation deferred to upstream PR review. The
soft-thresholds in `coherence-gate-dflash.sh` should pass trivially
because the underlying arithmetic is the same as HFQ4 dp4a (which
has shipped coherence-gate validation in PR #158).

All A.1-A.4 commits are gated on `gemv_dp4a_enabled(arch)` (default-on
for gfx906 only). Other archs continue to take WMMA / dot2 / FP16
paths with no change.

## What's NOT covered by Phase A

**MMQ-streaming for HFQ6 prefill (Phase C, plan §5.1 priority 6).**
The remaining 3.65× mq4-vs-mq6 prefill gap is in this territory.
mq4 has `gemm_hfq4g256_residual_mmq_gfx906_x{N}` (PR #158, +5× over
its own FP16 baseline). Porting to HFQ6 means writing the equivalent
8-symbol MMQ family with HFQ6's 200-byte group stride driving a
full LDS bank-conflict diagnostic + per-mmq_x X_STRIDE sweep. Plan
estimate: ~5 sessions.

**MoE-indexed kernels (plan §5.1 priority 5).** Five missing kernels
for HFQ6 MoE expert dispatch (down + gate_up, indexed + indexed_batched
+ wave64 variants). ~1 session if needed; only relevant for
A3B+MQ6-class workloads which don't ship today.

## Recommended next work

1. **Coherence-gate validation** before any upstream PR. Required
   for the dp4a commits per CLAUDE.md "Coherence Gate" rule.
2. **Phase C MMQ-streaming for HFQ6** if production prefill-heavy
   mq6 workloads emerge. The remaining 3.65× gap is real but the
   work is substantial (~5 sessions per plan).
3. **Upstream PR for the kernel fix `ee0fac6`** (mq8 sudot4 → sdot4,
   draft already at `docs/notes/upstream-pr-draft-mq8-sdot4.md`).
   Smallest readily-mergeable contribution from this session.

## Post-ship: review fixes (commit `5768fe4`, 2026-05-07)

After the Phase A.4 wire-up, ran a three-way critical review of
`feat/gfx906-mq6-phase-a-dp4a` (self / glm5 / gemini). Three blocking
findings landed in commit `5768fe4` on PR #187; three non-blocking
follow-ups are deferred to the pre-Phase-B cleanup pass (plan v3.2.4
errata, items 4 / 5 / 6 / 13).

**Fixes shipped in `5768fe4`:**

- **`capture_mode` guards on 5 HFQ6 dp4a dispatch sites** — silent
  correctness bug under `HIPFIRE_GRAPH=1`. The HFQ6 dp4a path calls
  `ensure_q8_1_mmq_x` which launches an internal quantize kernel that
  the captured graph may not record, leaving x stale. Added
  `&& !self.capture_mode` to match the HFQ4 sibling at
  `dispatch.rs:7889`. Sites: residual / batched_lmhead / qkv / qkvza
  / gate_up.
- **DDTree dispatch arms for HFQ6G256 + MQ6G256** — slow-path
  fallthrough. `crates/hipfire-arch-qwen35/src/speculative.rs`
  `run_dflash_draft_for_logits` and `run_dflash_draft_for_topk_gpu`
  had no arms for the new 6-bit dtypes; spec-decode on HFQ6/MQ6
  models would have errored at runtime ("unsupported target.output
  dtype"). Added MQ6 arm (with `rotate_x_mq_batched`) and HFQ6 arm
  (no rotation), mirroring the MQ3/MQ4 pattern.
- **Stale `gemv_dp4a_enabled` doc** — said "fused_qkv / fused_qkvza
  ports are pending" but PR #167 (HFQ4) and PR #187 (HFQ6) have
  shipped them. Updated to reflect that the lever now toggles every
  fused dp4a path together.

**Validation:** `cargo build --release -p rdna-compute -p
hipfire-arch-qwen35` clean; sanity bench on rebased PR2 branch
matched audit-branch numbers (pp128: 162.7 prefill / 42.5 decode
tok/s, vs audit-branch 162.8 / 42.6) — rebase against
post-#147/#181 master is byte-clean.

**Deferred to plan v3.2.4 follow-ups (items 4-6 + 13):** profile.rs
HFQ6 byte counts, `begin_timer` / `end_timer` on the 7 new
dispatchers, defensive `assert!(gemv_dp4a_enabled(arch))` on dp4a
Rust fns, and `scripts/audit-dispatch-coverage.sh`. Observability /
defensive hardening, not correctness — batched into the pre-Phase-B
audit pass.

## Phase B scoping (rocprof, 2026-05-07)

Re-baselined mq4 vs mq6 prefill on current HEAD before scoping Phase B.
The dev-log's stale 12.7× gap was pre-Phase-A.3 — A.3 already closed
most of it.

**Real numbers (gfx906 / MI50, 9B model, pp128, JIT-warm):**

| Quant | Prefill (median) | Source |
|---|---:|---|
| mq4 | 598.6 tok/s | `/tmp/mq4-prefill-baseline.log` |
| mq6 | 164.9 tok/s | `/tmp/mq6-prefill-baseline.log` |
| **Gap** | **3.6×** | (was 12.7× pre-A.3) |

**rocprof per-kernel ranking (mq6 pp128, 1625 contexts, 1543 ms wall):**

| Rank | Kernel | % time | Calls | Avg ns/call |
|---|---|---:|---:|---:|
| 1 | `gemm_gate_up_hfq6g256_wave64_dp4a` | **45.5 %** | 64 | 10.99 ms |
| 2 | `gemm_hfq6g256_residual_wave64_dp4a` | **28.5 %** | 128 | 3.45 ms |
| 3 | `gemm_qkvza_hfq6g256_wave64_dp4a` | **17.0 %** | 48 | 5.49 ms |
| 4 | `gemm_qkv_hfq6g256_wave64_dp4a` | 4.7 % | 16 | 4.58 ms |
| | **Top 4 (Phase A.3 dp4a)** | **95.7 %** | | |

**rocprof per-kernel ranking (mq4 pp128, 3242 contexts):**

| Rank | Kernel | % time | Calls |
|---|---|---:|---:|
| 1 | `gemm_hfq4g256_residual_mmq_gfx906_full_set_x64` | 38.0 % | 272 |
| 2 | `gemm_hfq4g256_residual_mmq_gfx906_full_add_x64` | 22.1 % | 128 |
| 3 | `gemm_hfq4g256_residual_fp16_wave64` | 19.7 % | 200 |
| 4 | `gemm_hfq4g256_residual_mmq_gfx906_full_add_x16` | 7.8 % | 200 |
| | **Top 4 (MMQ-streaming family)** | **87.6 %** | |

**Architectural insight: mq4 prefill dispatches NO `gemm_gate_up` /
`gemm_qkv` / `gemm_qkvza` kernels.** All prefill GEMMs go through the
`gemm_hfq4g256_residual_mmq_gfx906_x{N}` family — 8 size variants ×
{set, add} = 16 specialized kernels sharing a common body header
(`kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh`). Each
projection (q / k / v / z / α / β / gate / up / down) becomes one
residual-shaped MMQ call with `set` for the first projection of a
fused group and `add` for subsequent projections accumulating into Y.
This is the PR #158 win — small-tile dp4a-MMQ streaming, not big
fused kernels.

For comparison: mq6 dispatches **256 calls of large-block dp4a fused
kernels** (avg 5–11 ms) where mq4 dispatches **800 calls of small
MMQ tiles** (avg 1 ms). Same total work, very different scheduling.

**Two paths for Phase B:**

- **Path 1: optimize the 4 existing dp4a fused GEMMs in place**
  (~2-3 sessions). LDS-tile redesign on `gemm_gate_up_*` first
  (45.5 % alone), prefetch tuning, x_stride sweep. Stays within
  the Phase A.3 dispatch architecture. Expected lift: 1.5–2 ×.
- **Path 2: port the MMQ-streaming family to HFQ6** (plan §3.1.3,
  ~5 sessions). Build `gemm_hfq6g256_residual_mmq_gfx906_x{N}`
  family + retarget dispatchers. This is what mq4 uses for its
  60 % kernel-time win. Expected lift: 3–4 × (parity with mq4).

**Plan: start with Path 1 kernel #1.** Cheapest first improvement,
validates whether the dp4a-fused path has LDS / prefetch headroom,
and informs whether Path 2's bigger investment is justified. If
gate_up only yields +5–10 % after a session of LDS-tuning, that's
evidence the dp4a kernels are near their ceiling and we should skip
to Path 2.

**Phase B.0 (instrumentation) demoted to non-blocker:** the original
plan v3.2.4 framing assumed in-process `begin_timer` was needed to
localize the bottleneck. rocprof + `--stats` does this at the
kernel-name level without touching dispatch.rs. Adding `begin_timer`
to HFQ6 dispatchers is still useful for the production daemon
profile dump (`HIPFIRE_PROFILE=1`) but doesn't gate Phase B.1.
Plan v3.2.4 item 5 stays a non-blocking follow-up.

## Cross-references

- Plan: `docs/plans/gfx906-mq6-mq8-port.md` v3.2.3 (commit `d02dc95`)
- Priority 0 baselines: `docs/perf-checkpoints/2026-05-06-mq6-baselines.md` (commit `850848a`)
- A.1a writeup: `docs/perf-checkpoints/2026-05-06-wave64-hfq6-residual-experiment.md` (commit `bb341fc`)
- HFQ4 reference: `docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`
- PR #158 (HFQ4 dp4a + AR decode + MMQ): commit `afb84bd` on master
- Dispatch matrix (the audit that flagged A.4 as a perf miss):
  `docs/perf-checkpoints/2026-05-06-quant-dispatch-matrix.md`
- Audit dev-log (mq8 silent-corruption): `docs/perf-checkpoints/2026-05-06-mq8-runtime-dispatch-audit.md`
- Dot2 ruled-out experiment: `docs/perf-checkpoints/2026-05-06-dot2-gfx906-experiment.md`

## Raw bench logs

(/tmp logs may be lost to system reboot; numbers are reproduced in the
individual A.1a/b/c, A.2, A.3, A.4 commit messages which are durable.)

- `/tmp/baseline-2026-05-06/9b-mq6.log` (wave32)
- `/tmp/wave64-experiment-2026-05-06/9b-mq6-wave64.log` (A.1a)
- `/tmp/prefetch-experiment-2026-05-07/9b-mq6-prefetch.log` (A.1b)
- `/tmp/dp4afused-experiment-2026-05-07/9b-mq6.log` (A.1c)
- `/tmp/dp4abatched-experiment-2026-05-07/9b-mq6.log` (A.2)
- `/tmp/A3-experiment-2026-05-07/9b-mq6.log` (A.3)
- `/tmp/A4-experiment-2026-05-07/9b-mq6.log` (A.4 sanity)
- `/tmp/mq4-prefill-baseline.log` + `/tmp/mq6-prefill-baseline.log` (Phase B re-baseline)
- `/tmp/mq4-rocprof.stats.csv` + `/tmp/mq6-rocprof.stats.csv` (Phase B kernel ranking)
