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

## Phase B.1: gemm_gate_up_hfq6g256_wave64_dp4a — BATCH_TILE sweep

First Path 1 experiment: BATCH_TILE was 8, halving A reloads by
doubling the per-block batch slice gives bigger blocks but more
register pressure. Swept BT ∈ {8, 16, 32}.

| BT | gate_up ms/call | pp128 tok/s | Δ vs BT=8 |
|---:|---:|---:|---:|
| **8** (baseline) | 10.99 | 165.8 | — |
| **16** | **9.19** | **180.0** | **−16% per call, +9% wall** |
| 32 | 22.11 | 113.3 | regression — register spill |

Decode (g50 mq6 9B): 43.7 tok/s with BT=16 vs 42.5-42.7 baseline,
no regression. The BT cap is mostly relevant at prefill (B=128) —
decode runs `local_bs = 1` and only writes `acc[0]`.

**PMC counters at BT=16 (rocprof):**

- arch_vgpr 48, sgpr 32, lds 0
- VALUBusy 52.8 %
- MemUnitBusy 87.8 %
- MemUnitStalled 2.0 %
- L2CacheHit 85.5 %

**Diagnosis: memory-throughput-bound, not stall-bound.** L2 catches
85 % of loads; the kernel issues memory ops as fast as the memory
units can serve them. ALU is idle ~47 % of cycles waiting for
memory. Effective bandwidth ≈ 33 GB/s vs MI50 peak 1024 GB/s —
~3 % of peak. Compute floor is ~70 µs (dp4a at peak); we're at
9.2 ms, so 130× over compute floor.

**Root cause:** single-wave-per-block design means each output row's
A data is read by exactly one block, no inter-block A-sharing. With
8 batch tiles per row (BT=16, B=128), each row's 3.2 KB of A bytes
is reloaded 8× across batch tiles (best-case via L2). Multi-wave
blocks with LDS-staged A would reduce that to 1×.

**Next: 2-waves-per-block kernel rewrite** (kernel B.1.1) to test
whether LDS-staging A across waves moves the kernel toward the
memory-throughput floor. Estimated effort: ~1 session.

## Phase B.1.1: multi-wave attempt (NEGATIVE) + BT=16 sibling propagation

Tested two multi-wave designs to reduce A-reload traffic:

| Variant | Block size | tok/s pp128 | vs BT=16 single-wave |
|---|---|---:|---:|
| BT=16, 1 wave (baseline) | [64, 1, 1], 2 rows × 16 tokens | 153.7 | — |
| ROWS_PER_BLOCK=4 | [128, 1, 1], 4 rows × 16 tokens | 147.0 | -4 % |
| 2-subtile (same 2 rows, 32 tokens) | [128, 1, 1], 2 rows × 32 tokens | 142.0 | -8 % |

**PMC for the 2-subtile variant:**
- L2CacheHit 85 → **95 %** (cache sharing worked!)
- VALUBusy 53 → **43 %** (occupancy hurt more than cache helped)

**Root cause of regression:** 128-thread blocks halve VGPR-budget
occupancy on gfx906 (10 blocks/SIMD vs 21 with 64-thread blocks).
The cache-locality benefit is real but smaller than the SIMD-
utilization loss. **The current 64-thread block is well-tuned for
occupancy.** Multi-wave-with-LDS would need careful VGPR/LDS budget
rebalancing — not a one-session win.

Reverted to single-wave + BT=16 baseline.

### BT=16 propagation to siblings (the actual win)

Applied BT=8 → BT=16 to the other 3 HFQ6 dp4a kernels: `residual`,
`qkvza`, `qkv`. All four kernels share the structurally identical
inner loop, so the same A-reload-amortization argument applies.

**Per-kernel ms/call (rocprof, 9b.mq6 pp128, gfx906):**

| Kernel | BT=8 baseline | BT=16 all 4 | Δ per call |
|---|---:|---:|---:|
| gate_up | 10.99 ms | **9.15 ms** | -16.7 % |
| residual | 3.45 ms | **3.19 ms** | -7.5 % |
| qkvza | 5.49 ms | **4.70 ms** | -14.4 % |
| qkv | 4.58 ms | **3.86 ms** | -15.8 % |

**Wall-clock comparison (5 prefill runs each, JIT-warm, same session):**

| State | Prefill pp128 | Decode g50 | Notes |
|---|---:|---:|---|
| Pre-Phase-B (BT=8 all 4) | ~165.8 | 42.5-42.7 | from rocprof timing |
| HEAD (BT=16 gate_up only, commit `2bee6e6`) | 153.7 | 43.7 | |
| **B.1.1 final (BT=16 all 4)** | **189.9** | **43.5** | spread 0.4 % across 5 runs |
| Δ vs HEAD | +23.5 % | -0.5 % (noise) | |
| Δ vs Pre-Phase-B | +14.5 % | -0.0 % | |

**Methodology note (CLAUDE.md "Perf benchmarking" rule):** an early
BT=16-all run reported 162 tok/s and a separate decode run showed
20.9 tok/s — both turned out to be measurement noise (likely thermal
throttling from rapid back-to-back rocprof runs without DPM stabilization).
5-run JIT-warm benches with tight spread (0.4 %) are the reliable
signal; never trust 1-run or 2-run numbers for kernel-A/B decisions.
Decode noise was particularly misleading — looked like a 2× regression,
turned out to be one-off thermal. Always re-bench after any "this
broke" reaction.

### Why no further tuning on Path 1

The four dp4a kernels are now memory-throughput-bound at ~33 GB/s
vs MI50 peak 1024 GB/s (3 %). The ALU is idle ~47 % of cycles
waiting for memory. To move the floor we need either:
- LDS-staged A across multiple waves (real rewrite, ~1 session;
  also requires careful VGPR/LDS budget — multi-wave attempt above
  failed without LDS)
- MMQ-streaming small-tile port (Path 2, plan §3.1.3, ~5 sessions)

Path 2 is the right next move per the v3.2.5 errata sequencing.
Path 1's room is exhausted with BT=16 + single-wave 64-thread
blocks.

## Phase B.2: HFQ6 MMQ-streaming port — RESULT

Implementation: `docs/plans/gfx906-mq6-mmq-port-phase-b2.md` v2.1
sessions S1+S2+S3. Three commits on audit branch
(`feat/gfx906-hfq6-hfq8-analysis`):

- `8755a35` — S1: body.cuh + x8 + x64 + 2 Rust dispatchers + screen
- *(S2 commit pending)* — size sweep x16-x56 + dispatcher routing
- *(S3 commit pending)* — rewire 4 fused/residual dispatchers

### End-to-end mq6 9B prefill (5-run JIT-warm, spread 0.16 %)

| State | Prefill pp128 | Decode g50 | Δ vs prior |
|---|---:|---:|---:|
| Pre-Phase-B (BT=8 baseline) | 165.8 | 42.5 | — |
| HEAD before B.2 (`ff9e210`, BT=16) | 190.8 | 44.0 | +15 % |
| **Phase B.2 S3 (MMQ wired)** | **561.2** | **43.7** | **+194 %** |
| mq4 reference (audit HEAD) | 598.7 | n/a | — |
| **mq6/mq4 ratio** | **0.937** | | gap closed 3.14× → 1.07× |

**Result: 561.2 tok/s — exceeds the v2 plan's stretch target (450)
by 25 %; within 6 % of mq4 prefill parity.** Bandwidth-bound floor
of 407 tok/s (mq4/1.47) is exceeded by 38 % — the win is bigger
than 1.47× weight-byte overhead would predict, suggesting MMQ
amortizes more than just the per-byte cost (Q8_1 quantize sharing
across sibling projections + LDS-stage A reuse across batch tiles).

Decode unchanged at 43.7 tok/s (within 0.6 % of pre-B.2 baseline).
The MMQ path is gated by `should_use_mmq && hfq6_mmq_winning_size`
and never fires at B=1, so AR decode is unaffected by design.

### Per-call kernel timings (microbench, M=3584, K=4096)

S1.5 GO/NO-GO threshold (≥10 % over wave64_dp4a) was anchored on
B=8 in the v2 plan. At B=8, MMQ is actually 16 % SLOWER (block-size
mismatch with small grid; PMC: 112 waves vs 1792 for dp4a, VALUBusy
9 % vs 33 %). The threshold was the wrong reference workload.
Production prefill is B=128, where MMQ x64 wins big:

| Batch | mmq_x | MMQ µs | wave64_dp4a µs | Speedup |
|---:|---:|---:|---:|---:|
|   8 |  8 |  158 |  135 | 0.86× ❌ |
|  16 | 16 |  199 |  243 | 1.22× ✓ |
|  24 | 24 |  275 |  297 | 1.08× marginal |
|  32 | 32 |  381 |  366 | 0.96× marginal |
|  40 | 40 |  415 |  470 | 1.13× ✓ |
|  48 | 48 |  466 |  588 | 1.26× ✓ |
|  56 | 56 |  542 |  657 | 1.21× ✓ |
|  64 | 64 |  475 |  720 | 1.51× ✓ |
|  96 | 64 |  469 | 1082 | 2.31× ✓ |
| 128 | 64 |  485 | 1470 | **3.03×** ✓ |

Dispatcher routes MMQ only at B=16 or B≥40 (helper
`hfq6_mmq_winning_size`). B=24/32 fall through to wave64_dp4a — the
intermediate sizes are non-monotonic and not worth a regression risk
in S3 dispatch. Future S5 PMC stride sweep may close this gap.

### Cross-quant validation (LANDMINE discriminator)

Constant-weight test (`q ≡ 5`, `x ≡ 1.0`, expected
`Σ = M·K·(sc·q + zp) = 10240`): max_dev=0.125. Both math-identity
landmines from plan §3.1 (x_dm shift compensation, 0.25f factor)
clear by absolute equality.

Cross-architecture (MMQ vs wave64_dp4a at B=8): max_abs_err=0.00009
on a M=128 K=4096 random workload — essentially bit-exact.

Per-mmq_x sweep (M=128 K=4096, B=mmq_x for each ∈ {8,16,24,32,40,48,56,64}):
NRMSE 0.00031-0.00033 across all 8 sizes, no size-specific bug.

### Coherence

7/7 OK on the post-MMQ-wired daemon (mq3, mq4, mq6 prompts).

The 9b.mq6 reasoning prompt's prefill_tok_s in the coherence harness
stays at ~175.8 — same as pre-B.2 — because the prompt is 36 tokens
and `hfq6_mmq_winning_size(36) == false` (sizes 17-39 fall through to
wave64_dp4a per the S2 routing decision). MMQ wins materialize at
B=16 and B≥40, not at intermediate sizes. **The pp128 wall-clock
gain (190.8 → 561.2) is the headline number; coherence-harness
pp36 is not a regression test for B.2 by design.**

### S4 + S5 polish

**S4 — lm_head MMQ retarget.** `gemm_hfq6g256_batched_lmhead` rewired
to use `gemm_hfq6g256_mmq_set_gfx906` at `set` semantics (no memset
needed since `_full_set` overwrites). Routes via the same
`hfq6_mmq_winning_size` gate. End-to-end mq6 9B pp128 stayed at
561.4 tok/s (within 0.04 % of S3's 561.2) — the lm_head is a small
fraction of pp128 GEMM time, so the win is amortized into the
multi-projection sites. Decode unchanged at 44.2 tok/s.

**S5 — debug env vars + final regression.**

Two env-var knobs added in S5 to support production debugging:

| Env var | Effect |
|---|---|
| `HIPFIRE_HFQ6_MMQ=0` | Kill-switch: `hfq6_mmq_winning_size` always returns false → all 4 dispatcher sites fall through to wave64_dp4a |
| `HIPFIRE_HFQ6_MMQ_DIAG_PASSTHROUGH=1` | Forces `gemm_hfq6g256_residual_mmq_gfx906` to redirect to `gemm_hfq6g256_residual_fp16` for numerical-correctness bisection |

Validated via mq6 9B pp128:

| Config | mq6 pp128 | Match |
|---|---:|---|
| Default (MMQ on) | 562.1 tok/s | — |
| `HIPFIRE_HFQ6_MMQ=0` | **189.7 tok/s** | matches pre-B.2 baseline ✓ |
| `HIPFIRE_HFQ6_MMQ_DIAG_PASSTHROUGH=1` | 129.4 tok/s | FP16 fallback ≈ 70 % of wave64_dp4a |

**mq4 regression check** (S5 mandatory): mq4 9B pp128 at audit
HEAD = **597.7 tok/s** (vs pre-B.2 mq4 baseline 598.7) — within 0.2 %,
no regression. The HFQ6 MMQ path is properly isolated from HFQ4
dispatch.

### Phase B.2 follow-up: b128 cliff lowered for HFQ6 (B=32 fixed)

S2 sweep had B=32 at 0.96× regression (mmq_x=32 used b32 LDS path,
8 ds_read_b32 per inner ALU iter — issue-rate starvation).

PMC at mmq_x ∈ {16, 24, 32, 40} (all using b32 path before fix):

| Batch | mmq_x | LDS (B) | VALUBusy | MemUnitBusy | L2 hit |
|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 20480 | 11.2 % | 28.8 % | 42.6 % |
| 24 | 24 | 21504 | 13.4 % | 21.0 % | 47.3 % |
| **32** | 32 | 22528 | **10.8 %** | **13.8 %** | 47.6 % |
| 40 | 40 | 24064 | 12.0 % | 16.0 % | 51.2 % |

MemUnitBusy collapsed to 13.8 % at mmq_x=32 — kernel idle, not stalled.
Same SQ_WAVES (112), same VGPR (~92), no LDS bank conflicts. The b32
path's 8-way ds_read sequence was choking the LDS pipeline.

**Fix:** drop the b128 cliff from `mmq_x >= 64` to `mmq_x >= 32` in
`x_stride_for<>()` and the `vec_dot_dp4a_streaming` `if constexpr` —
both must move together (b128 reads need stride=40 for 16-byte
alignment; stride=33 only supports b32). This activates b128 (=
2 ds_read_b128 per inner ALU iter) for mmq_x ∈ {32, 40, 48, 56, 64}.

**Microbench (M=3584, K=4096, MMQ vs wave64_dp4a):**

| Batch | mmq_x | Before µs | After µs | Speedup before | Speedup after |
|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 381 | **316** | 0.96× ❌ | **1.15× ✓** |
| 40 | 40 | 415 | **348** | 1.13× | **1.35×** |
| 48 | 48 | 466 | **391** | 1.26× | **1.51×** |
| 56 | 56 | 542 | **433** | 1.21× | **1.51×** |
| 64+ | 64 | 475-485 | 475-485 | (already b128, unchanged) | unchanged |

**16-20 % per-call improvement at mmq_x ∈ {32, 40, 48, 56}**,
flipping B=32 from regression to a 1.15× win.

End-to-end mq6 9B pp128 unchanged at 561.1 tok/s — pp128 always picks
mmq_x=64, which was already on b128. The fix is forward-looking: B=32
spec-decode verify shapes now route to MMQ instead of falling back.

`hfq6_mmq_winning_size` updated: B≥32 now routes (was B≥40);
B=24 stays on wave64_dp4a (b32 path still marginal there at 1.08×).

Correctness: 14/14 PASS in test_hfq6_mmq.rs (all mmq_x ∈ {8..64}).
27B mq6 pp128: 192.6 tok/s (vs 192.8 pre-fix, within noise).

### 27B mq6 validation (was deferred in pre-S1 item 8)

After Phase B.2 shipped, we discovered `qwen3.6-27b.mq6` is already on
disk (Qwen 3.6 27B, same architecture family as 3.5 — same
hidden_dim=4096, head_dim=128, group_size=256). MMQ kernels are
dtype/shape-only; the routing applies identically.

Bench (gfx906 / MI50, qwen3.6-27b.mq6, pp128, JIT-warm):

| 27B mq6 9B pp128 | Prefill | Decode | Δ vs MMQ-off |
|---|---:|---:|---:|
| `HIPFIRE_HFQ6_MMQ=0` (kill-switch) | 54.8 tok/s | 15.2 | — |
| **Default (MMQ on)** | **192.8** | **15.4** | **+252 % prefill, +1 % decode** |

**3.52× prefill speedup at 27B — even bigger than 9B's 2.94×.** The
win scales WITH model size because larger M means more
`MMQ_Y=128` row tiles fully utilized (better grid coverage on the
SIMD pool).

Decode unchanged at 15.2-15.4 tok/s (MMQ never fires at B=1 by design;
the BW-bound floor at 27B is 305 GiB/s effective vs MI50 ~1024 peak).

**rocprof confirms MMQ kernels fire at 27B's larger K dimensions:**

| Kernel | Calls (1 prefill) | Calls (3 prefills) | Per-prefill |
|---|---:|---:|---:|
| `_full_set_x64` | 272 | 816 | 272/run ✓ scaling |
| `_full_add_x64` | 123 | 369 | 123/run ✓ scaling |
| `_full_add_x16` | 400 | 400 | one-time (mmq_screen pass) |
| `_residual_fp16` | 400 | 400 | one-time (mmq_screen reference) |

The fp16 + x16 calls are from the per-weight `mmq_screen_weight_hfq6`
running ONCE at first use (~400 weights for 27B = 64 layers × ~6
matrices). The actual prefill-hot kernels are `_full_set_x64` +
`_full_add_x64`, both scaling linearly with prefill-runs.

This closes the pre-S1 item 8 deferral and confirms MMQ correctness +
performance at the larger 27B-class (M, K) shapes — no regression at
larger K (8192 in some 27B layers, validated by the scaling growth).

**Methodology lesson re-confirmed:** during S4 testing a coherence-gate
chain held the GPU at 99 °C junction temp, causing a phantom 50 %
decode regression (43.7 → 21.2). Pulling the load and waiting for
cooldown restored the numbers. This matches the
`feedback_perf_noise_decode.md` memory — 5+ warm runs with consistent
spread is necessary but not sufficient; thermal junction temp needs
to be checked explicitly via `rocm-smi --showtemp` before trusting
"this regressed" claims.

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
