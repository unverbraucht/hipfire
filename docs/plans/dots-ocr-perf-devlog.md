# dots.ocr Phase 6 (perf) — dev log

Goal: bring dots.ocr end-to-end latency on a real page toward vLLM's
**~15s**. Branch `feat/dots-ocr-phase-6-wmma-gemm`. Deploy target gfx1100
(RX 7900 XTX), system ROCm 7.2.3.

Companion design doc: [`dots-ocr.perf-investigation.md`](dots-ocr.perf-investigation.md).
Per-kernel timing is the tool of record — PMC memory counters
(`FETCH_SIZE`, `GL2C_*`) read flat zero on this gfx1100 under rocprofv1/v2/v3
(driver/GFXOFF-gated; see `docs/perf-checkpoints/2026-05-27-*` once written).

## 2026-05-27 — honest end-to-end breakdown (the measurement-bug fix)

`ocr_e2e` reported prefill at `0.1s / 3696 tok/s` and decode at `2.3 tok/s`.
Both were **measurement artifacts**: the batched prefill enqueues async and
returns before the GPU work runs, so the prefill timer measured only host
submission, and the first post-prefill `argmax` then drained ~27s of pending
prefill compute — which the generate timer charged to *decode*. Added a
`device_synchronize()` after the prefill call (`ocr_e2e.rs`); decode-loop is
now timed in isolation.

Honest breakdown, smoke image (`dots_ocr_smoke_001.jpg`, 5095 prompt
positions = 4880 visual + 215 text), 32 decode tokens:

| stage | time | share |
|---|---|---|
| vision encoder | **49.7 s** | 64% |
| prefill (5095 pos) | **27.4 s** (185 tok/s) | 35% |
| decode | 0.4 s (**78 tok/s**) | <1% |

Decode is **not** the bottleneck — it's already ~78 tok/s. (The decode
hipGraph capture/replay work this session is gated behind
`HIPFIRE_DECODE_GRAPH` and gives ~0 on gfx1100 because decode here isn't
host-dispatch-bound; it's a deliverable for the dispatch-bound gfx1151 box.
Parked in `git stash@{0}` + `~/decode-hipgraph-wip/`, not committed.)

## Root causes (both are the same problem: no real tiled GEMM on RDNA3)

**Prefill = 99% Q8 GEMM.** Per-category timing (`HIPFIRE_PREFILL_TIMING=1`):
FFN GEMM 23.1s + QKV GEMM 4.0s + causal-WMMA attention 0.4s. The
`gemm_q8_0_batched_chunked` WMMA path is gated `is_rdna4()`, so gfx1100
(RDNA3) falls to `gemm_q8_0_batched` — a GEMV-style kernel (one block per
output row, no weight reuse across the batch → the weight matrix is
re-streamed from DRAM for every one of 5095 batch rows). Attention is fine;
the §14.5 causal-WMMA win already landed.

**Vision = naive WMMA GEMM + redundant transpose.** rocprofv2 kernel-trace:
`gemm_f16_wmma` 26.4s (171×154ms) + vision attention 18.8s (42×449ms) +
`transpose_f32` 4.1s (one per GEMM, layout fixup). `gemm_f16_wmma` *uses*
WMMA but is naively tiled (one wave per 16×16 output tile, re-reads operands
from DRAM every K-step, no LDS staging / no reuse) — matches the perf doc's
"K-tile=16 vs 256, L2 hit <1%, DRAM-bound".

## Plan (ranked by leverage)

1. **Prefill: wire the existing gfx11 WMMA Q8 GEMM kernels** —
   `gemm_qkv_q8_0_wmma`, `gemm_gate_up_q8_0_wmma`, `gemm_q8_0_residual_wmma`
   (all RDNA3 `_w32`, already proven in the qwen35 production path) — into
   `forward_prefill_batch_embeds`. These are real tiled GEMMs (16×16 output
   tiles, `blockIdx.y` = batch tile → weight reuse across the batch). NOT a
   port; just not currently used by the dots.ocr prefill. **Doing the fused
   variants (QKV in one launch, gate+up in one launch) first** — same as
   qwen35 prefill.
2. **Vision GEMM**: replace naive `gemm_f16_wmma` with an LDS-tiled +
   register-blocked WMMA GEMM; drop the per-GEMM `transpose_f32`.
3. **Vision attention** (§14.1–14.4 — only ever planned, never implemented,
   incl. on gfx1151): async V-load, V_lds transpose, N=256, M=128.

## rocprofv2 on gfx1100 (this session)
- Runs; gives per-kernel timestamps + VGPR/SGPR/LDS/occupancy. ✅
- Memory counters dead (`FETCH_SIZE`=0, `GL2C_HIT` unsupported) — same as v3.
- One metric per pass (small buffer); multi-counter aborts.
- **Crashes on the full prefill** (`AqlPacket` assertion — the many-dispatch
  bug). Profiles vision fine; for prefill, use in-engine category timing
  (`HIPFIRE_PREFILL_TIMING=1`) instead.
