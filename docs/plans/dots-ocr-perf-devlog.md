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

## 2026-05-27 — prefill WMMA Q8 wired: 27.4s → 1.1s (≈25×), F1=1.000

Wired the fused gfx11 WMMA Q8 GEMMs into `forward_prefill_batch_embeds`:
`gemm_qkv_q8_0_wmma` (fused QKV), `gemm_gate_up_q8_0_wmma` (fused gate+up),
`gemm_q8_0_residual_wmma` for o_proj + down (folds the residual add into the
GEMM). Gated on all-Q8 weights + WMMA arch + K%32==0; falls back to the old
`proj()` (GEMV) path otherwise. Mirrors the qwen35 production prefill path; the
kernels were already proven there, only the wiring is new.

Result (smoke image, full layout decoded to EOS):
- **prefill 27.4s → 1.1s (4804 tok/s) ≈ 25×**
- grade vs vLLM: **F1=1.000, 13/13 regions, text exact-match 13/13 — PASS**
- new end-to-end: vision **49.6s** + prefill 1.1s + decode **62.3s** (4633 tok @ 74 tok/s)

**Decode is now the largest full-page component** (62s > vision 49.6s) — the
earlier "decode is negligible" held only for the 32-token smoke window. Next
levers: vision GEMM (same naive-WMMA class of fix as prefill) and decode
throughput (the parked hipGraph work targets dispatch-bound boxes; on gfx1100
decode is compute-bound at ~74 tok/s).

## 2026-05-27 — vision GEMM: register-blocked mb4 WMMA, 49.6s → 39.8s, F1=1.000

New kernel `gemm_f16_wmma_mb4.hip`: NB=4 register blocking over N (one 16×64
output panel per block, the W tile loaded once per K-step and reused across 4
N-subtiles) + **transposed [N,M] output** (folds away the per-GEMM
`transpose_f32`). Wired into `linear_f16`, `linear_f16_no_bias`, and the fc13
SwiGLU (two mb4 GEMMs on the fc1/fc3 weight halves → silu → no transpose).

Result: vision **49.6s → 39.8s** (~20%), F1=1.000, 13/13, text exact 13/13.
Modest vs the 4× the weight-traffic cut predicted — the vision GEMM isn't
purely weight-DRAM-bound at these shapes (X traffic + the 4-accumulator VGPR
cost trims occupancy); NB=8 / M-blocking / LDS staging are follow-ups. New
e2e: vision 39.8s + prefill 1.1s + decode 62.3s.

**Bug caught during bring-up (your "stride" instinct):** fc13 first produced a
decode attractor. Root cause was NOT the kernel (bit-exact vs `gemm_f16_wmma`
at every shape incl. non-64-divisible N) — it was the weight slice. `fc13_proj`
is `DType::Raw` (1-byte stride) holding F16 data, so `sub_offset` takes BYTE
offsets; `sub_offset(interm*h)` landed fc3 mid-fc1. Fix: `sub_offset(interm*h*2)`
(F16 bytes). The naive path never noticed because `gemm_f16_wmma` reads the ptr
as `_Float16*` regardless of dtype. Lesson: `sub_offset` on a `Raw` tensor is
byte-addressed — multiply element indices by the real element size.

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
