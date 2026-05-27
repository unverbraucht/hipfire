# dots.ocr Phase 6 (perf) — dev log

Goal: bring dots.ocr end-to-end latency on a real page toward vLLM's
**~15s**. Branch `feat/dots-ocr-phase-6-wmma-gemm`. Deploy target gfx1100
(RX 7900 XTX), system ROCm 7.2.3.

Companion design doc: [`dots-ocr.perf-investigation.md`](dots-ocr.perf-investigation.md).
Per-kernel timing is the tool of record — PMC memory counters
(`FETCH_SIZE`, `GL2C_*`) read flat zero on this gfx1100 under rocprofv1/v2/v3
(driver/GFXOFF-gated). The dated entries below are a reverse-chronological log;
**start with the handoff block, then the ranked headroom at the bottom.**

## Status & handoff (2026-05-27)

Branch `feat/dots-ocr-phase-6-wmma-gemm` (off master `09e94b25`), 9 commits, all
landed with **F1=1.000** vs the vLLM reference. The dots.ocr F1 grade is the
**only** correctness gate that matters here — the repo's `coherence-gate.sh`
runs qwen35 models and does NOT exercise the qwen2/dots-ocr forward path, so any
change to these kernels/forward must be re-graded:

```
cargo build --release -p hipfire-arch-dots-ocr --example ocr_e2e
./target/release/examples/ocr_e2e --hfq /data/hipfire/dots-ocr.q8.hfq \
  --image benchmarks/images/dots_ocr_smoke_001.jpg \
  --prompt-json benchmarks/references/dots_ocr_smoke_001.json \
  --prefill batch > /tmp/our.txt 2>/tmp/our.err          # decodes to EOS (~4633 tok)
python3 scripts/grade_dots_ocr_e2e.py --our /tmp/our.txt \
  --ref benchmarks/references/dots_ocr_smoke_001_vllm.json  # want F1=1.000, 13/13
```

Honest end-to-end (smoke image, full layout to EOS) vs vLLM's ~15s target:

| stage | session start | now | how |
|---|---|---|---|
| vision  | 49.6s | **32.2s** | mb4 WMMA GEMM + attention de-spill |
| prefill | 27.4s | **1.1s**  | wired gfx11 WMMA Q8 GEMMs (25×) |
| decode (4633 tok) | 62.3s | 62.3s | characterized, near kernel floor |
| **total** | ~139s | **~96s** | |

(`ocr_e2e` reports honest per-stage times since the async-drain timer fix; the
decode-loop is timed in isolation. The "vision 49.7 / prefill 27.4" baselines
above are post-fix — pre-fix the tool mis-reported prefill as 0.1s and decode
as 2.3 tok/s; see the first dated entry.)

**Tools / repro:**
- In-engine per-stage timing: `HIPFIRE_PREFILL_TIMING=1` (prefill GEMM/attn
  split), `HIPFIRE_DECODE_TIMING=1` (per-token forward vs argmax).
- Microbenches: `cargo run --release -p rdna-compute --example bench_gemv_q8`
  (decode GEMV bandwidth by shape), `… --example bench_decode_attention --seq
  5100 --iters 100` (decode attention µs/call; honors `HIPFIRE_GQA_CHUNK`),
  `… --example check_mb4` (mb4 GEMM bit-exactness, incl. sub_offset halves).
- Profiler: `rocprofv2 --kernel-trace -d OUT -o t -- <bin>` gives per-kernel
  timestamps + VGPR/SGPR/LDS (vision profiles fine; **crashes on prefill/long
  decode** — the AqlPacket many-dispatch bug; use the in-engine timers there).
  Prepend a `/tmp/hipcc-stub` (`#!/bin/sh\nexit 0`) to PATH so the compiler
  version-probe doesn't deadlock under the profiler.
- Static register/LDS/spill (the decisive data the profiler can't see — memory
  PMC counters are DEAD on this gfx1100 across all profiler versions): the
  `gfx-kernel-metadata` skill on `.hipfire_kernels/gfx1100/<kernel>.hsaco`
  (`clang-offload-bundler` unbundle → `llvm-readelf --notes` → vgpr_count /
  vgpr_spill_count / group_segment_fixed_size). gfx1100 = 1024 VGPR/SIMD,
  16 max waves/SIMD, 64 KB LDS/CU.

Model: `/data/hipfire/dots-ocr.q8.hfq` (Q8 text weights, F16 vision weights).
Text decoder: hidden 1536, 28 layers, 12 heads / 2 kv-heads (GQA 6:1), head_dim
128, interm 8960. Vision: embed 1536, 42 blocks, interm 4224, ~19520 patches.

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

## 2026-05-27 — decode profiling: attention-bound (7ms), cheap levers exhausted

Decode 74 tok/s ≈ 13.5ms/token GPU. rocprofv2 trace (1-token) + a synthetic
`bench_gemv_q8` give the split:

| decode kernel | time/token | note |
|---|---|---|
| `attention_flash_gqa_partial` (28×) | **~7ms** | 250µs/call @ ctx 5095, F32 KV ~38 GB/s — occupancy-bound (80 blocks < 96 CUs) |
| `gemv_q8_0_wide` (qkv/o, M=1536) | ~1ms | **33% peak** (small-M latency-dominated) |
| gemv gate/up/down (M=8960) | ~2.5ms | 67–73% peak — already good |
| misc (rmsnorm/silu/rope/add) | ~3ms | — |

`bench_gemv_q8` (gfx1100, ~960 GB/s peak): qkv/o (M1536) 319 GB/s (33%),
gate/up (M8960) 667 (70%), down 639 (67%), lm_head 697 (73%). Small-M is
latency-dominated (only ~2.5 MB moved); large-M saturates fine.

**Cheap decode levers confirmed exhausted:**
- `HIPFIRE_GQA_CHUNK` sweep: 128→270µs, 64→276µs, 32→321µs (+ maxdiff 8.0, the
  reduce assumes chunk≥64), 16→416µs. More blocks is *worse*, not better —
  occupancy-via-chunks is a dead end on gfx1100 too (gfx1151 prediction
  falsified). flash (480 blocks) is *slower* (424µs) than gqa (80, 270µs)
  because gqa's 6× KV-reuse wins despite fewer blocks.
- Q8 KV cache: already rejected (project memory — same wall as F32; decode
  attention is dispatch/occupancy-bound, not KV-byte-bound).

Decode attention (7ms, the dominant block) needs a structural redesign — more
parallelism without losing the GQA KV-reuse — not a knob. The small-M gemv
(33% peak) is improvable (multi-row-per-wave) but only ~1ms, ~4% of decode.
Decode is near its kernel floor on gfx1100 with the current attention.

## 2026-05-27 — NEGATIVE: dropping attention V_lds (49KB→17KB) is a no-op

Hypothesis: the 49 KB dynamic LDS (V_lds 32 KB + S_lds 16 KB) caps the attention
to 1 workgroup/CU, so dropping V_lds (read V from DRAM in phase C) → 17 KB →
3 wg/CU should hide latency via occupancy. Tested (kernel + dispatch shared_mem
both updated): vision **32.8s vs 32.2s — neutral** (within noise), and it
reintroduced 13 VGPR spills. **Reverted.**

Lesson: the vision attention is **latency-bound, not occupancy-bound** — the
LDS-staged V was *itself* the latency-hiding mechanism (on-chip SRAM, faster
than L2), so trading it for occupancy is a wash at best. The 3× wave headroom
didn't recover the exposed L2/DRAM V-read latency. LDS reduction is not a lever
for this kernel; V staging stays.

## 2026-05-27 — vision attention de-spill: 926 spills → 0, vision 39.8s → 32.2s

The 926-VGPR spill was caused by **full `#pragma unroll` of the two inner 8-way
d-chunk WMMA loops** (phase-A QK and phase-C SV) — the compiler kept 8 live
`b_reg` (half16_t) copies + the [8] accumulator arrays, blowing past the 256 cap.
Dropping the unroll factor to **4** (`#pragma unroll 4`) reuses fewer `b_reg`:
VGPR 256+926spill → **214, 0 spill**. unroll 1 also de-spills (VGPR 166) but
loses ILP (vision 36.4s); unroll 4 keeps ILP and is the sweet spot (**32.2s**).
VGPR headroom up to ~256 is free here because LDS (49 KB) already caps occupancy
to 1 wg/CU. (Also folded out the redundant `o_acc_per_dc[8]` — but that was a
no-op; the compiler already fused it into O_frags, spill count was unchanged by
it.) F1=1.000, 13/13. **Vision now 32.2s** (attention ~18.8s → ~12s).

Still occupancy-capped by the 49 KB dynamic LDS (V_lds 32 KB + S_lds 16 KB) →
1 workgroup/CU. Reducing LDS ≤ 32 KB → 2 wg/CU is the next attention lever.

## 2026-05-27 — vision profiling: where the remaining 39.8s goes + the knobs

rocprofv2 kernel-trace (timing+occupancy) + `gfx-kernel-metadata` skill (static
`.hsaco`, exposes spills the profiler can't). gfx1100 = 1024 VGPR/SIMD, 16 max
waves/SIMD, 64 KB LDS/CU. (No `rocprofv2` in /opt/rocm-7.13 — only rocprofv3,
counters still dead — so 7.2.3 rocprofv2 is the best available; memory counters
dead everywhere, but timing + the static occupancy/spill data are enough.)

Vision now splits ~evenly between two kernels:

| kernel | total | VGPR | spill | LDS | verdict |
|---|---|---|---|---|---|
| `gemm_f16_wmma_mb4` | 20.9s | 72 | **0** | 0 | clean; NOT occupancy-bound (14 waves/SIMD possible) → **memory/reuse-bound** |
| `attention_dflash_..._v3_f32` | 18.8s | **256** | **926** | 49 KB dyn | **catastrophic register spill** + LDS caps to 1 wg/CU (~12% occ) |

**The single best vision knob: the attention kernel's 926-VGPR spill.** The
M=64×N=128 f16-K/V WMMA tile over-provisions registers → 926 spills to scratch
(VRAM round-trip per spill) — the real reason it's 449 ms/call (×42 = 18.8s),
beyond just low occupancy. Lever: cut register pressure (smaller query tile M,
or stage accumulator/state in LDS). The §14.1–14.4 plans (V-load focused) don't
target the spill directly; the spill is the bigger lever.

**mb4 GEMM (20.9s)** is spill-free with occupancy headroom → memory-bound; its
lever is more operand reuse (2D M+N register blocking or LDS staging), not
occupancy. NB=8 alone wouldn't help (not VGPR-capped).

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

## Plan — done vs remaining

**Done this session (all F1=1.000):**
- [x] Fix `ocr_e2e` async-drain measurement bug (`9ae0f08e`) — honest per-stage times.
- [x] **Prefill** WMMA Q8 GEMM: wired the proven gfx11 `gemm_qkv_q8_0_wmma` /
      `gemm_gate_up_q8_0_wmma` / `gemm_q8_0_residual_wmma` into
      `forward_prefill_batch_embeds` (`fb767260`). 27.4s → 1.1s (25×).
- [x] **Vision GEMM**: new `gemm_f16_wmma_mb4.hip` (NB=4 register block +
      transposed output, drops the per-GEMM transpose), wired into
      `linear_f16`/`linear_f16_no_bias`/fc13 (`4ed42b7f`). 49.6s → 39.8s.
- [x] **Vision attention de-spill**: `#pragma unroll 4` on the inner WMMA loops
      killed the 926-VGPR spill (`7ca958de`). 39.8s → 32.2s.
- [x] **Decode characterized** (`7471722f`): attention-bound (~7ms/token), cheap
      levers (GQA-chunk, Q8-KV, V_lds-drop, hipGraph) all confirmed dead.

**Remaining headroom (ranked for the next agent):**

1. **Decode attention structural redesign** — ~7ms/token, the single largest
   block end-to-end. `attention_flash_gqa` is occupancy-bound: grid =
   `n_kv_heads(2) × n_chunks(40)` = **80 blocks** underfilling 96 CUs, reading
   F32 KV at ~38 GB/s. DEAD knobs (do not retry): `HIPFIRE_GQA_CHUNK` (more
   blocks → worse, breaks <64), per-head flash (480 blocks, slower — GQA's 6×
   KV-reuse wins), Q8 KV (same wall). NEEDS a decomposition that ADDS
   parallelism WITHOUT losing the GQA reuse — e.g. also split the head_dim
   (d-chunks) across blocks (2 kv × 40 chunks × N d-groups) with a reduce that
   recombines, or a streaming single-pass kernel with more concurrent waves.
   Files: `kernels/src/attention_flash_gqa.hip`, dispatch
   `dispatch.rs::attention_flash_gqa`, heuristic
   `qwen2.rs::forward_step_after_x` (gqa when `n_kv<n_heads && pos+1>=4096`).
   Validate: `bench_decode_attention` maxdiff vs flash must be ~0, THEN F1.
2. **Vision attention §14.1–14.4** (perf-investigation.md — planned, never
   implemented): async V-load (`global_load_lds`), V_lds transpose, N=256.
   Kernel already de-spilled to unroll-4/VGPR-214. NOTE: LDS *reduction* is a
   proven no-op (latency-bound, V staging is the latency hider) — don't retry.
3. **Vision mb4 GEMM** (20.9s; 33% peak at M=1536, 67–73% at M≥8960): 2D (M+N)
   register blocking or LDS staging for more operand reuse. Spill-free with
   occupancy headroom → memory-reuse-bound. `kernels/src/gemm_f16_wmma_mb4.hip`.
4. **Decode small-M gemv** (qkv/o at 33% peak, ~1ms, ~4% of decode): multi-row-
   per-wave to amortize launch/ramp. Low ROI. `kernels/src/gemv_q8_0_wide.hip`.
5. **Decode hipGraph** — parked in `git stash@{0}` + `~/decode-hipgraph-wip`
   (gated `HIPFIRE_DECODE_GRAPH`). No-op on gfx1100 (decode compute-bound), but
   a real win for the dispatch-bound gfx1151 box.

**Gotchas for the next agent:**
- `sub_offset` on a `DType::Raw` tensor is BYTE-addressed — `fc13_proj` is Raw
  F16, so multiply element offsets by 2 (this caused a decode attractor).
- `hipStreamBeginCapture` records without executing — a capture must be
  followed by a replay or logits stay stale + kv_cache_write never lands.
- Any forward-path change → re-grade F1 (coherence-gate does NOT cover dots).

## rocprofv2 on gfx1100 (this session)
- Runs; gives per-kernel timestamps + VGPR/SGPR/LDS/occupancy. ✅
- Memory counters dead (`FETCH_SIZE`=0, `GL2C_HIT` unsupported) — same as v3.
- One metric per pass (small buffer); multi-counter aborts.
- **Crashes on the full prefill** (`AqlPacket` assertion — the many-dispatch
  bug). Profiles vision fine; for prefill, use in-engine category timing
  (`HIPFIRE_PREFILL_TIMING=1`) instead.
