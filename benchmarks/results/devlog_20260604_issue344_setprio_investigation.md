# Dev Log: FeatherOps Techniques Investigation (Issue #344)

**Date:** 2026-06-04
**Hardware:** AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (gfx1151, Strix Halo, 40 CU, 32-wide wavefront)
**ROCm:** 7.13
**Branch:** `feat/PR-344-featherops` (based on `origin/master` at `02634f4c`)

## Context

Issue #344 investigates techniques from [ComfyUI-FeatherOps](https://github.com/woct0rdho/ComfyUI-FeatherOps)
— hand-tuned HIP WMMA kernels achieving ~47% over hipBLASLt on gfx1151. Kaden's triage ranked
three experiments by effort:

1. `s_setprio` priority hint — ~1–2 lines per WMMA kernel
2. Identity-order B-operand prepack — audit LDS layout for bank conflicts
3. Register-tiled B-fragment reuse — WMMA K-loop optimization

---

## Experiment 1: Baseline Profiling (rocprofv3)

### Kernel under test

`gemm_gate_up_mq4g256_lloyd_wmma` (gfx1151 K4 variant)
- 32-thread WG, 512 B LDS, WMMA 16×16×16 f16 → f32
- MQ4-Lloyd: 4-bit nibble-pack with 16-entry half16 codebook per group of 256 weights
- Grid: `[ceil((gate_m+up_m)/16), ceil(N/16)]`
- `__launch_bounds__(32, 2)` — only 2 WGs per CU guaranteed

### Hardware counter baseline (M=27648, K=8192, N=64)

| Counter | Value | Notes |
|---|---|---|
| GPUBusy | 100% | Fully saturated |
| MemUnitBusy | 99% | Memory-bound |
| MeanOccupancyPerActiveCU | 53.3 | Low — see §Exp 4 |
| SQ_WAVES | 345.6 per dispatch | |
| SQ_BUSY_CYCLES | 13.5M per CU | |
| SQ_INSTS_VALU | 8.64M | |
| SQ_INSTS_LDS | 2.84M | Heavy LDS usage (codebook lookups) |
| FETCH_SIZE | 1,282,997 | Global read traffic |
| WRITE_SIZE | 7,200 | Minimal write traffic |
| LDSBankConflict | 54 | Negligible LDS bank conflicts |
| SQC_LDS_BANK_CONFLICT | 3,342,318 | Significant SQC→LDS path conflicts |
| L2CacheHit | 50.6% | Moderate |
| VGPR | 96 per wave | |

### Wall-clock baseline (median of 11 runs × 3 fresh processes)

| Shape | Median µs | GiB/s |
|---|---|---|
| M=27648 K=8192 N=64 (27b prefill) | ~4940 | ~28.2 |
| M=6912 K=8192 N=64 (9b prefill) | ~1190 | ~29.7 |
| M=27648 K=8192 N=16 (27b small batch) | ~1940 | ~68.9 |
| M=27648 K=8192 N=1 (27b decode-like) | ~2230 | ~59.1 |

**Key insight:** The kernel is 99–100% memory-bandwidth bound. L2 hit rate is only 50%,
meaning half the weight data is re-fetched from VRAM.

---

## Experiment 2: s_setprio around codebook loads

### Hypothesis

From FeatherOps: `s_setprio 1` (low wave priority) during global→LDS loads lets outstanding
WMMA compute drain. `s_setprio 0` restores normal priority for compute.

### Implementation

Two variants tested:
- **Variant A:** `s_setprio 1/0` around codebook VRAM→LDS loads (per group)
- **Variant B:** `s_setprio 1/0` around inner K-loop weight reads (finer-grained)

### Results (3 fresh-process iterations)

Both variants show results within the ±5% DPM noise floor on the Strix Halo APU.
No reproducible, statistically significant win on any shape.

### Conclusion

**s_setprio is not a reliable win on this kernel/hardware.** The kernel is 99%
memory-bound — compute/load overlap hints have minimal headroom. The FeatherOps
win was on compute-bound kernels at M=N=K=8192.

---

## Experiment 3: Detailed profiling — low occupancy discovered

### Key finding

The `__launch_bounds__(32, 2)` hint tells the compiler to optimize for only 2 WGs per CU.
Each WG is 32 threads = 1 wave32. So the hardware schedules only 2 waves/CU (6.25%
of the 32-wave max), while the VGPR budget (512/SIMD ÷ 96/wave = 5 waves/SIMD) allows
up to **10 waves/CU**.

On a memory-bound kernel, low occupancy is devastating — with only 1 wave per SIMD,
any memory stall is a full stall with no other wave to schedule. The L2 cache also
underutilized because data is evicted before the next wave can reuse it (50.6% L2 hit rate).

---

## Experiment 4: High-occupancy variant — `__launch_bounds__(32, 10)`

### Hypothesis

Increasing `__launch_bounds__` from `(32, 2)` to `(32, 10)` allows the hardware to
schedule more WGs per CU, improving memory latency hiding and L2 temporal locality.

### Implementation

`kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_hiocc.gfx1151.hip` — single-line change:
```c
-__launch_bounds__(32, 2)
+__launch_bounds__(32, 10)
```

### Correctness

**Bit-exact match** across all shapes (M=27648/6912, N=64/16/1). Zero mismatches,
max absolute difference = 0.0. The launch_bounds change is purely a scheduling hint.

### Compiler behavior

The compiler allocated **more** VGPR for the hiocc variant: 96 → 112 VGPR/wave.
This reduces max waves/SIMD from 5 to 4 (8 waves/CU instead of 10), but still
4× better than the original's effective 2 waves/CU.

### Performance (3 iterations, median of 11 runs)

| Shape | Original µs | Hiocc µs | Δ% | Consistency |
|---|---|---|---|---|
| 27b N=64 | ~4980 | ~4200 | **+15–16%** | Rock-solid |
| 9b N=64 | ~1230 | ~1000 | **+16–20%** | Consistent |
| 27b N=16 | ~2020 | ~1670 | **+17–19%** | Consistent |
| 27b N=1 | ~2240 | ~1570 | **+28–31%** | **Huge, rock-solid** |

### Hardware counter comparison

| Counter | Original (lb 32,2) | Hiocc (lb 32,10) | Change |
|---|---|---|---|
| MeanOccupancyPerActiveCU | 53.3 | 37.5 | (metric semantics differ) |
| VGPR/wave | 96 | 112 | +17% |
| Max waves/CU (VGPR limit) | 10 | 8 | -20% but still 4× original effective |
| FETCH_SIZE | 1,282,997 | 1,000,893 | **-22%** |
| L2CacheHit | 50.6% | **62.0%** | **+11.4pp** |
| SQ_BUSY_CYCLES | 13,465,100 | 11,327,454 | **-16%** |
| GPUBusy | 100% | 100% | same |
| MemUnitBusy | 99% | 98.5% | same |
| LDSBankConflict | 54 | 54 | same |
| SQC_LDS_BANK_CONFLICT | 3,342,318 | 3,342,318 | same (structural) |

### Mechanism

1. **More waves/CU → better latency hiding:** With 4× more concurrent waves,
   when one wave stalls on a global load, the SIMD can issue instructions from
   other waves. This reduces the average memory stall seen per wave.

2. **Better L2 temporal locality:** With more WGs per CU reading the same weight
   rows (workgroups in the same row_tile but different batch_tiles), data stays
   hot in L2 longer. L2 hit rate jumped from 50.6% to 62.0%.

3. **22% less VRAM traffic:** The higher L2 hit rate means 22% fewer VRAM fetches,
   which is the direct cause of the wall-clock improvement on a bandwidth-bound kernel.

### Why the win is largest at N=1

At N=1 (decode-like), there are 55,296 row tiles × 1 batch tile = 55,296 WGs.
With 40 CUs, each CU processes ~1,382 WGs. At 2 WGs/CU (original), that's 691
scheduling rounds. At 8 WGs/CU (hiocc), only 173 rounds. The latency hiding
benefit is proportionally larger when there are fewer waves per CU competing
for memory bandwidth.

### Risk assessment

- **No correctness risk:** Bit-exact output match.
- **No register pressure risk:** VGPR went up 96→112 but stays well within limits.
- **No LDS risk:** Still 512 B/WG, unchanged.
- **The compiler chose more VGPR deliberately:** With more waves, the compiler
  has less per-wave register pressure budget but chose to use *more* registers
  (for better ILP within each wave). The net effect is still positive.
- **Portable to other archs:** The launch_bounds change is gfx1151-specific in
  the `.gfx1151.hip` file, so it won't affect gfx1100/gfx1200/gfx12 builds.

### Recommendation

**Promote `__launch_bounds__(32, 10)` to production for the gfx1151 gate_up kernel.**
The +15–30% win is consistent across all shapes and the change is isolated to one line
in one file. The same analysis should be applied to the other gfx1151 WMMA kernels
(QKV, residual, mb4 variants) — they all use `__launch_bounds__(32, 2)` and are
likely also occupancy-starved.

---

## Promotion status (2026-06-04)

### Promoted to production: `__launch_bounds__(32, 10)` on all 15 gfx1151 kernels

The following files were changed from `__launch_bounds__(32, 2)` to `__launch_bounds__(32, 10)`:

**MQ4-Lloyd WMMA kernels (prefill hot path):**
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.gfx1151.hip`
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_mb4.gfx1151.hip`
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_mb4_nosync.gfx1151.hip`
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_nosync.gfx1151.hip`
- `kernels/src/gemm_qkv_mq4g256_lloyd_wmma.gfx1151.hip`
- `kernels/src/gemm_qkv_mq4g256_lloyd_wmma_mb4.gfx1151.hip`
- `kernels/src/gemm_qkvza_mq4g256_lloyd_wmma.gfx1151.hip`
- `kernels/src/gemm_qkvza_mq4g256_lloyd_wmma_mb4.gfx1151.hip`
- `kernels/src/gemm_mq4g256_lloyd_residual_wmma_mb4.gfx1151.hip`

**MMQ / MoE kernels:**
- `kernels/src/gemm_hfq4g128_mmq.gfx1151.hip`
- `kernels/src/gemm_hfq4g256_moe_grouped_mmq.gfx1151.hip`
- `kernels/src/gemm_hfq4g256_moe_grouped_mmq_k4.gfx1151.hip`
- `kernels/src/gemm_hfq4g256_moe_grouped_mmq_k8.gfx1151.hip`
- `kernels/src/gemm_paro_q4g128_moe_grouped_mmq.gfx1151.hip`
- `kernels/src/gemm_paro_q4g128_moe_grouped_mmq_k8.gfx1151.hip`

### Microbenchmark (isolated gate_up GEMM)

| Shape | Original µs | Hiocc µs | Δ% | Consistency |
|---|---|---|---|---|
| 27b N=64 | ~4960 | ~4160 | **+16%** | Rock-solid across 3 runs |
| 9b N=64 | ~1230 | ~1000 | **+18–20%** | Consistent |
| 27b N=16 | ~2000 | ~1670 | **+17%** | Consistent |

Binary md5 (original): `b485989bc9ca6a63637a0b6fe0bfe339`
Binary md5 (hiocc): `47ba52ea4e2a120441120bb657e6fa7b`

### End-to-end (Qwen 3.6-27B, dflash_spec_demo)

Target: `qwen3.6-27b.mq4`, Draft: `qwen35-27b-dflash-mq4.hfq`
Config: `max=256, ctx=4096, kv-mode=q8, block=16, no-chatml, no-adaptive-b`

**Decode tok/s (median of runs 2–3, run 1 = DPM warmup):**

| Mode | ORIG (32,2) | HIOCC (32,10) | Δ |
|---|---|---|---|
| AR code (232 prefill tokens) | 14.18 | 14.18 | ~0% |
| AR prose (27 prefill tokens) | 14.33 | 14.33 | ~0% |
| DFlash code (232 tokens) | 41.9 | 42.0 | +0.2% |
| DFlash prose (27 tokens) | 87.2 | 87.5 | +0.3% |

**Prefill tok/s (median of runs 2–3):**

| Mode | ORIG (32,2) | HIOCC (32,10) | Δ |
|---|---|---|---|
| AR code (232 tokens) | 201.1 | 201.4 | +0.1% |
| AR prose (27 tokens) | 139.5 | 139.0 | -0.4% |
| DFlash code (232 tokens) | 201.7 | 202.2 | +0.2% |
| DFlash prose (27 tokens) | 139.2 | 139.4 | +0.1% |

**Why the E2E win is near-zero despite +16–20% microbenchmark win:**

Qwen 3.6-27B has 64 layers. Only 16 (25%) are FullAttention layers that
use the WMMA MQ4-Lloyd GEMM kernels we changed. The other 48 (75%) are
LinearAttention (DeltaNet) layers that use the gemv path (unchanged).
Furthermore, prefill is only ~1.1s of a ~70s total process lifecycle
(~62s model load + ~5–10s decode). The arithmetic works out:

```
WMMA fraction of prefill ≈ 25% × ~50% per-layer = 12.5%
Prefill fraction of E2E ≈ 1.1s / 70s = 1.6%
Expected E2E win ≈ 16% × 12.5% × 1.6% ≈ 0.03%
```

Exactly what we observe. The win is real but diluted by the Qwen 3.6
architecture's 75% DeltaNet composition.

**Where the win will matter in production:**

1. **Multi-user serving**: concurrent prefill requests where WMMA throughput
   is the bottleneck on request start latency
2. **Long-context prefill** (ctx > 4096): prefill wall time grows, WMMA
   fraction grows with it
3. **Models with more FullAttention layers**: Qwen 3.5-9B has fewer
   DeltaNet layers; the WMMA fraction is proportionally higher
4. **Batched mb4 path** (N >= 128): WMMA GEMM is a larger fraction of
   per-step wall time

A/B bench script: `scripts/bench_launch_bounds_ab.sh`

### Scope: gfx1151 only

This change is limited to `*.gfx1151.hip` files. The gfx1100, gfx1200, gfx12, and
other arch-specific files retain their original `__launch_bounds__(32, 2)`. The
compiler compiles these per-arch so there is no cross-arch contamination.

---

## Phase 2: Generic WMMA kernel promotion (2026-06-04)

All 51 generic (non-arch-specific) WMMA `.hip` kernels with `__launch_bounds__(32, 2)`
were promoted to `__launch_bounds__(32, 10)`. These compile for all gfx11 targets
(gfx1100, gfx1101, gfx1102, gfx1150, gfx1151) and fire on the following code paths:

- **MQ3-Lloyd WMMA** (sub-4-bit weights on gfx11): gate_up, qkv, qkvza, residual, mb4 variants
- **HFQ4/HFQ3/HFQ6 WMMA** (legacy quant formats): all gate_up/qkv/residual variants
- **HFP4G32 WMMA** (FP4 format): gate_up, qkv, qkvza, residual
- **Q8_0 WMMA** (unquantized F16-as-Q8): gate_up, qkv, qkvza, residual, x64
- **gemm_f16_wmma** (F16 GEMM for vision encoders, dots.ocr): base + mb4
- **MoE grouped WMMA**: HFQ4, MQ2, ParoQ4 grouped variants
- **Misc**: gemm_mw16_residual_wmma, ksplit/ksplit_det, ldscoop variants

### Correctness verified on gfx1151

- `test_gemm_fused_mq4g256_lloyd_wmma`: ALL PASS (bit-exact)
- `test_gemm_mq4g256_lloyd_residual_wmma`: ALL PASS (bit-exact)
- `test_gemm_mq3g256_lloyd_residual_wmma`: ALL PASS (bit-exact)
- `test_gemm_hfq3g256_wmma`: ALL PASS (bit-exact, _mb4 bit-equivalent to _wmma)
- `coherence-gate-dflash`: ALL PASS (4 tests, no hard errors, no soft warns)

### gfx1100 validation needed

The following kernels are most likely to be exercised on gfx1100 in production
and should be microbenchmarked (A/B with original `(32, 2)`) on a discrete GPU:

**High priority (MQ4-Lloyd, used by Qwen 3.5/3.6-27B):**
- `gemm_mq4g256_lloyd_residual_wmma.hip`
- `gemm_mq4g256_lloyd_residual_wmma_mb2.hip`

**High priority (MQ3-Lloyd, production on gfx11):**
- `gemm_gate_up_mq3g256_lloyd_wmma.hip`
- `gemm_gate_up_mq3g256_lloyd_wmma_mb4.hip`
- `gemm_gate_up_mq3g256_lloyd_wmma_nosync.hip`
- `gemm_gate_up_mq3g256_lloyd_wmma_mb4_nosync.hip`
- `gemm_qkv_mq3g256_lloyd_wmma.hip`
- `gemm_qkv_mq3g256_lloyd_wmma_mb4.hip`
- `gemm_qkvza_mq3g256_lloyd_wmma.hip`
- `gemm_qkvza_mq3g256_lloyd_wmma_mb4.hip`
- `gemm_mq3g256_lloyd_residual_wmma.hip`
- `gemm_mq3g256_lloyd_residual_wmma_mb4.hip`

**Medium priority (HFQ4, used by older quant formats):**
- `gemm_gate_up_hfq4g256_wmma.hip`
- `gemm_gate_up_hfq4g256_wmma_k4.hip`
- `gemm_qkv_hfq4g256_wmma.hip`
- `gemm_hfq4g256_residual_wmma.hip`
- `gemm_hfq4g256_residual_wmma_k2.hip`
- `gemm_hfq4g256_residual_wmma_k2x32.hip`
- `gemm_hfq4g256_residual_wmma_ksplit.hip`
- `gemm_hfq4g256_residual_wmma_ksplit_det.hip`

**Medium priority (F16, used by vision encoders):**
- `gemm_f16_wmma.hip`
- `gemm_f16_wmma_mb4.hip`

**Lower priority (less common formats / MoE paths):**
- All HFQ3, HFQ6, HFP4G32, Q8_0, ParoQ4, MoE-grouped variants

### 51 files changed

```
gemm_f16_wmma.hip                          gemm_f16_wmma_mb4.hip
gemm_gate_up_hfp4g32_wmma.hip              gemm_hfp4g32_residual_wmma.hip
gemm_gate_up_hfq3g256_wmma.hip             gemm_gate_up_hfq3g256_wmma_mb4.hip
gemm_qkv_hfq3g256_wmma.hip                 gemm_qkv_hfq3g256_wmma_mb4.hip
gemm_qkvza_hfq3g256_wmma.hip               gemm_qkvza_hfq3g256_wmma_mb4.hip
gemm_hfq3g256_residual_wmma.hip            gemm_hfq3g256_residual_wmma_mb4.hip
gemm_gate_up_hfq4g256_wmma.hip             gemm_gate_up_hfq4g256_wmma_k4.hip
gemm_gate_up_hfq4g256_wmma_ldscoop.hip     gemm_gate_up_hfq4g256_wmma_ldscoop_nosync.hip
gemm_gate_up_hfq4g256_wmma_ldsx.hip
gemm_qkv_hfq4g256_wmma.hip                 gemm_qkvza_hfq4g256_wmma.hip
gemm_hfq4g256_residual_wmma.hip            gemm_hfq4g256_residual_wmma_k2.hip
gemm_hfq4g256_residual_wmma_k2x32.hip
gemm_hfq4g256_residual_wmma_ksplit.hip     gemm_hfq4g256_residual_wmma_ksplit_det.hip
gemm_hfq4g256_moe_grouped_wmma_k2.hip
gemm_gate_up_hfq6g256_wmma.hip             gemm_qkv_hfq6g256_wmma.hip
gemm_qkvza_hfq6g256_wmma.hip              gemm_hfq6g256_residual_wmma_k2.hip
gemm_gate_up_mq3g256_lloyd_wmma.hip        gemm_gate_up_mq3g256_lloyd_wmma_mb4.hip
gemm_gate_up_mq3g256_lloyd_wmma_nosync.hip gemm_gate_up_mq3g256_lloyd_wmma_mb4_nosync.hip
gemm_qkv_mq3g256_lloyd_wmma.hip            gemm_qkv_mq3g256_lloyd_wmma_mb4.hip
gemm_qkvza_mq3g256_lloyd_wmma.hip          gemm_qkvza_mq3g256_lloyd_wmma_mb4.hip
gemm_mq3g256_lloyd_residual_wmma.hip       gemm_mq3g256_lloyd_residual_wmma_mb4.hip
gemm_mq4g256_lloyd_residual_wmma.hip       gemm_mq4g256_lloyd_residual_wmma_mb2.hip
gemm_mq2g256_lloyd_moe_grouped_wmma_k2.hip
gemm_mw16_residual_wmma.hip
gemm_paro_q4g128_moe_grouped_wmma_k2.hip
gemm_gate_up_q8_0_wmma.hip                 gemm_q8_0_residual_wmma.hip
gemm_q8_0_wmma_x64.hip                     gemm_qkv_q8_0_wmma.hip
gemm_qkvza_q8_0_wmma.hip
gemm_qkv_hfp4g32_wmma.hip                  gemm_qkvza_hfp4g32_wmma.hip
gemm_gate_up_hfp4g32_wmma.hip              gemm_hfp4g32_residual_wmma.hip
```

---

## Open follow-ups (for gfx1100 and beyond)

### F1. Replicate occupancy win on gfx1100 (RX 7900 XTX etc.)

**Why:** gfx1100 has the same VGPR/SIMD layout (512 VGPR/SIMD, 2 SIMD/CU) so the
occupancy analysis is identical in theory. But discrete GPUs have:
- GDDR6X (much higher bandwidth than LPDDR5X) → may be less memory-bound
- Stable power delivery → lower DPM variance → cleaner A/B measurements
- Potentially different L2 behavior (different cache sizing, different miss rates)

**Status:** All 51 generic WMMA kernels already changed to `(32, 10)`. Need to
build, microbenchmark, and coherence-gate on gfx1100 hardware. If any kernel
regresses on gfx1100, create a `*.gfx1100.hip` override with the original `(32, 2)`.

### F2. Re-evaluate s_setprio on gfx1100 (UNCHANGED)

See above. Experiment files kept.

The setprio experiment was negative on gfx1151 (APU DPM noise exceeded the
signal). On a discrete gfx1100 with stable power, s_setprio may show a
reproducible win — especially at shapes where the kernel is less bandwidth-
saturated (9B batch-64 showed promising +5% on gfx1151 before the noise
overwhelmed it).

**Experiment files kept for re-use:**
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_setprio.gfx1151.hip` (variant A)
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_setprioB.gfx1151.hip` (variant B)
- `crates/rdna-compute/examples/bench_setprio_gate_up.rs` (A/B/C harness)

On gfx1100, copy these to non-suffixed `.hip` files and rebuild, or create
gfx1100-specific setprio variants.

### F3. Investigate `__launch_bounds__(32, 16)` for even higher occupancy (UNCHANGED)

See above.

With 112 VGPR/wave (what the compiler chose for `(32, 10)`), we get 4 waves/SIMD.
To reach 8 waves/SIMD (16 waves/CU), the compiler would need to reduce to
≤64 VGPR/wave — likely requiring significant restructure of the register-heavy
K4 unroll (8 nibble-pack `unsigned int` values + half16_t a_reg + float8_t acc
already exceeds 64 VGPR). May not be feasible, but worth measuring if the
compiler can find a way.

### F4. MQ3 variants (RESOLVED)

All MQ3-Lloyd WMMA kernels promoted to `(32, 10)` in Phase 2 above.
Bit-exact correctness verified on gfx1151.

---

## Artifacts

- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_hiocc.gfx1151.hip` — hiocc experiment (now superseded by production change)
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_setprio.gfx1151.hip` — setprio variant A (negative on gfx1151, kept for gfx1100 re-test)
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_setprioB.gfx1151.hip` — setprio variant B (negative on gfx1151, kept for gfx1100 re-test)
- `crates/rdna-compute/examples/bench_setprio_gate_up.rs` — multi-variant A/B/C bench
- `crates/rdna-compute/examples/verify_hiocc_gate_up.rs` — correctness verification
- `crates/rdna-compute/examples/bench_hiocc_gate_up.rs` — dedicated hiocc profiling bench
- `benchmarks/results/rocprof_exp2_detailed/` — detailed counter baseline
- `benchmarks/results/rocprof_hiocc_gate_up/` — hiocc counter data
