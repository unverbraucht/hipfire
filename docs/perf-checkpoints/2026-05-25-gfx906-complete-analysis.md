# gfx906 Decode/Prefill Optimization — Complete Analysis

**Date:** 2026-05-25
**Hardware:** AMD Instinct MI50 (gfx906, Vega 20, 60 CUs, 1024 GB/s HBM2)
**Baseline:** Qwen 3.5 9B MQ4 (HFQ4-G256)

This document maps the complete optimization landscape for gfx906, covering the 15-phase
investigation that already shipped, the remaining open levers, the dispatch routing,
and prioritized recommendations for the next session.

---

## 1. Current State — What Already Shipped

### 1.1 Prefill MMQ Redesign (5× speedup)

| Metric | Pre-redesign | Post-redesign | Speedup |
|---|---|---|---|
| pp32  | 136 tok/s | 313 tok/s | 2.30× |
| pp512 | 142 tok/s | 714 tok/s | 5.02× |
| vs llama.cpp | 3.29× slower | 95% parity | — |

**Structural changes (all permanent):**
- **nwarps=2 → nwarps=4** (256 threads/WG). Doubles the in-flight dp4a issue
  from the wave scheduler; the 4-wave topology lets the compiler extract
  cross-iter ILP that was serial at nwarps=2.
- **Runtime mmq_x dispatch.** 8 kernel variants (`_x{8,16,24,32,40,48,56,64}`)
  selected greedily from `batch_size`. Each variant has a matched `_full_*_x{N}`
  for the aligned case (M%128==0 && batch_size%mmq_x==0).
- **Option C+pad Window Streaming.** 4 syncs/group (load → compute →
  load → compute) instead of Option B's 16 syncs/group. 8 sub-blocks
  computed back-to-back within each window with no internal syncs.
- **Per-mmq_x X_STRIDE.** `x_stride = mmq_x >= 32 ? 40 : 33` — stride-40
  enables `ds_read_b128` (16-B alignment every row) at mmq_x≥32; stride-33
  avoids bank conflicts at mmq_x<32 where b128 issue rate overwhelms
  the LDS pipeline. Both cliff values validated by PMC bank-conflict counters.

### 1.2 Decode: ILP-Prefetch on `gemv_residual` (+7.3%)

Kernel: `gemv_hfq4g256_residual_wave64_prefetch`
- **Software-pipelined** prologue/steady/epilogue: quad q+1's 12 dwords
  (sc, zp, pk×4) issued *before* quad q's FMA chain runs.
- **Result:** +4.8% on the kernel (51.9 → 54.4 tok/s), +7.3% end-to-end.
- **Root cause** (Phase 6 PMC reframe): not L2-cache warming — instruction-
  issue serialization. VALUBusy jumped 25.6% → 33.3% as the scheduler
  got independent loads to overlap with FMA chains.

The other three fused kernels (`fused_gate_up`, `fused_qkv`, `fused_qkvza`)
**did not benefit** from the same prefetch pattern:
- `fused_qkv`/`fused_qkvza`: +7 VGPR moves them across a wave boundary
  (46 → 53 VGPR, 5 → 4 waves/SIMD). The 25% occupancy drop cancels the
  ILP gain.
- `fused_gate_up`: stays at 5 waves/SIMD but the warp-id row routing
  (warp 0 = gate row, warp 1 = up row) gives implicit MLP the prefetch
  can't improve further.

### 1.3 Decode: dp4a Port (+15.4% cumulative, +24.3% total)

Four fused GEMVs ported to `v_dot4_i32_i8` (dp4a), all using Q8_1
pre-quantized activations from `ensure_q8_1_mmq_x`:

| Kernel | Decode Share | Δ per-call | Δ end-to-end |
|---|---:|---:|---:|
| `fused_gate_up_dp4a` | 25.5% | ~25-30% | +7.1% |
| `fused_qkv_dp4a` | 2.7% | similar | +0.2% |
| `fused_qkvza_dp4a` | 9.8% | similar | +0.5% |
| `gemm_hfq4g256_wave64_dp4a` (LM-head) | 17.1% | ~70% | +7.0% |

**Cumulative at HEAD:** 50.7 → 63.0 tok/s (+24.3% total, +7.0% from LM-head alone).
**Stock llama.cpp Q4_K_M = 61.55 tok/s — hipfire now at parity or slightly above.**

### 1.4 DFlash: MMQ Cutover 16→8

Changed `should_use_mmq` min_batch from 16 to 8 on gfx906:

| Workload | min_batch=16 | min_batch=8 | Δ |
|---|---:|---:|---:|
| DFlash humaneval-0 (27B) | 12.28 | 20.24 | +64.8% |
| DFlash lru_cache (27B) | 10.93 | 15.19 | +39.0% |
| DFlash coherence prose | 10.52 | 12.67 | +20.4% |
| DFlash ddtree-prose | 5.40 | 19.21 | +256% |
| AR decode 9B (B=1) | 59.4 | 59.2 | flat |

Root cause: adaptive-b at B=12/14 (67% of calls) fell through to FP16
wave64 at min_batch=16. The residual batched GEMM is structurally cheaper
than the non-residual prefill sweep, so the cutover crosses below.

### 1.5 DFlash: LM-head dp4a Port (+11.8-12.5% on DFlash 27B)

The LM-head GEMM (M=vocab=152k, K=hidden=5120) is the single largest
kernel. dp4a's 75% x-traffic reduction (18 KB f32 → 4.6 KB Q8_1) has
its biggest per-call effect on the largest-K kernel.

| Test | before | after | Δ |
|---|---:|---:|---:|
| 27B-3.5 / lru_cache | 35.65 | 39.85 | +11.8% |
| 27B-3.5 / humaneval_0 | 41.97 | 47.21 | +12.5% |
| 27B-3.6 / humaneval_0 | 22.25 | 24.83 | +11.6% |

---

## 2. Dispatch Routing Map

### 2.1 Prefill (batched GEMM, B ≥ 2)

```
gemm_hfq4g256_residual(a, x, y, M, K, N)
  ├── N >= 8 (gfx906 min_batch) → mmq_screen_weight
  │     ├── safe → gemm_hfq4g256_residual_mmq_gfx906(a, x, y, M, K, N)
  │     │          ├─ ensure_q8_1_mmq_x(x, N, K) → x_q8_ptr
  │     │          ├─ mmq_x = greedy(N) ∈ {8,16,24,32,40,48,56,64}
  │     │          ├─ is_full = M%128==0 && N%mmq_x==0
  │     │          └─ kernel: {full_add_x{mmq_x} | _x{mmq_x}}
  │     │             (body.cuh: nwarps=4, dp4a, Option C+pad, per-mm q_x stride)
  │     └── unsafe → gemm_hfq4g256_residual_fp16_wave64(a, x, y, M, K, N)
  │                    (2 rows/WG, 4-quad interleave, no dp4a)
  └── N < 8 → fp16 wave64 (B=1→7 below MMQ cutover)
```

### 2.2 Decode (GEMV, B = 1)

```
gemv_hfq4g256(a, x, y, M, K)
  ├── gemv_dp4a_enabled("gfx906") → true (default)
  │     └── NOT taken for the *residual* gemv (dispatch doesn't route
  │          gemv_hfq4g256_residual through gemv_dp4a_enabled)
  └── gemv_prefetch_enabled("gfx906") → true (default)
        └── gemv_hfq4g256_residual_wave64_prefetch
             (ILP-prefetch variant, 2 rows/WG, 4-quad interleave)

fused_gate_up_hfq4g256(a_gate, a_up, x, y_gate, y_up, ...)
  └── gemv_dp4a_enabled("gfx906") → true
        └── fused_gate_up_hfq4g256_wave64_dp4a
             (Q8_1 pre-quantize + dp4a inner loop, 2 rows/WG)

fused_qkv_hfq4g256(...) / fused_qkvza_hfq4g256(...)
  └── gemv_dp4a_enabled("gfx906") → true
        └── fused_qkv*_hfq4g256_wave64_dp4a
             (Q8_1 pre-quantize + dp4a, same topology)

gemm_hfq4g256_wave64 (LM-head, batched)
  └── gemv_dp4a_enabled("gfx906") → true
        └── gemm_hfq4g256_wave64_dp4a
             (BATCH_TILE=16, Q8_1, dp4a, residual write-back)
```

### 2.3 Key Env Var Overrides

| Env Var | Default | Effect |
|---|---|---|
| `HIPFIRE_GEMV_DP4A` | ON (gfx906) | Toggle all fused dp4a GEMVs on/off |
| `HIPFIRE_GEMV_PREFETCH` | ON (gfx906) | Toggle ILP-prefetch on residual gemv |
| `HIPFIRE_MMQ` | auto | Force MMQ on/off globally |
| `HIPFIRE_MMQ_MIN_BATCH` | 8 (gfx906) / 256 (others) | Override MMQ cutover |
| `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` | OFF | Isolate Q8_1 quantize cost |
| `HIPFIRE_FP16` | ON (default) | Global FP16 fast-path enable |
| `HIPFIRE_FP16_LAYER_MIN/MAX` | — | Per-layer FP16 sweep (issue #302) |
| `HIPFIRE_HFQ3_DP4A` | OFF | Experimental HFQ3 wave32 dp4a on gfx10 |
| `HIPFIRE_HFQ3_MMQ` | OFF | Experimental HFQ3 wave32 MMQ on gfx10 |

---

## 3. Remaining Levers — Ranked by Impact

### 3.1 `v_dot8_i32_i4` (int4 dot product) — Highest Remaining Lever

**Estimate:** +12-18% end-to-end on decode; +20-30% per dp4a kernel.

**Mechanism:** gfx906 has `v_dot8_i32_i4` — packs 8× int4 × int8 in one
instruction vs dp4a's 4× int8 × int8. Halves the dp4a count per output.

**Current bottleneck:** Phase 8 showed the dp4a *issue rate* is the hardware
ceiling (48.5% VALUBusy at mmq_x=64, compiler extracting all available ILP).
The dp4a port landed at 85% of decode on optimized paths; `v_dot8_i32_i4`
would directly halve the instruction count feeding that ceiling.

**Implementation cost:**
1. Weights stay as 4-bit nibbles (already).
2. Q8_1 activations need to become Q4_1 (int4 x_int) — format change.
   The `(n-8)` shift correction becomes simpler (int4 is already in the
   right signed range).
3. New `block_q4_1_mmq` struct: 4-bit activations + half2 ds. Smaller
   per-block (72 B vs 144 B Q8_1) — doubles weight reuse per L2 cache line.
4. All dp4a kernels need inner-loop rewrite: `__builtin_amdgcn_sdot8`
   replaces 2× `__builtin_amdgcn_sdot4` per sub-block.
5. Correctness: CPU reference at 4 shapes + coherence gate.

**Risk:** High. Format change has blast radius across all fused dp4a
kernels (6 kernels × 2 quant variants = 12 kernel source files).
But the math is simpler: no `(n-8)` shift, no zp_eff folding.

**Priority:** P1 if there's a dedicated perf session; ~2 work sessions
for full port + correctness.

### 3.2 Issue #172 — DFlash Dispatch Overhead

**Estimate:** +5-10% end-to-end on DFlash (three small levers combined).

Current state: ~17% of steady-state decode is inter-kernel gap (250 ms /
1500 ms for 7768 kernels in 1.5s window). Median gap 10.2 μs, p99 1450 μs.

Three levers:
1. **Hoist `fillBufferAligned` zero-memsets** out of the cycle loop.
   Currently per-cycle on output buffers. Move to buffer allocation time.
   Est: +1-3% (reduces kernel count by ~2 per cycle).
2. **Async D2H of `argmax_buf`.** The DFlash argmax D2H is currently
   synchronous per-cycle. HIP graph or `hipMemcpyAsync` with deferred
   sync could overlap it with the next cycle's compute.
   Est: +1-2% (argmax is ~128 bytes/token, small but per-cycle).
3. **GPU-side accept/reject.** Currently a small CPU-side comparison
   after D2H. Moving to GPU (kernel or graph) removes one D2H round-trip.
   Est: +1% (small, mechanical).

**Risk:** Low. All mechanical, no kernel changes.

**Priority:** P1 for a DFlash-focused session; ~1 work session.

### 3.3 Issue #173 — Bench Harness Daemon Reuse

**Estimate:** ~8× battery wallclock reduction (no production impact).

Cold-load of 17 GB target+drafter = ~56s per invocation, 0.7s actual decode.
Wrap `hipfire serve` with a warm daemon instead of per-invocation cold load.

**Risk:** Zero (development tooling only).

**Priority:** P2 — high ROI for iteration speed, no production impact.

### 3.4 Cross-Arch Validation (gfx908 / MI300x)

**Estimate:** Unknown (0-25% depending on each arch's bottleneck mix).

All dp4a + prefetch kernels are gated `arch == "gfx906"`. gfx908 (MI100)
has the same wave64 + dp4a builtin. MI300x (gfx94x) has MFMA but the
small-batch dp4a path could still help below the WMMA cutover.

**Action:** Flip the gate to include the target arch + bench.
Zero new code, needs hardware access.

**Risk:** Low. Gated by `gemv_dp4a_enabled()` — can be tested behind
the env var toggle.

**Priority:** P2 if hardware available.

### 3.5 HFQ3/HFQ6 Prefetch + dp4a Port

**Estimate:** +5-15% per kernel (workload-dependent decode share).

**HFQ3 (3-bit, 104 B/group):** Same prefetch lever is mechanical (identical
software-pipeline shape). dp4a port needs an HFQ3-aware nibble decoder
(3-bit values unpack to int8, then dp4a — no native int3 dot).
Weight unpack is slightly different: 8 K-elements per lane = 3 bytes
packed across 24 bits.

**HFQ6 (6-bit, 200 B/group):** Prefetch is mechanical. dp4a port is
awkward: 6-bit values don't pack neatly into int8 for dp4a (16/3 per
int). The `(n-0)` shift (no correction needed for unsigned q∈[0,63])
makes the math simpler than HFQ4, but the packing is unusual.

**Decision criteria:** Port only if (a) the quant is used in a production
path at measurable decode share, AND (b) PMC shows the same ILP/memory-
bound regime that gave wins on MQ4.

**Priority:** P3 — only if workload demands it.

---

## 4. Ruled-Out Levers (With Diagnostic Evidence)

### 4.1 `__launch_bounds__` Tuning (P2')

**Ruled out by occupancy audit.** All four decode kernels are well under
the VGPR ceiling (29-46 VGPR, 0 spills). Theoretical waves/SIMD is 5-8
with cap 32/51. No occupancy ceiling to tune.

### 4.2 Y-tile Prefetch on Prefill MMQ (Phase 8a)

**Result: -0.4% regression.** L2 hit on `full_*_x64` is already 69-81%,
so the "warm L2 for next iter" lever has no slack. iacopPBK's lever
may have applied to an earlier MMQ revision with less aggressive Y reuse.

### 4.3 2-Accumulator Split in `vec_dot_dp4a_streaming` (Phase 8b)

**Result: -2.1% regression.** VALUBusy moved right (+1.6pp) but throughput
went down. The compiler was already extracting cross-iter ILP from the
outer (i,j) loops. At mmq_x=64, the inner kernel runs 16 independent
dp4a chains per (i,j); adding 2 more chains per (i,j) didn't unlock
more issue rate (gfx906 has 1 dp4a issue port/cycle/warp) and the final
integer-add merge added latency.

### 4.4 SGPR Hoisting via `readfirstlane` (P4')

**Result: 0% net, worse disassembly.** `__shfl` lowered to `ds_bpermute`
(LDS round-trip, 27× calls), and `readlane` increased SGPR count (14→31)
but global loads went up by 4. Structural blocker: wave-uniformity
assumed wave = warp, but 2-rows-per-WG topology means each warp sees
a *different* sc/zp address. The compiler's vector-to-scalar pass only
fires when the *whole wave* is uniform.

**Stays valid for:** 1-row-per-WG kernels (e.g., `gemv_hfq4g256_wide`)
where the whole wave truly sees the same address.

### 4.5 dp4a on `gemv_residual` (GEMV, B=1)

**Ruled out by cost/benefit math.** The prefetch variant already won
(+4.8%). dp4a on the single-token GEMV adds ~1 work session for an
estimated 0-5% lift (weights are ~90% of fetched bytes, so x-traffic
reduction is only ~10% of total HBM traffic on this kernel).

**Reconsider if:** VALUBusy pushes above ~70% (another round of prefetch
tuning shifts the bottleneck to ALU) or DFlash mid-batch (B=2-15)
becomes a significant decode share.

---

## 5. Kernel Source Inventory

### 5.1 Prefill MMQ (gfx906 only)

| File | Role |
|---|---|
| `gemm_hfq4g256_residual_mmq_gfx906_body.cuh` | Shared body: load + compute + writeback templates |
| `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}.hip` | 8 wrappers, each instantiates `mmq_body_templated<MMQ_X, true, -1>` |

### 5.2 Decode GEMV (single-token, B=1)

| File | Path | Status |
|---|---|---|
| `gemv_hfq4g256_residual_wave64_prefetch.hip` | `gemv_*_residual` (prefetch variant) | Shipped, default |
| `gemv_hfq4g256_residual_wave64.hip` | Fallback (non-prefetch) | Available, not used by default |

### 5.3 Fused GEMV (decode + small-batch prefill)

| Kernel | FP Path | dp4a Path | Dispatch Gate |
|---|---|---|---|
| `fused_gate_up_hfq4g256` | `*_wave64.hip` | `*_wave64_dp4a.hip` | `gemv_dp4a_enabled` |
| `fused_qkv_hfq4g256` | `*_wave64.hip` | `*_wave64_dp4a.hip` | `gemv_dp4a_enabled` |
| `fused_qkvza_hfq4g256` | `*_wave64.hip` | `*_wave64_dp4a.hip` | `gemv_dp4a_enabled` |
| `gemm_hfq4g256` (LM-head) | `*_wave64.hip` | `*_wave64_dp4a.hip` | `gemv_dp4a_enabled` |
| `gemm_hfq4g256_residual` | `*_wave64.hip` | `*_residual_wave64_dp4a.hip` | `gemv_dp4a_enabled` |
| `fused_gate_up_hfq6g256` | `*_wave64.hip` | `*_wave64_dp4a.hip` | `gemv_dp4a_enabled` |
| `fused_qkv_hfq6g256` | `*_wave64.hip` | `*_wave64_dp4a.hip` | `gemv_dp4a_enabled` |
| `fused_qkvza_hfq6g256` | `*_wave64.hip` | `*_wave64_dp4a.hip` | `gemv_dp4a_enabled` |
| `gemm_hfq6g256_residual` | `*_wave64.hip` | `*_residual_wave64_dp4a.hip` | `gemv_dp4a_enabled` |

### 5.4 HFQ3 dp4a (experimental, gfx10 only)

| File | Gate |
|---|---|
| `gemm_gate_up_hfq3g256_dp4a.gfx1030.hip` | `hfq3_dp4a_enabled` |
| `gemm_qkv_hfq3g256_dp4a.gfx1030.hip` | `hfq3_dp4a_enabled` |

---

## 6. Structural Observations

### 6.1 Q8_1 Staging Bottleneck

`ensure_q8_1_mmq_x` runs `quantize_q8_1_mmq_ds4` on every dp4a call.
This is a ~128-thread/WG kernel at `((K+1023)/1024, batch_size)` grid.
For K=5120, B=1: grid is 5×1 = 5 WGs. For K=5120, B=64: grid is 5×64
= 320 WGs.

The quantize cost is paid once and the Q8_1 buffer is reused across
sibling projections (gate + up share the same xq). This amortization
means the quantize cost is ~1x per layer rather than ~4x (one per
fused output). **But** for DFlash with adaptive-b < 8 (the B=12/14
sweet spot), the MMQ path is taken and the quantize is paid per-verify
cycle. At 7768 kernels in a 1.5s window, the quantize kernel is a
non-trivial fraction.

**Opportunity:** If Q4_1 replaces Q8_1 (see §3.1), the quantize output
is half the size (72 B/block vs 144 B/block) — the quantize kernel is
faster and the staging buffer is smaller.

### 6.2 dp4a Math Correctness

The dp4a path uses a math identity that folds the HFQ4 zero-point
correction into a single per-block term:

```
sum_k (sc * n_k + zp) * x_k
= sc * sum_k((n_k - 8) * x_k) + (zp + 8*sc) * sum_k(x_k)
```

The `(n-8)` shift moves unsigned nibbles [0,15] to signed [-8,7]
matching the int8 lane convention of Q8_1 activations. This is
**correct but fragile**: if any future quant format changes the nibble
range (e.g., unsigned-only weights), the shift direction changes.

The correctness validation (existing) is:
- `test_gemm_hfq4_dp4a` CPU reference: max abs error <1e-2, mean rel <0.05%
- Coherence gate: 7/7 clean (4 DFlash + mq3 + mq6 + baseline)

### 6.3 The MMQ Screening Threshold

`mmq_screen_threshold` is 0.50 on gfx906 (vs 0.10 default on other archs).
This was validated as safe: the dp4a+Q8_1 path has different error
characteristics than WMMA, and the 0.50 threshold was calibrated against
coherence gate results. The arch-conditional override is at `Gpu::init`.

---

## 7. Recommended Next Actions

### 7.1 Immediate (If Perf Session Available)

1. **`v_dot8_i32_i4` probe (Phase A).** 1-day spike:
   - Check `__builtin_amdgcn_sdot8` availability on current ROCm version.
   - Sketch the Q4_1 struct layout and inner-loop rewrite for one kernel
     (e.g., `fused_gate_up_dp4a`).
   - If ROCm supports it, the full port is the single highest-leverage
     remaining change.

2. **Issue #172 (DFlash dispatch overhead).** Low-risk, high-ROI:
   - Audit the per-cycle `fillBufferAligned` calls in `speculative.rs`.
   - Profile the argmax D2H path (rocprof kernel-trace, filtered to
     memcpy/memset).
   - Estimate per-lever impact before implementation.

3. **Issue #173 (bench daemon reuse).** ~2 hours, 8× iteration speed.

### 7.2 If `v_dot8_i32_i4` is Not Available

1. **Revisit dp4a on `gemv_residual`.** The prefetch variant won at B=1
   but dp4a could still help at mid-batch (B=2-15) where the ILP-prefetch
   variant wasn't ported. Estimate: +3-5% on mid-batch DFlash verify.

2. **Wider `n_tokens` batched dp4a for the fused kernels.**
   The current dp4a fused kernels are B=1 GEMV. Mid-batch range
   (B=2-15) has no dp4a-batched-GEMV variant. Porting `fused_gate_up_dp4a`
   to a batched variant (mirroring `gemm_hfq4g256_residual_wave64_dp4a`
   which already handles B>1) would capture the DFlash B=12/14 path.
   Estimate: +2-3% end-to-end.

3. **`gated_delta_net` tuning.** 4.6% decode share, wave64-native.
   A focused PMC pass could find 10-30% per-call lift (~0.5-1.4%
   end-to-end). Bottom of the list but mechanical.

### 7.3 Cross-Arch (If Hardware Available)

1. **gfx908 (MI100).** Same wave64 + dp4a. Flip `gemv_dp4a_enabled`
   to include gfx908 + bench. Zero new code.

2. **MI300x (gfx94x).** The dp4a path could help at small batches
   below the WMMA cutover (B=8-255). Test behind the env var toggle.

---

## 8. Appendices

### A. Final Scoreboard (Post-Phase-15)

| Workload | tok/s | BW (GiB/s) | Δ vs start |
|---|---:|---:|---:|
| AR decode 9B | **63.0** | ~292 | +24.3% |
| DFlash 27B humaneval | **24.83** | — | +90% |
| Prefill pp512 9B | ~714 | — | 5.02× |
| vs llama.cpp 9B | 63.0 vs 61.55 | — | parity |

### B. PMC Data Summary (Post-Optimization)

| Kernel | VALU% | MemStall% | L2 Hit |
|---|---:|---:|---:|
| `gemv_residual_prefetch` | 33.3 | 1.9 | ~45% (est.) |
| `fused_gate_up_dp4a` | ~35 | ~2 | ~45% (est.) |
| `gemm_*_mmq_x64` | 48.5 | 0.3 | 69% |
| `gemm_*_mmq_full_add_x64` | 39.6 | 0.1 | 81% |
| `gemm_hfq4g256_dp4a` (LM-head) | — | — | — |

(The LM-head dp4a PMC was not captured post-Phase-14; the 70% per-call
lift was measured by tok/s delta, not per-kernel counters.)

### C. Kernel Metadata (Occupancy Audit)

| Kernel | VGPR | SGPR | LDS | Spills | Waves/SIMD |
|---|---:|---:|---:|---:|---:|
| `gemv_hfq4g256_residual_wave64` | 29 | 14 | 0 | 0 | 8 |
| `gemv_hfq4g256_residual_wave64_prefetch` | 46 | 14 | 0 | 0 | 5 |
| `fused_gate_up_hfq4g256_wave64` | 29 | 20 | 0 | 0 | 8 |
| `fused_qkv_hfq4g256_wave64` | 46 | 20 | 0 | 0 | 5 |
| `fused_qkvza_hfq4g256_wave64` | 46 | 22 | 0 | 0 | 5 |
| `mmq_body<64>` (nwarps=4) | 94 | — | 30.7 KiB | 0 | 2 |

Prefetch adds +17 VGPR (staged-load buffer for next quad). dp4a kernels
have similar VGPR profiles (the Q8_1 pointers replace float pointers,
no additional VGPR pressure).
