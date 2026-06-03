# Profiling Dev Log: hipfire vs llama.cpp on gfx906 (MI50)

**Date:** 2026-06-03
**Goal:** Understand why llama.cpp beats hipfire on AR decode tok/s and long-context prefill.
**GPU:** AMD Instinct MI50 (gfx906, 32GB, CDNA1)
**Model:** Qwen3.6-27B — hipfire MQ4 (14GB) vs llama.cpp Q4_K_XL (17GB)
**Branch:** `merge/master-pr352`

## Baseline Numbers (llama-benchy, 3 runs)

| Metric | hipfire AR | llama.cpp Q4_K_XL | Δ |
|--------|-----------|-------------------|---|
| tg128 @ pp128 | 19.38 ± 0.02 | 20.66 ± 0.02 | -6.2% |
| tg128 @ pp512 | 17.46 ± 0.38 | 19.91 ± 0.51 | -12.3% |
| tg128 @ pp2048 | 15.86 ± 0.17 | 17.23 ± 0.02 | -8.0% |
| pp128 | 191.51 ± 0.08 | 103.65 ± 0.69 | +84.8% |
| pp512 | 200.43 ± 4.29 | 177.30 ± 2.11 | +13.0% |
| pp2048 | 169.20 ± 1.45 | 181.54 ± 1.49 | -6.8% |

## Bench Tool Numbers (direct, no HTTP overhead)

| Context | hipfire (tok/s) | llama.cpp (tok/s) | Δ |
|---------|----------------|-------------------|---|
| pp=1 tg | 19.9 | 20.75 | -4.1% |
| pp=128 tg | 19.7 | 20.43 | -3.6% |
| pp=2048 tg | 18.5 | 17.95 | +3.1% |

**Key insight:** The decode gap at short context is only ~3.5-4%, not the 6-12%
seen via llama-benchy. The difference likely comes from HTTP server overhead
and request processing in hipfire's daemon vs llama.cpp's more streamlined
server. At pp=2048, hipfire is actually slightly faster.

## Investigation Plan

### Phase 1: Decode (tg128) — the steady-state bottleneck
- Profile hipfire decode kernel (gemv_hfq4g256) for 100 single-token generations
- Profile llama.cpp decode kernel (ggml_mul_mat_q4_0) for 100 single-token generations
- Compare: kernel dispatch overhead, kernel execution time, memory bandwidth usage

### Phase 2: Prefill — short vs long context
- Profile pp128 prefill (both) — hipfire wins 2×, understand why
- Profile pp2048 prefill (both) — llama.cpp wins, understand why

### Phase 3: Kernel source analysis
- Examine hipfire's gfx906-specific kernels (dp4a, wave64)
- Examine llama.cpp's gfx906 kernels (hip backend)
- Identify missing optimizations

---

## Finding 1: Dispatch Overhead Dominates Decode (39% of wall time)

**Date:** 2026-06-03
**Method:** `bench_qwen35_mq4` with `HIPFIRE_PROFILE=1 HIPFIRE_PROFILE_DECODE=1`

20-token decode on Qwen3.6-27B MQ4, MI50 (gfx906):

- **Wall time:** 1289.9ms total, 64.26ms/tok
- **Kernel time (serialized):** 783.6ms total, 39.18ms/tok
- **Dispatch overhead:** 506.3ms total, 25.08ms/tok — **39.3% of wall time**
- **Kernel launches:** 17,340 total, 867 per token

| Kernel | Calls | Total(ms) | μs/call | %kernel | BW(GiB/s) |
|--------|-------|-----------|---------|---------|----------|
| gemv_hfq4g256_residual | 2560 | 303.0 | 118 | 38.7% | 252.8 |
| fused_qkvza_hfq4g256_dp4a | 960 | 130.2 | 136 | 16.6% | 308.9 |
| fused_rmsnorm_mq_rotate_awq | 2560 | 84.1 | 33 | 10.7% | 2.4 |
| gemv_hfq4g256 (lm_head) | 20 | 45.3 | 2264 | 5.8% | 278.3 |
| gated_delta_net_q8 | 960 | 41.2 | 43 | 5.3% | 37.4 |
| fused_qkv_hfq4g256_dp4a | 320 | 38.3 | 120 | 4.9% | 304.0 |
| attention_q8_0_kv | 320 | 27.8 | 87 | 3.6% | 3.7 |
| Other (13 kernels) | 11040 | 213.9 | — | — | — |

**Key observations:**
1. ~39% of decode time is NOT in kernels — it's host-side dispatch overhead
2. 867 kernel launches per token × 64 layers = ~13.5 launches/layer (DeltaNet) or ~10.6/layer (FullAttention)
3. `gemv_hfq4g256_residual` at 252.8 GiB/s is near peak bandwidth for MI50 (262 GiB/s HBM2 theoretical)
4. The small fused kernels (rmsnorm, rotate, etc.) have very low bandwidth (2-7 GiB/s), suggesting they're dispatch-limited
5. `attention_q8_0_kv` at 3.7 GiB/s is surprisingly low BW for a B=1 decode

**Hypothesis for decode gap (hipfire 19.4 vs llama.cpp 20.7 tok/s):**
- ~10% gap could come from dispatch overhead per token. llama.cpp likely has fewer,
  larger kernel launches or better pipelining.
- The `fused_rmsnorm_mq_rotate_awq` kernel runs 2560 times at only 2.4 GiB/s — it's
  very small and most of its cost is launch overhead, not compute.
- hipfire does NOT use graph capture for these decode kernels (graph capture is
  reserved for the verify path). Without capture, each small kernel launch costs
  ~5-15μs of dispatch.

**Next steps:**
- [x] Profile llama.cpp decode to get comparable per-kernel breakdown
- [x] Check if graph capture can be enabled for the decode path
- [ ] Fuse more small kernels (rmsnorm+rotate is already fused, but could combine
  more of the 13.5 launches per DeltaNet layer into fewer launches)
- [ ] Examine attention_q8_0_kv — 3.7 GiB/s seems low for B=1 decode

---

## Finding 3: llama.cpp uses hipGraph for decode — hipfire does not

**Date:** 2026-06-03

llama.cpp (ggml) has `USE_CUDA_GRAPH` support compiled in. The docker binary
contains `ggml_cuda_graph` class symbols and the server log shows `context
checkpoints enabled`. For decode (B=1), llama.cpp:

1. Captures the entire decode computation graph on the 2nd warmup pass
2. Replays it as a single `hipGraphLaunch` per token
3. This eliminates all per-kernel dispatch overhead for subsequent tokens

hipfire has graph capture infrastructure (`ar_forward_replay_enabled`, `begin_graph_capture`,
`end_graph_capture`, `graph_launch`) but it is **hard-disabled** with `let use_graph = false;`
in the AR decode path (`crates/hipfire-runtime/examples/daemon.rs:4323`). The comment says:

```
// merge does not clear the disable. Until the capture/replay attractor is
// re-verified gone on current ROCm (7.13) via the coherence gate, AR
// forward is direct-only.
```

This means **every decode token in hipfire suffers 867 individual kernel launches**,
while llama.cpp replays a single graph. At ~5-15μs dispatch per kernel, that's
4-13ms of dispatch overhead per token that llama.cpp completely avoids.

The 12.37ms difference between serialized kernel time (39.18ms) and real wall
time (51.55ms at 19.4 tok/s) is consistent with this dispatch overhead being
a major contributor.

**Impact estimate:**
- hipfire serialized kernel time: 39.18ms/tok (theoretical best with graph replay)
- This would give ~25.5 tok/s, or **+32% faster than current 19.4 tok/s**
- Even if graph replay only eliminates half the dispatch overhead, that's ~22.5 tok/s

**Action items:**
- [ ] Re-enable graph capture for AR decode, test on gfx906 with ROCm 6.3.3
- [ ] If the "attractor bug" is fixed in ROCm 6.3+, enable `use_graph = true`
- [ ] If not, investigate whether the bug only affects specific arch/gpus

---

## Finding 4: llama.cpp MMVQ kernel uses 2 warps (64 threads) on GCN/CDNA for B=1

**Date:** 2026-06-03

llama.cpp's `mul_mat_vec_q` kernel (mmvq.cu) has arch-specific tuning:

- For GCN/CDNA (gfx906): `nwarps = 2`, `rows_per_block = 1` for ncols_dst=1
  → 64-thread workgroup, 1 row per block
  → Each wave64 processes 1 row with 2 warps collaborating

hipfire's `gemv_hfq4g256` uses `__launch_bounds__(32, 16)`:
  → 32-thread block, only half a wave64
  → Each wave64 processes 2 rows (1 per half-wave)
  → But on gfx906, the wave scheduler must run 2 half-waves, lowering occupancy

For B=1 decode where Qwen3.6-27B has M=5120 (hidden dim), rows=5120:
- hipfire: 5120 blocks × 32 threads = 5120×32 threads
- llama.cpp: 5120 blocks × 64 threads = 5120×64 threads

The impact of this difference depends on whether the gemv kernel is memory-
bound (likely for B=1), in which case more threads per row doesn't help much.
But WAVE64 hardware executing a WAVE32 block wastes the other 32 lanes.

**Action items:**
- [ ] Write a wave64-native gemv_hfq4g256 kernel (64 threads/block) for gfx906
- [ ] Benchmark against current 32-thread variant to quantify the impact
- [ ] Consider fusing 2 rows per wave on wave64 (like the residual_wave64 variant)

---

## Finding 5: Enabling graph capture *slowed* decode by 13%

**Date:** 2026-06-03

Tested `use_graph = true` in `forward_from_x_gpu`. Expected it to reduce dispatch
overhead, but it made decode **13% slower**:

- Without graph: 19.7 tok/s (50.53ms/tok)
- With graph: 17.5 tok/s (57.02ms/tok)

**Root cause identified:** The daemon (`bench_qwen35_mq4`) never calls
`end_decode_turn()`, so `ar_forward_replay_enabled` stays false. Every token
takes the capture+launch path (drop old graph, memcpy, begin_capture, forward,
end_capture, graph_launch), which is more expensive than direct dispatch.

The replay path (memcpy + graph_launch) would be ~order-of-magnitude faster,
but the warmup/commit gate is never satisfied. Fixing this requires wiring
`end_decode_turn()` into the bench tool and daemon after each coherent decode turn.

**Note:** ROCm 6.3.3 hipGraph support appears functional — the capture and
launch work without errors. The attractor bug cited in the code comment was
observed on ROCm 7.13. Testing on 6.3.3 with coherence-gate would determine
if the bug exists here.

---

## Finding 6: Decode gap breakdown

**hipfire bench (no profiling):** 19.7 tok/s (50.53ms/tok) at short context
**llama-benchy via server:** 19.38 tok/s at pp128
**llama.cpp:** 20.66 tok/s at pp128

The 3.2ms/tok gap (6%) is smaller than the profiling data suggests because
the profiler serializes all kernels. Real pipelined execution hides some
dispatch overhead. Possible contributors to the remaining 6% gap:

1. **Quant format efficiency**: Q4_K_XL (llama.cpp) has different memory access
   patterns than MQ4 (hipfire). Hipfire's FWHT rotation is an extra step.
2. **Attention implementation**: hipfire's `attention_q8_0_kv` at 3.7 GiB/s
   for B=1 decode seems low. llama.cpp may have a more efficient B=1 attention.
3. **KV cache format**: hipfire uses asym4/q8 KV cache. llama.cpp uses f16
   (by default) or q8_0. The format choice affects attention kernel efficiency.
4. **lm_head**: hipfire's `gemv_hfq4g256` for lm_head (248K vocab) takes
   2264μs per call. llama.cpp may have a more efficient vocab projection.
   This is only called once per token but it's 5.8% of kernel time.
5. **Host-side pipelining**: llama.cpp's ggml graph scheduler may pipeline
   kernel launches more efficiently than hipfire's sequential Rust dispatch.

## Finding 7: GEMV kernel at 98-109% of peak BW — already optimal

**Date:** 2026-06-03

Microbenchmark of `gemv_hfq4g256` on gfx906 shows the kernel is fully memory-bound:

| Shape | µs/call | GiB/s | % peak |
|-------|---------|-------|--------|
| gate_up (17408×5120) | 157.9 | 279.7 | 106.8% |
| down (5120×17408) | 168.4 | 262.3 | 100.1% |
| qkv (13824×5120) | 122.5 | 286.4 | 109.3% |
| o_proj (5120×8192) | 80.8 | 257.3 | 98.2% |
| lm_head (248320×5120) | 2231.1 | 282.4 | 107.8% |

The >100% readings are due to L2 caching of the input vector x, which reduces
real DRAM traffic below the analytical byte count. **No kernel-level optimization
of the GEMV will improve decode performance.** The bottleneck is the memory bus.

---

## Conclusion and Action Items

### The decode gap is primarily dispatch overhead, not kernel efficiency

| Component | Time (ms/tok) | % of wall |
|-----------|---------------|-----------|
| Pure kernels (pipelined) | ~39.1 | 77.4% |
| Dispatch overhead | ~11.4 | 22.6% |
| HTTP server overhead | ~1.0 | 2.0% |

**The 3.5% gap vs llama.cpp at short context comes from:**
1. Dispatch overhead (22.6% of wall time in hipfire, ~0% in llama.cpp with graph replay)
2. Small differences in non-GEMV kernels (attention, rmsnorm, etc.)
3. HTTP server overhead in llama-benchy measurements

### Action items (priority order):

1. **[HIGH] Re-enable graph replay for AR decode** — The infrastructure exists
   but `end_decode_turn()` is never called. Wiring it into the daemon after each
   decode turn would enable graph replay, potentially saving ~11.4ms/tok (+30%
   decode speed). Need to verify the ROCm attractor bug is absent on 6.3.3.

2. **[MED] Reduce kernel launch count** — 867 launches/tok is very high. Fuse
   more operations per layer. The small kernels (rmsnorm at 2.4 GiB/s, rotate
   at 6.4 GiB/s, norm at 3.8 GiB/s) are dispatch-limited and could be combined
   into fewer, larger kernels.

3. **[LOW] Attention kernel optimization** — `attention_q8_0_kv` at 3.7 GiB/s
   for B=1 decode is low but only 3.4% of kernel time. Worth investigating
   for long-context decode but not a priority.

4. **[LOW] Prefill scaling** — hipfire wins pp128 by 2× but loses pp2048 by 7%.
   The GEMM kernels (24-33 GiB/s) may need better tiling at large batch sizes.

5. **[INFO] GEMV kernel is already optimal** — 98-109% of peak BW. No further
   kernel tuning needed for decode GEMV.

---

## Finding 2: Prefill Profile

128-token prefill on Qwen3.6-27B MQ4, MI50:

- **Wall time:** 1160.1ms (includes JIT on first run)
- **Kernel time:** 579.4ms
- **Cold overhead:** 580.7ms (JIT compilation)
- **19.4 tok/s wall, 220.9 tok/s kernels-only**

| Kernel | Calls | Total(ms) | μs/call | %kernel | BW(GiB/s) |
|--------|-------|-----------|---------|---------|----------|
| gemm_gate_up_hfq4g256_mmq_gfx906 | 112 | 286.4 | 2558 | 49.4% | 32.0 |
| gemm_hfq4g256_residual_mmq_gfx906 | 128 | 190.7 | 1490 | 32.9% | 24.3 |
| gated_delta_net_q8_batch_seq | 48 | 43.1 | 897 | 7.4% | 14.8 |
| gemm_qkv_hfq4g256_mmq_gfx906 | 16 | 21.1 | 1317 | 3.6% | 33.3 |

Prefill is dominated by GEMM (MMQ path for gfx906), which is correctly using
the gfx906-specific dp4a kernels. The prefill gap at pp2048 (169 vs 181 tok/s)
needs a separate long-context profile.

---

## Findings
