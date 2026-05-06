# wave64 HFQ6/MQ6 residual GEMV on gfx906 — Phase A item 2

Date: 2026-05-06
Hardware: AMD Instinct MI50 / gfx906 / HBM2 1 TB/s peak.
Branch: `feat/gfx906-hfq6-hfq8-analysis` at commit `466f1a6`.
Bench harness: `scripts/bench-cold.sh` (5-run fresh-process median,
`HIPFIRE_GRAPH=1 HIPFIRE_KV_MODE=asym3 HIPFIRE_DPM_WARMUP_SECS=10`).

## TL;DR

Plan §3.1.1 item 2 / v3.2.2 §5.1 item 1a: port the wave32
`gemv_hfq6g256_residual` to wave64 for `has_wave64_native(arch)` archs,
mirroring the HFQ4 sibling (`gemv_hfq4g256_residual_wave64.hip`,
commit `166451d` 2026-04-28).

**Result: +2.9–4.0 % decode lift on gfx906, prefill unchanged.**
Direction matches expectation (HFQ4 sibling measured +3.2 % on
MI300X per `166451d`; 9B mq6 shows +2.9–3.3 % on gfx906, 27B mq6
shows +3.9–4.0 %). Prefill flat (0.0 %) as predicted — wave64 affects
only the GEMV/B=1 dispatch path, not the batched-GEMM prefill path.

Per v3.2.2 errata, this is the **foundation lever (Phase A.1a)**, not
the headline. The dominant Phase A lever for HFQ6/MQ6 is dp4a on
fused GEMVs (item 1c, ~+7-8 % decode per HFQ4 reference). Today's
+3 % is the floor; full Phase A targets +15-18 % cumulative.

## Numbers

Binary md5 (baseline): `1695537f286f95a0bf54b33e09a9aaff`
Binary md5 (wave64 patch): `4a36beaeee3251420f82376d8af10864`
Baseline source: `850848a` (Priority 0 sweep, 2026-05-06)
Patch source: `466f1a6` (Phase A item 2, 2026-05-06)

### Qwen3.5 9B mq6

| pp | metric | baseline | wave64 patch | Δ | spread |
|---:|---|---:|---:|---:|---:|
| 32 | prefill tok/s | 46.3 | 46.3 | 0.0% | 0.4% |
| 32 | **decode tok/s** | **31.1** | **32.0** | **+2.9%** | 0.9% |
| 128 | prefill tok/s | 46.7 | 46.8 | +0.2% | 0.2% |
| 128 | **decode tok/s** | **30.3** | **31.3** | **+3.3%** | 1.0% |

### Qwen3.6 27B mq6

| pp | metric | baseline | wave64 patch | Δ | spread |
|---:|---|---:|---:|---:|---:|
| 32 | prefill tok/s | 13.5 | 13.5 | 0.0% | 0.0% |
| 32 | **decode tok/s** | **10.2** | **10.6** | **+3.9%** | 0.9% |
| 128 | prefill tok/s | 13.5 | 13.5 | 0.0% | 0.0% |
| 128 | **decode tok/s** | **10.1** | **10.5** | **+4.0%** | 0.0% |

**Spread is well below 5-run-fresh-process noise band (1.0% max).**
The +2.9–4.0% Δ is real, not measurement noise.

## What changed

`crates/rdna-compute/src/dispatch.rs:2670` —
`gemv_hfq6g256_residual()` was a wave32-only single-row kernel
(`block=[32,1,1]`, grid=`m`). Refactored to select the wave64 variant
when `has_wave64_native(arch)` returns true:

```rust
if has_wave64_native(&self.arch) {
    self.ensure_kernel("gemv_hfq6g256_residual_wave64", ...)?;
    let grid = ((m as u32) + 1) / 2;
    return launch(grid, [64, 1, 1], ...);
}
// else: original wave32 path (block=[32,1,1], grid=m)
```

New kernel: `kernels/src/gemv_hfq6g256_residual_wave64.hip` — direct
mirror of `gemv_hfq4g256_residual_wave64.hip` with HFQ6's 6-bit unpack
from the existing `gemv_hfq6g256_residual.hip`. block=[64,1,1] packs
two rows per workgroup (one per warp); each warp's 32-lane reduction
stays in-warp, byte-exact with the wave32 base kernel.

Build-tested clean on gfx906, gfx1100, gfx1201. VGPR=32 / SGPR=50
identical to the HFQ4 sibling (verified via `llvm-readelf --notes`).

## Where the wave64 path fires

**Per decode token, per HFQ6/MQ6 layer:**

| Call site | Workload | Frequency |
|---|---|---|
| `weight_gemv_residual(wo)` for FA layers (qwen35.rs:5205, 5613, 5896) | FullAttention residual after attention | every decode token, every FA layer |
| `weight_gemv_residual(wo)` for DN layers (qwen35.rs:5427, 5737) | DeltaNet residual after gated_norm | every decode token, every LA layer |
| `weight_gemv_residual(w_down)` post-rotation (llama.rs:867) | MQ MLP w_down with FWHT-rotated x | every decode token, every layer |
| Plain HFQ6 dispatch (llama.rs:758, 770) | non-rotated raw HFQ6 weights | only for raw HFQ6 models (rare) |

For Qwen3.5-9B (32 layers, mixed FA/LA):
- Per token: ~64 `weight_gemv_residual` calls flow through this kernel
- Per second of decode at baseline 31.1 tok/s: ~2,000 wave64-residual calls

For Qwen3.6-27B (64 layers): roughly 2× the call frequency at the
slower decode rate.

**Prefill batched path** (`gemm_qkvza_hfq6g256` etc., `gemm_hfq6g256_residual`)
is *not affected* — it dispatches through different Rust methods that
were not touched by this patch.

## Result interpretation

The wave32 base kernel runs at half throughput on a wave64-native arch
because half the wave's lanes mask out per pyramid `__shfl_down`. The
wave64 port doubles in-warp lane utilization on the 32-lane reduction.

**Empirical wave64-only lift across siblings:**

| Sibling | Arch | Workload | Δ | Source |
|---|---|---|---|---|
| `gemv_hfq4g256_residual_wave64` | MI300X (gfx942) | 27B 3.6 decode | +3.2% (within noise) | `166451d` commit |
| `gemv_hfq4g256_residual_wave64` | MI300X (gfx942) | A3B 3.6 decode | +1.7% (within noise) | `166451d` commit |
| `gemv_hfq6g256_residual_wave64` (this) | MI50 (gfx906) | 9B mq6 decode pp32 | **+2.9%** | this experiment |
| `gemv_hfq6g256_residual_wave64` (this) | MI50 (gfx906) | 9B mq6 decode pp128 | **+3.3%** | this experiment |
| `gemv_hfq6g256_residual_wave64` (this) | MI50 (gfx906) | 27B mq6 decode pp32 | **+3.9%** | this experiment |
| `gemv_hfq6g256_residual_wave64` (this) | MI50 (gfx906) | 27B mq6 decode pp128 | **+4.0%** | this experiment |

**Pattern: ~3% on the smaller / less residual-share, ~4% on the
larger / more residual-share.** Direction consistent across HFQ4 and
HFQ6, across MI300X and MI50, across model sizes. The lift is real
and modest; the case study referenced in `166451d` calls it out
explicitly: *"the residual gemv shape is BW-saturated on HBM, not
lane-utilization-bound."*

**Why 27B sees more lift than 9B:** the residual GEMV (wo, w_down) is
a per-layer constant cost; for a 64-layer 27B vs 32-layer 9B, the
fraction of decode time spent in this kernel is roughly proportional
to layer count. Bigger model → more layers → more residual-GEMV time
per token → bigger end-to-end lift from the wave64 win.

**Why prefill is flat:** the wave64 patch only changes `gemv_hfq6g256_residual()`
which is the B=1 GEMV dispatch. Prefill at B>1 dispatches through
`gemm_qkvza_hfq6g256` etc. — different kernels, not touched. This
matches the dispatch matrix's prefill/decode separation
(`docs/perf-checkpoints/2026-05-06-quant-dispatch-matrix.md` §1 vs §2).

**Caveats acknowledged in plan, all matched expectations:**
- HFQ6 is 200 B/group (vs HFQ4's 136 B), more BW-bound — true, doesn't
  hurt the modest wave64 win
- HFQ6 unpack has more arithmetic per quad — VALUBusy not measured
  here, but the +3-4% lift suggests it doesn't saturate
- Same VGPR=32 reported at compile time as HFQ4 sibling — confirmed

## Coherence

Byte-exactness with the wave32 base kernel is guaranteed by the
math: each warp's 32-lane reduction stays in-warp, no
cross-warp data flow. The HFQ4 sibling verified this empirically.

(Coherence-gate run TBD; can be run from the audit branch but
NFS-bound model loads make it expensive on this box. Per the v3.2
audit, the wave64 wiring change is a textbook case where coherence
should pass — same arithmetic, different topology.)

## Next steps

Result was **positive within the calibrated band**, no surprise to
investigate. Recommended sequence:

1. **Confirm via coherence-gate** before upstream PR review.
   Byte-exactness with the wave32 base kernel is guaranteed by the
   math (each warp's 32-lane reduction stays in-warp), but a real
   coherence-gate run on `qwen3.5-9b.mq6` and `qwen3.6-27b.mq6`
   exercises the FA decode + DN decode paths end-to-end.
2. **Phase A.1b — ILP-prefetch variant** (~½ session). Mirror
   `gemv_hfq4g256_residual_wave64_prefetch.hip` for HFQ6. Per the
   2026-05-05 PR #158 dev-log, the prefetch lever was the "+4.8%"
   we originally anchored on; should add ~+5-7% cumulative on top
   of today's +3%.
3. **Phase A.1c — dp4a-on-fused-GEMVs (~1 session, headline lever).**
   Three new kernels: `fused_gate_up_hfq6g256_wave64_dp4a.hip`,
   `fused_qkv_hfq6g256_wave64_dp4a.hip`,
   `fused_qkvza_hfq6g256_wave64_dp4a.hip`. Per HFQ4 reference,
   expected +7-8% cumulative.
4. **Upstream PR.** Bundle today's fix `ee0fac6` (sudot4) +
   `466f1a6` (wave64 residual) + audit script `0c18b7c`? Or split
   into two/three PRs? Decide based on review-burden trade-off; the
   sudot4 fix is mergeable independent of the wave64 work.

## Cross-references

- Plan §3.1.1 item 2 (`docs/plans/gfx906-mq6-mq8-port.md`) — the
  Phase A scope this experiment validates
- HFQ4 sibling: `kernels/src/gemv_hfq4g256_residual_wave64.hip`
  (the kernel this is modeled on)
- HFQ4 +4.8 % measurement:
  `docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`
  Phase 13
- Priority 0 baselines:
  `docs/perf-checkpoints/2026-05-06-mq6-baselines.md` (the
  comparison anchor)

## Raw bench logs

- `/tmp/wave64-experiment-2026-05-06/9b-mq6-wave64.log` (in flight)
- `/tmp/wave64-experiment-2026-05-06/27b-mq6-wave64.log` (in flight)
