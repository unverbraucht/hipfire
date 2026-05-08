# Phase B.2 — HFQ6 MMQ-streaming port (gfx906)

**Status:** scoping (2026-05-08)
**Author:** Claude Opus 4.7
**Predecessor:** Phase B.1 (BT=8→16 on dp4a fused kernels, +14.5 % cumulative; commits `2bee6e6` + `ff9e210`).
**Reference impl:** `kernels/src/gemm_hfq4g256_residual_mmq_gfx906_*.hip` + `_body.cuh` (PR #158).

## 1. Goal

Close the remaining ~3.15× prefill gap between mq4 (598.6 tok/s) and
mq6 (189.9 tok/s) on the 9B model at pp128 by porting the
PR-#158-style MMQ-streaming kernel family to HFQ6. Target: **mq6
prefill within 30 % of mq4 prefill** (≥ 460 tok/s). Stretch: parity
with mq4's bandwidth-bound floor (mq6 has 1.47× more weight bytes per
output element, so a perfect port would land at ~407 tok/s — the
30 % target is realistic).

## 2. What we're porting (mq4 reference)

The mq4 prefill on gfx906 dispatches **only** kernels from the
`gemm_hfq4g256_residual_mmq_gfx906_x{N}` family — fused gate_up /
qkv / qkvza are NOT used. Every projection (q / k / v / z / α / β /
gate / up / down) becomes one residual-shaped MMQ call:
`set` for the first projection of a fused group (zeros Y), `add` for
subsequent ones accumulating into Y.

**Topology:**

- Block: `(64, 4, 1)` = 256 threads = 4 wave64s
- Grid: `(M / 128, N / mmq_x, 1)` (1-D rows × 1-D batch tiles)
- mmq_y = 128 (rows per block — fixed)
- mmq_x ∈ {8, 16, 24, 32, 40, 48, 56, 64} (batch tile width — runtime
  selected: greedy `mmq_x = min({8,…,64} | mmq_x ≥ batch_size})` for
  small batch, else 64)
- LDS budget per block: 19 KiB at mmq_x=8, up to 30 KiB at mmq_x=64
  (within the 32 KiB → 2 WGs/CU target on the 64 KiB/CU cap)

**Per-output-tile algorithm (Option C "Window Streaming"):**

```
for kg in 0..(K / 256):                       # one HFQ4 group per outer iter
    for window in 0..2:                       # 2 Q8_1 blocks per group
        load_q8_1_tile_coalesced<mmq_x>(...)  # 128B per row × mmq_x cols
        load_hfq4_tile_streaming<x_stride>(...)  # 64B nibbles × MMQ_Y rows
        __syncthreads()
        for sub in 0..4:                      # 4 sub-blocks of 32 K-elements
            vec_dot_dp4a_streaming<mmq_x>(...)
        __syncthreads()
write_back_residual_templated<mmq_x, need_check>(Y, sum, ..., add)
```

Key design choices:

- **LDS-staged X (activations) AND A (weights).** The 8-byte sc/zp
  per HFQ4 group plus 64 B of unpacked nibble pairs goes into LDS once
  per window; all 4 wavefronts read from LDS for the inner dp4a loop.
  This is the architectural difference from the dp4a fused kernels
  (which keep A in registers via the 1-wave-per-row design).
- **Compile-time `add` modes.** `_full_add_x{N}` and `_full_set_x{N}`
  variants compile out the runtime `if (add)` in writeback —
  hot-loop branch elimination.
- **`x_stride` per-mmq_x.** mmq_x ≥ 64 uses stride 40 (b128 ds_read,
  4-way bank conflict, b128 issue rate dominates). mmq_x < 64 uses
  stride 33 (b32 ds_read, 0-way bank conflict). PMC-validated in
  `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`.
- **mmq_screen safety net.** `mmq_screen_weight()` runs each weight
  through MMQ vs FP16 reference at first use; if NRMSE > threshold,
  the dispatcher falls through to FP16 wave64 instead. Catches
  pathological quant groups (#87).

## 3. Deltas: HFQ4 → HFQ6 quant

| Property | HFQ4 | HFQ6 | Impact on port |
|---|---|---|---|
| Bits per weight | 4 | 6 | Unpack changes — 6 source bytes → 8 unsigned weights, vs 4 bytes → 8 signed nibbles |
| Group size (K) | 256 | 256 | **Same.** Outer-loop count unchanged. |
| Group bytes | 136 | **200** | Per-row bytes scale 1.47×. LDS x_qs same size (decoded ints); only the load path widens. |
| Header bytes | 8 (sc + zp) | 8 (sc + zp) | **Same.** |
| Weight bytes per group | 128 | 192 | Window split: HFQ4 = 64 B/window × 2 windows. HFQ6 = 96 B/window × 2 windows. |
| Sign convention | nibble - 8 (signed) | unsigned q ∈ [0, 63] | **Skip the -8 shift.** Math identity is `acc = sc·sumi + zp·sum_x` (no `zp_eff`, no `0.25` factor like the fused version). Simpler arithmetic. |
| Decoded int8 packing | 4 nibbles → 1 int (8-bit shift each) | 4 q6 → 1 int (q6 fits in int8 since ≤63) | **Same int_a/int_b layout** — both use 4 int8 packed in int32. |

**Key insight:** the HFQ4 MMQ kernel's `int_a` / `int_b` layout
(packed int8, dp4a-consumable) is identical to what HFQ6 produces.
Only the **unpack step** differs (6 source bytes per lane vs 4). The
`vec_dot_dp4a_streaming`, `load_q8_1_tile_coalesced`, and
`write_back_residual_templated` stages are byte-for-byte reusable.

## 4. Implementation plan (5 sessions estimated)

### Session 1: HFQ6 unpack helper + smallest variant (mmq_x=8)

Goal: prove the architecture works on the simplest size before
touching the full sweep. Land:

- `kernels/src/gemm_hfq6g256_residual_mmq_gfx906_body.cuh` — copy
  HFQ4 body, replace `load_hfq4_tile_streaming` with the HFQ6
  unpack pattern (3 source bytes per int_a + 3 per int_b, q6
  decode = `(b0 & 63)`, `(b0>>6) | ((b1&0xF)<<2)`, etc., already
  implemented in `gemm_hfq6g256_residual_wave64_dp4a.hip` lines
  79-86 — port that same shift algebra into the streaming loader).
- `kernels/src/gemm_hfq6g256_residual_mmq_gfx906_x8.hip` — wrapper
  with `MMQ_X_VAL=8` and `__launch_bounds__(256, 2)`.
- `dispatch.rs::gemm_hfq6g256_residual_mmq_gfx906()` — analog of
  the HFQ4 fn, single-mmq_x path for now.
- Wire mmq_screen for HFQ6 (parallel of `mmq_screen_weight` —
  analogous threshold, FP16-wave64 reference).
- Test: numerical parity vs `gemm_hfq6g256_residual_wave64_dp4a` at
  batch_size=8 on a single q/k/v projection, NRMSE < 0.5 %.

**Risk:** the HFQ6 unpack is more complex (6 → 8 q6 values vs 4 → 8
nibbles per lane). The streaming loader's tid → byte-offset arithmetic
needs careful adaptation. Window split ALSO changes: HFQ4 splits 128 B
of nibbles into 2 × 64 B windows; HFQ6 splits 192 B of bytes into 2 ×
96 B windows. Verify the per-tid load distribution (HFQ4: 8 uints/tid;
HFQ6: 12 bytes/tid or 3 uints/tid — needs recalc).

**Validation:** new `crates/hipfire-runtime/examples/test_hfq6_mmq.rs`
modeled on the existing `test_hfq6_gemm.rs` with MMQ vs wave64-dp4a
NRMSE comparison.

### Session 2: Size sweep + per-mmq_x bring-up

Add the remaining 7 size variants (x16, x24, x32, x40, x48, x56,
x64). Each is a 5-line wrapper around the body header. Risks:

- **`x_stride_for<mmq_x>()` may need re-tuning.** HFQ6's per-row
  ints in x_qs is `groups_per_row * 32` (same as HFQ4 since both
  decode 256 K → 64 ints/row/group). Same b128-vs-b32 tradeoff
  applies, but the unpack overhead is heavier for HFQ6 — the
  threshold may be >64 instead of ≥64. PMC sweep needed.
- **LDS budget.** Same shape as HFQ4 — `MMQ_Y * x_stride * 4 +
  1024 + mmq_x * Y_STRIDE * 4`. Need to re-verify ≤ 32 KiB on the
  larger mmq_x with whatever x_stride wins.

**Validation:** rocprof shows all 8 variants with reasonable
runtime (no >5× outliers), `mmq_screen_weight` passes for all
weight matrices at startup, mq6 prefill bench shows monotonic
improvement.

### Session 3: Wire up dispatch + retarget gate_up / qkv / qkvza

Replace the HFQ6 fused-dp4a dispatch sites with **multiple residual
MMQ calls**:

- `gemm_qkvza_hfq6g256` at B>1 dispatches → 4 calls of
  `gemm_hfq6g256_residual_mmq_gfx906`: one `_full_set` for q
  followed by `_full_add` for k/v + z/β/α (the existing residual
  output buffer is reused as the accumulator).
- Same for `gemm_qkv_hfq6g256` (3 calls: set q, add k, add v).
- Same for `gemm_gate_up_hfq6g256` (2 calls: set gate, add up).

This is the architectural change — instead of one big fused kernel,
N small MMQ calls per layer. Mirrors the mq4 dispatch pattern at
prefill.

**Validation:** coherence-gate clean on 9b.mq6, prefill rocprof
shows the new MMQ kernels accounting for the bulk of GEMM time.

### Session 4: Fast paths for full / non-full + lm_head

The `_full_set_x{N}` and `_full_add_x{N}` variants only fire when
`m % 128 == 0 && batch_size % mmq_x == 0`. The `_x{N}` (data-dependent
`add`) variant handles the residual for the non-full case (last batch
tile when `batch_size % mmq_x != 0`, etc.).

Wire batched_lmhead too: `gemm_hfq6g256_batched_lmhead` currently
dispatches to wave64_dp4a; for prefill should switch to MMQ for the
big-K W_out matrix.

**Validation:** all unit shapes (M=3584/3072/4096, K=4096) hit the
`_full_*` fast path; coherence still clean.

### Session 5: Polish + perf validation

- mmq_screen threshold tuning for HFQ6 (HFQ4 uses NRMSE 0.005;
  HFQ6 quantization noise floor is lower so threshold should
  drop — empirical).
- HIPFIRE_MMQ_K_FILTER + HIPFIRE_MMQ_LAYER_FILTER + HIPFIRE_MMQ_DIAG_*
  env-var debug knobs (parallel of HFQ4 implementation).
- profile.rs `hfq6g256_weight_bytes` helper (plan v3.2.4 item 5
  — closes that follow-up).
- Final mq6 9B pp128 bench, dev-log writeup, plan v3.2.6 errata.

**Validation:** mq6 9B pp128 ≥ 460 tok/s (target), no decode
regression on g50, coherence 7/7, mq6 reasoning prompt
prefill_tok_s reaches mq4-class numbers (~330 in the coherence
harness vs current 175).

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| HFQ6 unpack in LDS-staging path is materially slower than HFQ4 due to 1.5× source bytes | Profile session-1 prototype before committing to size sweep. If x8 doesn't beat the existing wave64_dp4a, redesign before scaling out. |
| LDS budget overflow at large mmq_x with HFQ6's heavier unpack | Reuse HFQ4's x_qs LDS layout exactly (decoded ints, same 32/row/group). Only the load step widens, not the storage. |
| mmq_screen false-positives on edge-case HFQ6 weights | Reuse HFQ4 mmq_screen infrastructure exactly, just keyed by HFQ6 weight pointer. Existing kbloack-fallback path catches pathological matrices. |
| Capture_mode silent breakage | All HFQ6 dp4a sites already have `&& !self.capture_mode` (commit `5768fe4`); MMQ port inherits this. Port adds the same guard to the new dispatcher. |
| DDTree spec-decode breakage on HFQ6 lm_head MMQ | Existing arms in `speculative.rs` (commit `5768fe4`) call the dispatcher fn, not a specific kernel — they'll route to MMQ automatically once the dispatcher does. |
| Per-mmq_x x_stride tuning takes longer than scoped | Start with HFQ4's exact stride choices; only re-tune if PMC shows clear bank-conflict regression. The 5-session estimate already includes 1 session of polish/tuning. |

## 6. Decision points

- **Session 1 GO/NO-GO:** if the x8 prototype gives <30 % uplift over
  current wave64_dp4a at batch_size=8, abort the port and look for
  alternatives (e.g. multi-wave LDS-staged dp4a kernel, which we
  rejected in B.1.1 due to occupancy loss but might be worth
  revisiting with proper LDS budget rebalance).
- **Session 3 GO/NO-GO:** if rewiring gate_up / qkv / qkvza to
  multiple MMQ calls regresses (e.g. extra kernel-launch overhead
  outweighs the per-kernel speedup), keep the fused kernels and
  only ship MMQ for the residual sites (which already use one call
  per projection).

## 7. Out of scope (deferred)

- HFQ8 MMQ (plan §3.2.5 — explicitly ruled out).
- MoE-indexed HFQ6 MMQ (plan v3.2.4 follow-up item 4 — ~1
  additional session post-Phase-B).
- LDS bank-conflict tuning beyond HFQ4's existing per-mmq_x
  thresholds (only re-tune if PMC flags a regression).

## 8. References

- Reference impl: `kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh`
- HFQ6 6-bit unpack pattern: `kernels/src/gemm_hfq6g256_residual_wave64_dp4a.hip:79-86`
- mmq_screen: `dispatch.rs:1263 (mmq_screen_weight)`
- HFQ4 dispatch: `dispatch.rs:7005 (gemm_hfq4g256_residual_mmq_gfx906)`
- Phase B.1 cumulative writeup: `docs/perf-checkpoints/2026-05-07-phase-a-cumulative-mq6.md` (Phase B.1.1 section)
- PR #158 design doc: `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`
- v3.2.5 errata (Phase B framing): `docs/plans/gfx906-mq6-mq8-port.md` v3.2.5
