# HFP4 / MFP4 — RDNA3 Acceleration Plan (Phase B-1)

**Target:** gfx1100 (Navi 31, RX 7900 XTX) primary, gfx1151 (Strix Halo APU) secondary.
**Base:** master @ 5716dcf — post #224 (HFP4G32 v1) + #225 (MFP4G32 v1).
**Branch:** `feat/hfp4-mfp4-rdna3-accel`.
**Why now:** v1 MFP4G32 ships at **−12% decode / −90% prefill** vs MQ4 on gfx1100 (9B canonical bench: 113 vs 128.8 tok/s decode; 116 vs 1134 tok/s prefill). The architectural payoff for the FP element family was scoped to RDNA4 FP8-WMMA (deferred to v2). RDNA3 currently has no acceleration story for HFP4/MFP4 — this plan builds one.

## Scope

Three named accel vectors, ranked by ROI on RDNA3. All apply equally to HFP4G32 and MFP4G32 (same kernel — MFP4 is HFP4 + offline FWHT in the weight encoding).

### Vector 1 — Batched WMMA-FP16 prefill (the −90% gap)

**Bottleneck.** `gemv_hfp4g32` is the only kernel in the family today. Prefill falls back to per-token GEMV in a loop, which is why pp goes from MQ4's 1134 tok/s to HFP4's 116 tok/s on 9B/gfx1100. RDNA3 has FP16 WMMA (`v_wmma_f32_16x16x16_f16`), and the MQ4 family already uses it — `gemm_qkv_hfq4g256_wmma`, `gemm_gate_up_hfq4g256_wmma`, `gemm_hfq4g256_residual_wmma` all live in `kernels/src/`. Port that recipe.

**Recipe (mirrors HFQ4G256).**
1. Dequant prologue — load 16-wide FP4 nibble strip → unpack → multiply by UE8M0 block-exp (`v_ldexp_f32` or the existing `bit_cast<float>(e << 23)` trick at `gemv_hfp4g32.gfx1100.hip:103`) → cast to FP16, stage in LDS.
2. WMMA 16×16×16 inner loop on FP16 staged tile vs FP16 x tile.
3. Apply row_scale_a once at output stage (folded into the FP16→f32 store).

**Kernels to add.**
- `kernels/src/gemm_qkv_hfp4g32_wmma.hip` — fused QKV projection.
- `kernels/src/gemm_gate_up_hfp4g32_wmma.hip` — fused gate+up FFN (uses `row_scale_b`).
- `kernels/src/gemm_hfp4g32_residual_wmma.hip` — generic residual-add GEMM (down-proj, attn out-proj).

**Dispatch.** Add WMMA arms in `dispatch.rs` parallel to the existing HFQ4 ones around `dispatch.rs:4477` (`gemm_qkv_hfq4g256`). Reuse the same arch detection + `HIPFIRE_GEMV_*` toggle conventions. MFP4 routes through the same kernels — the FWHT is already baked into the weight codes; runtime applies the matching FWHT to x via the existing `mq_rotate_x` hoist (same as decode today, `dispatch.rs:3088`).

**Expected:** Closes 9B prefill gap to within ~10–20% of MQ4 — i.e. ~900–1000 tok/s on 9B/gfx1100 (vs current 116). Not parity with MQ4 (no FP8-WMMA-tier win on RDNA3) but recovers the 10× regression.

**Risk.** LDS pressure on the dequant→stage pattern; resolved precedent in `gemm_gate_up_hfq4g256_wmma_ldsx.hip` (LDSX variant tried and shipped). Coherence-gate must pass on 9B + 27B before commit.

### Vector 2 — SGPR-LUT / direct-bit-cast E2M1 decode (the −12% gap)

**Bottleneck.** Current GEMV uses an LDS-resident E2M1 magnitude LUT (`__shared__ _Float16 lut[16]` at `gemv_hfp4g32.gfx1100.hip:42`). 8 LDS loads per quad iter. The deferred v2 list in #224 explicitly named "SGPR-LUT / direct-bit-cast E2M1 decode" as the closer for the −12% HFP4 ALU gap vs HFQ4.

**Two candidate decodes.**
- **(a) SGPR-LUT.** Move the 16-entry FP16 LUT into SGPR-resident constants (compile-time array). Removes the LDS round-trip but each lookup is still a gather.
- **(b) Direct bit-cast.** E2M1 nibble → FP16 in 3–4 VALU ops without any table:
  - sign bit → FP16 sign bit
  - For magnitude: subnormal codes 0/1 (= 0.0 / 0.5) handled with one branch-free select; normal codes 2..7 map exp_field {0,0,1,1,2,2,3} + mantissa onto FP16 (exp_bias_diff = FP16_BIAS − E2M1_BIAS) via shift + OR.

Variant (b) is what the deferred list is pointing at — same trick the kernel already uses for UE8M0 (`bit_cast<float>(e << 23)`). On gfx1100 the LDS LUT is roughly free at v1 occupancy, so the win measured here will be small (~3–6%); the bigger payoff is gfx1151 where LDS pressure is tighter, and dispatch-side it lets us drop the `__syncthreads()` after the LUT init.

**Apply to.** GEMV (`gemv_hfp4g32.gfx1100.hip`) AND every Vector-1 prefill kernel — the dequant prologue lives in both.

**Expected:** Decode 113 → ~120 tok/s on 9B/gfx1100 (closes most of the −12% gap). Larger win on gfx1151.

**Risk.** Subnormal handling correctness — `+0.0` and `−0.0` need to bit-cast to the right FP16 sign-zero, and `0.5` is the only non-zero subnormal in E2M1. Unit test in `test_gemv_hfp4g32.rs` already has K-sweeps with row-varying weights — extend it with an explicit "every-nibble-once" weight pattern to catch sign-zero and subnormal regressions.

### Vector 3 — Multi-row GEMV fanout (`_mb4`-style)

**Precedent.** Commit 659afc7 (`feat(mq3): _mb4 batch-tile fanout for HFQ3 + MQ3-Lloyd (+77% / +68% gfx1151 9B)`) ports a multi-row tile fanout to the MQ3 family. Same pattern applies to HFP4/MFP4 GEMV: amortize x-vector load across N output rows.

**Path.** Add `gemv_hfp4g32_mb4` (or the existing `HIPFIRE_GEMV_ROWS={1,2,4,8}` selector at `dispatch.rs:30`). Mostly a structural transform of the existing kernel — 4× output row registers, x is loaded once and reused.

**Expected:** Decode +30–60% on gfx1151, +10–20% on gfx1100 (gfx1100 already gets close to BW-bound at 9B/MQ4; HFP4's ALU cost leaves more headroom). This is a **drop-in kernel substitution** — runtime selection happens via `HIPFIRE_GEMV_ROWS` already.

**Sequencing.** Land Vector 1 first; Vector 3 stacks on top of the post-Vector-2 kernel.

## Out of scope

- WMMA-FP8 hero kernel on gfx1201 — RDNA4-only. Separate workstream (the actual deferred v2 from #224/#225).
- HFP4G16 / HFP4G64 ablations — orthogonal format question; doesn't affect this perf gap.
- HFP8E4M3G32 / HFP8E5M2G32 — different format family, separate plan.
- MFP4 FWHT fusion (the −2 to −5% MFP4-vs-HFP4 decode gap). Smaller than every vector here. Defer.
- gfx906 (CDNA1 / MI50) — wave64 + dp4a path is its own track (see `gemm_qkv_hfq4g256_wave64_dp4a.hip`). HFP4 LUT decode doesn't fit dp4a directly; revisit if/when CDNA1 gets a Vector-1-equivalent.

## Phasing

| Phase | Vector | Deliverable | Gate |
|---|---|---|---|
| B1-1 | 1 | `gemm_qkv_hfp4g32_wmma` correctness + 9B/27B coherence | coherence-gate.sh PASS, byte-exact vs reference GEMV |
| B1-2 | 1 | `gemm_gate_up_hfp4g32_wmma` + `gemm_hfp4g32_residual_wmma` | same |
| B1-3 | 1 | Wire all three into `dispatch.rs` MFP4/HFP4 prefill route | 9B prefill ≥ 800 tok/s on gfx1100 |
| B2 | 2 | Direct-bit-cast E2M1 decode in GEMV + Vector-1 prologue | every-nibble-once correctness test PASS, decode ≥ 118 tok/s on 9B/gfx1100 |
| B3 | 3 | `gemv_hfp4g32_mb4` + selector wiring | gfx1151 9B decode +30% over Vector-2 baseline |

Each phase is a discrete commit + bench row in `docs/perf-bench/`. Per `docs/methodology/perf-benchmarking.md`, every claimed win is verified across a fresh process via `scripts/probe_commits.sh PARENT HEAD` before commit. Coherence-gate runs automatically via the pre-commit hook on any kernel change.

## Open questions (resolve before B1-1)

1. Does `gemm_qkv_hfq4g256_wmma.hip` use a row_scale broadcast that maps cleanly to HFP4's per-row-per-block scale structure, or do we need a different scale-staging pattern? (Read kernel before porting.)
2. RDNA3 `v_wmma_f16_16x16x16_f16` vs `v_wmma_f32_16x16x16_f16` — HFQ4's WMMA path picks one; verify which and match it for byte-comparable accumulation.
3. Does the existing `prefill_scaling_rdna3` branch (origin) introduce changes that affect the dispatch geometry for MMQ auto-routing at batch_size ≥ 256 (commit dd2581b)? If yes, rebase onto its tip rather than master.

## Memory + bench discipline

- Per `feedback_rebaseline_before_cross_arch_compare.md`: every perf claim in this branch must be measured **on this hardware** in a fresh process. No transplanting historical numbers across gfx targets.
- Canonical bench config (post-2026-04-26 default-on `prompt_normalize`): 9B Qwen3.5, `--no-chatml --kv-mode asym3`, max=120, PEP-8 strict prompt. Expected MQ4 reference: 199 tok/s τ=10.36 on 7900 XTX. Drift >5% from this is a regression.
- Multi-agent GPU coordination via `gpu-lock.sh` (auto-acquired by hooks).
