# Issue #247 — Q8 flash vs non-flash attention divergence at the 2048 dispatch boundary

**Status:** investigation plan, not yet executed
**Branch:** `docs/issue-247-investigation`
**Hardware target for repro:** gfx1151 (Strix Halo APU), ROCm 7.12 — same setup that surfaced the report. gfx906 (the dev box) can run the same kernels but reference dumps need to be regenerated.
**Related prior fix:** PR #264 (commit 533cc0c) — `attention_flash_q8_0_tile.hip` partials-stride + head_dim-coverage bugs. The defect described in #247 is *post-#264* and is a different issue.

## What we know

- Comparison artifact (gfx1151, Q3.5-0.8B q8f16, Q8 KV, chunk-0 of 2048 positions):

  | pos range | rL2 (hipfire vs HF BF16) | kernel |
  |---|---:|---|
  | 0..2046 | 0.02-0.15, typical ~0.07 | `attention_q8_0_kv` (non-flash) |
  | **2047** | **0.7475** | `attention_flash_q8_0` |

  Next-worst position 1497 sits at 0.146 — pos 2047 is ~5× any other position. Post-gate (stage 12) at pos 2047: rL2 0.8075 / cosine 0.59.

- Dispatch (`crates/hipfire-arch-qwen35/src/qwen35.rs:6700-6717`):
  ```rust
  let use_flash = gpu.capture_mode
      || s.flash_mode == 2
      || (s.flash_mode == 1 && pos + 1 >= 2048)
      || pos + 1 > 15000;
  ```
  At pos=2047, the FIRST position that meets `pos+1 >= 2048`, the kernel swaps from `attention_q8_0_kv` (single-pass softmax over full row) to `attention_flash_q8_0_tile` + `attention_flash_q8_0_reduce` (tile + 2-pass reduce). Tile size = 128 → 16 fully-filled tiles, no partial-last-tile case at this exact position.

## Top-of-stack reading

The 10× spike at the kernel-swap boundary, combined with the recency of PR #264 fixing a different class of bug in the same tile kernel, is suspicious for a *uniformly worse* flash-kernel implementation that only becomes visible at the dispatch boundary. A targeted experiment can falsify this in one daemon run.

## Kernel surface review (already done in conversation)

Files inspected:
- `kernels/src/attention_q8_0_kv.hip:1-101` — single-pass softmax baseline. Per-thread sequential FMA across 8 blocks × 32 lanes.
- `kernels/src/attention_flash_q8_0_tile.hip:25-169` — wave32 tile. Phase A wave-cooperative dot via `__shfl_xor`; Phase D per-lane V accumulation owning 4 dims, loops over `n_halves = (head_dim+127)/128`.
- `kernels/src/attention_flash_q8_0_reduce.hip:25-92` — 2-pass reduce: global_max across tiles, then `Σ tile_out * exp(tile_max − global_max)`, normalize by `Σ tile_sum * corr`.

Findings from the stare:

1. **Partial-last-tile is NOT the bug at pos 2047** — seq_len=2048 is exactly 16 × tile_size=128, all tiles full.
2. **Q-load coverage** in tile kernel: `for half in 0..n_halves: q_lds[half*128 + tid*4 + 0..3] = q_head[…]` — covers head_dim=256 fully for Qwen3.5.
3. **Phase-A wave-reduce** leaves the full sum on every lane after `__shfl_xor` butterfly. Only `tid==0` writes `scores[t_local]`. `__syncthreads()` at line 104 publishes the LDS write to Phase B. Single-writer is correct but unusual; verify LDS visibility semantics on gfx1151.
4. **`global_sum` in reduce kernel** is lane-local but identical across lanes because all 32 lanes execute the same loop with the same scalar tile data on `half==0`. Each lane reads its own copy when computing `inv_sum`. Correct.
5. **`if (tile_sum <= 0.0f) continue;`** guards in both Pass 1 and Pass 2 of reduce — skips uninitialized/empty tiles. If `partials` is stale, this is the only line keeping the reduce honest. The tile kernel always writes `p[1] = tile_sum` for tiles `tile_start < seq_len`, so for the live n_tiles range this guard is a no-op. Not the bug.
6. **Dot-product summation order differs** between baseline (sequential per-thread) and flash (wave-cooperative). Reorder of 128 FP32 terms with FP16-scale × INT8 inputs is ~1e-4 relative — far below rL2=0.75. Not the cause.

## Plan — experiments in order of payoff per unit cost

### Step 1 — falsify "uniformly worse flash" hypothesis (cheapest, most informative)

Run the chunk-0 dump with `s.flash_mode = 2` (force flash on every position) and recompare per-position rL2 vs the existing HF dump. Three outcomes triage the bug:

| rL2 on positions 0..2046 with `flash_mode=2` | Diagnosis | Next step |
|---|---|---|
| Stays ~0.07 like baseline | Bug is specific to the 2048-position case or 16-tile boundary | Step 2 (boundary inspection) |
| Jumps to ~0.7 across all positions | Flash kernel is **uniformly wrong**; pos 2047 is just where dispatch reveals it | Step 3 (single-tile sanity) |
| Pos-dependent shape (e.g. only positions where seq_len % tile_size == 0) | Tile-fill interaction | Step 3 + correlate with seq_len mod tile_size |

Implementation: env-gate already exists in spirit — `s.flash_mode` is set per session. Either:
- (a) Add a `HIPFIRE_FORCE_FLASH=1` env read at session init that sets `flash_mode=2` (~3 lines in daemon.rs / arch-qwen35 state init), OR
- (b) Use the existing dump tooling from PR #76f92d01 (`HIPFIRE_DUMP_LA_STAGES`) and just patch the dispatch boundary to `pos + 1 >= 0` temporarily (one-line local diff, no commit).

Use (a) for a re-runnable experiment.

### Step 2 — bit-level diff at a non-boundary position (if Step 1 says boundary-specific)

Force `flash_mode=2`, dump `s.fa_attn_out` at pos=1024 (mid-chunk, no special-case behavior). Then force `flash_mode=0`, dump same. Element-wise diff:
- `np.abs(flash - nonflash).max()` and per-dim distribution.
- Identify whether divergence is uniform across head_dim (suggests scale/softmax bug) or concentrated in a contiguous range (suggests half-loop / wave-coverage bug).

### Step 3 — single-tile sanity (if Step 1 says flash-uniformly-worse)

Force flash at a short prompt (pos=64, seq_len=64 < tile_size=128). With one tile:
- Pass 1: global_max = tile_max
- Pass 2: corr = exp(0) = 1 for the single tile
- The reduce becomes a pass-through of partials[0]

If single-tile flash matches non-flash → tile-combination math (cross-tile correction or normalization) is the bug.
If single-tile flash disagrees → per-tile QK Phase A or V Phase D is the bug; drill into wave-cooperative dot product.

### Step 4 — Phase D ordering check (only if Step 3 points at V)

Phase D's V accumulation is intra-lane sequential across tokens (line 154-163 of `attention_flash_q8_0_tile.hip`):
```c
for (int t_local = 0; t_local < tile_len; t_local++) {
    float w = scores[t_local];
    // …
    out0 += w * (vs * (float)((signed char)vb[2 + bj_base + 0]));
    // …
}
```
Token-major order is FP32 catastrophic-cancellation-resistant for sane weight distributions, but if `scores[t_local]` has large dynamic range (e.g. one peaked attention head), running sum can lose precision in low-weight tail. The non-flash kernel iterates the same way (line 93-97 of `attention_q8_0_kv.hip`), so this should match. Confirm both orderings are identical, then rule out.

### Step 5 — write a regression test (after fix lands)

Add a unit test that compares `attention_flash_q8_0` vs `attention_q8_0_kv` on a fixed random input with seq_len ∈ {64, 128, 2047, 2048, 2049}. Asserts max element-wise diff < 1e-3 and rL2 < 0.01. Lives in `crates/hipfire-runtime/examples/test_q8kv*.rs` family (test_q8kv.rs already covers q8 kv basics; extend it). Gates future regressions of this class.

## Out of scope for this plan

- **Engine-drift residual floor** (~0.08 nats on Q3.5-0.8B): tracked separately at `docs/investigations/2026-05-12-engine-drift-floor/` and continued on `origin/feat/engine-drift-investigation-tooling`. The flash boundary mismatch is one localized contributor, not the floor itself.
- **Long-context user-visible quality** (>4K tokens): claimed in the issue as "probable user-visible symptom — degraded coherence past 2K". Verifying this needs a separate coherence/PPL sweep at 4K/8K/32K with `flash_mode=1` vs `flash_mode=0`, post-fix. Add to follow-up issues if Step 1-4 confirm the kernel bug.

## Artifacts needed on gfx1151

The reporter's machine has these locally; they are NOT in the repo:
- `/data/cache/hipfire/audit-2026-05-13/hip_fa3_chunk0.bin` (~177 MB)
- `/data/cache/hipfire/audit-2026-05-13/hf_fa3_chunk0.bin` (~177 MB)
- `scripts/compare_la_stages.py` — present on `origin/feat/engine-drift-investigation-tooling` (commit 14738db4, 2026-05-13), NOT on master. Either cherry-pick that commit onto this branch before running, or rebase this branch on top of it.

## Suggested commit cadence

1. (this commit) plan checked in to `docs/plans/issue-247-q8-flash-boundary.md`.
2. Add `HIPFIRE_FORCE_FLASH` env-gate (Step 1 enabler).
3. Run Step 1, commit findings to `docs/investigations/2026-05-17-issue-247-q8-flash-boundary/01-flash-mode-2-sweep.md`.
4. Branch on Step 1 outcome → Steps 2 or 3.
5. Fix commit referencing the diagnosed root cause.
6. Regression test commit (Step 5).
