# Dev log 2026-05-06 — Lloyd-Max codebook extension from MQ3 to MQ4

**Branch:** `lloyd-max-mq3-spike` — fetched from upstream PR #115, then
**ported onto post-0.1.20 master** (the modular split of `engine/` into
`hipfire-runtime/` + per-arch crates). The branch is now
`upstream/master + 1 commit` (the ported Lloyd-Max work; see
`feat(mq-lloyd): port Lloyd-Max MQ3/MQ2 codebooks onto post-modular master`).
**Target hardware:** gfx1100 (7900 XTX) — same as PR maintainer, so the
PR's tok/s and ppl numbers are directly comparable.

## TL;DR

PR #115 lands per-block N-entry fp16 Lloyd-Max codebooks for MQ3 (8
centroids) and MQ2 (4 centroids), replacing uniform asymmetric `scale·q
+ zero` reconstruction on the FWHT-rotated MQ family. **Extending the
same machinery to MQ4 (16 centroids) is a credible Pareto win** between
current MQ4 (136 B/group) and MQ6 (200 B/group), at +17.6% bandwidth and
projected 10–25% ppl reduction over uniform MQ4 on Qwen 3.5+. The
existing PR's perf gate (3.2× decode regression from naive switch
dispatch) is the same blocker MQ4 will hit harder; solving it once for
MQ3 should solve both.

## Where this fits in the prior HF/MQ damage analysis

Earlier session analysed the actual per-weight error of HFQ4-G256 /
HFQ6-G256 by reading the quantizer (`crates/hipfire-quantize/src/main.rs:580`
and `:917`). Both are plain G256 asymmetric uniform: one f32 scale + f32
min per 256 weights, equispaced bins. SNR ladder relative to Q8_1:

| Format        | Bins | Δ vs group range | RMSE / R | SNR (theoretical) | Multiple of Q8_1 RMSE |
|---------------|-----:|-----------------:|---------:|-------------------|----------------------:|
| HF4 (uniform) |   15 |          R/15    |   1.92%  | ~34 dB            | ~17×                  |
| HF6 (uniform) |   63 |          R/63    |   0.46%  | ~46 dB            | ~4×                   |
| Q8_1 (asym)   |  255 |         R/255    |   0.11%  | ~58 dB            | 1× (reference)        |

Two failure modes for uniform asym at G256:
1. One outlier inflates `range` for all 255 neighbours — most bins go to
   waste over the wide gap.
2. Even on well-behaved blocks, equispaced cells are MSE-optimal only
   for uniform distributions; post-FWHT weights are roughly Gaussian.

PR #115 swaps the rule for `rec(idx) = codebook[idx]`, where `codebook`
is fitted per-block by Lloyd's algorithm (k-means in 1-D, MSE-min). This
is exactly the "Path D Lloyd-Max non-uniform codebooks" mentioned in
`docs/QUANTIZATION.md:35`. It attacks both failure modes at once — it
re-places centroids around the actual mass of the distribution.

## What the PR delivers

Empirical wikitext2 perplexity (gfx1100, ctx=2048, 2039 tokens scored,
from `benchmarks/results/lloyd_max_findings_20260501.md`):

| size | MQ4   | uniform MQ3 | **Lloyd-MQ3** | Lloyd factor | Lloyd-MQ3 vs MQ4 |
|------|------:|------------:|--------------:|-------------:|-----------------:|
| 0.8B | 25.65 |      301.06 |    **155.22** |        1.94× |            6.05× |
| 4B   | 12.73 |       45.24 |     **22.56** |        2.01× |            1.77× |
| 9B   | 10.34 |       42.03 |     **18.52** |        2.27× |            1.79× |

Lloyd-MQ3 at 9B is the closest sub-4-bit format hipfire has to MQ4
quality — closing roughly half the gap between uniform-MQ3 and MQ4. This
matches the analytic prediction: at 3 bits the codebook has only 8 cells
and Lloyd captures most of the placement error vs the uniform grid.

MQ2-Lloyd: 41–55× ppl reduction over uniform MQ2, but 9B absolute floor
stays at ppl=2,163 vs MQ4's 10.34 — bit-width (not codebook placement)
is binding at 2 bpw. PR keeps it research-only.

## Storage layout (extending the table to MQ4)

Each Lloyd block is `2^B × fp16 centroids` + `B × 256 / 8` index bytes.

| Format        | Uniform B/group | Lloyd B/group                        | Lloyd overhead |
|---------------|----------------:|-------------------------------------:|---------------:|
| MQ2 / Lloyd-MQ2 |              72 | 72  (8 hdr + 64 idx)                |             0% |
| MQ3 / Lloyd-MQ3 |             104 | **112** (16 hdr + 96 idx)           |          +7.7% |
| **MQ4 / Lloyd-MQ4** |         136 | **160** (32 hdr + 128 idx)          |     **+17.6%** |
| MQ6 / Lloyd-MQ6 |             200 | 328 (128 hdr + 192 idx)             |           +64% |

The crossover for MQ4 is the right anchor: **Lloyd-MQ4 (160 B) sits
between uniform MQ4 (136) and MQ6 (200)**. If quality also lands between
MQ4 and MQ6, it's Pareto-favourable — and that's the bet.

## Quality projection for Lloyd-MQ4

Lloyd's gain over uniform compresses as bit-width rises (uniform grid
already absorbs the tail well at 16 cells). Analytic + cross-checked
against the PR's 3-bit numbers:

- **3 bits:** Lloyd ~2.27× ppl improvement at 9B (observed).
- **4 bits:** expect 10–25% ppl reduction.
- **6 bits:** marginal; not worth the +64% bandwidth.

Concrete projection on Qwen 3.5+ at 9B: `MQ4 ppl 10.34 → Lloyd-MQ4 ppl
~8.0–9.3`. That places it between current MQ4 and MQ6 quality at a
bandwidth between MQ4 and MQ6.

For dense Llama-class models without FWHT calibration, the Lloyd lift
percentage is similar (Lloyd attacks per-block placement error
regardless of distribution shape), but the starting point is HF4, which
is worse than MQ4 to begin with. **Lloyd-HF4 may be the more interesting
deliverable for the dense-model audience** — quantitatively the same
implementation, no FWHT, applied to non-Qwen targets.

## Implementation map (what to add)

The Lloyd-Max plumbing in PR #115 is fully end-to-end for MQ3/MQ2; MQ4
follows the same template. Files to touch (mirroring the PR's MQ3
additions):

1. **Quantizer** — `crates/hipfire-quantize/src/main.rs`. Add
   `quantize_mq4g256_lloyd`. Identical to `quantize_mq3g256_lloyd`
   except: `cb: [f32; 16]`, 16 percentile init points (1/32, 3/32, …,
   31/32), 4-bit index packing (2 indices per byte instead of 3-bit
   cross-byte).
2. **DType + qt code** — pick next free `QuantType` value (qt=21
   alongside qt=19/20 in the PR).
3. **HIP GEMV** — `kernels/src/gemv_mq4g256_lloyd.hip`. Mirror
   `gemv_mq3g256_lloyd.hip`. **The naive switch-on-index dispatch will
   be worse at 16 cases than 8** — see perf gate below.
4. **Engine load arms** — `crates/engine/src/qwen35.rs` /
   `crates/engine/src/llama.rs` (for HF4-Lloyd variant) — qt=21 wired
   into `load_lm_head`, `load_weight_tensor`, DeltaNet CPU
   `load_any_as_f32`, and `weight_gemv*` arms.
5. **Dispatch wrappers** — `crates/rdna-compute/src/dispatch.rs`:
   `gemv_mq4g256_lloyd[_with_rotate]`.
6. **Research-opt-in guards** — `--allow-mq4-lloyd` /
   `HIPFIRE_ALLOW_MQ4_LLOYD=1` in the quantizer until ship gates clear.

Reference quantizer skeleton (8-centroid version from
`crates/hipfire-quantize/src/main.rs`, post-PR-#115 patch — adapt to 16
centroids for MQ4):

```rust
// Initial centroid placement: 8 evenly-spaced percentiles.
let mut sorted: [f32; 256] = group;
sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
let mut cb: [f32; 8] = [0.0; 8];
for k in 0..8 {
    let frac = (2 * k + 1) as f32 / 16.0;
    let idx = ((frac * 255.0).round() as usize).min(255);
    cb[k] = sorted[idx];
}
// Lloyd loop: max_iter=8, early-exit on no-change.
// Final: sort centroids ascending, remap indices, fp16-pack header.
```

## Perf gate — the actual blocker

PR #115 explicitly says: "quality result is real but decode perf gate
isn't cleared." Lloyd-MQ3 decodes at 44 tok/s on 9B vs uniform MQ3's 141
tok/s — **3.2× regression** from the 8-way switch in
`gemv_mq3g256_lloyd.hip` dequant.

For MQ4 the situation is worse before it's better:

- 8-way switch → **16-way switch**. Naive `switch(idx)` on RDNA inner
  loops scales badly past ~4 cases (loses unroll budget, pressures
  scalar regs).
- The fix proposed in the PR (LDS-resident codebook table) is *more*
  obviously correct for MQ4: 16 fp16 = 32 B per wave in LDS, one
  `ds_read_b16` per weight per thread, K4 unroll mirroring
  `gemv_hfq3g256.gfx1100.hip`. 32 B in LDS is nothing; this is the
  right design at 16 cells.
- **WMMA prefill path doesn't exist for MQ3/MQ2/MQ4-Lloyd yet.** PR file
  list adds WMMA kernels only for HFQ3, not for the Lloyd variants. For
  Lloyd-MQ4 to ship as a default it needs a WMMA prefill kernel; the
  codebook lookup in the dequant phase of WMMA is solvable but more
  work than the GEMV path.

## What changed in the prior analysis

Two updates to last session's writeup based on PR #115 evidence:

1. **The "what would move HF4 closer to Q8_1" list** previously ranked
   sub-block scales (Q4_K-style hierarchy) ahead of Lloyd-Max codebooks.
   PR #115 shows Lloyd alone produces a 2.27× ppl improvement at 3 bits
   without sub-block scale machinery. Lloyd-MQ4 reuses the
   index-into-table pattern the PR already lands; sub-block scales would
   require a second header layer and complicate kernels. **Lloyd is
   probably the better single bet** for this codebase at 4 bits.
2. **The "stays at Q4-grade quality" verdict for HF4/MQ4** was
   conditional on uniform quantization. Lloyd-MQ4 reframes the question:
   at +17.6% bandwidth, gap-to-Q8_1 narrows from ~17× RMSE to a
   projected ~10–12× RMSE — still not Q8_1 territory, but a meaningful
   step up.

## Plan for the gfx1100 box

Order of work:

1. Land K4-unroll + LDS-resident codebook fix on PR #115's existing
   `gemv_mq3g256_lloyd.hip` — clears the MQ3 perf gate and establishes
   the kernel template MQ4 will reuse. Speed target: ≥120 tok/s on 9B
   Lloyd-MQ3 (PR's stated gate).
2. Implement `quantize_mq4g256_lloyd` — same Lloyd loop, 16 cells, 4-bit
   indices.
3. Run the PR's `crates/engine/examples/perplexity.rs` harness on Qwen
   3.5+ at 0.8B / 4B / 9B for Lloyd-MQ4. Same harness as the PR's table,
   so deltas are directly comparable.
4. Decision point on the data:
   - **Lloyd-MQ4 ppl < uniform MQ4 ppl by ≥10%**: ship it. +17.6%
     bandwidth is less than the gap to MQ6 and likely better quality
     than MQ6.
   - **<5% improvement**: don't ship. Lloyd's value concentrates at low
     bit-width.
5. If shipping: implement WMMA prefill kernel for Lloyd-MQ4 (gate for
   prefill perf parity).

## Verification rules from CLAUDE.md that bind here

- **Coherence gate** (`./scripts/coherence-gate.sh`) on any kernel /
  quant-format / dispatch change.
- **Perf benchmarks must be cross-process** (`scripts/probe_commits.sh`)
  — within-session A/B has ±10–15% drift from DPM/thermal state. Apply
  to any tok/s claim when comparing Lloyd vs uniform decode.
- **Speed-gate** must pass before commit on any kernel-perf-relevant
  change.
- The PR's gates are strictly compatible: ≥120 tok/s decode (perf) +
  4-prompt coherence battery (quality) on 4B and 9B.

## Open questions / risks

- **WMMA prefill design at 16 cells.** Codebook fits trivially in LDS
  but the WMMA inner loop dequants in fp16 lanes; need to verify
  `ds_read` scheduling doesn't stall the matrix path.
- **Quantize wall time.** Single-thread Lloyd's "didn't finish in 5+
  min" on 9B per the PR — rayon-parallel over output blocks brought it
  to ~85s. MQ4's 16-cell version doubles centroid-update work; expect
  ~120–150s on 24-core for 9B. Acceptable for a CPU-only tool but
  worth measuring early.
- **Lloyd-MQ4 on dense (non-Qwen) HF4 base.** Worth a side experiment:
  same quantize / kernel without FWHT (Lloyd-HF4). This is the deliverable
  for the Llama-class audience and the implementation cost is the same
  binary minus the rotation step.

## References

- PR #115: <https://github.com/Kaden-Schutt/hipfire/pull/115>
- Branch: `lloyd-max-mq3-spike` (local, fetched from upstream).
- `benchmarks/results/lloyd_max_findings_20260501.md` — PR's full
  empirical writeup.
- `docs/plans/mq-sub4bit-research-queue.md` Q1.5 — canonical research
  log.
- `docs/plans/mq-sub4bit-prd.md` Phase 1.5 — PRD-level entry.
- `crates/hipfire-quantize/src/main.rs:580` — `quantize_hfq4g256` (the
  uniform baseline being replaced).
- `kernels/src/gemv_mq3g256_lloyd.hip` — naive switch-dispatch reference
  whose perf is the open ship gate.
- `kernels/src/gemv_hfq3g256.gfx1100.hip` — K4-unroll pattern to mirror
  for the LDS-codebook fix.
