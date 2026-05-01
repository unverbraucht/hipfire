# Case studies — wins, losses, and the methodology in action

Five worked examples from the actual hipfire git log. Each shows
the workflow from `playbook.md` running on real engineering — a
mix of decisive wins, fake wins caught by discipline, and silent
corruption caught by gating. The lessons are the durable artifact;
the numbers will move as the engine evolves.

---

## §1 — wave64 CDNA3 port (decisive 2× win)

**Commit**: `4105035` — "perf(cdna3): full wave64 port of all hot
HFQ4 kernels — MI300X decode 48.6 → 96 tok/s"

**Bottleneck**: MI300X (gfx94x) is wave64 native, but hipfire's
HFQ4 kernels were wave32. On a wave64 wave running a wave32 kernel,
half the lanes silently mask out — the kernel produces correct
output but at 50% effective throughput.

**Lever**: per-arch wave64 variant. Ten kernels ported with
2-rows-per-block wave64 lane decomposition.

**Numbers**:

```
A3B decode pre-port:  48.6 tok/s on MI300X (gfx942)
A3B decode post-port: 96.0 tok/s — matches 7900 XTX on the same model
```

**Validation path**:
- Channel-test against CPU reference on synthetic HFQ4 weights
  (caught a wave-lane-mapping bug pre-merge).
- Coherence-gate ran against the standard model matrix.
- Speed-gate on the MI300X showed no regression on RDNA archs
  (the wave64 path is gated by `arch.starts_with("gfx94")`).

**Lesson**: wave-size mismatch is a 2× perf cliff, not a small
inefficiency. Worth a proper port any time the target arch's wave
size differs from your kernel's. The pattern (separate `.wave64.hip`
or `.gfx942.hip` file) keeps RDNA dispatch unaffected.

---

## §2 — nontemporal weight-load fake win (caught by clean-baseline bisect)

**Commits**: `0532579` (the candidate) → `34eb024` (the revert).

**Setup**: an experiment to use `__builtin_nontemporal_load` for
weight reads on hot decode kernels, intuition being that decode
weights are streaming-read (each token re-reads them once) and
shouldn't pollute L2.

**Initial measurement** (within-session A/B): +2.0% decode tok/s on
9B MQ4. Looked plausible, committed.

**Bisect against committed speed-gate baseline** (April 12 anchor):
**−13% decode**. 131 → 113 tok/s on 7900 XTX 9B MQ4.

The within-session A/B happened in a GPU state already skewed by
many preceding bench runs — preceding warmup put L2 in a state
where the nontemporal change *appeared* to win, but a fresh process
with a cold cache showed the actual regression.

**Hypothesis** (in the revert commit message): on RDNA3, the
nontemporal load path bypasses cache-line allocation but ALSO
defeats wave-level coalescing/prefetch behavior the default load
path gets for free. Each wave was issuing one coalesced 128-byte
transaction for 32 packed-u32 weight reads; the nontemporal hint
broke that coalescing pattern.

**Lessons**:
1. **Always bisect against the committed baseline**, not your last
   bench run. The speed-gate baseline file
   (`tests/speed-baselines/<arch>.txt`) exists for exactly this reason.
2. **Hypothesis without measurement = noise**. The nontemporal
   intuition was reasonable on paper. The hardware behavior was
   different.
3. **Reverts are first-class commits**. The revert commit message
   captures the WHY so the next contributor doesn't try the same
   thing for the same reasons.

---

## §3 — k2x32 wider-row variant (null result, kept for posterity)

**Commit**: `f670e16` — "experiment(gemm): k2x32 wider-row lm_head —
null result"

**Hypothesis**: on the M=248320 lm_head kernel, a 32-row block
(versus the 16-row default) would halve block count and amortize
X-fragment loads across 2 WMMA issues per K-tile.

**Result**: 46% **slower** at the target shape. 1564 µs (k2 baseline)
→ 2280 µs (k2x32). Effective BW dropped from 446 GB/s to 307 GB/s.

**Root cause**: doubled accumulator (`float8_t × 2`) plus 4× dequant
live ranges pushed wave register pressure past the compiler's
budget, forcing spills or reducing effective occupancy. 310 GB/s
(32% of 960 peak) signals latency-bound, not BW-bound — more
parallel WMMAs don't help when you can't pipeline them.

**Why kept**: the kernel + `HIPFIRE_WO_WMMA_VARIANT=k2x32` env
override stayed in the tree even though auto-dispatch routes around
it. A future revisit with LDS-staged B-share + manual register
budgeting might unlock it. The negative result is a known-checkpoint
that future tuning passes don't have to re-discover.

**Lesson**: register pressure is the gating constraint past a
certain point. More parallel work does not help when the compiler
can't pipeline the issue chain. When you measure a kernel at
~30% peak BW and the obvious "do more" lever loses, the bottleneck
is latency, not BW — different lever class.

---

## §4 — gfx11 WMMA C-mapping silent corruption (caught only by channel-test)

**Commit**: `b7ac66a` — "wmma correctness fix + MQ6 family +
cross-arch prefill + gate framework"

**Setup**: gfx11 (RDNA3) WMMA was the WMMA workhorse for hipfire
since the v0.1.4 line. The C-output mapping
(`acc[j] = C[2*j + (tid>>4)][tid & 15]`) was silently wrong for
**~6 weeks**.

**How it stayed hidden**:
- All speed-gates passed — the kernel produced numbers, just wrong
  ones in the same ballpark.
- Coherence-gates passed — output was English-shaped, on-topic-ish,
  no panics or zero-tokens or attractor loops.
- Functional tests passed — comparing kernel output to itself
  doesn't catch a systematic mapping error.
- Real-model tok/s didn't regress noticeably — quality degradation
  was within "MQ4 is lossy by nature" range.

**How it got caught**: a channel-test that compared kernel output
**element by element against a CPU reference on synthetic
deterministic inputs** flagged a row-mod-16 pattern of mismatches.
The histogram diagnostic that landed in PR #56's gfx12 channel
tests is the tool that would have caught this in 30 seconds.

**Lessons**:
1. **Channel-test is the load-bearing correctness gate**, not
   speed-gate or coherence-gate. The other two are weaker signals
   that miss systematic errors.
2. **Per-lane mappings are silent-corruption magnets.** WMMA, MFMA,
   and any cooperative-thread reduction has implicit mapping
   conventions that you can get wrong without any obvious symptom.
3. **The row-mod-16 histogram diagnostic is reusable** — every
   future WMMA / MFMA channel-test should include it (it would
   have caught this in seconds).

The arch-port skill (`.skills/hipfire-arch-port/`) explicitly cites
this commit as the cautionary tale for new contributors. PR #56
followed that guidance and avoided the trap entirely on gfx12.

---

## §5 — 27B DFlash perf recovery (root-causing a real regression)

**Commit**: `9a2c667` — "perf-recovery: restore 27B DFlash perf +
flip prompt_normalize default ON + DFlash speed-gate"

**Setup**: 27B DFlash decode regressed 30-40% suddenly. Looked
catastrophic.

**Investigation path** (over 6 hours of bisecting):

1. Suspect rocBLAS — null. `HIPFIRE_ROCBLAS_OFF=1` made no difference.
2. Suspect DKMS / firmware — null. `dmesg` clean, kernel firmware
   versions matched.
3. Suspect mold / sccache — null. Clean rebuild reproduces.
4. Suspect DPM / thermal — null. `pp_dpm_sclk` looked normal.
5. **Found it**: prompt structure. A whitespace-cleanup edit to a
   bench script changed `\n\n\n` → `\n\n`. Same prompt by token
   count, totally different by token sequence. τ collapsed from
   9.42 to 8.07; tok/s from 199 to 161.

**Lessons** (now codified in CLAUDE.md and AGENTS.md):
1. **Prompt structure dictates τ.** One newline character can swing
   τ by 17%. Embed prompts as committed files, record prompt md5
   alongside results.
2. **Tight stddev on a spec-decode bench is suspicious, not
   reassuring.** The "before" measurement had tight stddev
   suggesting a deterministic attractor; real acceptance is wider.
3. **Bisect attribution is hard when the cause is in the test
   harness, not the engine.** Always reproduce the regression on
   a different prompt before deep-diving the engine.

The fix: implement engine-side `\n{3,}` → `\n\n` collapse default-on
(`prompt_normalize` config key, commit `9a2c667`). +24% τ on PEP-8-
style code prompts vs the opt-out path.

---

## §6 — wave64 residual gemv on MI300X (small win, BW-saturation ceiling)

**Commit**: this branch — "perf(cdna3): wave64 port of gemv_hfq4g256_residual"

**Bottleneck**: rocprof on 27B 3.6 mq4 decode (50 gen tokens, asym3 KV)
showed `gemv_hfq4g256_residual.kd` at 19.2% of GPU time — the largest
non-wave64 kernel after the 2026-04-17 (`4105035`) wave64 port. The
original commit ported 10 hot HFQ4 kernels but missed the residual
variants of gemv (`_residual` and `_wide`).

**Hypothesis**: wave64 port should give 1.5-2× per-call speedup
(matching the original commit's win on the same kernel family).

**Lever**: §1 wave-size port. New `gemv_hfq4g256_residual_wave64.hip`
with 2-rows-per-block layout (warp_id selects row, lane drives the
32-lane reduction unchanged). Dispatch routes via `has_wave64_native(arch)`.

**Numbers**:

```
27B 3.6 decode pre-port:  66.0 tok/s on MI300X (gfx942)
27B 3.6 decode post-port: 68.1 tok/s     (+3.2%, within noise)

per-call kernel time:
  pre-port:   28783 ns/call (single-row wave32 on wave64 hardware)
  post-port:  25222 ns/call (two-rows-per-block wave64)              -12.4%

A3B 3.6 decode pre/post: 194.6 → 198.0 tok/s (+1.7%, within noise)
```

**Why the small wall-clock delta despite -12% kernel time**: residual
gemv on this shape (M ~ 5120, K ~ 5120, single output row per warp32) is
**bandwidth-bound, not lane-bound**. Each row already saturates a wide
HBM3 read on MI300X regardless of wave size — the wave32 kernel was
issuing one coalesced 128-byte transaction every 32 packed-u32 weight
reads, and the new wave64 kernel pays the same BW for half the lanes.
The 12% per-call drop is real (less ALU pressure on the unused upper
lanes) but the wall-clock is dominated by the BW transfer, not the
compute pipeline.

**Lesson**: wave64 port wins biggest on kernels that are
**lane-utilization-bound** (multi-row fused projections like qkv, where
each lane has its own row-output to compute). On per-row gemv shapes
that are already BW-saturated, the win is incremental — ship it because
it's correctness-preserving and additive with future fusion work, but
don't expect 2× decode.

**Cross-arch**: gated by `has_wave64_native(&self.arch)`, so
gfx908/gfx940/gfx941/gfx942 only. RDNA archs unchanged. Speed-gate on
gfx1100 should pass byte-exact.

---

## §7 — LDS-staged X share on gate_up (null result, kept opt-in for posterity)

**Commit**: `feb16a1` — "experiment(gate_up): LDS-staged X share
variant — pp512 prefill -12% (null result)"

**Variant kernel**: `kernels/src/gemm_gate_up_hfq4g256_wmma_ldsx.hip`,
opt-in via `HIPFIRE_GATE_UP_VARIANT=ldsx`. Investigation tracked in
issue #60 (which has the v2 plan + three independent adversarial
reviews recorded in the comment thread).

**Bottleneck identified**: per-wave VMEM latency in front of WMMA B in
the baseline `gemm_gate_up_hfq4g256_wmma` inner loop. ISA dump shows
`s_waitcnt vmcnt(0)` immediately before the second WMMA each
K-tile-pair iteration — the compiler couldn't schedule enough
independent work to hide the b_b load latency.

**Lever attempted**: LDS-staged X share — the "unfinished follow-up"
from §3's k2x32 lessons. Cooperative global → LDS load once per
K-tile-pair, then ds_read into the WMMA B operand. Theory: replace
the ~50 cycle `vmcnt(0)` VMEM stall with a ~20 cycle `lgkmcnt`
LDS-read stall, hidden by the dequant work that already sits between
the X load and WMMA.

**Design**: per-K-tile-pair LDS slab (1 KB stages, 2 KB
double-buffered) — chosen specifically to *avoid* the occupancy
collapse the v1 design sketch would have hit at 16 KB per block.
Cooperative load mapping splits the 16 batches × 32 K-element tile
across all 32 lanes (each loads its own unique 16 fp16) to fix the
wave-redundancy where lanes 0-15 and lanes 16-31 currently re-read
the same X columns.

**Numbers** (Qwen 3.5 9B MQ4, gfx1100, ROCm 7.2,
`HIPFIRE_PROFILE=1 ... --warmup 5 --gen 0`):

| | baseline gate_up µs/call | LDSX gate_up µs/call | Δ per-call |
|---:|---:|---:|---:|
| pp32 | 261 | 314 | **+20.3%** |
| pp128 | 895 | 1157 | **+29.3%** |
| pp512 | 1760 | 2415 | **+37.2%** |

| | baseline prefill tok/s | LDSX prefill tok/s | Δ |
|---:|---:|---:|---:|
| pp32 | 598 | 492 | −17.7% |
| pp128 | 792 | 760 | −4.0% |
| pp512 | 1155 | 1012 | −12.4% |

Effective BW collapses 178 → 64 → 41 GiB/s on gate_up as batch grows.
The kernel is *more* memory-bound at large M than the baseline, not
less.

**Validation path**:

- **Gate 0 (ISA inspection) — PASSED.** `hipcc -save-temps -S -O3
  --offload-arch=gfx1100` showed: 75 VGPRs (down from baseline 80),
  no `s_barrier` emitted (single-wave-per-block elides
  `__syncthreads()`), weight load preserved at the top of the
  inner loop, ds_read followed by ~145 instructions of dequant
  before WMMA. All four Gate 0 criteria from the v2 plan
  satisfied.
- **Gate 1 (microbench) — FAILED.** Per-call wall time regressed
  at every pp size. Issue #60 thread documents the full per-pp
  breakdown plus the comparison vs the baseline ISA.

**Why ISA-clean still regressed**: the baseline inner loop has
2 VMEM stalls per K-tile-pair (`vmcnt(2)` before WMMA A,
`vmcnt(0)` before WMMA B). The LDSX inner loop has 3 VMEM stalls
+ 2 LGKM stalls per iteration (vmcnt waits in the LDS-store phase,
vmcnt before dequant for the weight load, and lgkmcnt waits before
each WMMA). More stall events, smaller individual latencies.
Critically, the baseline's `vmcnt(0)` was already partially hidden
by wave-level ILP (2 waves/SIMD baseline → wave scheduler swaps to
sibling wave during the stall), so eliminating it didn't free as
much wall time as the static analysis suggested. Meanwhile the
new LDS round-trip costs were paid in full.

**Why kept**: the kernel + dispatch arm stay in the tree
(default-off opt-in via `HIPFIRE_GATE_UP_VARIANT=ldsx`) so future
revisits — possibly on RDNA4 (gfx12 gains `s_prefetch_data`) or
with a fundamentally different LDS layout (e.g., LDS holds
*dequantized* weights instead of X, making the lever go after the
A-side load not the B-side) — don't have to rebuild the
infrastructure from scratch. Mirrors the §3 k2x32 disposition.

**Three reviews caught the headline issues, two of them caught
issues that turned out to be moot, one caught the issue that
mattered most**:

- All three (Claude, Gemini, GLM-5): "32× X-load reduction" framing
  was wrong on multiple axes. Confirmed empirically.
- Gemini §1 + GLM §2: occupancy collapse risk if LDS budget grows
  to 16 KB/block. **Moot** — v2 design used 2 KB and Gate 0 ISA
  showed VGPRs went *down* not up.
- GLM §4: `__syncthreads()` would force a pipeline flush and
  destroy compiler scheduling freedom. **Moot** — single-wave
  blocks elide the barrier entirely.
- GLM §2: ~50% of `vmcnt(0)` stalls are already hidden by
  wave-level ILP. **This was the load-bearing critique.**
  Eliminating already-hidden stalls is the textbook recipe for a
  null result, and that's exactly what we got — except worse,
  because we replaced them with new stalls the wave scheduler
  couldn't hide as well.

**Lesson**: ISA inspection alone is insufficient. A clean ISA
(no barrier emitted, weight prefetch preserved, VGPR budget healthy)
predicted a net win that didn't materialize on the bench. The
missing piece was wave-scheduler-level latency hiding, which is
invisible in static analysis but dominant in wall-time measurement.
**Always pair ISA inspection with cycle-counting microbench before
committing to a kernel rewrite.** And if a stall you're trying to
remove is in a kernel that's already running at meaningful
SIMD-utilization (≥20%), assume the wave scheduler is hiding
*some* of it — your ceiling is smaller than the per-iteration
cycle count suggests.

**For future revisit**: don't re-try this exact design. If you want
to attack the same `vmcnt(0)` stall with a different mechanism,
the candidates are:

- Pre-dequantize the A-side into LDS (move the lever to weights, not
  X — A is read-once-per-row, X is read-once-per-batch, but A's
  dequant work is what's currently filling the stall window —
  removing it changes the schedule).
- gfx12 `s_prefetch_data` (per `levers.md §5`) — different
  hardware, different tradeoffs.
- Restructure the inner loop to issue more independent WMMAs in
  parallel, giving the scheduler more work to hide individual
  stalls behind. This loops back to the K4 / K8 deeper-pipelining
  discussion that's currently blocked on K4's correctness bug.

## How to add a case study

If you land a real perf win or revert worth documenting, append a
new §N section here. Required fields:

- **Commit** — the canonical commit hash.
- **Bottleneck** — what the profile said.
- **Lever** — which entry from `levers.md` you used.
- **Numbers** — before / after with binary md5 + prompt md5.
- **Validation path** — which gates ran, what they showed.
- **Lesson** — the durable insight a future contributor needs.

Negative results (null lift, fake win caught) are equally
valuable — they save the next person from re-running the same
experiment. Don't omit them just because they "didn't ship."
