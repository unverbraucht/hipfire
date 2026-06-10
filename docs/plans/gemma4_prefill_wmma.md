# Gemma 4 WMMA Prefill — Phase 6 Milestone 1

**Date:** 2026-06-09 (revised after profiling)
**Branch:** `feat/dispatch-unification-gemma4` (tip `d1b1a488`)
**Goal:** Route gemma4 prefill projections through batched WMMA GEMM, closing the prefill performance gap.

## 0 · Profiled baseline

rocprofv3 kernel-trace on 12B-Q8, 20-token prompt, per-token decode:

| Category | Calls | Time (ms) | % |
|---|---|---|---|
| **GEMV/GEMM (projections)** | 9,212 | 1,629.5 | **93.6%** |
| Normalization (rmsnorm) | 9,436 | 52.6 | 3.0% |
| Attention (tile + reduce) | 2,688 | 29.1 | 1.7% |
| Memory (copy/fill) | 6,214 | 9.2 | 0.5% |
| Elementwise (scale/add/mul/gelu) | 8,092 | 9.0 | 0.5% |
| RoPE | 1,344 | 5.9 | 0.3% |
| KV cache write | 2,688 | 4.7 | 0.3% |
| Embedding | 28 | 0.3 | 0.0% |
| Logits (softcap) | 28 | 0.2 | 0.0% |

**Projections are 93.6% of GPU time.** Per-token GEMV is the bottleneck. Attention at 1.7% is negligible for short prefill. Batching projections through WMMA is the correct target.

Each `gemv_q8_0` call takes 177µs average (including launch overhead). 9.6 GEMVs per token-layer × 48 layers ≈ 460 GEMV launches per token. With B=20 tokens that's 9,212 total. WMMA batched GEMM reduces this to ~7 per layer × 48 layers = 336 total launches.

Full profile data: `findings/gemma4_prefill_profile_12b_q8.md`

## 1 · Bug fixes already landed

Three critical bugs were fixed before this plan's implementation phase:

### Bug 1 (CRITICAL): `gemm_hfq4g256_wmma` missing F32→F16 conversion

The GPU method took `x_f16` by name but never verified or performed F32→F16 conversion. Callers passing F32 data (e.g. via `GemmFamily` dispatch) would silently produce garbage — F32 bytes reinterpreted as F16.

**Fix (commit `d1b1a488`):** Added `ensure_fp16_x` conversion matching the `gemm_q8_0_wmma` pattern. If input is already F16, skips conversion. Also added `launch_maybe_blob` + `KernargBlob` for graph-capture compatibility and profiling timer.

### Bug 2 (CRITICAL): `GemmFamily::resolve` had no arm for `DType::MQ4G256`

The dispatch arm returned `UnsupportedVariant` for MQ4G256 weights, crashing on 26B-A4B production model.

**Fix (commit `d1b1a488`):** Added `DType::MQ4G256 → GemmHfq4G256Wmma / GemmHfq4G256` mapping. MQ4G256 shares the same 136-byte/group layout as HFQ4G256 — same kernel binary.

### Bug 3 (CRITICAL): WMMA results not byte-identical to scalar

F16 input quantization loses ~3 mantissa bits vs F32 scalar GEMV. Cannot be default-ON without relaxing coherence criteria.

**Fix (commit `d1b1a488`):** Added `HIPFIRE_WMMA_PREFILL` env var gate, default OFF. Set `HIPFIRE_WMMA_PREFILL=1` to opt in.

## 2 · Adversarial review findings

Three reviews were produced: self-review (`findings/gemma4_prefill_wmma_plan_rev_glm5.md`), Gemini (`findings/gemma4_prefill_wmma_plan_rev_gemini.md`), and Claude (`findings/gemma4_prefill_wmma_plan_rev_claude.md`). Cross-review consolidation in Appendix A of the self-review.

Key findings incorporated:

| # | Finding | Source | Status |
|---|---|---|---|
| 1 | F32→F16 bug in `gemm_hfq4g256_wmma` | Self + Claude | **Fixed** (Bug 1) |
| 2 | MQ4G256 not in `GemmFamily` | Self | **Fixed** (Bug 2) |
| 3 | WMMA not byte-identical — cannot default-ON | Self + Claude | **Fixed** (Bug 3) |
| 4 | Adapt v2, don't write greenfield | Claude C4 | **Accepted** — restructure Step 2 |
| 5 | Stale F16 cache across layers | Self + Gemini G3 | **Open** — use `convert_fp16_x_uncached` or invalidate per-layer |
| 6 | Add lm_head to prefill, eliminate redundant re-run | Gemini G6 | **Accepted** — add to Step 2 |
| 7 | Drop 26B-A4B from Milestone 1 success criteria | Claude C5 | **Accepted** — MoE dominates, small gain expected |
| 8 | Add gfx1100 correctness gate | Claude C8 | **Accepted** — add to validation |

## 3 · Architecture

### 3.1 Current flow (per-token decode reused for prefill)

```
for each prompt token:
  forward_scratch_inner_lowered():
    for each layer:
      Step::Gemv (q_proj)     ← single-token GEMV, 177µs each
      Step::Gemv (k_proj)
      Step::Gemv (v_proj)
      Step::Attend (kv_write + flash_attn)  ← per-token, works with q8 ring-buffer
      Step::Gemv (o_proj)
      Step::Gemv (gate_proj)
      Step::Gemv (up_proj)
      Step::Gemv (down_proj)
    final_norm + lm_head (Step::Gemv) + softcap
```

9.6 GEMVs per token-layer × 48 layers × B tokens = 460B total GEMV launches.
93.6% of GPU time in GEMV, 177µs average each (mostly launch overhead + latency).

### 3.2 Target flow (batched WMMA prefill)

```
forward_prefill_batch_wmma():
  embed all tokens → pb_residual [B, dim]

  for each layer:
    // Batched projections via WMMA (336 total, vs 460B GEMV)
    rmsnorm_batched(pb_residual, ...) → pb_tmp
    GemmFamily::run(q_proj, pb_tmp → pb_q)     ← WMMA [B×q_dim, K] GEMM
    GemmFamily::run(k_proj, pb_tmp → pb_k)     ← (F32→F16 conversion inside)
    GemmFamily::run(v_proj, pb_tmp → pb_v)
    rmsnorm_batched + rope_batched_f32 (batched proportional RoPE already exists)

    // Per-token attention (unchanged — works with q8 ring-buffer)
    for each token:
      Step::Attend(kv_write + flash_attn)

    GemmFamily::run(o_proj, pb_q → pb_attn_out)
    rmsnorm_batched + residual_add

    // Dense FFN (12B) or per-token MoE (26B-A4B)
    rmsnorm_batched
    GemmFamily::run(gate_proj, ...) → pb_gate
    GemmFamily::run(up_proj, ...) → pb_up
    gelu_tanh + mul
    GemmFamily::run(down_proj, ...) → pb_ffn_out
    rmsnorm + residual_add + layer_scalar

  final_norm + lm_head + softcap (on last token only)
```

**Key design choices:**
- **Batched GEMM for projections, per-token attention.** Profile data confirms 93.6% in projections, 1.7% in attention. Optimizing attention (Finding 5/13 in the original review) is premature — it's not the bottleneck.
- **Per-token attention preserved.** The q8 ring-buffer KV write and flash attention work correctly per-token. No need for batched attention (which would require ring-buffer-aware batched kernels). This avoids the v1/v2 q8 KV bug entirely.
- **`GemmFamily::run()` auto-selects WMMA.** On HasWmma archs (gfx1100+), resolves to WMMA variant. On older archs, falls back to scalar.
- **`HIPFIRE_WMMA_PREFILL=1` gate.** WMMA F16 quantization is not byte-identical to scalar F32. Must be explicitly opted into until coherence criteria are relaxed.
- **C4 recommendation: adapt v2, don't rewrite from scratch.** The existing `forward_prefill_batch_v2` (gemma4.rs:2612) already has all the structure. Fix its KvTierInputs bug, wire `run_prefill_gemm` → WMMA path, done.

### 3.3 What changes from the existing code

The existing `run_prefill_gemm` (gemma4.rs:39) already routes through `GemmFamily` dispatch. With `HIPFIRE_WMMA_PREFILL=1`, it calls `GemmFamily::run()` which resolves to WMMA. With the gate OFF, it uses the explicit scalar key mapping.

The existing `forward_prefill_batch_v2` (gemma4.rs:2612) already has batched `rmsnorm_batched`, batched projection routing, per-token attention, per-token expert loop for MoE. It needs:
1. KvTierInputs bug fix (hardcoded `quant_asym3: true` → read from cache)
2. Replace `run_prefill_gemm` calls with the WMMA-enabled version
3. Add final norm + lm_head (eliminate redundant last-token re-run)
4. Fix F32→F16 caching across layers (use `convert_fp16_x_uncached` or invalidate)

## 4 · Implementation plan

### Step 1 — Fix v2 KvTierInputs + wire WMMA (2 hours)

Fix the existing `forward_prefill_batch_v2`:
- Replace hardcoded `quant_asym3: true` / `quant_q8: false` with dynamic cache reads
- Replace `run_prefill_gemm` calls with the WMMA-enabled path (already works via `HIPFIRE_WMMA_PREFILL=1`)
- Validate: short prompt produces coherent output with v2 path

### Step 2 — Fix F16 cache + add lm_head (1 hour)

- Use `convert_fp16_x_uncached` for prefill F32→F16 (the pointer-keyed cache in `ensure_fp16_x` is wrong for reused activation buffers across layers — same pointer, different data)
- Add final norm + lm_head (G6 recommendation: eliminate redundant last-token re-run from daemon)

### Step 3 — Daemon wiring (1 hour)

- Wire `forward_prefill_batch_v2` into the daemon behind `HIPFIRE_WMMA_PREFILL=1`
- Set `PREFILL_BATCH_THRESHOLD = 16` (use per-token decode for prompts ≤15 tokens)
- Last-token logits from v2 itself (no daemon re-run)

### Step 4 — Coherence validation (1 hour)

1. 12B Q8, short prompt ("Capital of France?") — argmax must match scalar path
2. 12B Q8, long prompt (1266 tokens) — summary must be coherent
3. 12B Q8, WMMA vs scalar — first ~26 tokens identical, then small divergence (expected)
4. 26B-A4B MQ4+Q8, short prompt — coherent
5. gfx1100 correctness (C8 recommendation) — if hardware available

### Step 5 — Perf measurement (30 min)

Measure tok/s on gfx1151 for:
- 12B Q8 × 20-token prompt
- 12B Q8 × 1266-token prompt

Expected: 5–10× improvement on projection-dominated path. Actual speedup depends on launch overhead reduction and WMMA compute throughput.

## 5 · Risks

| Risk | Mitigation |
|---|---|
| Stale F16 cache serves wrong data across layers | Use `convert_fp16_x_uncached`; or invalidate `fp16_x_source_ptr` between layers |
| v2 has an unknown bug beyond KvTierInputs (v1 garbage root cause still unknown) | v1 calls `sliding_layer_decode_impl` dynamically; v2 hardcodes. Test thoroughly. If v2 still produces garbage after KvTierInputs fix, root-cause before shipping |
| Per-token attention loop is still slow for long prefill (B>512) | Accept for Milestone 1. B=128 per chunk is the initial target. Batched attention is Milestone 2+ |
| WMMA F16 quantization changes numerical results — small divergence after ~26 tokens | Expected. Documented as opt-in. `HIPFIRE_WMMA_PREFILL=1` required |
| 26B-A4B MoE per-expert loop dominates prefill | Expected (C5). MoE batching is separate work. 26B-A4B gain will be small |
| `GemmFamily::run()` for Q8_0 on gfx1151 resolves to `GemmQ8_0Wmma` — new for this arch | `gemm_q8_0_wmma` has `ensure_fp16_x` and `has_wmma()` assertion. Works on gfx1151 but untested for prefill specifically |

## 6 · Out of scope

- ~~**Batched attention for prefill**~~ — **NO LONGER out of scope** (as of 2026-06-09 this is Milestone 2 / §8, the load-bearing long-context lever; A.1/A.2 are the in-progress, currently-broken attempts — see "Implementation Log & Current Status" below). Note the original "per-token is correct and fast enough (1.7%)" framing was based on the 20-tok profile; at 1279 tok per-token attention is the ~15s wall-clock floor.
- **MoE prefill batching** — per-token expert loop is adequate; indexed kernels handle decode
- **v1/v2 batched prefill debug** — preserved on `feat/gemma4-batched-prefill-jukefr` for reference
- **Default-ON WMMA** — requires relaxed coherence criteria (byte-identical → within epsilon)
- **gfx1100 validation** — if hardware is unavailable, skip for Milestone 1

---

*Revised 2026-06-09 after profiling and bug fixes; **2026-06-10 handover update** appended below (see "Implementation Log & Current Status"). Profile data in `findings/gemma4_prefill_profile_12b_q8.md`. The standalone adversarial reviews (glm5, gemini, claude) have been folded into the "Implementation Log & Current Status" section and removed. Bug fixes in commit `d1b1a488`.*
## 8 · Measured perf results (2026-06-09)

### Short prompt (17 tokens, "What is France?")

| Path | Prefill time | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---|---|---|
| Per-token decode | 1041ms | 16.3 | 13.9 | 1041ms |
| Batched scalar | 876ms | 19.4 | 15.7 | 876ms |
| **WMMA batched** | **160ms** | **106.2** | **16.9** | **160ms** |

**6.5× prefill speedup for short prompts.** TTFT drops from 1.04s to 0.16s.

### Long prompt (1279 tokens)

| Path | Prefill time | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---|---|---|
| Per-token decode | 93,610ms | 13.7 | 10.6 | 93.6s |
| Batched scalar | 93,659ms | 13.7 | 10.6 | 93.7s |
| WMMA batched | 93,668ms | 13.7 | 10.5 | 93.7s |

**0× improvement.** Per-token attention dominates wall-clock time at long contexts.

### Root cause: GPU utilization

rocprof on 1279-token prompt:
- GPU compute: 26,836ms (GEMV 23,870 + attn 2,966)
- Wall time: 93,610ms
- **GPU utilization: 28.7%** — the CPU is the bottleneck

The per-token attention loop issues ~700K HIP operations (1279 tokens × 48 layers × ~12 calls each). The GPU is idle 71% of the time waiting for the CPU to stage the next dispatch. Batched GEMM helps projections but doesn't reduce the attention dispatch count.

### Revised milestone plan

**Milestone 1 (SHIPPED):** WMMA batched projections for short/medium prefill. 6.5× for ≤32 tokens, diminishing returns for longer contexts. `HIPFIRE_WMMA_PREFILL=1` and `HIPFIRE_BATCHED_PREFILL=1` gates.

**Milestone 2 (NEXT):** Batched attention for long-context prefill. This is the critical missing piece for 1279+ token contexts. Options:
- **a)** Batched q8 KV write + batched flash attention (new ring-buffer-aware kernels)
- **b)** CPU-side pipelining — overlap attention dispatch with GEMV computation
- **c)** CuDNN-style flash attention with batched inputs (leverage ROCm library)

Each approach needs ring-buffer cache_capacity support for q8 sliding KV.

**Milestone 3:** 26B-A4B MoE batched prefill (currently gated out due to `apply_moe_branch_batched` token attractor).

---

# Implementation Log & Current Status (2026-06-09 → 2026-06-10)

> This section folds in the former standalone reviews
> `gemma4_prefill_wmma_plan_rev_claude.md` (Claude/Opus dev log) and
> `gemma4_prefill_wmma_plan_rev_gemini.md` (Gemini). Those files have been
> removed; this is the single authoritative record. All numbers gfx1151,
> 12B-Q8, canonical committed prompt `benchmarks/prompts/gemma4_longcontext_1200.txt`
> (1279 tok, md5 `6236cc470a2eefe9c1b34913f9fa9ea6`), warm-then-measure, fresh
> process per measure.

## ⚠ HANDOVER STATUS (2026-06-10): the batched-attention path is BROKEN

**Two coupled defects in the opt-in batched prefill path. Both opt-in flags
(`HIPFIRE_BATCHED_PREFILL`, `HIPFIRE_WMMA_PREFILL`) default OFF, so the default
per-token prefill path is correct and unaffected — this is a broken
experimental path, not a default-user regression.**

1. **A.1 (`aa66352f`, PUSHED UPSTREAM) corrupts at multi-chunk.** Batched
   full-layer (asym3) attention is wrong at `start_pos > 0`. It was validated
   only at 160 tok = a single chunk (`start_pos = 0`), which masked the bug.
   On any prompt > 128 tok under `HIPFIRE_BATCHED_PREFILL=1` it emits a
   `(No)(No)(No)…` attractor. **→ Worth notifying the maintainer (Kaden).**
2. **A.2 (`d5fc03d9`, WIP, known-broken) corrupts even single-chunk.** Batched
   q8 *sliding* attention produces `<audio|>****…` on a 35-tok single chunk —
   an additional defect on top of (1).

**Three-way control (identical binary, identical 1279-tok prompt):**

| Config | Prefill path | Result |
|---|---|---|
| default (no flags) | per-token `forward_scratch` | **COHERENT** ✓ |
| `HIPFIRE_GEMMA4_NO_SLIDING_BATCH=1` (= `aa66352f`) | per-token sliding + batched full | **GARBAGE** ✗ `(No)(No)` |
| `HIPFIRE_BATCHED_PREFILL=1` (A.2) | batched sliding + batched full | **GARBAGE** ✗ `<audio\|>` / `the the` |

The default path being coherent on the identical prompt isolates the
corruption to the **batched attention at `start_pos > 0`** (the only delta
between rows 1 and 2 is that full-layer attention is batched).

**Repro:** `HIPFIRE_BATCHED_PREFILL=1` + any prompt > 128 tok → attractor;
same prompt, no flag → coherent.

## Commits (on `feat/dispatch-unification-gemma4`, atop `d1b1a488`)

| commit | what | status |
|---|---|---|
| `ed6e68c3` | window/cap threading completion — made the tree build (3 missed call sites) | ok |
| `5e927530` | **chunking** — >128-tok prompts route through batched prefill in ≤128 windows | ok (per-token attn) |
| `bf44af1f` | 4th threading call site (q8 microbench example) | ok |
| `aa66352f` | **A.1** — batched full-layer (asym3) prefill attention, 1 launch/layer | **BROKEN @ multi-chunk** |
| `d5fc03d9` | **A.2** — batched q8 sliding attention (kernel port + wiring) | **KNOWN-BROKEN, do not ship** |

## Measured prefill (1279 tok)

| path | prefill | speedup | note |
|---|---|---|---|
| per-token baseline (default) | 92.8s | 1.00× | coherent |
| scalar-chunked (`HIPFIRE_BATCHED_PREFILL=1`, pre-A.1) | 69.3s | 1.34× | coherent — per-token attention, batched projections |
| + A.1 batched full @ **single chunk** (160 tok) | 68.0s | — | coherent at 160 tok ONLY; multi-chunk now known broken |
| + A.1 batched full @ **multi-chunk** (1279 tok) | 68.1s | — | ❌ `(No)(No)` attractor |
| wmma + kv-scalar | 19.1s | 4.86× | ❌ block attractor @ ~tok 30 |
| wmma full-F16 | 14.8s | 6.25× | ❌ attractor @ ~tok 12 |

## Established findings (each falsified the prior cheaper hypothesis)

1. **Sync-stall theory FALSIFIED.** Env-gated `active_stream` A/B gave **−7.5%**,
   not the predicted ~3×. The long-prefill bottleneck is CPU op-**submission**
   (~95 µs/op × ~700K ops ≈ 66.8s of the 66.8s idle), not synchronous-copy
   stalls (corroborated by ~200% daemon CPU during prefill). Async copies don't
   change the submit count → can't help. Reverted. The lever is **op-count
   reduction** (chunking + graph capture), not de-syncing.
2. **Chunking is the correct fix for submission cost** (`5e927530`) — collapses
   ~588K per-token GEMV launches into a few thousand batched GEMMs. Proven
   coherent across all 10 chunks of the 1279-tok prompt with **per-token**
   attention.
3. **WMMA F16 is not coherence-safe for long context** — corrupts every
   prefilled KV entry AND accumulates in the residual stream (o_proj/down_proj
   write F16 error into the residual every layer). K/V-scalar alone is
   INSUFFICIENT (delays the attractor tok 12→30, still block-loops at length).
   Coherent WMMA needs split-F16 (compensated activation staging) — deferred.
   *Process note: a 24-tok eyeball passed; the 160-tok check caught the block
   attractor. Always validate at length.*
4. **Per-token attention (~15s) is the floor** bounding every config and the
   real gap vs llama.cpp (~0.33s / 3925 tok/s). Batching it is the real lever
   (Milestone 2) — but is exactly where the current corruption lives.

## Ring-buffer hazard (constrains A.2 design — important)

The sliding KV is a **q8 ring of exactly `sliding_window` (1024) physical
slots** (`new_gpu_q8_capped`, `physical_cap = sliding_window`). Positions ≥1024
MUST wrap. In a batched "write-all-then-attend-all", a later token at pos `p'`
overwrites ring slot `p' % 1024`, which held position `p'−1024`. For
**window == cap == 1024**, that old position is always inside an earlier
token's window `[p−1023, p]` → corruption. So a batch is ring-safe **iff every
position < cap** (`start_pos + n_batch ≤ sliding_window`). Past that boundary,
window==cap forces size-1 safe batches (per-token). A.2 therefore batches only
the no-wrap regime and falls back to per-token for the wrapped tail. This is a
write/attend **ordering** constraint a masking change cannot fix. (NB: this
hazard is independent of, and does not explain, the `start_pos>0` and
single-chunk corruption above — those are real bugs in the batched path
itself.)

## Root-cause status (open) — for the next owner

- The corruption is **`start_pos > 0`-specific** for the full (asym3) path, and
  **even `start_pos == 0`** for the q8 sliding path → likely two distinct bugs,
  possibly a shared positions/offset root.
- **Reduce kernel ruled out** (`attention_flash_asym_reduce_batched.hip`): it
  iterates per-query `n_tiles = ceil(seq_len/tile)` and guards stale tiles with
  `tile_sum > 0`; the tile kernel writes every tile in `[0, n_tiles)` for full
  layers (window=0). Not stale-partials.
- `pb_positions` confirmed absolute (`start_pos + i`, `gemma4.rs:2692`).
- **Next diagnostic:** dump-and-diff the batched full-layer attention output
  vs. the per-token reference at layer 0 for `start_pos = 128` (chunk 1) — the
  smallest multi-chunk case — to localize the `start_pos>0` divergence. Then
  separately diff batched-q8-sliding vs per-token at `start_pos = 0`.
- Validation MUST be length-gated (>1024-tok prompt) through
  `coherence-gate.sh` + `coherence-gate-dflash.sh` before any A.x is called
  coherent — the failure mode is a block attractor past the window edge that a
  short smoke test misses.

## Isolation toggle (committed, for debugging)

`HIPFIRE_GEMMA4_NO_SLIDING_BATCH=1` forces the sliding layers back to per-token
while keeping A.1's batched full layers — i.e. reproduces `aa66352f` exactly.
Used for the three-way control above.
