# Gemma 4 branch merge into master — 2026-06-04

**Scope:** Merge `jukefr/gemma4-128k-ring-buffer` (62 unique commits,
e0381119…08e89542) into `upstream/master` (02634f4c). Record branch
selection rationale, conflict resolution decisions, and findings for
future work.

## Branch selection

Four Gemma 4 branches existed at triage time:

| Branch | Unique commits | Behind master | Crate layout | Conflicts into master |
|--------|---------------|---------------|-------------|----------------------|
| `upstream/gemma4` (Kaden) | 53 | 849 | ❌ old `crates/engine/` monolith | 19 (6 modify/delete, 5 file-location renames) |
| `upstream/gemma4-rebased-2026-05-07` (Kaden) | 21 | 790 | ✅ modular | 10 (content only) |
| `jukefr/gemma4-rebased-2026-05-18` | 29 | 724 | ✅ modular | 9 (content only) |
| `jukefr/gemma4-128k-ring-buffer` | 62 | 585 | ✅ modular | 14 (content only) |

Selected **`jukefr/gemma4-128k-ring-buffer`** because:

1. Strict superset of Kaden's rebased branch (15/22 patch-ids shared;
   the remaining 7 are content-equivalent rebases onto a newer master).
2. Strict superset of `jukefr/gemma4-rebased-2026-05-18` (28/30
   patch-ids shared).
3. Contains the MoE forward pass (433 refs), daemon/serve wiring (69
   refs), ring-buffer KV cache (13 refs), hipGraph capture (7 refs),
   and batched prefill — none of which exist in the other three branches
   at this fidelity.
4. Closest to master (585 behind vs 790+ for the others), minimizing
   merge surface.
5. Only 14 content-only conflicts vs the 19 structural conflicts (modify/delete + file renames) that `upstream/gemma4` would require.

Kaden's `upstream/gemma4` has **zero patch-id overlap** with all other
branches — its content was manually ported into the modular layout by
jukefr. It remains a useful reference for the original MoE work but is
not a viable merge candidate.

## Conflicts resolved (14 files)

### Kernel parameter additions (3 files)

The four `attention_flash_asym*_tile_batched.hip` kernels got
parameter additions from both sides:

| Kernel | HEAD (jukefr) added | master added | Resolution |
|--------|---------------------|--------------|------------|
| `asym2_tile_batched` | `int window_size` | `int v_mode` | Keep both: `v_mode, window_size` |
| `asym3_tile_batched` | `int window_size, int cache_capacity` | `int v_mode` | Keep all three: `v_mode, window_size, cache_capacity` |
| `asym4_tile_batched` | `int window_size` | `int v_mode` | Keep both: `v_mode, window_size` |

`v_mode` is declared but unused in the batched kernels (API consistency
with the non-batched variants). `window_size` and `cache_capacity` are
actively used in the Gemma 4 sliding-window + ring-buffer attention
paths.

### Workspace Cargo.toml (1 file)

Both branches added an arch crate to the workspace members list:
- HEAD: `hipfire-arch-gemma4`
- master: `hipfire-arch-deepseek4`

Resolution: keep both.

### cli/index.ts (1 file)

HEAD had a no-op comment (`// prefill_progress is a no-op on the
non-streaming path`). Master added `tool_calls` structured output
handling (parsing `msg.type === "tool_calls"` into OpenAI-compatible
function call format). Resolution: take master's version (actual
functionality).

### hipfire-arch-qwen35/speculative.rs (1 file)

HEAD added a trailing `0` argument to a `kv_compact_gather` call (new
`budget` parameter). Master reformatted the existing arguments.
Resolution: keep HEAD's extra `0` with master's formatting.

### rdna-compute/kernels.rs (1 file)

HEAD added three Gemma 4 kernel constants (`ROPE_PARTIAL_HALVED_SRC`,
`LOGIT_SOFTCAP_SRC`, plus the existing `ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC`).
Master reformatted the halfsplit constant with line breaks.
Resolution: take master's formatting for halfsplit, keep HEAD's three
additions.

### Remaining large conflicts (6 files)

These have multi-hundred-line conflict regions and need careful
section-by-section resolution:

| File | Conflict markers | Region sizes |
|------|-----------------|--------------|
| `Cargo.lock` | 12 | auto-generated |
| `crates/hipfire-quantize/src/main.rs` | 18 | quant format additions |
| `crates/hipfire-runtime/Cargo.toml` | 12 | dependency additions |
| `crates/hipfire-runtime/examples/daemon.rs` | 30 | Gemma 4 daemon wiring vs master features |
| `crates/hipfire-runtime/src/llama.rs` | 24 | loader/model additions |
| `crates/rdna-compute/src/dispatch.rs` | 15 | dispatch function additions |

**Status: in progress.**

## Findings for future work

### 1. DFlash WMMA / GQA kernels from dots.ocr don't replace asym batched

Master gained several new attention kernel families from the dots.ocr
and qwen2 work:

| Kernel family | head_dim | KV format | Sliding window |
|---|---|---|---|
| `attention_dflash_wmma_*` | ≤128 (most hard-coded) | F32/F16 flat | No |
| `attention_flash_gqa_fused` | flexible | F32 flat | No |
| `attention_gqa_warp` | 128 only | F32 flat | No |

Gemma 4 needs `head_dim=512` (full-attention layers), quantized KV
(`asym2/3/4` with Givens rotation), sliding-window + ring-buffer
support. None of the new kernels satisfy these constraints. The
existing `attention_flash_asym*_tile_batched.hip` family remains the
correct choice.

**Future optimization opportunity:** The **GQA warp-cooperative**
pattern (coalesced K-access via warp-shuffle-reduce) could inspire a
future variant of the asym decode path. Gemma 4 has heavy GQA
(32:16 sliding, 32:4 full). Currently the asym kernels do
one-head-per-block; a warp-cooperative GQA variant that dequants the
shared K/V once for multiple query heads would be a meaningful decode
win. This is a kernel-authoring task, not a wiring task.

### 2. HFQ4G128 WMMA sibling missing

`gemv_hfq4g128_moe_down_*` and `gemm_hfq4g128` are wave32 GEMV/GEMM
shaped only. No WMMA prefill sibling exists (unlike HFQ4G256 which has
`gemm_qkv_hfq4g256_wmma.gfx12.hip`). On gfx1201, Gemma 4 26B-A4B's
`down_proj` (HFQ4G128) prefill goes through wave32 GEMV even with
batching. A `gemm_hfq4g128_*_wmma.gfx12.hip` sibling would likely pull
another 2–4× on the down-projection prefill share. Noted in jukefr's
original evidence-debt doc.

### 3. Batched prefill v2 regression

Per jukefr's evidence-debt doc: the v2 batched-prefill path
(`forward_prefill_batch_v2`, commit `521161f8`) has a known token
attractor regression. The root cause was identified and fixed
(missed ceil(K/128) in `gemm_hfq4g128.hip`), achieving 180 tok/s
prefill. However the branch currently routes to v1 by default (109
tok/s) as a conservative measure. The v2 fix should be validated
post-merge before re-enabling.

---

*Last updated: 2026-06-04. Update this file as conflicts are resolved
and new findings emerge during the merge.*

## dispatch.rs strategy pivot (2026-06-04)

Master refactored the monolithic `dispatch.rs` (~1973 lines on master) into
separate modules:

| Module | Lines | Contents |
|--------|-------|----------|
| `attention.rs` | 9481 | All flash attention dispatch (asym2/3/4/fwht, q8, reduce, dflash) |
| `gemm.rs` | 19406 | Batched GEMM variants (hfq4, mq4, hfq6, paro4, WMMA, wave64) |
| `gemv.rs` | 7158 | Single-token GEMV + fused projections |
| `moe.rs` | 877 | MoE dispatch helpers |
| `norm.rs` | 2403 | rmsnorm, layernorm, etc. |
| `embedding.rs` | — | embedding lookups |
| `dispatch.rs` | 1973 | Core Gpu struct, init, precompile, profile |

Jukefr's branch has everything in one 18700-line `dispatch.rs`. Attempting
to merge these two structures produces 329+ compile errors from mismatched
function bodies and signatures.

### Correct approach: use master's modules, port gemma4 functions

14 functions from jukefr's branch need to be added to master's modules:

**Attention (`attention.rs`):**
- `attention_flash_asym2_window` — sliding-window asym2 attention
- `attention_flash_asym3_window` — sliding-window asym3 attention (decode)
- `attention_flash_asym3_batched_window` — sliding-window asym3 (prefill)
- `attention_flash_asym4_window` — sliding-window asym4 attention
- `attention_flash_q8_0_window` — sliding-window q8 attention

**GEMV/MoE (`gemv.rs` or `moe.rs`):**
- `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed` — HFQ4G128 MoE down
- `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched` — batched variant
- `gemv_hfq4g128_moe_down_residual_scaled_bucketed` — bucketed routing variant
- `gemv_q8_0_moe_down_residual_scaled_k8_indexed` — Q8 MoE down
- `gemv_hfq4g256_moe_gate_up_bucketed` — bucketed gate/up
- `gemv_mq4g256_moe_gate_up_k8_indexed` — indexed gate/up
- `moe_bucket_build` — routing bucket construction

**Other:**
- `logit_softcap_f32` → `norm.rs`
- `rope_partial_halved_f32` → `attention.rs` or new rope module

This port is a separate step from the conflict resolution. The current
branch state has dispatch.rs reset to master's version — the gemma4
dispatch functions need to be cherry-picked from jukefr's branch into
the appropriate module files.

## Function port completion (2026-06-04)

All 14 Gemma 4 dispatch functions have been ported from jukefr's
monolithic `dispatch.rs` into master's modular layout:

| Function | Target module | Status |
|---|---|---|
| `attention_flash_asym2_window` | attention.rs | ✅ |
| `attention_flash_asym3_window` | attention.rs | ✅ |
| `attention_flash_asym3_batched_window` | attention.rs | ✅ |
| `attention_flash_asym4_window` | attention.rs | ✅ |
| `attention_flash_q8_0_window` | attention.rs | ✅ |
| `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed` | gemv.rs | ✅ |
| `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched` | gemv.rs | ✅ |
| `gemv_hfq4g128_moe_down_residual_scaled_bucketed` | gemv.rs | ✅ |
| `gemv_q8_0_moe_down_residual_scaled_k8_indexed` | gemv.rs | ✅ |
| `gemv_hfq4g256_moe_gate_up_bucketed` | gemv.rs | ✅ |
| `gemv_mq4g256_moe_gate_up_k8_indexed` | gemv.rs | ✅ |
| `gemv_mq4g256_moe_gate_up_bucketed` | gemv.rs | ✅ (thin wrapper) |
| `moe_bucket_build` | moe.rs | ✅ |
| `logit_softcap_f32` | norm.rs | ✅ |
| `rope_partial_halved_f32` | attention.rs | ✅ |

### Additional changes required for compilation

1. **`launch_asym_flash_batched` signature extended** with `window_size: u32` and
   `cache_capacity: u32` params — threaded through kernel launch params and blob builders.
   All existing callers (qwen35 etc.) pass `0, 0` for these.

2. **`kv_cache_write_asym3_fused` and `kv_cache_write_asym3_batched`** gained
   `cache_capacity: u32` param for ring-buffer KV support. All existing callers pass `0`.

3. **`kv_cache_write_q8_0` and `kv_cache_write_q8_0_batched`** gained `cache_capacity: u32`
   param (added during initial merge). All existing callers pass `0`.

4. **`GraphState.ar_forward_warmed_up`** field added to `graph.rs` + initialized to `false`
   in `dispatch.rs` constructor — needed by Gemma 4 graph capture warmup.

5. **`QuantType::MG4G256`** discriminant changed from `30` to `25` to avoid collision with
   `MQ4G256Lloyd = 30` (master's addition). Engine treats MG4G256 as MQ4G256 at load time
   regardless of discriminant value.

6. **Graph API migration** in `gemma4.rs`: `gpu.begin_graph_capture()` →
   `gpu.graphs.begin_graph_capture(&gpu.hip, gpu.device_id, ...)` etc.
   Master moved graph state into `GraphState` struct accessed via `gpu.graphs`.

7. **`capture_mode` field access** in window attention functions changed from
   `self.capture_mode` to `self.graphs.capture_mode` (master's modular layout).

### Build status

- `cargo check --all-features` — **0 errors**, warnings only
- `cargo check -p hipfire-runtime --features "arch-qwen35 arch-gemma4 deltanet"` — **0 errors**
- `cargo check -p hipfire-quantize` — **0 errors**

### Remaining issues (not blocking compilation)

1. **daemon.rs `arch_id=7` conflict** — Both Gemma 4 and Qwen2 map to `arch_id=7`.
   Needs human decision on ID assignment or dispatch restructuring.

2. **Gemma 4 `gemma4.rs` calls `kv_cache_write_asym3_fused` with a `cache_capacity`
   arg** but the function's kernel launch params don't actually pass `cache_capacity` to
   the HIP kernel yet — the kernel itself needs a matching parameter. Currently a no-op
   passthrough for the ring-buffer path.

3. **`gemv_mq4g256_moe_gate_up_bucketed`** is a thin wrapper that delegates to
   `gemv_hfq4g256_moe_gate_up_bucketed` (MQ4 and HFQ4 share the same 136 B/group binary).
   If Gemma 4 later diverges, this will need a dedicated implementation.

