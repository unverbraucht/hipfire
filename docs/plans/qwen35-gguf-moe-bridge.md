# GGUF → hipfire-arch-qwen35 MoE bridge — known issues and follow-up plan

**Date:** 2026-05-10
**Status:** Investigation. Patches exist on `experiment/rmsnorm-fix-plus-pr214` (worktree `/tmp/hipfire-pr214-test`); not shipped.
**Trigger:** Attempted to validate PR #214 (K-map alternating mode) on Qwen3.6-35B-A3B by re-quantizing from a local BF16 GGUF source. Discovered the GGUF→hipfire-arch-qwen35-MoE pipeline is incomplete; pivoted to the safetensors path. This doc captures what was found so a future PR can finish the bridge.

## Patches that exist (in the worktree branch)

1. **Multi-shard GGUF loader** (`crates/hipfire-quantize/src/gguf_input.rs`)
   - Added `shard_idx` field to `TensorInfo`.
   - `GgufFile::open` detects `split.count` / `split.no` metadata, enumerates sibling shards by filename pattern (`-NNNNN-of-MMMMM.gguf`), opens each, builds a unified tensor list dispatched per-shard.
   - Fixed `MetaValue::as_u32` to also accept `U16/I16/U8/I8/I64` (unsloth GGUFs use `U16` for `split.no`/`split.count`).
   - Sibling-path computation is filename-based (regex-free), 1-based shard numbering per llama.cpp convention.
   - **Status: works empirically** — verified picks up both shards of Qwen3.6-35B-A3B-BF16, sees all 733 tensors.

2. **`qwen35moe` arch_id mapping** (`crates/hipfire-quantize/src/main.rs:2048`)
   - Added `"qwen3moe" | "qwen35moe" => 6` so newer-family GGUFs are tagged as MoE (arch_id=6) instead of falling through to llama (arch_id=0).

3. **3D MoE expert split pre-pass** (`crates/hipfire-quantize/src/main.rs:2129+`, ~210 lines)
   - Mirrors the safetensors-path's per-expert split (line ~2620).
   - Detects `blk.{N}.ffn_{gate,up,down}_exps.weight`, slices each per expert.
   - Fuses gate || up into `gate_up_proj` (matching vLLM's `chunk(2, dim=-2)` reading at `qwen3_5_mtp.py:264`).
   - Outputs per-expert tensors with engine-expected names and shapes:
     - `model.layers.{N}.mlp.experts.{X}.gate_up_proj.weight` shape `[2*mi, hidden]`
     - `model.layers.{N}.mlp.experts.{X}.down_proj.weight` shape `[hidden, mi]`
   - Per-expert quant choice respects K-map (Promote6 → MQ6G256).
   - Parallel via rayon.
   - **Status: works empirically** — output for layer 0 matches expected per-expert shapes, K-map alternating promotes layers 0/1/2/5/8/11/... as expected.

4. **GemmaRMSNorm `+1` fixup** (`crates/hipfire-quantize/src/main.rs:2061+`)
   - llama.cpp's `convert_hf_to_gguf.py:Qwen3NextModel.modify_tensors` adds `+1.0` to all norm weights at HF→GGUF conversion. Hipfire's `load_norm_weight` at `qwen35.rs:697` adds another `+= 1.0` at engine load time. Without compensation: double-bake.
   - The patch subtracts `1.0` from norm tensors when `arch_str == "qwen35moe" | "qwen35"`, so the .hfq matches the safetensors-derived convention.
   - **Status: applied but UNTESTED end-to-end** because the engine load fails before reaching forward (see issues below).

## Issues blocking a working GGUF→qwen35 MoE pipeline

### Issue A: missing tensor-name remappings — BLOCKS ENGINE LOAD

`gguf_to_safetensors_name` at `main.rs:1775` translates a fixed set of slot names. The Qwen3.5+ MoE architecture uses several names not in that map. After conversion, the resulting `.hfq` contains tensors at the wrong names, and `load_weight_tensor` at `qwen35.rs` panics on the first missing tensor at load time.

**Concrete missing translations:**

| GGUF name (raw) | Engine expects |
|---|---|
| `blk.{N}.ffn_gate_inp.weight` | `model.layers.{N}.mlp.gate.weight` (router) |
| `blk.{N}.ffn_gate_inp_shexp.weight` | `model.layers.{N}.mlp.shared_expert_gate.weight` |
| `blk.{N}.ffn_gate_shexp.weight` | `model.layers.{N}.mlp.shared_expert.gate_proj.weight` |
| `blk.{N}.ffn_up_shexp.weight` | `model.layers.{N}.mlp.shared_expert.up_proj.weight` |
| `blk.{N}.ffn_down_shexp.weight` | `model.layers.{N}.mlp.shared_expert.down_proj.weight` |
| `blk.{N}.attn_qkv.weight` | (FUSED in GGUF) → split into `model.layers.{N}.self_attn.{q,k,v}_proj.weight` |
| `blk.{N}.attn_gate.weight` | unclear — may be MoE-attention specific gate, not handled by current arch crate |
| `blk.{N}.attn_output.weight` | already mapped (`self_attn.o_proj`) |
| `blk.{N}.post_attention_norm.weight` | already maps to `post_attention_layernorm` (or close) |
| DeltaNet: `blk.{N}.ssm_a` | `?` |
| DeltaNet: `blk.{N}.ssm_alpha.weight` | `model.layers.{N}.linear_attn.in_proj_a.weight`? |
| DeltaNet: `blk.{N}.ssm_beta.weight` | `linear_attn.in_proj_b.weight`? |
| DeltaNet: `blk.{N}.ssm_conv1d.weight` | `linear_attn.conv1d.weight`? |
| DeltaNet: `blk.{N}.ssm_dt.bias` | ? |
| DeltaNet: `blk.{N}.ssm_norm.weight` | `linear_attn.norm.weight` |
| DeltaNet: `blk.{N}.ssm_out.weight` | `linear_attn.out_proj.weight`? |

**Action:** extend `gguf_to_safetensors_name` and add a separate "split fused" pass for `attn_qkv` similar to the MoE expert split. The DeltaNet mapping needs cross-referencing against the safetensors source to confirm.

### Issue B: `attn_qkv` is fused in GGUF, split in HF — needs same treatment as MoE experts

GGUF's `attn_qkv.weight` shape `[hidden, q_dim + 2*kv_dim]` (or similar — depends on Qwen3.5 MoE attention head config). Engine expects three separate tensors for q/k/v. Need to split.

This is a different shape calculation than gate||up: the Q/K/V dims are not equal (GQA: typically `n_q_heads * head_dim` for Q, `n_kv_heads * head_dim` for K and V). The split point is at `q_dim` and `q_dim + kv_dim`.

For Qwen3.6-35B-A3B per the GGUF metadata: `attn_qkv` is `[2048, 8192]` per the conversion log. With Qwen3.6-A3B's hidden=2048, n_q_heads=32, n_kv_heads=4, head_dim=128:
- Q dim = 32 * 128 = 4096
- K dim = 4 * 128 = 512
- V dim = 4 * 128 = 512
- Total fused = 5120 ≠ 8192. Doesn't match.

Maybe GQA config is different on A3B, or the hidden output of the qkv proj includes attention bias / other channels. Need to read the model config more carefully before splitting.

**Action:** read `arch_str=qwen35moe` config keys (`{arch}.attention.head_count`, `{arch}.attention.head_count_kv`, `{arch}.attention.key_length`, etc.) and compute split points.

### Issue C: norm fixup gating is too coarse

The patch gates the `subtract 1.0` on `arch_str == "qwen35moe" | "qwen35"`. Two problems:

1. **`qwen35` (without `moe`) doesn't exist as a llama.cpp arch tag.** The dense Qwen3.5/3.6 uses `qwen3` per the existing `match` in `gguf_to_safetensors_name`. The `qwen35` literal was a guess. Either remove (and rely solely on `qwen35moe`) or verify against llama.cpp PRs that introduced Qwen3.5 GGUF support.
2. **DeltaNet's `linear_attn.norm.weight` is RMSNormGated, NOT GemmaRMSNorm.** llama.cpp's converter does NOT add +1 to `linear_attn.norm.weight` (it's not in the list of GemmaRMSNorm tensors). My fixup naively subtracts from every norm — would corrupt these.

   GGUF tensor name for DeltaNet norm is `blk.{N}.ssm_norm.weight` (per the conversion log). The fix: scope the subtract to NOT include `ssm_norm`. A cleaner predicate: `name.contains("norm") && !name.contains("ssm_norm") && !name.contains("linear_attn.norm")`.

### Issue D: K-map plan summary undercounts experts

The K-map summary printed before quant ("F16: 161 / Q8: 2 / Promote6: 81 / Base: 489") counts each 3D expert tensor as one entry. After per-expert split, the actual quant-distribution is closer to:
- Promote6: 81 + (n_promoted_layers × 256 experts × 2 [gate_up + down]) = 81 + 16 × 256 × 2 = 8273 tensors
- Base: 489 - 120 (3D expert tensors removed) + (n_base_layers × 256 × 2)

Cosmetic only. The actual quant choices are correct.

### Issue E: total_params accounting double-counts

The expert pre-pass adds to `total_params` once for the produced output (per-expert tensors), and the main loop also adds to `total_params` for non-expert tensors. The 3D source tensors are skipped in the main loop, so input bytes are also undercounted. The summary line "Input size: 4941 MB" looks far too small for a 35B model. **Cosmetic, but should be fixed for correctness in error messages.**

### Issue F: Memory / spilling for large MoE

The expert pre-pass holds `hfq_tensors` in memory across all 40 layers × 256 experts = 20,480 tensors at ~1.5 MB each = ~30 GB. The user's gfx906 dev box has 30 GB system RAM. Empirically the 35B-A3B run fit (just), but it's tight.

The safetensors path uses `maybe_spill` to flush to a temp file at 2 GB threshold. The expert pre-pass does NOT call `maybe_spill`. Adding it would prevent OOM on larger models or under memory pressure.

### Issue G: GgufFile::open requires shard 1 specifically

If a user passes `*-00002-of-00002.gguf`, `open()` returns an error telling them to pass shard 1. Reasonable but rough UX. Could auto-redirect to shard 1 by swapping the filename's NNNNN field.

### Issue H: Worktree only — patches not on master

All of the above lives in `experiment/rmsnorm-fix-plus-pr214` worktree at `/tmp/hipfire-pr214-test`. The branch contains the rmsnorm fix (already shipping in `fix/qwen35-moe-final-norm`) plus PR-214 cherrypicks plus the GGUF patches. Need to extract the GGUF patches into a clean follow-up PR off master once Issues A-F are resolved.

## Suggested follow-up PR sequence

1. **Issue A first** — add tensor-name remappings. Simplest: extend `gguf_to_safetensors_name` for the listed slots. Verify by re-running the quantizer and checking with `compare_hfq` against a safetensors-derived reference.

2. **Issue B** — `attn_qkv` split. Pre-pass like the MoE expert split, but using config-derived q/k/v dims. Test against same safetensors reference.

3. **Issue C** — narrow the norm fixup predicate. Add explicit exclude list for `ssm_norm` / DeltaNet-style names.

4. **`compare_hfq` validation** — for each model variant we care about (3.5-9B dense, 3.5-27B dense, 3.6-A3B MoE), produce both safetensors-derived and GGUF-derived `.hfq`, run `compare_hfq` to confirm NRMSE < 1e-3 per tensor. Sanity check that quantization is byte-identical when sources are equivalent (after BF16↔F32 round-trip).

5. **Issues D, E, F, G, H** — polish. Not blocking once the bridge is correct.

## Files referenced

- `crates/hipfire-quantize/src/gguf_input.rs` — split-shard loader
- `crates/hipfire-quantize/src/main.rs` — arch_id mapping, expert pre-pass, norm fixup
- `crates/hipfire-quantize/src/main.rs:1775 gguf_to_safetensors_name` — name translation table
- `crates/hipfire-arch-qwen35/src/qwen35.rs:1645+ load_moe_ffn_into` — engine's expert loader
- `crates/hipfire-arch-qwen35/src/qwen35.rs:1651` — router `mlp.gate.weight`
- `crates/hipfire-arch-qwen35/src/qwen35.rs:1657-1659` — shared expert gate/up/down
- `crates/hipfire-runtime/examples/compare_hfq.rs` — NRMSE comparison tool (added in this work)
- llama.cpp `convert_hf_to_gguf.py:4865 Qwen3NextModel.modify_tensors` — `+1` bake source
