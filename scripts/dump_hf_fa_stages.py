#!/usr/bin/env python3
"""HF transformers oracle for FA pipeline per-stage hidden state.

Captures intermediate state at each of the 14 boundaries used by
hipfire's `HIPFIRE_DUMP_FA_STAGES` instrumentation. Writes the same
append-only on-disk format as the LA-stages probe so
`scripts/compare_la_stages.py` (or a thin FA-name override) can ingest
both files side-by-side.

Stage IDs (must match hipfire FA branch in qwen35.rs):
  0  pre-rmsnorm residual (DecoderLayer input)
  1  post-input_layernorm
  2  post-q_proj (raw Q+gate, 2× wide, interleaved per-head)
  3  post-k_proj
  4  post-v_proj
  5  post-deinterleave Q          (flat n_heads × head_dim)
  6  post-deinterleave gate       (flat n_heads × head_dim)
  7  post-q_norm
  8  post-k_norm
  9  post-RoPE Q
  10 post-RoPE K
  11 post-attention (pre-gate)    (n_heads × head_dim)
  12 post-sigmoid-mul gate        (n_heads × head_dim)
  13 post-wo + residual (block exit)

Usage:
    dump_hf_fa_stages.py --model <hf_dir> --ref <kldref> \
                         --chunk N --layer L --out <path>

Layer L must be a FullAttention layer (layer_types[L] == "FullAttention").
On Qwen3.5-0.8B these are layers 3, 7, 11, 15, 19, 23 (every 4th after
the initial 3 LA layers).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


HEADER_FMT = "<8I"
HEADER_BYTES = struct.calcsize(HEADER_FMT)


def write_record(f, stage_id, layer_idx, pos, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().to(dtype=torch.float32).contiguous().cpu().numpy()
    else:
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    n = arr.size
    hdr = struct.pack(HEADER_FMT, layer_idx, pos, stage_id, n, 0, 0, 0, 0)
    f.write(hdr)
    f.write(arr.astype(np.float32, copy=False).tobytes())


def read_chunk_tokens(ref_path, chunk_idx):
    with open(ref_path, "rb") as f:
        if f.read(8) != b"HFKLDR\0\0":
            sys.exit("bad ref magic")
        hdr = f.read(24)
        n_ctx = struct.unpack("<I", hdr[4:8])[0]
        n_chunk = struct.unpack("<I", hdr[12:16])[0]
        if chunk_idx >= n_chunk:
            sys.exit(f"chunk {chunk_idx} >= n_chunk {n_chunk}")
        f.seek(32 + chunk_idx * n_ctx * 4)
        return n_ctx, list(struct.unpack(f"<{n_ctx}I", f.read(n_ctx * 4)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--layer", type=int, default=3,
                   help="Target FA layer index (default 3 — first FA layer)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu",
                   help="cpu|cuda — HF transformers device (default cpu for reproducibility)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ctx, tokens = read_chunk_tokens(Path(args.ref), args.chunk)
    print(f"loaded {n_ctx} tokens from chunk {args.chunk}", flush=True)
    print(f"target FA layer {args.layer}", flush=True)

    print(f"loading HF model BF16 ({args.device}): {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)

    decoder_layers = (
        model.model.language_model.layers
        if hasattr(model.model, "language_model")
        else model.model.layers
    )
    target_layer_module = decoder_layers[args.layer]
    if not hasattr(target_layer_module, "self_attn"):
        sys.exit(
            f"layer {args.layer} has no self_attn — likely a LinearAttention layer."
        )

    out_file = open(out_path, "wb")
    print(f"opened {out_path} for writing", flush=True)

    attn_module = target_layer_module.self_attn
    head_dim = attn_module.head_dim
    n_heads = attn_module.config.num_attention_heads
    n_kv_heads = attn_module.config.num_key_value_heads
    layer_idx = args.layer

    orig_attn_forward = attn_module.forward

    def recording_attn_forward(hidden_states, position_embeddings,
                                attention_mask=None, past_key_values=None,
                                **kwargs):
        # Replicates Qwen3_5Attention.forward body but captures
        # intermediates per-position. Single-batch only.
        B, T, _ = hidden_states.shape
        assert B == 1, "this probe assumes batch_size=1"
        input_shape = (B, T)
        hidden_shape = (B, T, -1, head_dim)

        # Stage 1 — post-input_layernorm (we receive this as input).
        h_f32 = hidden_states.detach().float()[0]  # [T, hidden]
        for t in range(T):
            write_record(out_file, 1, layer_idx, t, h_f32[t])

        # Stage 2 — post-q_proj (raw, 2× wide).
        q_full = attn_module.q_proj(hidden_states)
        q_full_f32 = q_full.detach().float()[0]  # [T, n_heads * 2 * head_dim]
        for t in range(T):
            write_record(out_file, 2, layer_idx, t, q_full_f32[t])

        # Stage 3 — post-k_proj.
        k_raw = attn_module.k_proj(hidden_states)
        k_raw_f32 = k_raw.detach().float()[0]
        for t in range(T):
            write_record(out_file, 3, layer_idx, t, k_raw_f32[t])

        # Stage 4 — post-v_proj.
        v_raw = attn_module.v_proj(hidden_states)
        v_raw_f32 = v_raw.detach().float()[0]
        for t in range(T):
            write_record(out_file, 4, layer_idx, t, v_raw_f32[t])

        # Q chunk into Q + gate.
        # HF: q_full.view([..., 2*head_dim]) then chunk on dim=-1
        q_reshaped = q_full.view(*input_shape, -1, head_dim * 2)
        query_states, gate = torch.chunk(q_reshaped, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)  # [B, T, n_heads * head_dim]

        # Stage 5/6 — post-deinterleave Q + gate (flat layouts).
        q_split_f32 = query_states.reshape(*input_shape, -1).detach().float()[0]
        gate_f32 = gate.detach().float()[0]
        for t in range(T):
            write_record(out_file, 5, layer_idx, t, q_split_f32[t])
            write_record(out_file, 6, layer_idx, t, gate_f32[t])

        # Per-head norms.
        query_states = attn_module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attn_module.k_norm(k_raw.view(hidden_shape)).transpose(1, 2)
        value_states = v_raw.view(hidden_shape).transpose(1, 2)

        # Stage 7/8 — post q/k_norm (flatten to match hipfire's layout).
        # query_states is now [B, n_heads, T, head_dim]; flatten to [B, T, n_heads*head_dim].
        q_norm_f32 = query_states.transpose(1, 2).reshape(*input_shape, -1).detach().float()[0]
        k_norm_f32 = key_states.transpose(1, 2).reshape(*input_shape, -1).detach().float()[0]
        for t in range(T):
            write_record(out_file, 7, layer_idx, t, q_norm_f32[t])
            write_record(out_file, 8, layer_idx, t, k_norm_f32[t])

        # RoPE.
        cos, sin = position_embeddings
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Stage 9/10 — post-RoPE Q + K.
        q_rope_f32 = query_states.transpose(1, 2).reshape(*input_shape, -1).detach().float()[0]
        k_rope_f32 = key_states.transpose(1, 2).reshape(*input_shape, -1).detach().float()[0]
        for t in range(T):
            write_record(out_file, 9, layer_idx, t, q_rope_f32[t])
            write_record(out_file, 10, layer_idx, t, k_rope_f32[t])

        # KV cache update (no math).
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn_module.layer_idx
            )

        # Attention compute — use eager fallback for reproducibility.
        # Manually compute scaled dot-product attention to avoid Flash dispatch.
        # Repeat-interleave KV heads if GQA.
        if n_heads // n_kv_heads > 1:
            ratio = n_heads // n_kv_heads
            key_states = key_states.repeat_interleave(ratio, dim=1)
            value_states = value_states.repeat_interleave(ratio, dim=1)
        attn_scale = attn_module.scaling
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * attn_scale
        # Causal mask for prefill: positions j > i get -inf.
        T_q = query_states.shape[-2]
        T_kv = key_states.shape[-2]
        if T_q > 1 and T_kv == T_q:
            mask = torch.triu(
                torch.full((T_q, T_kv), float("-inf"), dtype=attn_weights.dtype),
                diagonal=1,
            )
            attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)  # [B, n_heads, T, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)

        # Stage 11 — post-attention (pre-gate).
        attn_pre_f32 = attn_output.detach().float()[0]
        for t in range(T):
            write_record(out_file, 11, layer_idx, t, attn_pre_f32[t])

        # Gate.
        attn_output = attn_output * torch.sigmoid(gate)

        # Stage 12 — post-sigmoid-mul.
        attn_post_f32 = attn_output.detach().float()[0]
        for t in range(T):
            write_record(out_file, 12, layer_idx, t, attn_post_f32[t])

        # o_proj.
        output = attn_module.o_proj(attn_output)
        return output, None

    attn_module.forward = recording_attn_forward

    # DecoderLayer hooks for stage 0 (input) and stage 13 (output).
    def decoder_pre_hook(module, args, kwargs):
        if args:
            hs = args[0]
        else:
            hs = kwargs.get("hidden_states")
        if hs is None:
            return
        hs_f32 = hs.detach().float()[0]
        for t in range(hs_f32.shape[0]):
            write_record(out_file, 0, layer_idx, t, hs_f32[t])

    def decoder_post_hook(module, args, kwargs, output):
        # Note: this captures POST-MLP residual due to DecoderLayer.forward
        # returning the post-MLP hidden. The Phase 1c bug was the same; for
        # FA-3 we don't have a clean post-wo-pre-FFN capture without
        # patching DecoderLayer.forward more invasively. Recorded but flag
        # in the compare output that stage 13 includes the MLP block.
        if isinstance(output, tuple):
            out_t = output[0]
        else:
            out_t = output
        out_f32 = out_t.detach().float()[0]
        for t in range(out_f32.shape[0]):
            write_record(out_file, 13, layer_idx, t, out_f32[t])

    pre_handle = target_layer_module.register_forward_pre_hook(
        decoder_pre_hook, with_kwargs=True
    )
    post_handle = target_layer_module.register_forward_hook(
        decoder_post_hook, with_kwargs=True
    )

    input_ids = torch.tensor([tokens], dtype=torch.long)
    if args.device != "cpu":
        input_ids = input_ids.to(args.device)

    print(f"running forward on {n_ctx} tokens (this may take 2-3 min on CPU)...",
          flush=True)
    with torch.no_grad():
        _ = model(input_ids)
    print("forward done", flush=True)

    pre_handle.remove()
    post_handle.remove()
    attn_module.forward = orig_attn_forward
    out_file.close()
    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(f"wrote {out_path} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
