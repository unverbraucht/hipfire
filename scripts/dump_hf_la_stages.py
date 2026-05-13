#!/usr/bin/env python3
"""HF transformers oracle for LA pipeline per-stage hidden state.

Captures intermediate state at each of the 16 boundaries used by
hipfire's `HIPFIRE_DUMP_LA_STAGES` instrumentation in
`crates/hipfire-arch-qwen35/src/qwen35.rs`. Writes the same append-only
on-disk format so `scripts/compare_la_stages.py` can ingest both files
side-by-side.

Stage IDs (must match hipfire):
  0  pre-rmsnorm residual (DecoderLayer input)
  1  post-input_layernorm (linear_attn input)
  2  post-in_proj_qkv          (raw mixed_qkv, pre-conv)
  3  post-in_proj_z            (raw z, pre-reshape)
  4  post-in_proj_a            (raw a, pre-softplus)
  5  post-in_proj_b            (raw b, pre-sigmoid)
  6  post-sigmoid_alpha_gate gated alpha  (= g = -exp(A_log)*softplus(a+dt_bias))
  7  post-sigmoid_alpha_gate gated beta   (= sigmoid(b))
  8  post-conv1d_silu_split q_raw          (pre-l2norm)
  9  post-conv1d_silu_split k_raw          (pre-l2norm)
  10 post-conv1d_silu_split v
  11 post-l2norm+scale q
  12 post-l2norm k
  13 post-recurrence  (core_attn_out, pre-gated_norm)
  14 post-gated_norm  (core_attn_out, post self.norm(_, z))
  15 post-wo + residual (DecoderLayer output)

Usage:
    dump_hf_la_stages.py --model <hf_dir> --ref <kldref> \
                         --chunk N --layer L --out <path>

The HF call replays the same tokens that produced hipfire's stage dump.
Layer L must be a LinearAttention layer (layer_types[L] == "LinearAttention");
0, 1, 2, 4, 5 are typical LA layers on Qwen3.5-0.8B/9B.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# ----- record format (identical to hipfire's dump_la_stage) -----
# u32 layer_idx, u32 pos, u32 stage_id, u32 n_elems, u32×4 reserved
# f32 × n_elems
HEADER_FMT = "<8I"
HEADER_BYTES = struct.calcsize(HEADER_FMT)


def write_record(f, stage_id, layer_idx, pos, arr):
    """arr: 1-D torch tensor or numpy array, f32."""
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
    p.add_argument("--layer", type=int, default=0,
                   help="Target LA layer index (default 0 — cleanest input)")
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
    print(f"target LA layer {args.layer}", flush=True)

    print(f"loading HF model BF16 ({args.device}): {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)

    # The decoder layers live under model.model.language_model.layers for
    # multimodal-wrapped LMs, or model.model.layers for the bare LM.
    decoder_layers = (
        model.model.language_model.layers
        if hasattr(model.model, "language_model")
        else model.model.layers
    )
    target_layer_module = decoder_layers[args.layer]
    if not hasattr(target_layer_module, "linear_attn"):
        sys.exit(
            f"layer {args.layer} has no linear_attn — likely a FullAttention layer."
        )

    out_file = open(out_path, "wb")
    print(f"opened {out_path} for writing", flush=True)

    # ------- monkey-patch linear_attn.forward to capture stages 1-14 -------
    dn_module = target_layer_module.linear_attn
    orig_dn_forward = dn_module.forward
    head_k_dim = dn_module.head_k_dim
    head_v_dim = dn_module.head_v_dim
    n_v_heads = dn_module.num_v_heads
    n_k_heads = dn_module.num_k_heads
    qk_scale = 1.0 / (head_k_dim ** 0.5)
    layer_idx = args.layer

    def l2norm(x, dim=-1, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

    def recording_linear_attn_forward(hidden_states, cache_params=None,
                                       attention_mask=None):
        # hidden_states: [B=1, T, hidden] post input_layernorm
        # Replicates the body of Qwen3_5GatedDeltaNet.forward but captures
        # intermediates. Single-batch, no cache_params (we pass the full
        # 2048-token chunk through prefill mode — chunk_gated_delta_rule
        # handles the recurrence internally).
        B, T, _ = hidden_states.shape
        assert B == 1, "this probe assumes batch_size=1"

        h_f32 = hidden_states.detach().float()[0]  # [T, hidden]
        # Stage 1 — post-input_layernorm (per-position).
        for t in range(T):
            write_record(out_file, 1, layer_idx, t, h_f32[t])

        # Stage 2 — in_proj_qkv output (raw mixed_qkv, pre-conv).
        mixed_qkv = dn_module.in_proj_qkv(hidden_states)
        mq_f32 = mixed_qkv.detach().float()[0]  # [T, qkv_dim]
        for t in range(T):
            write_record(out_file, 2, layer_idx, t, mq_f32[t])

        # Stage 3 — in_proj_z output (raw z, pre-reshape).
        z = dn_module.in_proj_z(hidden_states)
        z_f32 = z.detach().float()[0]  # [T, v_dim]
        for t in range(T):
            write_record(out_file, 3, layer_idx, t, z_f32[t])

        # Stage 4 — in_proj_a output (raw a, pre-softplus).
        a = dn_module.in_proj_a(hidden_states)
        a_f32 = a.detach().float()[0]  # [T, n_v_heads]
        for t in range(T):
            write_record(out_file, 4, layer_idx, t, a_f32[t])

        # Stage 5 — in_proj_b output (raw b, pre-sigmoid).
        b = dn_module.in_proj_b(hidden_states)
        b_f32 = b.detach().float()[0]  # [T, n_v_heads]
        for t in range(T):
            write_record(out_file, 5, layer_idx, t, b_f32[t])

        # Continue with the model's actual forward to produce correct
        # downstream outputs, then capture the intermediates inline.
        # We need to replicate the post-projection path because we want
        # the conv1d+SiLU output and the recurrence output. Easiest: just
        # call the original forward — but we lose access to the in-between
        # state. So instead, we re-implement the body here.

        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        # Conv1d + SiLU. The HF code path uses causal_conv1d_fn when available,
        # else F.silu(self.conv1d(...)). We force the F.silu path for
        # reproducibility (no causal_conv1d dep on CPU).
        if dn_module.causal_conv1d_fn is not None:
            mixed_qkv_t = dn_module.causal_conv1d_fn(
                x=mixed_qkv_t,
                weight=dn_module.conv1d.weight.squeeze(1),
                bias=dn_module.conv1d.bias,
                activation=dn_module.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv_t = F.silu(
                dn_module.conv1d(mixed_qkv_t)[:, :, : mixed_qkv_t.shape[-1]]
            )

        mixed_qkv = mixed_qkv_t.transpose(1, 2)  # [B, T, k_dim*2 + v_dim]
        key_dim = dn_module.key_dim
        value_dim = dn_module.value_dim
        query, key, value = torch.split(
            mixed_qkv, [key_dim, key_dim, value_dim], dim=-1
        )

        # Stage 8 — q_raw (pre-l2norm, pre-reshape).
        q_f32 = query.detach().float()[0]  # [T, k_dim]
        for t in range(T):
            write_record(out_file, 8, layer_idx, t, q_f32[t])
        # Stage 9 — k_raw.
        k_f32 = key.detach().float()[0]
        for t in range(T):
            write_record(out_file, 9, layer_idx, t, k_f32[t])
        # Stage 10 — v.
        v_f32 = value.detach().float()[0]
        for t in range(T):
            write_record(out_file, 10, layer_idx, t, v_f32[t])

        # Reshape to per-head.
        query = query.reshape(B, T, -1, head_k_dim)
        key = key.reshape(B, T, -1, head_k_dim)
        value = value.reshape(B, T, -1, head_v_dim)

        # Stage 7 — sigmoid(b) = beta.
        beta = b.sigmoid()
        beta_f32 = beta.detach().float()[0]
        for t in range(T):
            write_record(out_file, 7, layer_idx, t, beta_f32[t])

        # Stage 6 — gated alpha g = -exp(A_log) * softplus(a + dt_bias).
        # `.float()` matches HF: cast to fp32 to avoid -inf at fp16/bf16.
        g = -dn_module.A_log.float().exp() * F.softplus(
            a.float() + dn_module.dt_bias
        )
        g_f32 = g.detach().float()[0]
        for t in range(T):
            write_record(out_file, 6, layer_idx, t, g_f32[t])

        # GQA repeat-interleave if n_v != n_k.
        if dn_module.num_v_heads // dn_module.num_k_heads > 1:
            ratio = dn_module.num_v_heads // dn_module.num_k_heads
            query = query.repeat_interleave(ratio, dim=2)
            key = key.repeat_interleave(ratio, dim=2)

        # Stage 11/12 — post-l2norm+scale Q/K (matches use_qk_l2norm_in_kernel
        # internal step inside chunk_gated_delta_rule). Q gets scaled by
        # 1/sqrt(head_k_dim); K is just l2-normed.
        q_l2 = l2norm(query.float(), dim=-1, eps=1e-6) * qk_scale  # [B, T, H, D]
        k_l2 = l2norm(key.float(), dim=-1, eps=1e-6)
        q_l2_f32 = q_l2.detach()[0]  # [T, H, D]
        k_l2_f32 = k_l2.detach()[0]
        for t in range(T):
            # Flatten H × D back into k_dim (post-GQA-expand). The hipfire
            # buffer s.dn_q_raw at this stage is also pre-repeat-interleave
            # (the qk_l2_norm runs BEFORE repeat_interleave_qk in hipfire).
            # For matched-head models (Q3.5-0.8B has n_v == n_k), this is
            # bit-equivalent; for n_v > n_k, hipfire dumps the pre-expand
            # buffer at stage 11/12 while HF's chunk_rule sees post-expand.
            # We approximate by dumping the un-expanded HF version when
            # n_v == n_k; otherwise the user must interpret stage 11/12
            # rel_L2 with care.
            write_record(out_file, 11, layer_idx, t, q_l2_f32[t])
            write_record(out_file, 12, layer_idx, t, k_l2_f32[t])

        # Recurrence — use HF's chunk_gated_delta_rule with l2norm_in_kernel.
        core_attn_out, _ = dn_module.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Stage 13 — post-recurrence (core_attn_out, pre-norm).
        # core_attn_out is [B, T, H, head_v_dim] before reshape.
        cao_pre_norm_f32 = core_attn_out.detach().float()[0].reshape(T, -1)
        for t in range(T):
            write_record(out_file, 13, layer_idx, t, cao_pre_norm_f32[t])

        # Reshape + gated norm.
        core_attn_out = core_attn_out.reshape(-1, dn_module.head_v_dim)
        z_flat = z.reshape(-1, dn_module.head_v_dim)
        core_attn_out = dn_module.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(B, T, -1)

        # Stage 14 — post gated_norm (pre out_proj).
        cao_post_norm_f32 = core_attn_out.detach().float()[0]
        for t in range(T):
            write_record(out_file, 14, layer_idx, t, cao_post_norm_f32[t])

        # Final projection.
        output = dn_module.out_proj(core_attn_out)
        return output

    dn_module.forward = recording_linear_attn_forward

    # ------- forward hook on DecoderLayer for stages 0 and 15 -------
    def decoder_pre_hook(module, args, kwargs):
        # args[0] (or kwargs["hidden_states"]) is the pre-norm residual input.
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
        # output is the post-residual hidden state (DecoderLayer returns either
        # a tensor or a tuple — Qwen3.5 returns tensor as primary output).
        if isinstance(output, tuple):
            out_t = output[0]
        else:
            out_t = output
        out_f32 = out_t.detach().float()[0]
        for t in range(out_f32.shape[0]):
            write_record(out_file, 15, layer_idx, t, out_f32[t])

    pre_handle = target_layer_module.register_forward_pre_hook(
        decoder_pre_hook, with_kwargs=True
    )
    post_handle = target_layer_module.register_forward_hook(
        decoder_post_hook, with_kwargs=True
    )

    # ------- run forward -------
    input_ids = torch.tensor([tokens], dtype=torch.long)
    if args.device != "cpu":
        input_ids = input_ids.to(args.device)

    print(f"running forward on {n_ctx} tokens (this may take 1-2 min on CPU)...",
          flush=True)
    with torch.no_grad():
        _ = model(input_ids)
    print("forward done", flush=True)

    pre_handle.remove()
    post_handle.remove()
    dn_module.forward = orig_dn_forward
    out_file.close()
    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(f"wrote {out_path} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
