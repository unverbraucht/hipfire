#!/usr/bin/env python3
"""HF transformers oracle for DeltaNet recurrence INPUTS at layer L.

Captures Q/K/V/g/β at the call boundary to `chunk_gated_delta_rule`
inside `Qwen3_5GatedDeltaNet.forward` — the same boundary that hipfire's
`dump_dn_inputs` writes for. Applies the same post-pipeline transforms
(l2norm + 1/sqrt(head_k_dim) scale + repeat_interleave) so the output
binary aligns with hipfire's dn_q / dn_k / dn_v / dn_alpha / dn_beta
state at the recurrence kernel input.

Output format mirrors `dump_dn_inputs` in `qwen35.rs` exactly:
  per call (per position):
    [u32×8 header] layer_idx, pos, n_v_heads, head_dim, qkv_elems,
                   alphabeta_elems, reserved×2
    [f32 qkv_elems]   q  (post l2norm + scale, post repeat-interleave)
    [f32 qkv_elems]   k  (post l2norm, post repeat-interleave)
    [f32 qkv_elems]   v
    [f32 ab_elems]    alpha (= g  = -exp(A_log) * softplus(a + dt_bias))
    [f32 ab_elems]    beta  (= sigmoid(b))

Usage:
    dump_hf_dn_inputs.py --model <hf_dir> --ref <kldref> \
                        --chunk N --layer L --out <path>
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--out", required=True)
    return p.parse_args()


def read_chunk_tokens(ref_path: Path, chunk_idx: int):
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


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Matches HF's l2norm in transformers/models/qwen3_5/modeling_qwen3_5.py:226-230."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def main():
    args = parse_args()
    n_ctx, tokens = read_chunk_tokens(Path(args.ref), args.chunk)
    print(f"loaded {n_ctx} tokens from chunk {args.chunk}, target layer {args.layer}", flush=True)

    print(f"loading HF model BF16: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)

    # Find the target Qwen3_5GatedDeltaNet module at the requested layer
    # index. The decoder layers live under model.model.language_model.layers
    # for the conditional-gen wrapper, or model.model.layers for the bare LM.
    decoder = (
        model.model.language_model.layers
        if hasattr(model.model, "language_model")
        else model.model.layers
    )
    target_layer = decoder[args.layer]
    if not hasattr(target_layer, "linear_attn"):
        sys.exit(
            f"layer {args.layer} has no linear_attn module — it may be a "
            f"FullAttention layer in the LA-FA hybrid layout. "
            f"Pick an LA-typed layer (e.g. 0, 1, 2, 4, 5, …)."
        )
    dn_module = target_layer.linear_attn
    n_v_heads = dn_module.num_v_heads
    n_k_heads = dn_module.num_k_heads
    head_v_dim = dn_module.head_v_dim
    head_k_dim = dn_module.head_k_dim
    print(
        f"target DeltaNet: n_v_heads={n_v_heads} n_k_heads={n_k_heads} "
        f"head_v={head_v_dim} head_k={head_k_dim}",
        flush=True,
    )
    assert head_v_dim == head_k_dim, (
        f"mixed head_dim not supported by this script "
        f"(head_v={head_v_dim} head_k={head_k_dim})"
    )

    # Monkey-patch the chunk_gated_delta_rule call to capture its inputs.
    # We hold onto the original ref, replace with a recorder, then restore
    # after forward.
    captured: dict = {}
    orig_chunk = dn_module.chunk_gated_delta_rule

    def recording_chunk(query, key, value, g, beta, initial_state=None,
                        output_final_state=False, use_qk_l2norm_in_kernel=True,
                        **kwargs):
        # Shapes from HF (before the rule's internal l2norm):
        #   query, key, value: [B, T, H, D]  (post-repeat_interleave for q/k)
        #   g, beta:           [B, T, H]
        # We capture as numpy, apply hipfire's matching pipeline transforms
        # (l2norm on q/k + scale on q), then dump per-position.
        captured["q"] = query.detach().float().cpu()  # [1, T, H, D]
        captured["k"] = key.detach().float().cpu()
        captured["v"] = value.detach().float().cpu()
        captured["g"] = g.detach().float().cpu()      # [1, T, H]
        captured["beta"] = beta.detach().float().cpu()
        captured["use_l2norm"] = use_qk_l2norm_in_kernel
        # Run the real implementation so the model produces sensible outputs
        # (the forward continues after this call). The chunked rule will
        # re-apply l2norm internally; that's fine.
        return orig_chunk(
            query, key, value, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )

    dn_module.chunk_gated_delta_rule = recording_chunk

    input_ids = torch.tensor([tokens], dtype=torch.long)
    print(f"running forward...", flush=True)
    with torch.no_grad():
        _ = model(input_ids)
    print("forward done", flush=True)

    dn_module.chunk_gated_delta_rule = orig_chunk

    if not captured:
        sys.exit("no capture — layer index may be wrong or DN module never ran")

    q = captured["q"][0]      # [T, H, D]
    k = captured["k"][0]
    v = captured["v"][0]
    g = captured["g"][0]      # [T, H]
    beta = captured["beta"][0]

    T, H, D = q.shape
    print(f"captured T={T} H={H} D={D}, beta shape={tuple(beta.shape)}", flush=True)

    # Apply hipfire's post-fused_qk_l2_norm_scale transforms:
    #   - l2norm on q and k (eps=1e-6, dim=-1, per-head)
    #   - scale q by 1 / sqrt(D)
    if captured["use_l2norm"]:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    scale = 1.0 / (D ** 0.5)
    q = q * scale

    # Dump per-position in the same record layout as hipfire's dump_dn_inputs.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qkv_elems = H * D
    ab_elems = H
    with open(out_path, "wb") as f:
        for pos in range(T):
            hdr = struct.pack(
                "<8I",
                args.layer, pos, H, D, qkv_elems, ab_elems, 0, 0,
            )
            f.write(hdr)
            f.write(q[pos].contiguous().numpy().astype(np.float32).tobytes())
            f.write(k[pos].contiguous().numpy().astype(np.float32).tobytes())
            f.write(v[pos].contiguous().numpy().astype(np.float32).tobytes())
            f.write(g[pos].contiguous().numpy().astype(np.float32).tobytes())
            f.write(beta[pos].contiguous().numpy().astype(np.float32).tobytes())

    sz = out_path.stat().st_size
    print(
        f"wrote {out_path} ({sz / 1024 / 1024:.1f} MB, {T} records)",
        flush=True,
    )


if __name__ == "__main__":
    main()
