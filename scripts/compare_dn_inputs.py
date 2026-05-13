#!/usr/bin/env python3
"""Per-tensor compare of DeltaNet recurrence INPUTS between hipfire dump
(from `dump_dn_inputs` in qwen35.rs) and HF transformers dump (from
`dump_hf_dn_inputs.py`).

Bisect strategy: if Q matches but V differs → upstream-of-V kernel is the
bug. If alpha matches but beta differs → fused_sigmoid_alpha_gate beta side
is wrong. Etc.

Reports per-tensor (Q / K / V / alpha / beta):
  - mean relative L2 across positions
  - max relative L2 across positions
  - mean cosine similarity across positions
  - per-position-bucket trend

Usage:
    compare_dn_inputs.py --hipfire <hipfire.bin> --hf <hf.bin>
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hipfire", required=True)
    p.add_argument("--hf", required=True)
    return p.parse_args()


def read_dn_dump(path: Path):
    """Returns list[dict] one entry per recorded position with
    {layer_idx, pos, n_v_heads, head_dim, q, k, v, alpha, beta}.
    """
    out = []
    with open(path, "rb") as f:
        raw = f.read()
    p = 0
    while p < len(raw):
        if p + 32 > len(raw):
            sys.exit(f"{path}: truncated header at byte {p}")
        layer_idx, pos, nvh, hd, qkv, ab, _, _ = struct.unpack(
            "<8I", raw[p : p + 32]
        )
        p += 32
        body_bytes = (3 * qkv + 2 * ab) * 4
        if p + body_bytes > len(raw):
            sys.exit(f"{path}: truncated body at byte {p}")
        body = np.frombuffer(raw[p : p + body_bytes], dtype=np.float32)
        p += body_bytes
        q = body[0:qkv].reshape(nvh, hd)
        k = body[qkv : 2 * qkv].reshape(nvh, hd)
        v = body[2 * qkv : 3 * qkv].reshape(nvh, hd)
        alpha = body[3 * qkv : 3 * qkv + ab]
        beta = body[3 * qkv + ab : 3 * qkv + 2 * ab]
        out.append(
            dict(
                layer_idx=layer_idx, pos=pos, nvh=nvh, hd=hd,
                q=q, k=k, v=v, alpha=alpha, beta=beta,
            )
        )
    return out


def per_tensor_stats(name: str, a_stack: np.ndarray, b_stack: np.ndarray):
    """a_stack, b_stack: [T, *tensor_shape]. Flatten to per-position vectors."""
    T = a_stack.shape[0]
    A = a_stack.astype(np.float64).reshape(T, -1)
    B = b_stack.astype(np.float64).reshape(T, -1)
    diff = A - B
    a_norm = np.linalg.norm(A, axis=-1)
    b_norm = np.linalg.norm(B, axis=-1)
    diff_norm = np.linalg.norm(diff, axis=-1)
    rel_l2 = diff_norm / np.maximum(a_norm, 1e-12)
    cos = np.einsum("ij,ij->i", A, B) / np.maximum(a_norm * b_norm, 1e-12)
    print(
        f"  {name:>6}  mean rel_L2 = {float(rel_l2.mean()):.4f}  "
        f"max = {float(rel_l2.max()):.4f}  "
        f"mean cos = {float(cos.mean()):.6f}  "
        f"min cos = {float(cos.min()):.6f}"
    )


def main():
    args = parse_args()
    hip = read_dn_dump(Path(args.hipfire))
    hf = read_dn_dump(Path(args.hf))
    if len(hip) != len(hf):
        sys.exit(f"record count mismatch: hipfire {len(hip)} vs hf {len(hf)}")
    if hip[0]["layer_idx"] != hf[0]["layer_idx"]:
        sys.exit(
            f"layer_idx mismatch: hipfire L{hip[0]['layer_idx']} vs "
            f"hf L{hf[0]['layer_idx']}"
        )
    if (hip[0]["nvh"], hip[0]["hd"]) != (hf[0]["nvh"], hf[0]["hd"]):
        sys.exit(
            f"shape mismatch: hipfire {hip[0]['nvh']}×{hip[0]['hd']} vs "
            f"hf {hf[0]['nvh']}×{hf[0]['hd']}"
        )
    T = len(hip)
    L = hip[0]["layer_idx"]
    H = hip[0]["nvh"]
    D = hip[0]["hd"]
    print(
        f"DN-inputs compare: layer {L}, {T} positions, "
        f"n_v_heads={H} head_dim={D}\n"
    )

    # Stack tensors across positions for batched per-tensor stats.
    q_h = np.stack([r["q"] for r in hip], axis=0)
    q_o = np.stack([r["q"] for r in hf], axis=0)
    k_h = np.stack([r["k"] for r in hip], axis=0)
    k_o = np.stack([r["k"] for r in hf], axis=0)
    v_h = np.stack([r["v"] for r in hip], axis=0)
    v_o = np.stack([r["v"] for r in hf], axis=0)
    a_h = np.stack([r["alpha"] for r in hip], axis=0)
    a_o = np.stack([r["alpha"] for r in hf], axis=0)
    b_h = np.stack([r["beta"] for r in hip], axis=0)
    b_o = np.stack([r["beta"] for r in hf], axis=0)

    per_tensor_stats("Q", q_h, q_o)
    per_tensor_stats("K", k_h, k_o)
    per_tensor_stats("V", v_h, v_o)
    per_tensor_stats("alpha", a_h, a_o)
    per_tensor_stats("beta", b_h, b_o)

    # Quick per-position-bucket break for whichever tensor is the worst.
    print()
    print("Per-position bucketed rel_L2 (early / mid / late):")
    for name, h_t, o_t in [
        ("Q", q_h, q_o), ("K", k_h, k_o), ("V", v_h, v_o),
        ("alpha", a_h, a_o), ("beta", b_h, b_o),
    ]:
        A = h_t.astype(np.float64).reshape(T, -1)
        B = o_t.astype(np.float64).reshape(T, -1)
        diff = A - B
        rel = np.linalg.norm(diff, axis=-1) / np.maximum(
            np.linalg.norm(A, axis=-1), 1e-12
        )
        early = float(rel[:128].mean())
        mid = float(rel[1024:1152].mean())
        late = float(rel[1920:2048].mean())
        print(
            f"  {name:>6}  [0..127]={early:.4f}  [1024..1151]={mid:.4f}  "
            f"[1920..2047]={late:.4f}"
        )


if __name__ == "__main__":
    main()
