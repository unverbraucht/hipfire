#!/usr/bin/env python3
"""F2-follow-up — fp64 reference threshold for l2norm drift (stages 11/12).

Pattern-hunt plan said stage 12 (post-l2norm k) is the highest upstream
drift at 0.0118 rL2 with no bf16-cast confound (F2 result). This script
checks whether that 0.0118 is HF's fp32-arithmetic floor (i.e. hipfire ≡
fp64 ref, like the rmsnorm story) or whether hipfire has its own
arithmetic drift away from the fp64 ideal.

For each engine, compute the per-head fp64 reference l2norm + scale on
that engine's own stage 8/9 input, then compare against that engine's
stage 11/12 dump. The (engine fp32 vs its own fp64 ref) rL2 tells us
the engine's intrinsic l2norm arithmetic floor — independent of which
upstream stage 8/9 the engine fed into l2norm.

HF formula (modeling_qwen3_5.py:135-136, called from dump_hf_la_stages.py:250):
    def l2norm(x, dim=-1, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    q_post = l2norm(query.float(), dim=-1, eps=1e-6) * qk_scale
    k_post = l2norm(key.float(), dim=-1, eps=1e-6)

For Qwen3.5-0.8B: n_heads_k = 16, head_k_dim = 128, k_dim = 2048,
qk_scale = 1.0 / sqrt(128).

Inputs:
    --hf-dump        HF stages dump (stages 8, 9, 11, 12 needed)
    --hip-dump       hipfire stages dump (same stages needed)

Outputs: prints per-engine fp32-vs-fp64 rL2 and the cross-engine rL2
for stages 11 and 12.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


HEADER_FMT = "<8I"
HEADER_BYTES = struct.calcsize(HEADER_FMT)

# Qwen3.5-0.8B linear-attention dims (matches hipfire-arch-qwen35 config).
N_HEADS_K = 16
HEAD_K_DIM = 128
K_DIM = N_HEADS_K * HEAD_K_DIM  # 2048
EPS = 1e-6
QK_SCALE = 1.0 / np.sqrt(HEAD_K_DIM)  # 1/sqrt(128)


def read_dump(path):
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i < len(data):
        if i + HEADER_BYTES > len(data):
            raise ValueError(f"truncated header at offset {i}")
        layer, pos, stage, n, _, _, _, _ = struct.unpack_from(HEADER_FMT, data, i)
        i += HEADER_BYTES
        nbytes = n * 4
        arr = np.frombuffer(data[i:i + nbytes], dtype=np.float32).copy()
        i += nbytes
        out[(int(layer), int(stage), int(pos))] = arr
    return out


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = a - b
    num = np.sqrt((diff * diff).sum())
    den = max(np.sqrt((b * b).sum()), 1e-12)
    return float(num / den)


def l2norm_fp64(x_flat_f32, scale_f64=1.0):
    """Per-head fp64 l2norm of a flat [n_heads*head_dim] vector. Returns fp32."""
    x = x_flat_f32.astype(np.float64).reshape(N_HEADS_K, HEAD_K_DIM)
    sq = (x * x).sum(axis=-1, keepdims=True)
    inv_norm = 1.0 / np.sqrt(sq + EPS)
    out = x * inv_norm * scale_f64
    return out.reshape(-1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dump", required=True)
    p.add_argument("--hip-dump", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--n-positions", type=int, default=2048)
    args = p.parse_args()

    print(f"reading HF dump: {args.hf_dump}")
    hf = read_dump(Path(args.hf_dump))
    print(f"  {len(hf)} records")
    print(f"reading hipfire dump: {args.hip_dump}")
    hip = read_dump(Path(args.hip_dump))
    print(f"  {len(hip)} records")

    rl_hf_q_vs_ref = []     # HF stage 11 vs fp64-ref(HF stage 8)
    rl_hf_k_vs_ref = []     # HF stage 12 vs fp64-ref(HF stage 9)
    rl_hip_q_vs_ref = []    # hipfire stage 11 vs fp64-ref(hipfire stage 8)
    rl_hip_k_vs_ref = []    # hipfire stage 12 vs fp64-ref(hipfire stage 9)
    rl_hip_q_vs_hf = []     # hipfire stage 11 vs HF stage 11 (audit number)
    rl_hip_k_vs_hf = []     # hipfire stage 12 vs HF stage 12 (audit number)
    # Bonus: hipfire stage 11/12 vs fp64-ref(HF stage 8/9). This isolates
    # input drift from kernel arithmetic: if hipfire ≡ fp64-of-HF-input then
    # hipfire's l2norm arithmetic is bit-faithful to fp64 ideal AND the
    # divergence between engines is entirely upstream (stage 8/9 drift).
    rl_hip_q_vs_ref_hf_input = []
    rl_hip_k_vs_ref_hf_input = []

    n_proc = 0
    for pos in range(args.n_positions):
        keys = [(args.layer, s, pos) for s in (8, 9, 11, 12)]
        if not all(k in hf and k in hip for k in keys):
            continue

        # HF side
        hf_q_in  = hf[(args.layer,  8, pos)]
        hf_k_in  = hf[(args.layer,  9, pos)]
        hf_q_out = hf[(args.layer, 11, pos)]
        hf_k_out = hf[(args.layer, 12, pos)]
        # hipfire side
        hip_q_in  = hip[(args.layer,  8, pos)]
        hip_k_in  = hip[(args.layer,  9, pos)]
        hip_q_out = hip[(args.layer, 11, pos)]
        hip_k_out = hip[(args.layer, 12, pos)]

        # fp64 references per engine's own input
        ref_hf_q  = l2norm_fp64(hf_q_in,  scale_f64=QK_SCALE)
        ref_hf_k  = l2norm_fp64(hf_k_in,  scale_f64=1.0)
        ref_hip_q = l2norm_fp64(hip_q_in, scale_f64=QK_SCALE)
        ref_hip_k = l2norm_fp64(hip_k_in, scale_f64=1.0)

        rl_hf_q_vs_ref.append(rel_l2(hf_q_out, ref_hf_q))
        rl_hf_k_vs_ref.append(rel_l2(hf_k_out, ref_hf_k))
        rl_hip_q_vs_ref.append(rel_l2(hip_q_out, ref_hip_q))
        rl_hip_k_vs_ref.append(rel_l2(hip_k_out, ref_hip_k))
        rl_hip_q_vs_hf.append(rel_l2(hip_q_out, hf_q_out))
        rl_hip_k_vs_hf.append(rel_l2(hip_k_out, hf_k_out))
        rl_hip_q_vs_ref_hf_input.append(rel_l2(hip_q_out, ref_hf_q))
        rl_hip_k_vs_ref_hf_input.append(rel_l2(hip_k_out, ref_hf_k))
        n_proc += 1

    if n_proc == 0:
        sys.exit("no overlapping positions")

    def st(label, arr):
        a = np.array(arr)
        print(f"  {label:<55} n={len(a):>4}  mean={a.mean():.6f}  "
              f"max={a.max():.6f}  min={a.min():.6f}")

    print(f"\n=== STAGE 11 (post-l2norm+scale q) — n_processed={n_proc} ===")
    st("HF stage 11 vs fp64-ref(HF stage 8)        [HF fp32 floor]", rl_hf_q_vs_ref)
    st("hip stage 11 vs fp64-ref(hip stage 8)      [hip fp32 floor]", rl_hip_q_vs_ref)
    st("hip stage 11 vs fp64-ref(HF stage 8)       [hip vs HF-input ref]", rl_hip_q_vs_ref_hf_input)
    st("hip stage 11 vs HF stage 11                [audit cross-engine]", rl_hip_q_vs_hf)

    print(f"\n=== STAGE 12 (post-l2norm k) — n_processed={n_proc} ===")
    st("HF stage 12 vs fp64-ref(HF stage 9)        [HF fp32 floor]", rl_hf_k_vs_ref)
    st("hip stage 12 vs fp64-ref(hip stage 9)      [hip fp32 floor]", rl_hip_k_vs_ref)
    st("hip stage 12 vs fp64-ref(HF stage 9)       [hip vs HF-input ref]", rl_hip_k_vs_ref_hf_input)
    st("hip stage 12 vs HF stage 12                [audit cross-engine]", rl_hip_k_vs_hf)

    # ---- INPUT DRIFT — compare upstream stage 8/9 between engines ----
    rl_q_in = []
    rl_k_in = []
    for pos in range(args.n_positions):
        for s, dst in [(8, rl_q_in), (9, rl_k_in)]:
            k = (args.layer, s, pos)
            if k in hf and k in hip:
                dst.append(rel_l2(hip[k], hf[k]))

    print(f"\n=== UPSTREAM INPUT DRIFT (cross-engine, audit baseline) ===")
    st("hip stage 8 vs HF stage 8 (q_raw)", rl_q_in)
    st("hip stage 9 vs HF stage 9 (k_raw)", rl_k_in)

    # ---- INTERPRETATION ----
    print("\n=== INTERPRETATION ===")
    hf_q_floor = float(np.mean(rl_hf_q_vs_ref))
    hf_k_floor = float(np.mean(rl_hf_k_vs_ref))
    hip_q_floor = float(np.mean(rl_hip_q_vs_ref))
    hip_k_floor = float(np.mean(rl_hip_k_vs_ref))
    print(f"HF's intrinsic l2norm fp32 floor:    Q={hf_q_floor:.6f}  K={hf_k_floor:.6f}")
    print(f"hipfire's intrinsic l2norm fp32 floor: Q={hip_q_floor:.6f}  K={hip_k_floor:.6f}")
    print(f"  → if both ≪ audit cross-engine number, l2norm is at its fp32 floor")
    print(f"  → in that case the 0.0087 / 0.0118 'drift' is upstream input difference")
    print(f"     propagated through near-identical kernels, not l2norm kernel divergence")


if __name__ == "__main__":
    main()
