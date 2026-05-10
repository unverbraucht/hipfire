#!/usr/bin/env python3
"""Corrected analysis: MQ4 vs Q4_K with proper FWHT-256.

The previous version had a buggy vectorized FWHT (reshape misinterpreted pair indices
at stride > 1). This uses the slow but correct reference loop.
"""
import json
import struct

import numpy as np


def gen_fwht_signs(seed, n=256):
    state = seed
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = 1.0 if (state >> 16) & 1 else -1.0
    return out


def fwht_256_correct(blocks, signs1, signs2):
    """Correct FWHT — matches main.rs:430-446 reference loop."""
    x = (blocks * signs1[None, :]).copy()
    stride = 1
    while stride < 256:
        i = 0
        while i < 256:
            for j in range(stride):
                a = x[:, i + j].copy()
                b = x[:, i + j + stride].copy()
                x[:, i + j] = a + b
                x[:, i + j + stride] = a - b
            i += stride * 2
        stride *= 2
    return x * (signs2[None, :] * 0.0625)


def fwht_256_fast(blocks, signs1, signs2):
    """Fast vectorized FWHT using a different correct decomposition.
    Each butterfly stage operates on contiguous pairs after a stride-aware
    reshape: at stride s, treat data as [batch, 256/(2s), 2, s] but ONLY
    when reading along the LAST dim corresponds to the stride-pair offset.
    Actually, the correct interpretation is:
        v.reshape(batch, n_pairs, stride, 2) where pair[0,1] are at offset stride apart
    Wait — the butterfly pairs (i, i+stride). In a stride=2 step on a 256-vec,
    the pairs are (0,2), (1,3), (4,6), (5,7), ... so contiguous-index pairs
    have offset 2. Reshape [batch, 64, 2, 2] interprets as 64 groups of 4
    contiguous elements; within each group, pair (idx 0 with idx 2) and
    (idx 1 with idx 3). That's [..., 2, 2] where last axis is "within pair"
    and the second-to-last is "first/second of pair". Equivalent to
    swapaxes(-2, -1) of my original. Let me just do the loop in pure numpy
    via explicit slicing.
    """
    x = blocks * signs1[None, :]
    stride = 1
    while stride < 256:
        # Process all pairs at this stride simultaneously.
        # Pairs: (i, i+stride) for i in [0, 2*stride, 4*stride, ...] AND offsets [0..stride)
        # i.e. for i_block in 0..(256/(2*stride)), j in 0..stride: pair (i_block*2*stride + j, i_block*2*stride + j + stride)
        x = x.reshape(x.shape[0], -1, 2 * stride)
        a = x[:, :, :stride].copy()
        b = x[:, :, stride:].copy()
        x[:, :, :stride] = a + b
        x[:, :, stride:] = a - b
        x = x.reshape(x.shape[0], 256)
        stride *= 2
    return x * (signs2[None, :] * 0.0625)


def quant_uniform_4bit(blocks):
    lo = blocks.min(axis=1, keepdims=True)
    hi = blocks.max(axis=1, keepdims=True)
    rng = hi - lo
    scale = np.where(rng > 0, rng / 15.0, 1.0)
    q = np.clip(np.round((blocks - lo) / scale), 0, 15)
    return lo + q * scale


def quant_per_32(blocks):
    n = blocks.shape[0]
    sub = blocks.reshape(n, 8, 32)
    lo = sub.min(axis=2, keepdims=True)
    hi = sub.max(axis=2, keepdims=True)
    rng = hi - lo
    scale = np.where(rng > 0, rng / 15.0, 1.0)
    q = np.clip(np.round((sub - lo) / scale), 0, 15)
    return (lo + q * scale).reshape(n, 256)


def f16_array(raw, n):
    return np.frombuffer(raw[: 2 * n], dtype=np.float16).astype(np.float32)


def bf16_array(raw, n):
    u16 = np.frombuffer(raw[: 2 * n], dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).astype(np.uint32).view(np.float32)


def main():
    import os
    snap = "/home/kread/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
    path = os.path.join(snap, "model-00001-of-00026.safetensors")

    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        body_offset = 8 + header_size

        signs1 = gen_fwht_signs(42)
        signs2 = gen_fwht_signs(1042)

        # First sanity-check: fast FWHT matches reference and is orthonormal
        rng = np.random.default_rng(0)
        x_test = rng.normal(0, 0.02, size=(4, 256)).astype(np.float32)
        y_ref = fwht_256_correct(x_test, signs1, signs2)
        y_fast = fwht_256_fast(x_test, signs1, signs2)
        norm_ref = np.linalg.norm(y_ref - x_test)
        diff = np.linalg.norm(y_ref - y_fast)
        norm_preserve = abs((y_ref**2).sum() - (x_test**2).sum())
        print(f"FWHT sanity:")
        print(f"  ||fast - ref|| = {diff:.6e}")
        print(f"  ||y_ref||² - ||x||² = {norm_preserve:.6e}  (should be ~0)")
        if diff > 1e-5:
            print("FAST FWHT IS WRONG, using slow reference")
            fwht = fwht_256_correct
        else:
            print("Fast FWHT validated, using it")
            fwht = fwht_256_fast
        print()

        targets = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.mlp.gate.weight",  # router (small)
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.visual.merger.linear_fc1.weight",
        ]

        print(f"{'tensor':<60} {'shape':<14} "
              f"{'mq4 mse':>11} {'q40 mse':>11} {'q4k mse':>11} "
              f"{'mq4/q40':>9} {'mq4/q4k':>9}")
        print("-" * 145)

        for name in targets:
            if name not in header:
                print(f"  (missing: {name})")
                continue
            info = header[name]
            shape = info["shape"]
            dtype = info["dtype"]
            doff = info["data_offsets"]
            n_elem = shape[0] * shape[1]
            f.seek(body_offset + doff[0])
            raw = f.read(2 * n_elem)
            arr = f16_array(raw, n_elem) if dtype == "F16" else bf16_array(raw, n_elem)

            if shape[1] % 256 != 0:
                print(f"  (skipping non-256-aligned: {name} shape {shape})")
                continue

            n_blocks = n_elem // 256
            arr = arr[: n_blocks * 256].reshape(n_blocks, 256)

            # Subsample for speed
            if n_blocks > 4096:
                idx = np.linspace(0, n_blocks - 1, 4096, dtype=int)
                arr = arr[idx]
                n_blocks = arr.shape[0]

            # MQ4: rotate, quantize, inverse-rotate. The inverse FWHT is the same
            # operation with signs1 and signs2 swapped (per kernels/turbo_common.h:57).
            rot = fwht(arr, signs1, signs2)
            rot_q = quant_uniform_4bit(rot)
            rec = fwht(rot_q, signs2, signs1)
            mq4_mse = float(((arr - rec) ** 2).mean())

            # Q4_0 (per-32 sub-block, single scale, no rotation)
            q40 = quant_per_32(arr)
            q40_mse = float(((arr - q40) ** 2).mean())

            # Q4_K-like (also per-32 here as upper bound; real Q4_K is similar but with
            # quantized 6-bit scales-of-scales that cost ~0.05 dB)
            q4k_mse = q40_mse  # same in this approximation

            shape_str = f"{shape[0]}x{shape[1]}"
            print(f"{name[:60]:<60} {shape_str:<14} "
                  f"{mq4_mse:>11.4e} {q40_mse:>11.4e} {q4k_mse:>11.4e} "
                  f"{mq4_mse/q40_mse:>8.2f}x {mq4_mse/q4k_mse:>8.2f}x")


if __name__ == "__main__":
    main()
