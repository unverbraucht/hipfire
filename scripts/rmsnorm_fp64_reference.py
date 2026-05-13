#!/usr/bin/env python3
"""D0.3 — fp64 reference threshold for rmsnorm drift (pattern-hunt plan).

Computes the irreducible fp32-arithmetic floor for Qwen3.5 input_layernorm
at LA-0 by running rmsnorm in fp64 with HF's exact formula on the same
bit-exact input HF used. Reports rL2 of three pairs:

    (HF stage-1 fp32)        vs (fp64 ref, HF BF16 weight)
        => HF's own fp32-arithmetic floor for this kernel + this input
    (hipfire stage-1 fp32)   vs (fp64 ref, HF BF16 weight)
        => hipfire's TOTAL drift from the fp64 oracle (matches HF setup)
    (hipfire stage-1 fp32)   vs (fp64 ref, hipfire's F16-stored weight)
        => hipfire's fp32 floor GIVEN its F16 weight storage choice
        (separates F16-vs-BF16 weight storage from kernel arithmetic noise)

Day 5 success target becomes ≤ 2× the first number (HF's own fp32 floor).

Inputs:
    --hf-dump        HIPFIRE_DUMP_LA_STAGES dump from
                     scripts/dump_hf_la_stages.py (has stage 0 + stage 1)
    --hip-stage1     Stage-1 dump from rmsnorm_isolated (--kernel rmsnorm_f32)
    --hf-model       HF model directory (has safetensors + config.json)
    --hfq-weight     Effective hipfire weight dumped via
                     `rmsnorm_isolated --dump-weight <path>` (post +1.0).
                     Flat f32 buffer, length = hidden_dim.

Outputs: prints the three rL2 numbers and per-position stats.

No GPU needed.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


HEADER_FMT = "<8I"
HEADER_BYTES = struct.calcsize(HEADER_FMT)


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
        if i + nbytes > len(data):
            raise ValueError(f"truncated record at offset {i}")
        arr = np.frombuffer(data[i:i + nbytes], dtype=np.float32).copy()
        i += nbytes
        out[(int(layer), int(stage), int(pos))] = arr
    return out


def rel_l2(a, b):
    """rL2(a, b) computed in fp64."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = a - b
    num = np.sqrt((diff * diff).sum())
    den = max(np.sqrt((b * b).sum()), 1e-12)
    return float(num / den)


def load_hf_norm_weight(hf_model_dir, layer):
    """Returns the BF16 input_layernorm.weight cast to fp32 (no +1.0 offset yet)."""
    import safetensors.torch
    import torch
    # Qwen3.5 nests the language model under `model.language_model.`. Try
    # multiple name conventions so the script works across model variants.
    candidates = [
        f"model.language_model.layers.{layer}.input_layernorm.weight",
        f"model.layers.{layer}.input_layernorm.weight",
        f"layers.{layer}.input_layernorm.weight",
    ]
    files = sorted(Path(hf_model_dir).glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {hf_model_dir}")
    # Just load the first file (Qwen3.5-0.8B is a single shard); if a
    # multi-shard model is used later, consult index json.
    if len(files) > 1:
        index_path = Path(hf_model_dir) / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"multi-shard model needs an index")
        wmap = json.loads(index_path.read_text())["weight_map"]
        for c in candidates:
            if c in wmap:
                shard = Path(hf_model_dir) / wmap[c]
                tensors = safetensors.torch.load_file(str(shard))
                w = tensors[c]
                break
        else:
            raise KeyError(f"none of {candidates} in weight_map")
    else:
        tensors = safetensors.torch.load_file(str(files[0]))
        w = None
        for c in candidates:
            if c in tensors:
                w = tensors[c]
                break
        if w is None:
            sample = list(tensors.keys())[:5]
            raise KeyError(f"none of {candidates} in {files[0]}; sample keys: {sample}")
    if w.dtype != torch.bfloat16:
        print(f"  WARNING: expected BF16, got {w.dtype}", file=sys.stderr)
    # Cast to fp32 (lossless for BF16→FP32).
    return w.to(dtype=torch.float32).numpy(), str(w.dtype)


def load_hfq_effective_weight(path):
    """Returns the effective (post-+1.0) hipfire weight as fp32. The file is
    produced by `rmsnorm_isolated --dump-weight <path>` — a flat f32 buffer."""
    raw = Path(path).read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"--hfq-weight file size {len(raw)} not multiple of 4")
    return np.frombuffer(raw, dtype=np.float32).copy()


def rmsnorm_fp64(x_f32, weight_plus_1_f64, eps_f64):
    """Reference rmsnorm in fp64, matching Qwen3.5RMSNorm.forward:
        variance = (x_fp32_cast_to_fp64).pow(2).mean(-1)
        out = x * rsqrt(variance + eps) * weight
    Output cast back to fp32 at the end (matching HF dump behavior).
    Returns fp32 ndarray.
    """
    x64 = x_f32.astype(np.float64)
    var = (x64 * x64).mean()
    rms = 1.0 / np.sqrt(var + eps_f64)
    out64 = x64 * rms * weight_plus_1_f64
    return out64.astype(np.float32)


def rmsnorm_fp64_kept64(x_f32, weight_plus_1_f64, eps_f64):
    """Same as rmsnorm_fp64 but returns fp64 (no final fp32 cast). For
    measuring the fp32 cast contribution separately."""
    x64 = x_f32.astype(np.float64)
    var = (x64 * x64).mean()
    rms = 1.0 / np.sqrt(var + eps_f64)
    return x64 * rms * weight_plus_1_f64


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dump", required=True,
                   help="HF LA-stages dump (has stage 0 + stage 1)")
    p.add_argument("--hip-stage1", required=True,
                   help="hipfire stage-1 dump from rmsnorm_isolated baseline")
    p.add_argument("--hf-model", required=True,
                   help="HF model dir (safetensors + config.json)")
    p.add_argument("--hfq-weight", required=True,
                   help="effective hipfire weight binary "
                        "(from `rmsnorm_isolated --dump-weight`)")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--n-positions", type=int, default=2048)
    p.add_argument("--eps", type=float, default=None,
                   help="rms_norm_eps; default = read from HF config.json")
    args = p.parse_args()

    layer = args.layer
    hf_dump = Path(args.hf_dump)
    hip_dump = Path(args.hip_stage1)
    hf_model = Path(args.hf_model)
    hfq_w_path = Path(args.hfq_weight)

    eps = args.eps
    if eps is None:
        cfg = json.loads((hf_model / "config.json").read_text())
        eps = float(cfg.get("rms_norm_eps", 1e-6))
    print(f"eps = {eps}")

    # ----- load dumps -----
    print(f"reading HF dump: {hf_dump}")
    hf_recs = read_dump(hf_dump)
    print(f"  {len(hf_recs)} records")
    print(f"reading hipfire stage-1 dump: {hip_dump}")
    hip_recs = read_dump(hip_dump)
    print(f"  {len(hip_recs)} records")

    # ----- load weights -----
    print(f"loading HF weight from {hf_model}")
    hf_w_raw, hf_w_dtype = load_hf_norm_weight(hf_model, layer)
    print(f"  shape={hf_w_raw.shape} dtype={hf_w_dtype} (cast to fp32, pre +1.0)")
    print(f"loading hipfire effective weight from {hfq_w_path}")
    hfq_w_eff_f32 = load_hfq_effective_weight(hfq_w_path)
    print(f"  shape={hfq_w_eff_f32.shape} (already post-+1.0)")

    # HF applies +1.0 inside forward; hipfire pre-applied it at load time.
    hf_w_eff_f64 = hf_w_raw.astype(np.float64) + 1.0
    hfq_w_eff_f64 = hfq_w_eff_f32.astype(np.float64)

    delta_w = hf_w_eff_f64 - hfq_w_eff_f64
    print(f"  weight rL2 (HF BF16 vs hipfire F16): "
          f"{np.sqrt((delta_w**2).sum() / max((hf_w_eff_f64**2).sum(), 1e-12)):.3e}")
    print(f"  weight max-abs-delta: {np.max(np.abs(delta_w)):.3e}")

    # ----- per-position comparison -----
    n_processed = 0
    rl_hf_vs_ref = []      # HF fp32 vs fp64 reference (HF weight)  — HF's own floor
    rl_hip_vs_ref = []     # hipfire fp32 vs fp64 reference (HF weight)  — hipfire's total drift
    rl_hip_vs_ref_hfq_w = []  # hipfire fp32 vs fp64 reference (hipfire F16 weight) — hipfire fp32 floor given F16 weight
    rl_hip_vs_hf_fp32 = [] # hipfire fp32 vs HF fp32  — sanity (should match compare_la_stages.py)

    for pos in range(args.n_positions):
        key_x = (layer, 0, pos)
        key_y_hf = (layer, 1, pos)
        key_y_hip = (layer, 1, pos)
        if key_x not in hf_recs or key_y_hf not in hf_recs or key_y_hip not in hip_recs:
            continue
        x_f32 = hf_recs[key_x]
        y_hf_f32 = hf_recs[key_y_hf]
        y_hip_f32 = hip_recs[key_y_hip]

        # fp64 reference with HF BF16 weight (matches HF's setup, modulo
        # arithmetic precision).
        ref_hf_f32 = rmsnorm_fp64(x_f32, hf_w_eff_f64, eps)
        # fp64 reference with hipfire's F16-stored weight (matches hipfire's
        # weight, modulo arithmetic precision).
        ref_hfq_f32 = rmsnorm_fp64(x_f32, hfq_w_eff_f64, eps)

        rl_hf_vs_ref.append(rel_l2(y_hf_f32, ref_hf_f32))
        rl_hip_vs_ref.append(rel_l2(y_hip_f32, ref_hf_f32))
        rl_hip_vs_ref_hfq_w.append(rel_l2(y_hip_f32, ref_hfq_f32))
        rl_hip_vs_hf_fp32.append(rel_l2(y_hip_f32, y_hf_f32))
        n_processed += 1

    if n_processed == 0:
        sys.exit("no positions had all three records (HF stage 0, HF stage 1, hipfire stage 1)")

    def stats(label, vals):
        arr = np.array(vals)
        return (
            f"{label:<55} n={len(arr):>4}  mean={arr.mean():.6f}  "
            f"max={arr.max():.6f}  min={arr.min():.6f}"
        )

    print()
    print("== Per-position rL2 (fp64 reference + sanity check) ==")
    print(stats("HF fp32 vs fp64-ref (HF BF16 weight)         ", rl_hf_vs_ref))
    print(stats("hipfire fp32 vs fp64-ref (HF BF16 weight)    ", rl_hip_vs_ref))
    print(stats("hipfire fp32 vs fp64-ref (hipfire F16 weight)", rl_hip_vs_ref_hfq_w))
    print(stats("[sanity] hipfire fp32 vs HF fp32              ", rl_hip_vs_hf_fp32))

    print()
    hf_floor = float(np.mean(rl_hf_vs_ref))
    hip_total = float(np.mean(rl_hip_vs_ref))
    hip_fp32_floor_given_hfq_w = float(np.mean(rl_hip_vs_ref_hfq_w))
    print(f"Day-5 success target (≤ 2× HF fp32 floor): "
          f"≤ {2 * hf_floor:.6f}")
    print(f"  current hipfire drift vs fp64-ref (HF weight): {hip_total:.6f}")
    print(f"  current hipfire fp32 floor (own weight): {hip_fp32_floor_given_hfq_w:.6f}")
    print(f"  ratio (hipfire total / HF floor): {hip_total / max(hf_floor, 1e-12):.2f}x")
    if hip_total <= 2 * hf_floor:
        print("  HIPFIRE ALREADY MEETS THE TARGET — pattern hunt may be solving a non-problem")
    else:
        residual_from_weight = hip_total - hip_fp32_floor_given_hfq_w
        print(f"  estimated F16-vs-BF16-weight-storage contribution: "
              f"{residual_from_weight:.6f} (= hipfire-total minus hipfire-own-weight floor)")
        print(f"  estimated kernel-arithmetic contribution: "
              f"{hip_fp32_floor_given_hfq_w:.6f}")


if __name__ == "__main__":
    main()
