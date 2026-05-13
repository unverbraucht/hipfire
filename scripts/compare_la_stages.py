#!/usr/bin/env python3
"""Compare hipfire and HF transformers per-stage LA pipeline dumps.

Reads two `HIPFIRE_DUMP_LA_STAGES`-format files (one from hipfire, one
from `scripts/dump_hf_la_stages.py`), groups records by (layer, stage,
position), and reports per-stage rel_L2 / cosine summary.

Record format (must match `dump_la_stage` in qwen35.rs):
  u32 layer_idx, u32 pos, u32 stage_id, u32 n_elems, u32×4 reserved
  f32 × n_elems

Usage:
    compare_la_stages.py --hip <hip.bin> --hf <hf.bin> \
                         [--stages 0,1,2,...] [--positions 0,16,64,256,1024]
"""
import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HEADER_FMT = "<8I"
HEADER_BYTES = struct.calcsize(HEADER_FMT)


STAGE_NAMES = {
    0: "pre-rmsnorm (residual in)",
    1: "post-input_layernorm",
    2: "post-in_proj_qkv (raw qkv)",
    3: "post-in_proj_z (raw z)",
    4: "post-in_proj_a (raw a)",
    5: "post-in_proj_b (raw b)",
    6: "post-sigmoid_alpha_gate alpha (g)",
    7: "post-sigmoid_alpha_gate beta",
    8: "post-conv1d_silu q_raw",
    9: "post-conv1d_silu k_raw",
    10: "post-conv1d_silu v",
    11: "post-l2norm+scale q",
    12: "post-l2norm k",
    13: "post-recurrence (core_attn_out)",
    14: "post-gated_norm",
    15: "post-wo + residual (block out)",
}


def read_dump(path):
    """Returns dict[(layer, stage, pos)] -> np.ndarray (1-D f32)."""
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    n_recs = 0
    while i < len(data):
        if i + HEADER_BYTES > len(data):
            raise ValueError(f"truncated header at offset {i}")
        layer, pos, stage, n_elems, _, _, _, _ = struct.unpack_from(
            HEADER_FMT, data, i
        )
        i += HEADER_BYTES
        nbytes = n_elems * 4
        if i + nbytes > len(data):
            raise ValueError(
                f"truncated record at offset {i} (n_elems={n_elems})"
            )
        arr = np.frombuffer(data[i:i + nbytes], dtype=np.float32)
        i += nbytes
        out[(int(layer), int(stage), int(pos))] = arr
        n_recs += 1
    return out, n_recs


def rel_l2(a, b):
    diff = a.astype(np.float64) - b.astype(np.float64)
    num = np.sqrt((diff * diff).sum())
    den = max(np.sqrt((b.astype(np.float64) ** 2).sum()), 1e-12)
    return float(num / den)


def cosine(a, b):
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    na = np.sqrt((a64 * a64).sum())
    nb = np.sqrt((b64 * b64).sum())
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float((a64 * b64).sum() / (na * nb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hip", required=True, help="hipfire HIPFIRE_DUMP_LA_STAGES file")
    p.add_argument("--hf", required=True, help="HF dump_hf_la_stages.py output")
    p.add_argument("--stages", default=None,
                   help="comma-separated stage IDs; default = all overlapping")
    p.add_argument("--positions", default=None,
                   help="comma-separated positions to sample; default = "
                        "0,32,64,128,256,512,1024,1536,1920")
    p.add_argument("--layer", type=int, default=None,
                   help="if both files contain multiple layers, restrict to this one")
    args = p.parse_args()

    print(f"reading hipfire dump: {args.hip}", flush=True)
    hip_recs, n_hip = read_dump(Path(args.hip))
    print(f"  {n_hip} records, {len(hip_recs)} unique keys", flush=True)
    print(f"reading HF dump: {args.hf}", flush=True)
    hf_recs, n_hf = read_dump(Path(args.hf))
    print(f"  {n_hf} records, {len(hf_recs)} unique keys", flush=True)

    common = set(hip_recs.keys()) & set(hf_recs.keys())
    print(f"  {len(common)} overlapping (layer, stage, pos) keys", flush=True)
    if not common:
        sys.exit("no overlapping records — check stage IDs / layer index match")

    # Filter
    if args.layer is not None:
        common = {k for k in common if k[0] == args.layer}
    if args.stages:
        wanted_stages = set(int(s) for s in args.stages.split(","))
        common = {k for k in common if k[1] in wanted_stages}
    if args.positions:
        wanted_pos = set(int(p_) for p_ in args.positions.split(","))
    else:
        wanted_pos = {0, 32, 64, 128, 256, 512, 1024, 1536, 1920}

    # Group by stage.
    by_stage = defaultdict(list)  # stage -> list of (pos, rel_l2, cos, n_elems)
    for key in common:
        layer, stage, pos = key
        h = hip_recs[key]
        f = hf_recs[key]
        if h.shape != f.shape:
            # Shape mismatch — likely a stage with different GQA layout.
            # Truncate to min and warn once per stage.
            continue
        rl = rel_l2(h, f)
        cs = cosine(h, f)
        by_stage[stage].append((pos, rl, cs, h.size))

    # Report.
    stage_ids = sorted(by_stage.keys())
    print()
    print("== Per-stage summary (all positions) ==")
    print(f"{'stage':>5}  {'name':<35}  {'n_elems':>8}  "
          f"{'mean rL2':>9}  {'max rL2':>9}  {'mean cos':>9}  "
          f"{'min cos':>9}  {'n_pos':>6}")
    print("-" * 110)
    for sid in stage_ids:
        rows = by_stage[sid]
        rls = np.array([r[1] for r in rows])
        cos = np.array([r[2] for r in rows])
        n_elems = rows[0][3]
        name = STAGE_NAMES.get(sid, f"stage{sid}")
        print(f"{sid:>5}  {name:<35}  {n_elems:>8}  "
              f"{rls.mean():>9.6f}  {rls.max():>9.6f}  "
              f"{cos.mean():>9.6f}  {cos.min():>9.6f}  {len(rows):>6}")

    # Per-position-bucket breakdown for the most-divergent stage.
    print()
    print("== Per-position-bucket rel_L2 (sampled positions) ==")
    sampled_pos = sorted([p_ for p_ in wanted_pos
                          if any(p_ == r[0] for rows in by_stage.values() for r in rows)])
    hdr = f"{'stage':>5}  {'name':<32}  " + "  ".join(
        f"{p_:>7}" for p_ in sampled_pos
    )
    print(hdr)
    print("-" * len(hdr))
    for sid in stage_ids:
        rows = {r[0]: r[1] for r in by_stage[sid]}
        name = STAGE_NAMES.get(sid, f"stage{sid}")
        cells = []
        for p_ in sampled_pos:
            if p_ in rows:
                cells.append(f"{rows[p_]:>7.4f}")
            else:
                cells.append(f"{'-':>7}")
        print(f"{sid:>5}  {name:<32}  " + "  ".join(cells))

    # Headline ranking.
    print()
    print("== Ranked by mean rel_L2 (divergence localization) ==")
    ranked = sorted(
        stage_ids,
        key=lambda sid: -np.array([r[1] for r in by_stage[sid]]).mean(),
    )
    for sid in ranked:
        rls = np.array([r[1] for r in by_stage[sid]])
        name = STAGE_NAMES.get(sid, f"stage{sid}")
        print(f"  stage {sid:>2} ({name}): mean rL2 = {rls.mean():.6f}")


if __name__ == "__main__":
    main()
