#!/usr/bin/env python3
"""Engine-floor subtraction-framework validation for KLD-based quant quality.

Tests whether `KLD(any hipfire quant) − KLD(hipfire Q8_floor)` gives a
clean "weight-quantization-only cost" comparable to llama.cpp's analogous
subtraction. Companion to the pattern-hunt investigation (which showed
hipfire's kernel arithmetic is at the fp64 ideal; the engine "drift" is
HF's bf16-cast pattern + DeltaNet recurrence amplification).

Inputs are existing HFKSEQ kldseq files produced by:
  - `eval_hipfire`     for hipfire candidates
  - `eval_gguf`        for llama.cpp anchors

Usage:
  scripts/kld_engine_floor_subtraction.py \\
      --hipfire-q8 <0.8B Q8 kldseq>  \\
      --hipfire-q8-9b <9B Q8 kldseq, optional>  \\
      --hipfire-08b-dir <dir of 0.8B quant kldseqs>  \\
      --hipfire-9b-dir <dir of 9B quant kldseqs>  \\
      --gguf-9b-dir <dir of llama.cpp 9B anchors>  \\
      --dense-control <Q3-0.6B Q8 kldseq, optional>

Without --hipfire-q8-9b, prints conservative-bracket extrapolation
(0.08 to 0.5 nats) for the 9B engine floor.

See `docs/investigations/2026-05-13-engine-drift-residual-audit/06-engine-floor-subtraction-validation.md`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Local import — kldref_format.py lives alongside the harness scripts.
sys.path.insert(
    0, str(Path(__file__).parent.parent / "benchmarks" / "quality-baselines" / "harness")
)
from kldref_format import read_per_seq_kld  # noqa: E402


def mean_kld(path, take=None):
    means, _, _ = read_per_seq_kld(path)
    if take:
        means = means[:take]
    return float(np.mean(means)), len(means)


def collect_dir(dir_path, pattern="*.kldseq"):
    if dir_path is None:
        return {}
    out = {}
    for f in sorted(Path(dir_path).glob(pattern)):
        out[f.stem] = f
    return out


def print_table(title, rows, headers):
    print(f"\n=== {title} ===\n")
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt.format(*[str(c) for c in r]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hipfire-q8", required=True,
                   help="hipfire Q3.5-0.8B Q8 kldseq (the measured DeltaNet engine floor)")
    p.add_argument("--hipfire-q8-9b", default=None,
                   help="hipfire Q3.5-9B Q8 kldseq (optional — without it, "
                        "the 9B cross-engine comparison uses a conservative bracket)")
    p.add_argument("--hipfire-08b-dir", default=None,
                   help="directory of Q3.5-0.8B hipfire quant kldseqs")
    p.add_argument("--hipfire-9b-dir", default=None,
                   help="directory of Q3.5-9B hipfire quant kldseqs")
    p.add_argument("--gguf-9b-dir", default=None,
                   help="directory of Q3.5-9B llama.cpp anchor kldseqs (filenames "
                        "contain 'gguf-' substring)")
    p.add_argument("--dense-control", default=None,
                   help="Q3-0.6B Q8 hipfire kldseq (no-DeltaNet dense control)")
    args = p.parse_args()

    floor_08b, floor_n = mean_kld(args.hipfire_q8)
    print(f"hipfire Q3.5-0.8B Q8 floor (DeltaNet engine floor): {floor_08b:.4f} (n={floor_n})")
    if args.dense_control:
        dense, dn = mean_kld(args.dense_control)
        print(f"hipfire Q3-0.6B Q8 dense control (no-DeltaNet):     {dense:.4f} (n={dn})")
        print(f"  → DeltaNet amplifies engine floor by {floor_08b / max(dense, 1e-6):.0f}×")

    if args.hipfire_08b_dir:
        rows = []
        for stem, p_ in collect_dir(args.hipfire_08b_dir).items():
            if "q8" in stem.lower():
                continue
            k, n = mean_kld(p_)
            rows.append([stem, f"{k:.4f}", f"{k - floor_08b:.4f}", f"{(k-floor_08b)/floor_08b*100:.0f}%"])
        rows.sort(key=lambda r: float(r[1]))
        print_table("Hipfire Q3.5-0.8B internal subtraction (KLD − Q8 floor)",
                    rows, ["variant", "KLD", "− Q8 floor", "% above Q8"])

    if args.hipfire_9b_dir and args.gguf_9b_dir:
        gguf = {}
        for stem, p_ in collect_dir(args.gguf_9b_dir).items():
            k, _ = mean_kld(p_)
            gguf[stem] = k
        # Find Q8_0 baseline
        q8_0 = next((k for stem, k in gguf.items() if "q8_0" in stem.lower()), None)
        if q8_0 is None:
            print("WARNING: no llama.cpp Q8_0 in --gguf-9b-dir; skipping cross-engine block")
            return

        print(f"\nllama.cpp Q3.5-9B Q8_0 engine floor: {q8_0:.4f}")
        rows_gguf = []
        for stem, k in sorted(gguf.items(), key=lambda x: x[1]):
            rows_gguf.append([stem, f"{k:.4f}", f"{k - q8_0:.4f}"])
        print_table("llama.cpp Q3.5-9B anchors (KLD − Q8_0)",
                    rows_gguf, ["variant", "KLD", "− Q8_0"])

        if args.hipfire_q8_9b:
            floor_9b, floor_9b_n = mean_kld(args.hipfire_q8_9b)
            print(f"\nhipfire Q3.5-9B Q8 floor (measured, n={floor_9b_n}): {floor_9b:.4f}")
            brackets = [floor_9b]
            bracket_labels = ["measured"]
        else:
            # Conservative bracket: 0.08 (= 0.8B Q8 floor, lower bound) to 0.5 (linear
            # scaling × 4× depth × discount, upper bound).
            brackets = [0.08, 0.5]
            bracket_labels = ["floor=0.08 (lo)", "floor=0.50 (hi)"]
            print(f"\nhipfire Q3.5-9B Q8 floor: NOT MEASURED — using conservative bracket")
            print(f"  lo: {brackets[0]:.2f}  hi: {brackets[1]:.2f}")

        rows = []
        for stem, p_ in collect_dir(args.hipfire_9b_dir).items():
            k, _ = mean_kld(p_)
            deltas = [k - b for b in brackets]
            rows.append([stem, f"{k:.4f}", *[f"{d:.4f}" for d in deltas]])
        rows.sort(key=lambda r: float(r[1]))
        print_table("Hipfire Q3.5-9B (raw KLD, then KLD − floor under each bracket)",
                    rows, ["variant", "raw KLD", *bracket_labels])

        # Cross-engine pairs at matching bit widths
        print("\n=== Cross-engine ratio at equivalent bit widths ===\n")
        print(f"  (hipfire Δ / llama.cpp Δ at the WORST-CASE hipfire floor — best for hipfire)")
        print()
        pairs = []
        hip9 = {stem: mean_kld(p_)[0] for stem, p_ in collect_dir(args.hipfire_9b_dir).items()}

        def pick(quants, *substrs):
            for stem, k in quants.items():
                if all(s in stem.lower() for s in substrs):
                    return (stem, k)
            return (None, None)

        # Match by quant family
        comparisons = [
            ("6-bit", ("mq6",), ("q6_k",)),
            ("4-bit (vs Q4_K_M)", ("mq4", "prefill"), ("q4_k_m",)),
            ("4-bit (vs UD-Q4_K_XL)", ("mq4", "prefill"), ("ud-q4_k_xl",)),
            ("3-bit (vs UD-Q3_K_XL)", ("mq3",), ("ud-q3_k_xl",)),
        ]
        rows_xc = []
        worst_floor = max(brackets)
        for label, hip_subs, gg_subs in comparisons:
            hs, hk = pick(hip9, *hip_subs)
            gs, gk = pick(gguf, *gg_subs)
            if hs is None or gs is None:
                continue
            hd = hk - worst_floor
            gd = gk - q8_0
            ratio = hd / max(gd, 1e-6)
            rows_xc.append([label, f"{hk:.4f}", f"{hd:.4f}", f"{gk:.4f}", f"{gd:.4f}", f"{ratio:.1f}×"])
        print_table("Cross-engine quant-quality ratio (after floor subtraction)",
                    rows_xc, ["bit width", "hip KLD", "hip Δ", "gguf KLD", "gguf Δ", "ratio"])


if __name__ == "__main__":
    main()
