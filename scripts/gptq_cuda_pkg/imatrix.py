"""Imatrix GGUF reader.

The imatrix file is a llama.cpp `--imatrix` output as a GGUF v3 file
containing per-linear-layer `.in_sum2` (F32 vector of length K) and
`.counts` (F32 scalar). Schema documented in
`crates/hipfire-quantize/src/main.rs::load_imatrix`.

Hipfire only consumes the `.in_sum2` entries (the calibration-averaged
sum of squared activations per channel — `N_tok · RMS_act²[j]`). The
`.counts` tensor is informational only.

Provides a `{ggml_name → numpy.ndarray[float32]}` map keyed by the BASE
tensor name (with the `.in_sum2` suffix stripped). MoE entries
(shape `[k, n_mat]` with `n_mat > 1`) are skipped with a warning;
hipfire's quantizer is dense-only as of master.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .names import to_ggml_name


def load_imatrix(path: Path) -> dict[str, np.ndarray]:
    """Read the imatrix GGUF; return `{ggml_name: in_sum2_array}`.

    Imports `gguf` lazily so the rest of the package stays usable
    without it installed (e.g. for unit tests of the algo module).
    """
    from gguf import GGUFReader  # type: ignore

    reader = GGUFReader(str(path))
    out: dict[str, np.ndarray] = {}
    skipped_moe = 0
    skipped_nonf32 = 0
    for t in reader.tensors:
        if not t.name.endswith(".in_sum2"):
            continue
        base = t.name[:-len(".in_sum2")]
        # dtype == 0 is F32 in ggml's enum. Anything else: skip.
        if int(t.tensor_type) != 0:
            skipped_nonf32 += 1
            continue
        # Shape is [K] for dense; [K, n_mat] for MoE.
        if len(t.shape) >= 2 and int(t.shape[1]) != 1:
            skipped_moe += 1
            continue
        arr = np.array(t.data, copy=True).astype(np.float32, copy=False)
        out[base] = arr
    if not out:
        raise RuntimeError(f"imatrix at {path} contains no .in_sum2 entries")
    if skipped_moe or skipped_nonf32:
        print(f"imatrix: loaded {len(out)} entries from {path} "
              f"(skipped {skipped_moe} MoE, {skipped_nonf32} non-F32)")
    return out


def imatrix_weights_for(
    imatrix_map: dict[str, np.ndarray],
    safetensors_name: str,
) -> np.ndarray | None:
    """Look up imatrix `.in_sum2` values for a safetensors weight name.

    Returns None when the tensor has no ggml mapping (norms, A_log,
    conv1d, dt_bias) OR the imatrix doesn't carry it (rare — calibration
    corpus didn't exercise the channel). Caller falls back to
    identity AWQ (s = 1) in either case.
    """
    ggml_name = to_ggml_name(safetensors_name)
    if ggml_name is None:
        return None
    return imatrix_map.get(ggml_name)
