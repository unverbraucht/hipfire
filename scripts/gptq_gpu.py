#!/usr/bin/env python3
"""GPTQ on GPU — one-shot quantize-time math for hipfire MQ4G256.

Replaces the ~14h CPU-bound GPTQ inner loop in
`crates/hipfire-quantize/src/gptq.rs` with a Python+PyTorch implementation
that runs on either CUDA (NVIDIA) or HIP/ROCm (AMD) via the standard
torch.cuda API. Target: full 9B GPTQ pass in 1-3h wall on dual 5070 Ti
or comparable AMD MI50 (per `docs/plans/gptq_cuda.md` /
`docs/plans/gptq_mi50.md`).

Produces a "precomputed GPTQ" manifest directory consumed downstream by
`hipfire-quantize --precomputed-gptq-path <dir>`:

  <output-dir>/
    weights.safetensors        # all model tensors (BF16); MQ4G256-eligible
                               # tensors are post-AWQ-scale + post-FWHT-rotate
                               # + post-GPTQ-update.
    awq_scales.safetensors     # F16 awq_scale vectors for AWQ-eligible tensors;
                               # key = "<weight_name>.awq_scale".
    frozen_grids.safetensors   # per-256-block (scale, min_val) pairs; key =
                               # "<weight_name>.grids", shape [n_blocks, 2] F16.
    manifest.json              # alpha / damp / source md5 / per-tensor stats.

Usage:
    python scripts/gptq_gpu.py \\
        --input ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \\
        --hessian /data/hipfire-refs/qwen3.5-9b-bf16.hessian.bin \\
        --imatrix benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.imatrix.gguf \\
        --alpha 0.55 \\
        --output ~/.hipfire/gptq-precomputed/qwen3.5-9b-mq4-awq-gptq-q8conv-f2/

Per-GPU round-robin scheduling: a tensor's `idx % n_devices` selects
which device runs it. Cards work asynchronously; CPU collects results
in tensor order. Plan §4.8 — `non_blocking=True` for cross-device hops.

Streaming RAM hygiene (plan §4.4): each tensor is loaded → GPU →
GPTQ'd → moved to CPU → output buffer → freed. System RAM peak under
~4 GB even on 9B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Local package import — allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gptq_gpu_pkg.algo import (  # noqa: E402
    compute_awq_scales,
    gen_fwht_signs,
)
from gptq_gpu_pkg.hfhs import HessianSidecar  # noqa: E402
from gptq_gpu_pkg.imatrix import imatrix_weights_for, load_imatrix  # noqa: E402
from gptq_gpu_pkg.names import (  # noqa: E402
    awq_eligible,
    to_hfhs_key,
)
from gptq_gpu_pkg.algo import CholeskyFailedError  # noqa: E402
from gptq_gpu_pkg.pipeline import gptq_one_tensor  # noqa: E402


# ─── Eligibility ──────────────────────────────────────────────────────────

def is_mq4g256_eligible(name: str, shape: tuple[int, ...]) -> bool:
    """Mirror of Rust's MQ4G256 routing for a "Base" tensor (no K-map
    override). True when this tensor would end up packed as MQ4G256 in
    a normal `--format mq4` Rust run.

    Excludes embed/lm_head/norms/conv1d/A_log/dt_bias and any tensor
    whose last-dim K is not divisible by 256.
    """
    if len(shape) != 2:
        return False
    if shape[1] % 256 != 0:
        return False
    if "embed_tokens" in name:
        return False
    if name.endswith("lm_head.weight"):
        return False
    if "conv1d" in name:
        return False
    if name.endswith("A_log") or name.endswith("dt_bias"):
        return False
    if name.endswith("norm.weight") or ".norm." in name:
        return False
    # MoE router goes to Q8 in Rust's default; skip GPTQ for it too.
    if name.endswith("mlp.gate.weight") or name.endswith("router.weight"):
        return False
    return True


# ─── Source model: index loader (safetensors index + per-tensor open) ────

def build_input_index(model_dir: Path) -> tuple[dict[str, Path], dict[str, tuple[int, ...]]]:
    """Resolve `model.safetensors.index.json` (sharded) or fall back to
    a single `.safetensors` file. Returns `(name→file_path, name→shape)`.
    """
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f:
            j = json.load(f)
        weight_map = j["weight_map"]
        file_map: dict[str, Path] = {n: model_dir / fn for n, fn in weight_map.items()}
    else:
        # Single-shard model.
        candidates = sorted(model_dir.glob("*.safetensors"))
        if len(candidates) != 1:
            raise IOError(
                f"no safetensors.index.json and {len(candidates)} .safetensors files in {model_dir}"
            )
        with safe_open(candidates[0], framework="pt") as st:
            file_map = {k: candidates[0] for k in st.keys()}

    # Per-tensor shape lookup (open each file once, query keys).
    shape_map: dict[str, tuple[int, ...]] = {}
    seen_files: set[Path] = set()
    files_grouped: dict[Path, list[str]] = {}
    for n, p in file_map.items():
        files_grouped.setdefault(p, []).append(n)
    for p, names in files_grouped.items():
        with safe_open(p, framework="pt") as st:
            for n in names:
                shape_map[n] = tuple(st.get_slice(n).get_shape())
        seen_files.add(p)
    return file_map, shape_map


def load_tensor(file_path: Path, name: str) -> torch.Tensor:
    """Load one tensor from its shard. Returns CPU tensor in its
    native dtype (BF16 for the Qwen3.5 sources here)."""
    with safe_open(file_path, framework="pt") as st:
        return st.get_tensor(name)


# ─── Manifest writer ──────────────────────────────────────────────────────

@dataclass
class TensorStats:
    name: str
    shape: tuple[int, ...]
    eligible: bool
    has_hessian: bool
    awq_eligible: bool
    effective_damp: float | None = None
    mse_vs_original: float | None = None
    clamps_below: int = 0
    clamps_above: int = 0
    wall_s: float = 0.0
    device: str = ""
    error: str | None = None


def write_manifest(
    output_dir: Path,
    *,
    weights_bf16: dict[str, torch.Tensor],
    awq_scales: dict[str, torch.Tensor],
    frozen_grids: dict[str, torch.Tensor],
    metadata: dict,
) -> None:
    """Write the four manifest files. Tensor dtypes:
      - weights: BF16
      - awq_scales: F16 (length K vectors keyed by `<name>.awq_scale`)
      - frozen_grids: F16 (shape [n_blocks, 2] keyed by `<name>.grids`)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(weights_bf16, str(output_dir / "weights.safetensors"))
    save_file(awq_scales, str(output_dir / "awq_scales.safetensors"))
    save_file(frozen_grids, str(output_dir / "frozen_grids.safetensors"))
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)


# ─── Source-file md5 for manifest provenance ──────────────────────────────

def md5_first_1mb(path: Path) -> str:
    """Cheap-ish source fingerprint for manifest provenance — first
    1 MiB md5. Full md5 of a 9B safetensors set is minutes; this is
    sufficient to detect "manifest produced against a different source"."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()


# ─── Orchestrator ─────────────────────────────────────────────────────────

def _start_watchdog(timeout_sec: int, last_progress: dict) -> None:
    """Daemon thread that SIGKILLs self if no `last_progress["ts"]` update
    happens within `timeout_sec` seconds.

    Why SIGKILL: when a CUDA kernel hangs (e.g. cuSOLVER pathological
    case on sm_120 we hit 2026-05-17), Python's main thread is blocked
    in `cudaStreamSynchronize` and ignores SIGTERM. SIGKILL is the only
    reliable way out. Recovery is via `--resume <dir>` on the next
    invocation, which picks up from the most recent checkpoint.
    """
    import threading, os, signal

    def _loop():
        while True:
            time.sleep(30)
            stalled_for = time.time() - last_progress["ts"]
            if stalled_for > timeout_sec:
                msg = (f"\n[watchdog] no per-tensor progress for "
                       f"{stalled_for:.0f}s > timeout {timeout_sec}s — "
                       f"SIGKILL'ing PID {os.getpid()}. "
                       f"Resume with `--resume <output_dir>`.\n")
                print(msg, file=sys.stderr)
                sys.stderr.flush()
                os.kill(os.getpid(), signal.SIGKILL)

    t = threading.Thread(target=_loop, daemon=True, name="gptq-watchdog")
    t.start()


def _load_resume(
    output_dir: Path, verbose: bool
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], set[str]]:
    """Load any existing partial manifest at `output_dir`.

    Returns (weights, awq_scales, frozen_grids, set_of_done_names).
    `set_of_done_names` is the set of weight tensor names present in
    `weights.safetensors` — these are skipped on the resumed run.

    All four files (weights, awq_scales, frozen_grids, manifest.json)
    are read best-effort: if any is corrupt or missing, we fall back to
    treating the corresponding piece as empty. A killed process during
    a checkpoint write may leave a truncated safetensors that fails to
    open — `_checkpoint_partial` writes to a `.tmp` sibling and renames
    atomically to avoid this, but ancient checkpoints from before that
    patch could still be corrupt.
    """
    weights: dict[str, torch.Tensor] = {}
    awq_scales: dict[str, torch.Tensor] = {}
    grids: dict[str, torch.Tensor] = {}
    done: set[str] = set()
    if not (output_dir / "weights.safetensors").exists():
        if verbose:
            print(f"[resume] no existing checkpoint at {output_dir} — starting fresh")
        return weights, awq_scales, grids, done

    def _safe_load(path: Path) -> dict[str, torch.Tensor]:
        if not path.exists():
            return {}
        try:
            with safe_open(str(path), framework="pt") as st:
                return {k: st.get_tensor(k) for k in st.keys()}
        except Exception as e:
            print(f"[resume] failed to load {path}: {e}; treating as empty",
                  file=sys.stderr)
            return {}

    weights = _safe_load(output_dir / "weights.safetensors")
    awq_scales = _safe_load(output_dir / "awq_scales.safetensors")
    grids = _safe_load(output_dir / "frozen_grids.safetensors")
    done = set(weights.keys())
    if verbose:
        print(f"[resume] {len(done)} tensors already in checkpoint; will skip them")
    return weights, awq_scales, grids, done


def _checkpoint_partial(
    output_dir: Path,
    *,
    weights_bf16: dict[str, torch.Tensor],
    awq_scales: dict[str, torch.Tensor],
    frozen_grids: dict[str, torch.Tensor],
    metadata: dict,
) -> None:
    """Write current state to `output_dir` as a partial manifest.

    Atomic write per file: save to `<name>.tmp`, then `os.rename`. POSIX
    rename is atomic — a kill during the rename leaves either the old
    file or the new one, never a half-written file.

    `metadata["partial"] = True` flags the manifest as incomplete; once
    the run finishes, the final `write_manifest` overwrites it with
    `partial` absent (full file).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, "partial": True}

    plans = [
        ("weights.safetensors", weights_bf16),
        ("awq_scales.safetensors", awq_scales),
        ("frozen_grids.safetensors", frozen_grids),
    ]
    for fname, payload in plans:
        tmp = output_dir / (fname + ".tmp")
        final = output_dir / fname
        save_file(payload, str(tmp))
        os.replace(tmp, final)
    # manifest.json — atomic via .tmp + rename too
    tmp_json = output_dir / "manifest.json.tmp"
    with open(tmp_json, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp_json, output_dir / "manifest.json")


def quantize_model(
    *,
    input_dir: Path,
    hessian_path: Path | None,
    imatrix_path: Path | None,
    output_dir: Path,
    alpha: float,
    initial_damp_ratio: float,
    max_damp_multiplier: float,
    devices: list[str],
    limit: int | None,
    skip_to: int,
    awq_f1_only: bool,
    n_bits: int,
    resume: bool,
    checkpoint_interval: int,
    watchdog_timeout_sec: int,
    verbose: bool,
) -> None:
    t_start = time.perf_counter()
    file_map, shape_map = build_input_index(input_dir)
    if verbose:
        print(f"[input] {len(file_map)} tensors from {input_dir}")

    sidecar = None
    if hessian_path is not None:
        sidecar = HessianSidecar(hessian_path)
        if verbose:
            print(f"[hessian] {len(sidecar.index)} entries from {hessian_path}")
    elif verbose:
        print("[hessian] not provided — every tensor will fall to the RTN-fallback path "
              "(per-256-block min/max with FWHT rotation, no Hessian-aware error propagation). "
              "Combined with `--imatrix`, this gives an AWQ-only quant.")

    imatrix = None
    if imatrix_path is not None:
        imatrix = load_imatrix(imatrix_path)
        if verbose:
            print(f"[imatrix] {len(imatrix)} entries from {imatrix_path}")

    # Sign tables per device (small but device-specific).
    signs1_per_device = {d: gen_fwht_signs(42, 256, device=d) for d in devices}
    signs2_per_device = {d: gen_fwht_signs(1042, 256, device=d) for d in devices}

    # ── Resume from existing partial manifest, if requested + present ──
    if resume:
        weights_out, awq_scales_out, frozen_grids_out, done_names = _load_resume(
            output_dir, verbose
        )
    else:
        weights_out: dict[str, torch.Tensor] = {}
        awq_scales_out: dict[str, torch.Tensor] = {}
        frozen_grids_out: dict[str, torch.Tensor] = {}
        done_names: set[str] = set()
    stats: list[TensorStats] = []

    # ── Watchdog: SIGKILL self if no per-tensor progress in N seconds ──
    last_progress = {"ts": time.time()}
    if watchdog_timeout_sec > 0:
        if verbose:
            print(f"[watchdog] enabled — will SIGKILL self after "
                  f"{watchdog_timeout_sec}s without per-tensor progress")
        _start_watchdog(watchdog_timeout_sec, last_progress)

    # Determine eligibility per tensor; build the processing list.
    eligible_names: list[str] = []
    for n, shape in sorted(shape_map.items()):
        if is_mq4g256_eligible(n, shape):
            eligible_names.append(n)
    if verbose:
        print(f"[plan] {len(eligible_names)} MQ4G256-eligible tensors "
              f"(out of {len(shape_map)} total)")

    if skip_to:
        eligible_names = eligible_names[skip_to:]
        if verbose:
            print(f"[plan] --skip-to {skip_to} → processing {len(eligible_names)} remaining")
    if limit:
        eligible_names = eligible_names[:limit]
        if verbose:
            print(f"[plan] --limit {limit} → processing {len(eligible_names)} tensors")

    # ── Resume: filter out already-completed tensors. The `done_names`
    # comes from the existing checkpoint's `weights.safetensors` keys —
    # they're already in `weights_out`, just skip the re-processing.
    if done_names:
        before = len(eligible_names)
        eligible_names = [n for n in eligible_names if n not in done_names]
        if verbose:
            print(f"[resume] {before - len(eligible_names)} of {before} "
                  f"eligible tensors already done in checkpoint; "
                  f"processing {len(eligible_names)} remaining")

    # ── Per-eligible-tensor: GPTQ on round-robin device ──
    for i, name in enumerate(eligible_names):
        device = devices[i % len(devices)]
        shape = shape_map[name]
        t0 = time.perf_counter()
        st = TensorStats(name=name, shape=shape, eligible=True,
                        has_hessian=False, awq_eligible=False, device=device)

        try:
            w_cpu = load_tensor(file_map[name], name)  # BF16, [M, K]
            assert w_cpu.dtype == torch.bfloat16, f"{name} expected BF16, got {w_cpu.dtype}"

            # Hessian — load to FP64 directly on the target device. Some
            # tensors won't have one (e.g. arch additions not on the
            # Python collector's whitelist) — fall back to RTN: identity
            # AWQ + identity actorder + no propagation. In practice that's
            # implemented by `gptq_one_tensor` with H = I * tiny (so
            # Cholesky succeeds, U^T U = (I+λI)^-1 ≈ I, propagation
            # weights ≈ 0). Cleaner: special-case it via direct pack.
            hfhs_key = to_hfhs_key(name)
            has_h = sidecar is not None and sidecar.has(hfhs_key)
            st.has_hessian = has_h

            # AWQ
            this_awq_eligible = awq_eligible(name, f1_only=awq_f1_only)
            st.awq_eligible = this_awq_eligible
            if this_awq_eligible and imatrix is not None:
                in_sum2 = imatrix_weights_for(imatrix, name)
                if in_sum2 is not None:
                    assert in_sum2.shape[0] == shape[1], \
                        f"{name}: imatrix K={in_sum2.shape[0]} vs weight K={shape[1]}"
                    s_cpu = compute_awq_scales(torch.from_numpy(in_sum2), alpha)
                    awq_scales_out[f"{name}.awq_scale"] = s_cpu.to(torch.float16)
                    s_gpu = s_cpu.to(device, dtype=torch.float64)
                    # Apply AWQ pre-scale to weights BEFORE GPTQ (mirrors
                    # Rust `apply_awq_prescale` which mutates weights in place).
                    w_cpu_f32 = w_cpu.to(torch.float32) * s_cpu.to(torch.float32).unsqueeze(0)
                    w_gpu = w_cpu_f32.to(device)
                else:
                    s_gpu = torch.ones(shape[1], dtype=torch.float64, device=device)
                    w_gpu = w_cpu.to(device, dtype=torch.float32)
            else:
                s_gpu = torch.ones(shape[1], dtype=torch.float64, device=device)
                w_gpu = w_cpu.to(device, dtype=torch.float32)

            # NOTE: `s_gpu` is passed to `gptq_one_tensor` for the
            # HESSIAN rescale only — weights are already AWQ-pre-scaled
            # in-place above.
            if has_h:
                h_gpu = sidecar.load_f64(hfhs_key, device=device)
                result = gptq_one_tensor(
                    w_gpu, h_gpu, s_gpu,
                    signs1_per_device[device], signs2_per_device[device],
                    initial_damp_ratio=initial_damp_ratio,
                    max_damp_multiplier=max_damp_multiplier,
                    n_bits=n_bits,
                    name=name,
                )
                del h_gpu

                st.effective_damp = result.effective_damp
                st.mse_vs_original = result.mse_vs_original
                st.clamps_below, st.clamps_above = result.clamps

                # Move post-GPTQ weights + grids back to CPU for safetensors write.
                w_out_bf16 = result.weights.to(torch.bfloat16).cpu()
                grids_f16 = result.frozen_grids.to(torch.float16).cpu()
                weights_out[name] = w_out_bf16
                frozen_grids_out[f"{name}.grids"] = grids_f16
            else:
                # No Hessian → emit RTN-equivalent: post-AWQ-scale +
                # post-FWHT-rotate weights, plus frozen grids computed
                # from THIS rotated weight. The "GPTQ" step is a no-op.
                # Same byte-level result Rust's `quantize_mq4g256`
                # would produce when called without a Hessian sidecar.
                from gptq_gpu_pkg.algo import (
                    apply_fwht_per_256_to_weights,
                    compute_frozen_block_grids,
                    quantize_mq4_with_grid,
                )
                w_rot = w_gpu.to(torch.float64).contiguous()
                apply_fwht_per_256_to_weights(
                    w_rot,
                    signs1_per_device[device],
                    signs2_per_device[device],
                )
                grids = compute_frozen_block_grids(w_rot.view(-1), n_bits=n_bits)
                # Per-element RTN dequant — produces a w_out that is
                # already on the grid, exactly what GPTQ's column loop
                # would do without OBS propagation.
                n_blocks_per_row = w_rot.shape[1] // 256
                row_idx = torch.arange(w_rot.shape[0], device=device).unsqueeze(1)
                col_idx = torch.arange(w_rot.shape[1], device=device).unsqueeze(0)
                bidx = row_idx * n_blocks_per_row + col_idx // 256
                scales = grids[bidx, 0]
                mins = grids[bidx, 1]
                w_out_f64 = quantize_mq4_with_grid(w_rot, scales, mins, n_bits=n_bits)
                st.mse_vs_original = ((w_out_f64 - w_rot) ** 2).mean().item()
                weights_out[name] = w_out_f64.to(torch.bfloat16).cpu()
                frozen_grids_out[f"{name}.grids"] = grids.to(torch.float16).cpu()

            # Free GPU memory immediately — alloc fragmentation otherwise
            # accumulates over 200+ tensors (plan §4.7).
            del w_gpu, s_gpu
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        except (CholeskyFailedError, torch.cuda.OutOfMemoryError) as e:
            # OOM and Cholesky-failed both go to the same RTN fallback:
            # the per-tensor GPU work failed, but the tensor is still
            # GPTQ-eligible; emit a plain MQ4/MQ3 pack on the AWQ-scaled
            # FWHT-rotated weights, no Hessian-aware error propagation.
            # OOM at K=17408 happens when allocator fragmentation pushes
            # `cholesky_inverse`'s 2.4 GB workspace allocation past the
            # 16 GB card limit (~15.5 GB cap with reserved fragments).
            # `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` mitigates
            # but doesn't always eliminate.
            st.error = f"{type(e).__name__}: {e}"
            print(f"[fallback] {name}: {type(e).__name__}; emitting RTN pack",
                  flush=True)
            # Free anything we can before the CPU fallback path
            try:
                del w_gpu, s_gpu
            except NameError:
                pass
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            from gptq_gpu_pkg.algo import (
                apply_fwht_per_256_to_weights, compute_frozen_block_grids,
            )
            w_rot = w_cpu.to(torch.float64).contiguous()
            signs1_cpu = gen_fwht_signs(42, 256)
            signs2_cpu = gen_fwht_signs(1042, 256)
            apply_fwht_per_256_to_weights(w_rot, signs1_cpu, signs2_cpu)
            grids = compute_frozen_block_grids(w_rot.view(-1), n_bits=n_bits)
            weights_out[name] = w_rot.to(torch.bfloat16)
            frozen_grids_out[f"{name}.grids"] = grids.to(torch.float16)

        except Exception as e:
            st.error = f"unhandled: {type(e).__name__}: {e}"
            raise

        st.wall_s = time.perf_counter() - t0
        stats.append(st)
        # Update watchdog timestamp — successful or fallback completion
        # both count as "progress", only a CUDA hang in cholesky_ex etc.
        # would prevent reaching this point.
        last_progress["ts"] = time.time()
        if verbose:
            kld_str = (f"mse={st.mse_vs_original:.4e}"
                       if st.mse_vs_original is not None else "mse=skip")
            damp_str = (f"damp={st.effective_damp:.2e}"
                        if st.effective_damp is not None else "no-H")
            print(f"  [{i+1:>3}/{len(eligible_names)}] {device} "
                  f"K={shape[1]:>5} M={shape[0]:>5} {kld_str} {damp_str} "
                  f"clamps={st.clamps_below+st.clamps_above:>6} "
                  f"t={st.wall_s:>6.2f}s  {name}")
            sys.stdout.flush()

        # ── Periodic checkpoint ──
        # Snapshot the current partial state to `output_dir` every N
        # eligible tensors. On a kill (watchdog or external), the next
        # `--resume` invocation picks up from this snapshot.
        if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
            t_ckpt = time.perf_counter()
            ckpt_meta = {
                "schema_version": 2,
                "source_model_dir": str(input_dir),
                "hessian_path": str(hessian_path) if hessian_path else None,
                "imatrix_path": str(imatrix_path) if imatrix_path else None,
                "alpha": alpha,
                "awq_f1_only": awq_f1_only,
                "n_bits": n_bits,
                "gptq_initial_damp_ratio": initial_damp_ratio,
                "gptq_max_damp_multiplier": max_damp_multiplier,
                "devices": devices,
                "n_tensors_processed_so_far": len(weights_out),
                "wall_seconds_so_far": time.perf_counter() - t_start,
            }
            _checkpoint_partial(
                output_dir,
                weights_bf16=weights_out,
                awq_scales=awq_scales_out,
                frozen_grids=frozen_grids_out,
                metadata=ckpt_meta,
            )
            if verbose:
                print(f"  [checkpoint] wrote partial manifest "
                      f"({len(weights_out)} tensors) in "
                      f"{time.perf_counter() - t_ckpt:.1f}s",
                      flush=True)
            # Refresh watchdog timestamp — the checkpoint write itself
            # can take 10-20s on /data NFS, shouldn't count as a stall.
            last_progress["ts"] = time.time()

    # ── Passthrough non-eligible tensors (embed, lm_head, norms, …) ──
    if verbose:
        print(f"[passthrough] copying {len(shape_map) - len(eligible_names)} non-eligible tensors")
    for n in sorted(shape_map):
        if n in weights_out:
            continue
        weights_out[n] = load_tensor(file_map[n], n)

    # ── Manifest ──
    manifest = {
        "schema_version": 2,  # v2 adds `n_bits` field; v1 manifests imply n_bits=4
        "source_model_dir": str(input_dir),
        "hessian_path": str(hessian_path),
        "imatrix_path": str(imatrix_path) if imatrix_path else None,
        "alpha": alpha,
        "awq_f1_only": awq_f1_only,
        "n_bits": n_bits,
        "gptq_initial_damp_ratio": initial_damp_ratio,
        "gptq_max_damp_multiplier": max_damp_multiplier,
        "devices": devices,
        "n_tensors_total": len(weights_out),
        "n_tensors_gptq": sum(1 for s in stats if s.has_hessian and s.error is None),
        "n_tensors_rtn_fallback": sum(1 for s in stats if not s.has_hessian or s.error is not None),
        "n_tensors_awq": len(awq_scales_out),
        "wall_seconds": time.perf_counter() - t_start,
        "tensors": [s.__dict__ for s in stats],
    }

    write_manifest(
        output_dir,
        weights_bf16=weights_out,
        awq_scales=awq_scales_out,
        frozen_grids=frozen_grids_out,
        metadata=manifest,
    )
    if verbose:
        print(f"[done] manifest written to {output_dir}")
        print(f"[done] total wall: {manifest['wall_seconds']:.1f}s "
              f"({manifest['n_tensors_gptq']} GPTQ, "
              f"{manifest['n_tensors_rtn_fallback']} RTN fallback, "
              f"{manifest['n_tensors_awq']} AWQ sidecars)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True,
                  help="HF snapshot directory with model.safetensors.index.json")
    p.add_argument("--hessian", type=Path, default=None,
                  help="HFHS sidecar (qwen3.5-*.hessian.bin). Optional: when "
                       "omitted, every tensor falls through to RTN-pack (no "
                       "GPTQ). Combined with `--imatrix`, this gives an "
                       "AWQ-only quant. Combined with neither, pure RTN.")
    p.add_argument("--imatrix", type=Path, default=None,
                  help="Imatrix GGUF for AWQ (omit to disable AWQ)")
    p.add_argument("--alpha", type=float, default=0.55,
                  help="AWQ alpha (master-doc default 0.55)")
    p.add_argument("--initial-damp-ratio", type=float, default=0.01,
                  help="initial damp / mean(diag(H)) (Rust default 0.01)")
    p.add_argument("--max-damp-multiplier", type=float, default=1.0,
                  help="damp cap / mean(diag(H)) (Rust default 1.0)")
    p.add_argument("--output", type=Path, required=True,
                  help="manifest output directory")
    p.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"],
                  help="CUDA devices for round-robin (default both 5070 Tis)")
    p.add_argument("--limit", type=int, default=None,
                  help="process only first N eligible tensors (smoke test)")
    p.add_argument("--skip-to", type=int, default=0,
                  help="skip first N eligible tensors (resume / partial)")
    p.add_argument("--awq-f1-only", action="store_true",
                  help="restrict AWQ to F1 set (no o_proj/down_proj/wo); A/B parity with HIPFIRE_AWQ_F1_ONLY=1")
    p.add_argument("--bits", type=int, default=4, choices=[3, 4],
                  help="Quantization bit width: 4 (MQ4G256, default, well-tested) or "
                       "3 (MQ3G256, uniform 3-bit — master-doc §5 warns uniform MQ3 may "
                       "collapse; pair with --alpha 0.55 AWQ for best chance). The manifest "
                       "writeback records n_bits so Rust's --precomputed-gptq-path dispatches "
                       "to the matching pack function automatically.")
    p.add_argument("--resume", action="store_true",
                  help="Load any existing partial manifest at --output and skip the "
                       "tensors already present in weights.safetensors. Survives a "
                       "mid-run kill (watchdog, OOM, manual) without re-processing "
                       "completed tensors. Compatible with --checkpoint-interval.")
    p.add_argument("--checkpoint-interval", type=int, default=25,
                  help="Write a partial manifest (atomic via .tmp + rename) every N "
                       "eligible tensors. 0 disables checkpointing. Default 25 — on a "
                       "27B run with ~506 eligible tensors, that's 20 checkpoints, "
                       "each costing ~10-20s on /data NFS. Lets a `--resume` pick up "
                       "with at most N-1 tensors of re-work after a kill.")
    p.add_argument("--watchdog-timeout-sec", type=int, default=900,
                  help="Force-SIGKILL self if no per-tensor progress in this many "
                       "seconds. Default 900 (15 min) — well above the legitimate "
                       "worst-case ~7 min for a K=17408 K³ Cholesky with 3 damp "
                       "retries. CUDA hangs (cuSOLVER pathological cases observed "
                       "2026-05-17 on sm_120) cannot be cancelled from within Python; "
                       "external kill via SIGKILL is the only recovery and pairs with "
                       "--resume on the next launch. Set 0 to disable.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    quantize_model(
        input_dir=args.input,
        hessian_path=args.hessian,
        imatrix_path=args.imatrix,
        output_dir=args.output,
        alpha=args.alpha,
        initial_damp_ratio=args.initial_damp_ratio,
        max_damp_multiplier=args.max_damp_multiplier,
        devices=args.devices,
        limit=args.limit,
        skip_to=args.skip_to,
        awq_f1_only=args.awq_f1_only,
        n_bits=args.bits,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        watchdog_timeout_sec=args.watchdog_timeout_sec,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
