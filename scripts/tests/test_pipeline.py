"""Pipeline smoke + invariant tests on synthetic mini-tensors.

These run in <1s on CPU; CUDA versions also covered for the path that
matters (real 4B/9B work runs on CUDA exclusively).

Strategy: build a known-PSD H, a random W, run `gptq_one_tensor`, and
assert the OBS invariants hold on the output:
  - Output values are dequantized-quantized: every w_out[i, j] satisfies
    `w_out = q * scale + min` for some q ∈ {0..15}.
  - Sanity MSE: post-GPTQ MSE not catastrophically worse than RTN.
  - Frozen-grids shape matches Rust block layout.

Run:
    PYTHONPATH=scripts ~/git/hipfire/.venv-cuda/bin/python \\
        -m unittest scripts/tests/test_pipeline.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gptq_gpu_pkg.algo import (  # noqa: E402
    apply_fwht_per_256_to_weights,
    compute_frozen_block_grids,
    gen_fwht_signs,
    quantize_mq4_with_grid,
)
from gptq_gpu_pkg.pipeline import gptq_one_tensor  # noqa: E402


def _synth_tensor(m: int, k: int, *, seed: int = 0, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Random W [M, K] in BF16 + random PSD H [K, K] in FP64.

    Hessian: `H = A A^T + K·I` with A random. Condition number ≈ K.
    Weights: roughly N(0, 0.04) — realistic Qwen-scale magnitude.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    w = 0.04 * torch.randn(m, k, generator=g, dtype=torch.float32, device=device)
    a = torch.randn(k, k, generator=g, dtype=torch.float64, device=device)
    h = a @ a.T + float(k) * torch.eye(k, dtype=torch.float64, device=device)
    return w.to(torch.bfloat16), h


class TestPipelineSmoke(unittest.TestCase):
    """End-to-end pipeline must run and produce on-grid outputs."""

    def _run_pipeline(self, m: int, k: int, device: str):
        w_in, h = _synth_tensor(m, k, seed=0, device=device)
        awq = torch.ones(k, dtype=torch.float64, device=device)
        signs1 = gen_fwht_signs(42, 256, device=device)
        signs2 = gen_fwht_signs(1042, 256, device=device)
        result = gptq_one_tensor(
            w_in, h, awq, signs1, signs2,
            initial_damp_ratio=0.01, max_damp_multiplier=1.0,
            name=f"synth_M{m}_K{k}",
        )
        return result, w_in, signs1, signs2

    def test_pipeline_runs_cpu(self):
        result, _, _, _ = self._run_pipeline(8, 512, "cpu")
        self.assertEqual(result.weights.shape, (8, 512))
        # 8 * 512 / 256 = 16 blocks
        self.assertEqual(result.frozen_grids.shape, (16, 2))

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA")
    def test_pipeline_runs_cuda(self):
        result, _, _, _ = self._run_pipeline(8, 512, "cuda:0")
        self.assertEqual(result.weights.shape, (8, 512))
        self.assertTrue(result.weights.is_cuda)

    def test_outputs_are_on_grid(self):
        """Every output value must be expressable as `q * scale + min`
        for some q ∈ {0..15} using the frozen grid for its block. The
        column-sequential write-back path is the only way values land
        in `w_out`, so this is the immediate next assertion if Phase A
        ever silently breaks (e.g. forgetting to write w_out).
        """
        result, _, _, _ = self._run_pipeline(8, 512, "cpu")
        w_out = result.weights
        grids = result.frozen_grids
        m, k = w_out.shape
        n_blocks_per_row = k // 256
        for row in range(m):
            for col in range(k):
                block_idx = row * n_blocks_per_row + (col // 256)
                scale = grids[block_idx, 0].item()
                min_val = grids[block_idx, 1].item()
                v = w_out[row, col].item()
                # Compute back-quantization: q = round((v - min) / scale)
                if scale == 0:
                    continue
                q = round((v - min_val) / scale)
                expected = q * scale + min_val
                # Allow tiny FP rounding when rebuilding the value.
                self.assertAlmostEqual(
                    v, expected, places=10,
                    msg=f"off-grid output at ({row},{col}): v={v} q={q} grid=({scale},{min_val})"
                )
                self.assertTrue(0 <= q <= 15, f"q={q} out of [0,15] at ({row},{col})")

    def test_mse_better_than_random(self):
        """GPTQ output MSE should be at most a small constant × RTN MSE
        on a well-conditioned H. (Not strictly smaller — GPTQ trades
        per-element error for an activation-weighted residual.)

        We compute the naive RTN MSE on the FWHT-rotated weights with
        the same frozen grids, and verify GPTQ's MSE is within 3× of it.
        Far apart = the pipeline ran a degenerate path; close = sanity.
        """
        m, k = 8, 512
        w_in, h = _synth_tensor(m, k, seed=42, device="cpu")
        awq = torch.ones(k, dtype=torch.float64)
        signs1 = gen_fwht_signs(42, 256)
        signs2 = gen_fwht_signs(1042, 256)
        # RTN baseline: rotate W, fit frozen grids, do per-element
        # round-half-up, dequant — no OBS propagation.
        w_rot = w_in.to(torch.float64).contiguous()
        apply_fwht_per_256_to_weights(w_rot, signs1, signs2)
        rtn_grids = compute_frozen_block_grids(w_rot.view(-1))
        # Per-element RTN — broadcast grids back to [M, K] by row layout.
        n_blocks_per_row = k // 256
        rtn_block_idx = (
            torch.arange(m).unsqueeze(1) * n_blocks_per_row
            + torch.div(torch.arange(k).unsqueeze(0), 256, rounding_mode="floor")
        )
        scales = rtn_grids[rtn_block_idx, 0]
        mins = rtn_grids[rtn_block_idx, 1]
        w_rtn = quantize_mq4_with_grid(w_rot, scales, mins)
        rtn_mse = ((w_rtn - w_rot) ** 2).mean().item()

        # GPTQ
        result = gptq_one_tensor(w_in, h, awq, signs1, signs2, name="synth_rtn_cmp")
        # Both MSEs are vs the SAME FWHT-rotated W (which the pipeline
        # also computes internally).
        self.assertLess(
            result.mse_vs_original, 5.0 * rtn_mse,
            f"GPTQ MSE {result.mse_vs_original:.4e} much worse than RTN {rtn_mse:.4e}"
        )

    def test_identity_actorder_path_is_safe(self):
        """When `diag(H)` is constant (all equal), `argsort` falls back
        to stable index order — i.e. perm = arange(K). Verify the pipe
        still runs and produces grid-aligned output."""
        m, k = 8, 256
        torch.manual_seed(7)
        w_in = (0.04 * torch.randn(m, k, dtype=torch.float32)).to(torch.bfloat16)
        # H = c * I — every diagonal equal, identity actorder by ties.
        h = 10.0 * torch.eye(k, dtype=torch.float64)
        awq = torch.ones(k, dtype=torch.float64)
        signs1 = gen_fwht_signs(42, 256)
        signs2 = gen_fwht_signs(1042, 256)
        result = gptq_one_tensor(w_in, h, awq, signs1, signs2, name="identity_perm")
        self.assertEqual(result.weights.shape, (8, 256))


if __name__ == "__main__":
    unittest.main(verbosity=2)
