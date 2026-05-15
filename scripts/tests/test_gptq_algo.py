"""Tests for `scripts/gptq_cuda_pkg/algo.py`.

Self-contained: every test verifies an INVARIANT of the math (not a
byte-exact match against Rust), so failures point at the actual broken
property rather than an FP-roundoff mismatch.

Run with:
    PYTHONPATH=scripts ~/git/hipfire/.venv-cuda/bin/python -m unittest \\
        scripts/tests/test_gptq_algo.py -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch

# Allow `python -m unittest scripts/tests/test_gptq_algo.py` from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gptq_cuda_pkg.algo import (  # noqa: E402
    CholeskyFailedError,
    apply_awq_rescaling_h,
    apply_fwht_per_256_to_weights,
    compute_awq_scales,
    compute_damped_inv_cholesky_upper,
    compute_frozen_block_grids,
    fwht_256_inplace,
    fwht_similarity_per_256_h,
    gen_fwht_signs,
    inverse_perm,
    quantize_mq4_with_grid,
    symmetrize_in_place,
    weight_mode_actorder,
)


def _random_psd(k: int, seed: int = 0, *, device="cpu") -> torch.Tensor:
    """Build a well-conditioned random PSD matrix in FP64."""
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(k, k, generator=g, dtype=torch.float64, device=device)
    return a @ a.T + float(k) * torch.eye(k, dtype=torch.float64, device=device)


class TestFwhtSigns(unittest.TestCase):
    """`gen_fwht_signs(42, 256)` must match Rust `gen_fwht_signs(42, 256)`
    byte-for-byte. We probe a handful of known values from the LCG to
    guard against a typo in the multiplier/increment/mask."""

    def test_first_few_signs_seed_42_match_rust(self):
        # Rust ground truth captured from a minimal LCG repro:
        #   fn main() {
        #     let mut state: u32 = 42;
        #     for _ in 0..8 {
        #       state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        #       println!("{}", if (state >> 16) & 1 == 1 { 1 } else { -1 });
        #     }
        #   }
        # Output: 1, 1, 1, 1, -1, 1, 1, -1
        expected = torch.tensor([1, 1, 1, 1, -1, 1, 1, -1], dtype=torch.float64)
        got = gen_fwht_signs(42, 8)
        torch.testing.assert_close(got, expected)

    def test_first_few_signs_seed_1042_deterministic(self):
        # Determinism: same seed twice = same output.
        a = gen_fwht_signs(1042, 16)
        b = gen_fwht_signs(1042, 16)
        torch.testing.assert_close(a, b)

    def test_signs_are_pm_one(self):
        s = gen_fwht_signs(1042, 256)
        self.assertEqual(s.shape, (256,))
        unique = torch.unique(s)
        self.assertEqual(set(unique.tolist()), {-1.0, 1.0})


class TestFwht256(unittest.TestCase):
    """Sanity: FWHT-256 is orthogonal w/ the 1/16 normalization, so it
    preserves the L2 norm. Round-trip would also work but invokes the
    same kernel twice — preservation is a stronger property check."""

    def test_orthogonal_preserves_norm(self):
        torch.manual_seed(0)
        x = torch.randn(256, dtype=torch.float64)
        signs1 = gen_fwht_signs(42, 256)
        signs2 = gen_fwht_signs(1042, 256)
        norm_before = x.norm().item()
        fwht_256_inplace(x.unsqueeze(0), signs1, signs2)
        norm_after = x.norm().item()
        self.assertAlmostEqual(norm_before, norm_after, places=10)

    def test_round_trip_inverse(self):
        # Forward then inverse (== forward with sign tables reversed and
        # appropriate inversion) should recover the original. Easier
        # check: with NEGATED signs, two FWHTs invert each other up to
        # the global ±1 from sign multiplications. Here we use the
        # cleaner property: applying FWHT twice with identical signs is
        # NOT identity (it's R · R · x where R isn't involutive due to
        # sign multiplications), so we explicitly invert by reversing
        # the sign tables. Going simpler: verify FWHT applied twice with
        # signs1=signs2=ones is identity (Hadamard^2 = I after 1/16
        # normalization applied twice).
        ones = torch.ones(256, dtype=torch.float64)
        x = torch.arange(256, dtype=torch.float64)
        x_orig = x.clone()
        fwht_256_inplace(x.unsqueeze(0), ones, ones)
        fwht_256_inplace(x.unsqueeze(0), ones, ones)
        # Two FWHTs with all-ones signs and 1/16 scaling each → identity.
        torch.testing.assert_close(x, x_orig, atol=1e-10, rtol=0)


class TestObsCholeskyInvariant(unittest.TestCase):
    """Core OBS-Cholesky test: verify
        `U^T · U == inv(P^T (H + λI) P)`
    holds to FP64 precision.

    This is the property that GPTQ depends on (master-doc rule, Rust
    `compute_damped_inv_cholesky_upper` doc). If this fails, every
    downstream test on real weights will fail too — and conversely, if
    this passes, Rust and Python disagreeing on packed bytes is an FP
    rounding artefact, not a math bug.
    """

    def _check_invariant(self, k: int, perm: torch.Tensor | None, seed: int = 0, device: str = "cpu"):
        h = _random_psd(k, seed=seed, device=device)
        result = compute_damped_inv_cholesky_upper(
            h, perm, initial_damp=0.0, max_damp_multiplier=1.0,
        )
        u = result.u
        damp = result.effective_damp

        # Build the matrix that U should be the inverse-upper-Cholesky of.
        eye_k = torch.eye(k, dtype=torch.float64, device=device)
        if perm is None:
            h_eff_damped = h + damp * eye_k
        else:
            h_eff_damped = h[perm][:, perm] + damp * eye_k

        # Property: `U^T · U @ (H_eff + damp*I) == I`.
        uTu_h = u.T @ u @ h_eff_damped
        residual = (uTu_h - eye_k).abs().max().item()
        # FP64 Cholesky on a K×K with cond ~K is good to ~K * ε ~ 1e-13;
        # we expect the chained matmul to amplify this by another K, so
        # tolerance ~K² * ε. For K=16, that's ~3e-14; K=64 ~6e-13.
        tol = max(1e-10, k * k * torch.finfo(torch.float64).eps)
        self.assertLess(
            residual, tol,
            f"OBS invariant violated (K={k}, device={device}, perm={'yes' if perm is not None else 'no'}): "
            f"max|U^T U (H+λI) - I| = {residual:.3e} > {tol:.3e}",
        )

    def test_invariant_identity_perm_k16(self):
        self._check_invariant(16, perm=None)

    def test_invariant_identity_perm_k64(self):
        self._check_invariant(64, perm=None)

    def test_invariant_nonidentity_perm_k16(self):
        # Reverse permutation — most rigorous "anything other than
        # identity" check.
        perm = torch.arange(15, -1, -1)
        self._check_invariant(16, perm=perm)

    def test_invariant_nonidentity_perm_k64(self):
        torch.manual_seed(1)
        perm = torch.randperm(64)
        self._check_invariant(64, perm=perm)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA")
    def test_invariant_cuda_k64(self):
        torch.manual_seed(2)
        perm = torch.randperm(64, device="cuda:0")
        self._check_invariant(64, perm=perm, device="cuda:0")

    def test_damp_escalation_on_rank_deficient(self):
        # Build a rank-1 matrix → Cholesky fails at damp=0, retries
        # with damp *= 10 escalation. We pass initial_damp=1e-20 (below
        # the ε·diag_mean clamp, so adaptive cascade kicks in).
        k = 8
        v = torch.randn(k, dtype=torch.float64)
        h_rank1 = v.outer(v)  # rank 1
        # Without damping this would fail; max_damp_multiplier must be
        # large enough that some damp value succeeds. diag_mean of
        # outer(v, v) is mean(v²) ~ 1; cap = 10 * 1 = 10 → some damp
        # value in {1e-20·10ⁿ} will succeed before n hits the cap.
        result = compute_damped_inv_cholesky_upper(
            h_rank1, perm=None, initial_damp=1e-20, max_damp_multiplier=10.0,
        )
        self.assertGreater(result.effective_damp, 0.0)

    def test_singular_even_with_max_damp_raises(self):
        # Definitively-non-PSD matrix where the damp cap can never recover.
        # H = diag([-10, -10]) → diag_mean = -10 → damp_cap = 0.5 * (-10) = -5.
        # The initial_damp clamp to ε·|diag_mean| is still positive (≈ 2e-15)
        # but `damp >= damp_cap` is satisfied at first iter (positive >= -5),
        # AND Cholesky of `-10*I + 2e-15*I` definitively fails (eigenvalue
        # ≈ -10). Adaptive cascade can't escape because the cap is below
        # the current damp — must raise immediately.
        k = 2
        h = -10.0 * torch.eye(k, dtype=torch.float64)
        with self.assertRaises(CholeskyFailedError):
            compute_damped_inv_cholesky_upper(
                h, perm=None, initial_damp=0.0, max_damp_multiplier=0.5,
            )


class TestAwqScales(unittest.TestCase):
    """`compute_awq_scales` must produce geo-mean-1 output (master-doc
    rule, plan §4.1 exponent formula)."""

    def test_geo_mean_is_one(self):
        torch.manual_seed(0)
        in_sum2 = (torch.randn(4096, dtype=torch.float32) ** 2 + 1e-4)
        s = compute_awq_scales(in_sum2, alpha=0.55)
        log_s_mean = s.log().mean().item()
        self.assertLess(abs(log_s_mean), 1e-12, f"geo-mean drift: log mean = {log_s_mean}")

    def test_alpha_zero_gives_identity(self):
        torch.manual_seed(1)
        in_sum2 = (torch.randn(128, dtype=torch.float32) ** 2 + 1e-4)
        s = compute_awq_scales(in_sum2, alpha=0.0)
        torch.testing.assert_close(s, torch.ones_like(s, dtype=torch.float64))

    def test_alpha_doubles_means_squared_in_log(self):
        # At alpha=1.0, log(s) = 0.5 * log(in_sum2) - mean(0.5 log).
        # At alpha=2.0, log(s) = 1.0 * log(in_sum2) - mean(1.0 log).
        # Property: log(s_α2) - log(s_α1)·2 should be a constant shift
        # that depends only on the geo-mean subtraction. Equivalent:
        # log(s_α2) / log(s_α1) → 2 in the limit of zero mean. Easier
        # observable: log(s_α2) is a linear scaling of log(s_α1) shifted
        # by a constant (the differences in geo-mean centring).
        torch.manual_seed(2)
        in_sum2 = (torch.randn(128, dtype=torch.float32) ** 2 + 1e-4)
        s1 = compute_awq_scales(in_sum2, alpha=1.0)
        s2 = compute_awq_scales(in_sum2, alpha=2.0)
        # log(s) is linear in alpha, so log(s_α2) == 2*log(s_α1) + shift.
        # The shift is a single constant; verify ratio is constant.
        diff = s2.log() - 2.0 * s1.log()
        self.assertLess(diff.std().item(), 1e-12)


class TestHessianTransforms(unittest.TestCase):
    """The AWQ rescaling + FWHT similarity + symmetrize chain must
    preserve specific invariants: AWQ rescale with s=1 is no-op,
    symmetrize idempotent on symmetric H, FWHT preserves trace."""

    def test_apply_awq_rescaling_h_identity_scales_noop(self):
        torch.manual_seed(0)
        h = torch.randn(8, 8, dtype=torch.float64)
        h_copy = h.clone()
        apply_awq_rescaling_h(h, torch.ones(8, dtype=torch.float64))
        torch.testing.assert_close(h, h_copy)

    def test_apply_awq_rescaling_h_doubling_scale_quarters(self):
        # H = 4 everywhere, s = 2 everywhere → H' = 4/(2*2) = 1.
        h = torch.full((3, 3), 4.0, dtype=torch.float64)
        apply_awq_rescaling_h(h, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        torch.testing.assert_close(h, torch.ones_like(h))

    def test_symmetrize_idempotent(self):
        torch.manual_seed(1)
        h = torch.randn(8, 8, dtype=torch.float64)
        h = 0.5 * (h + h.T)  # make symmetric to start
        h_copy = h.clone()
        symmetrize_in_place(h)
        torch.testing.assert_close(h, h_copy, atol=1e-13, rtol=0)

    def test_fwht_similarity_preserves_trace(self):
        # K = 256 (one block). FWHT-256 is orthogonal, so the similarity
        # transform preserves the spectrum, hence the trace.
        k = 256
        h = torch.diag((torch.arange(k, dtype=torch.float64) + 1.0))
        trace_before = h.diagonal().sum().item()
        signs1 = gen_fwht_signs(42, 256)
        signs2 = gen_fwht_signs(1042, 256)
        fwht_similarity_per_256_h(h, signs1, signs2)
        trace_after = h.diagonal().sum().item()
        self.assertAlmostEqual(trace_before, trace_after, places=8)


class TestFrozenGrids(unittest.TestCase):
    """Frozen-grid layout + value check against the Rust formula."""

    def test_scale_and_min_for_known_block(self):
        # Weights 0.0, 0.1, ..., 25.5 → min=0, max=25.5, range=25.5,
        # scale = 25.5 / 15 = 1.7. Matches Rust unit test
        # `frozen_grid_matches_quantize_mq4g256_formula`.
        w = torch.arange(0, 256, dtype=torch.float64) * 0.1
        grids = compute_frozen_block_grids(w)
        self.assertEqual(grids.shape, (1, 2))
        self.assertAlmostEqual(grids[0, 0].item(), 1.7, places=10)
        self.assertAlmostEqual(grids[0, 1].item(), 0.0, places=10)

    def test_constant_block_yields_unit_scale(self):
        # Range=0 → scale=1.0 (Rust convention to avoid divide-by-zero).
        w = torch.full((256,), 3.14, dtype=torch.float64)
        grids = compute_frozen_block_grids(w)
        self.assertEqual(grids[0, 0].item(), 1.0)
        self.assertEqual(grids[0, 1].item(), 3.14)


class TestQuantizeMq4(unittest.TestCase):
    def test_round_half_up(self):
        # Grid spacing 0.25: 0.0, 0.25, ..., 3.75 (min=0, scale=0.25).
        scale = torch.tensor(0.25, dtype=torch.float64)
        min_val = torch.tensor(0.0, dtype=torch.float64)
        q = quantize_mq4_with_grid(torch.tensor(0.1, dtype=torch.float64), scale, min_val)
        self.assertAlmostEqual(q.item(), 0.0)
        q = quantize_mq4_with_grid(torch.tensor(0.15, dtype=torch.float64), scale, min_val)
        self.assertAlmostEqual(q.item(), 0.25)
        q = quantize_mq4_with_grid(torch.tensor(10.0, dtype=torch.float64), scale, min_val)
        self.assertAlmostEqual(q.item(), 3.75)  # clamped to 15
        q = quantize_mq4_with_grid(torch.tensor(-1.0, dtype=torch.float64), scale, min_val)
        self.assertAlmostEqual(q.item(), 0.0)  # clamped to 0


class TestActorder(unittest.TestCase):
    def test_descending_order(self):
        diag = torch.tensor([1.0, 3.0, 2.0, 0.5], dtype=torch.float64)
        perm = weight_mode_actorder(diag)
        # Descending order: 3.0 (idx 1), 2.0 (idx 2), 1.0 (idx 0), 0.5 (idx 3)
        self.assertEqual(perm.tolist(), [1, 2, 0, 3])

    def test_inverse_perm_round_trip(self):
        torch.manual_seed(0)
        perm = torch.randperm(16)
        inv = inverse_perm(perm)
        # perm[inv[i]] should equal i for all i
        torch.testing.assert_close(perm[inv], torch.arange(16))


class TestFwhtWeightsRow(unittest.TestCase):
    def test_norm_preserved_on_row_split(self):
        # FWHT-256 on a 2-row weight matrix should preserve the L2 norm
        # of every row separately, since each row is FWHT'd by the same
        # orthogonal R.
        m, k = 2, 512  # 2 blocks of 256 per row
        torch.manual_seed(0)
        w = torch.randn(m, k, dtype=torch.float64)
        norms_before = w.norm(dim=1)
        signs1 = gen_fwht_signs(42, 256)
        signs2 = gen_fwht_signs(1042, 256)
        apply_fwht_per_256_to_weights(w, signs1, signs2)
        norms_after = w.norm(dim=1)
        torch.testing.assert_close(norms_before, norms_after, atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
