"""Diagnose the second-Cholesky failure on K=9216 down_proj on gfx906 MI50.

Background: the running 4B GPTQ on MI50 falls back to RTN for every
mlp.down_proj.weight tensor (K=9216) with the error "Second Cholesky on
H_inv failed; H_inv lost PSD due to FP drift in matmul". The first
Cholesky succeeds (info=0); cholesky_inverse runs; the chol(H_inv) call
returns info != 0.

This script loads ONE real down_proj Hessian and tests four variants of
the inversion path to find the cheapest fix.

Run inside the mixa3607 docker:
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    --shm-size=8g \
    -v /home/kread/git/hipfire:/hipfire \
    -v /data/hipfire-refs:/refs:ro \
    -w /hipfire \
    --entrypoint /bin/bash \
    mixa3607/vllm-gfx906:0.20.1-rocm-6.3.3-aiinfos \
    -c 'PYTHONPATH=scripts HIP_VISIBLE_DEVICES=0 python3 docs/investigations/2026-05-17-mi50-bringup/probe_second_cholesky.py'
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from gptq_gpu_pkg.hfhs import HessianSidecar  # noqa: E402


HESSIAN_PATH = Path("/refs/qwen3.5-4b-bf16.hessian.bin")
# Use layer 0; behavior was identical at layers 0/1/10/11/.../21 in the run.
TARGET_NAME = "model.language_model.layers.0.mlp.down_proj"
DEVICE = "cuda:0"


def damped_first_cholesky(h: torch.Tensor, initial_damp: float, max_mult: float) -> tuple[torch.Tensor, float]:
    """Replicates gptq_gpu_pkg.algo.compute_damped_inv_cholesky_upper's
    first-Cholesky retry loop. Returns (L, effective_damp)."""
    diag_mean = h.diagonal().mean().item()
    damp = max(initial_damp, torch.finfo(torch.float64).eps * max(diag_mean, 1.0))
    damp_cap = max_mult * diag_mean
    while True:
        damped = h.clone()
        damped.diagonal().add_(damp)
        l_try, info = torch.linalg.cholesky_ex(damped, upper=False)
        del damped
        if int(info.item()) == 0:
            return l_try, damp
        del l_try
        if damp >= damp_cap:
            raise RuntimeError(f"first cholesky failed at damp_cap={damp_cap:.6e}")
        damp = min(damp * 10.0, damp_cap)


def variant_baseline(l: torch.Tensor) -> dict:
    """A: current path -- cholesky_inverse -> second cholesky."""
    t0 = time.time()
    h_inv = torch.cholesky_inverse(l, upper=False)
    t_inv = time.time() - t0

    # Diagnose H_inv state before second chol
    asymmetry = (h_inv - h_inv.T).abs().max().item()
    min_diag = h_inv.diagonal().min().item()
    diag_mean = h_inv.diagonal().mean().item()

    t0 = time.time()
    l_hi, info = torch.linalg.cholesky_ex(h_inv, upper=False)
    t_chol = time.time() - t0
    return dict(
        info=int(info.item()),
        t_inv=t_inv,
        t_chol=t_chol,
        asymmetry=asymmetry,
        min_diag=min_diag,
        diag_mean=diag_mean,
    )


def variant_symmetrize(l: torch.Tensor) -> dict:
    """B: cholesky_inverse -> H_inv = 0.5*(H_inv + H_inv.T) -> second cholesky.

    Hypothesis: rocBLAS DGEMM accumulation order makes H_inv slightly
    non-symmetric; cholesky checks fail on min(diag) when the off-diagonal
    drift bleeds into the diagonal via the LDL elimination.
    """
    t0 = time.time()
    h_inv = torch.cholesky_inverse(l, upper=False)
    h_inv = 0.5 * (h_inv + h_inv.T)
    t_inv = time.time() - t0

    asymmetry = (h_inv - h_inv.T).abs().max().item()
    min_diag = h_inv.diagonal().min().item()
    diag_mean = h_inv.diagonal().mean().item()

    t0 = time.time()
    l_hi, info = torch.linalg.cholesky_ex(h_inv, upper=False)
    t_chol = time.time() - t0
    return dict(
        info=int(info.item()),
        t_inv=t_inv,
        t_chol=t_chol,
        asymmetry=asymmetry,
        min_diag=min_diag,
        diag_mean=diag_mean,
    )


def variant_symmetrize_plus_ridge(l: torch.Tensor, ridge_eps: float = 1e-8) -> dict:
    """C: symmetrize + add tiny ridge to diagonal to guarantee PSD."""
    t0 = time.time()
    h_inv = torch.cholesky_inverse(l, upper=False)
    h_inv = 0.5 * (h_inv + h_inv.T)
    diag_mean = h_inv.diagonal().mean().item()
    h_inv.diagonal().add_(ridge_eps * abs(diag_mean))
    t_inv = time.time() - t0

    t0 = time.time()
    l_hi, info = torch.linalg.cholesky_ex(h_inv, upper=False)
    t_chol = time.time() - t0
    return dict(
        info=int(info.item()),
        t_inv=t_inv,
        t_chol=t_chol,
        diag_mean=diag_mean,
        ridge_eps=ridge_eps,
    )


def variant_solve_triangular(l: torch.Tensor) -> dict:
    """D: solve_triangular(L, eye) -> L_inv -> H_inv = L_inv.T @ L_inv.

    This avoids cholesky_inverse's internal TRSM (which emits the
    'workspace too small' warning at large K). The math is identical
    mathematically; the implementation path differs.
    """
    k = l.shape[0]
    eye_k = torch.eye(k, dtype=torch.float64, device=l.device)

    t0 = time.time()
    l_inv = torch.linalg.solve_triangular(l, eye_k, upper=False)
    h_inv = l_inv.T @ l_inv
    del l_inv, eye_k
    t_inv = time.time() - t0

    asymmetry = (h_inv - h_inv.T).abs().max().item()
    min_diag = h_inv.diagonal().min().item()
    diag_mean = h_inv.diagonal().mean().item()

    t0 = time.time()
    l_hi, info = torch.linalg.cholesky_ex(h_inv, upper=False)
    t_chol = time.time() - t0
    return dict(
        info=int(info.item()),
        t_inv=t_inv,
        t_chol=t_chol,
        asymmetry=asymmetry,
        min_diag=min_diag,
        diag_mean=diag_mean,
    )


def main() -> int:
    print(f"=== second-Cholesky probe: {TARGET_NAME} ===")
    print(f"hessian: {HESSIAN_PATH}  device: {DEVICE}")
    print(f"torch: {torch.__version__}  hip: {torch.version.hip}")
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

    with HessianSidecar(HESSIAN_PATH) as sc:
        from gptq_gpu_pkg.names import to_hfhs_key
        key = to_hfhs_key(TARGET_NAME + ".weight")
        if not sc.has(key):
            print(f"FAIL: Hessian missing for key {key!r}")
            return 1
        e = sc.get(key)
        print(f"sidecar entry: K={e.k}  dtype={e.dtype}")
        h = sc.load_f64(key, device=DEVICE)

    print(f"H loaded: shape={tuple(h.shape)}  dtype={h.dtype}  device={h.device}")
    print(f"  diag mean={h.diagonal().mean().item():.6e}  min diag={h.diagonal().min().item():.6e}")
    print(f"  asymmetry={(h - h.T).abs().max().item():.6e}  (should be ~0; H is symmetric in storage)")

    # First Cholesky (this is the part that already works)
    print("\n--- first Cholesky on H + damp*I ---")
    t0 = time.time()
    l, eff_damp = damped_first_cholesky(h, initial_damp=0.01 * h.diagonal().mean().item(), max_mult=1.0)
    print(f"L computed in {time.time()-t0:.2f}s  effective_damp={eff_damp:.6e}")

    # Run each variant
    print("\n--- variant A: baseline (cholesky_inverse -> chol) [CURRENT BROKEN PATH] ---")
    a = variant_baseline(l)
    print(f"  result: info={a['info']}  t_inv={a['t_inv']:.2f}s  t_chol={a['t_chol']:.2f}s")
    print(f"  H_inv state: asymmetry={a['asymmetry']:.3e}  min_diag={a['min_diag']:.3e}  diag_mean={a['diag_mean']:.3e}")
    if a['info'] != 0:
        print(f"  → SECOND CHOLESKY FAILED (matches the production failure)")

    print("\n--- variant B: cholesky_inverse + symmetrize -> chol ---")
    b = variant_symmetrize(l)
    print(f"  result: info={b['info']}  t_inv={b['t_inv']:.2f}s  t_chol={b['t_chol']:.2f}s")
    print(f"  H_inv state: asymmetry_after_sym={b['asymmetry']:.3e}  min_diag={b['min_diag']:.3e}  diag_mean={b['diag_mean']:.3e}")
    if b['info'] == 0:
        print(f"  → ✓ SYMMETRIZE FIX WORKS")
    else:
        print(f"  → still fails")

    print("\n--- variant C: symmetrize + ridge (ε=1e-8 * diag_mean) -> chol ---")
    c = variant_symmetrize_plus_ridge(l, ridge_eps=1e-8)
    print(f"  result: info={c['info']}  t_inv={c['t_inv']:.2f}s  t_chol={c['t_chol']:.2f}s  ridge={c['ridge_eps']}")
    if c['info'] == 0:
        print(f"  → ✓ symmetrize+ridge works")
    else:
        print(f"  → still fails; trying ε=1e-6")
        c2 = variant_symmetrize_plus_ridge(l, ridge_eps=1e-6)
        print(f"  ε=1e-6 result: info={c2['info']}  t_chol={c2['t_chol']:.2f}s")

    print("\n--- variant D: solve_triangular path (L_inv.T @ L_inv) -> chol ---")
    d = variant_solve_triangular(l)
    print(f"  result: info={d['info']}  t_inv={d['t_inv']:.2f}s  t_chol={d['t_chol']:.2f}s")
    print(f"  H_inv state: asymmetry={d['asymmetry']:.3e}  min_diag={d['min_diag']:.3e}  diag_mean={d['diag_mean']:.3e}")
    if d['info'] == 0:
        print(f"  → ✓ SOLVE_TRIANGULAR PATH WORKS")
    else:
        print(f"  → still fails (TRSM workspace issue persists?)")

    print("\n=== SUMMARY ===")
    print(f"  A (current broken):           info={a['info']}")
    print(f"  B (symmetrize):               info={b['info']}")
    print(f"  C (symmetrize + 1e-8 ridge):  info={c['info']}")
    print(f"  D (solve_triangular):         info={d['info']}")

    # Recommendation
    if b['info'] == 0:
        print("\n  RECOMMENDED FIX: add `h_inv = 0.5 * (h_inv + h_inv.T)` after cholesky_inverse in algo.py:280")
    elif c['info'] == 0:
        print("\n  RECOMMENDED FIX: symmetrize + small diagonal ridge")
    elif d['info'] == 0:
        print("\n  RECOMMENDED FIX: switch to solve_triangular path (slower but correct on gfx906)")
    else:
        print("\n  NONE OF THE FIXES WORKED — deeper rocBLAS issue, escalate to upstream")

    return 0


if __name__ == "__main__":
    sys.exit(main())
