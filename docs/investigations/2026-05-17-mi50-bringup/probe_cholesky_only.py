"""Isolate the cholesky_ex failure mode at K=9216 on gfx906.

Variants:
  i. Real H_inv-shaped matrix from a real down_proj Hessian
  ii. Diagonal-dominant random K=9216 (clean PSD)
  iii. Identity matrix scaled
  iv. Same as iii but K=4096 / K=8192 / K=10240 / K=12288 to find the cliff
"""

from __future__ import annotations

import sys
import time
import torch

sys.path.insert(0, "scripts")


def try_cholesky(name: str, h: torch.Tensor, expected_pd: bool = True) -> int:
    k = h.shape[0]
    asym = (h - h.T).abs().max().item()
    min_d = h.diagonal().min().item()
    diag_mean = h.diagonal().mean().item()
    print(f"\n  {name} K={k}: dtype={h.dtype} dev={h.device} diag_mean={diag_mean:.3e} min_diag={min_d:.3e} asym={asym:.3e}")
    t0 = time.time()
    l, info = torch.linalg.cholesky_ex(h, upper=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    code = int(info.item())
    print(f"    cholesky_ex: info={code} (expected 0)  dt={dt:.2f}s")
    if code == 0 and expected_pd:
        # Verify
        reconstr = l @ l.T
        residual = (reconstr - h).abs().max().item()
        print(f"    L L^T vs H: max abs diff = {residual:.3e}")
    return code


def main():
    print(f"torch: {torch.__version__}  hip: {torch.version.hip}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    DEV = "cuda:0"

    # K sweep with a CLEAN PSD: diagonal-dominant random.
    print("\n=== K sweep: diagonal-dominant random PSD ===")
    for k in (2048, 4096, 6144, 8192, 9216, 10240, 12288):
        torch.manual_seed(0)
        a = torch.randn(k, k, dtype=torch.float64, device=DEV) * (1.0 / k**0.5)
        h = a @ a.T
        del a
        h.add_(torch.eye(k, dtype=torch.float64, device=DEV), alpha=0.5)
        try_cholesky(f"clean PSD", h)
        del h
        torch.cuda.empty_cache()

    # Scaled identity — must trivially succeed.
    print("\n=== K=9216 trivial: 2 * I ===")
    h = 2.0 * torch.eye(9216, dtype=torch.float64, device=DEV)
    try_cholesky("2I", h)
    del h
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
