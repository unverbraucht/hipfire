"""Check pytorch build configuration and CPU Cholesky availability."""
import torch, time

print(f"torch {torch.__version__}  hip {torch.version.hip}")
cfg = torch.__config__.show()
for tag in ("USE_MAGMA", "USE_LAPACK", "USE_BLAS", "USE_NUMPY", "USE_ROCM", "BLAS_INFO"):
    for line in cfg.splitlines():
        if tag in line:
            print(f"  {line.strip()}")
            break

print("\n--- CPU cholesky K=9216 (FP64) ---")
torch.manual_seed(0)
K = 9216
a = torch.randn(K, K, dtype=torch.float64) * (1.0 / K**0.5)
h = a @ a.T
del a
h.add_(torch.eye(K, dtype=torch.float64), alpha=0.5)
t0 = time.time()
try:
    l, info = torch.linalg.cholesky_ex(h, upper=False)
    dt = time.time() - t0
    code = int(info.item())
    print(f"  CPU cholesky_ex: info={code}  t={dt:.2f}s")
    if code == 0:
        resid = (l @ l.T - h).abs().max().item()
        print(f"  L@L.T vs H residual: {resid:.3e}  (FP64 expected ~1e-13)")
        # try cholesky_inverse
        t0 = time.time()
        h_inv = torch.cholesky_inverse(l, upper=False)
        print(f"  cholesky_inverse t={time.time()-t0:.2f}s")
        resid2 = (h_inv @ h - torch.eye(K, dtype=torch.float64)).abs().max().item()
        print(f"  (H_inv @ H) - I residual: {resid2:.3e}")
        # try second cholesky
        l2, info2 = torch.linalg.cholesky_ex(h_inv, upper=False)
        print(f"  second cholesky on H_inv: info={int(info2.item())}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
