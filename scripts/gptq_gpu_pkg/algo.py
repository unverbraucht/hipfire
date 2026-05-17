"""Pure-math primitives for the CUDA GPTQ pipeline.

Mirrors the Rust reference at `crates/hipfire-quantize/src/gptq.rs` and
`crates/hipfire-quantize/src/main.rs` (FWHT-256 + AWQ helpers). All
linear algebra is FP64 — FP32 Cholesky on K≥4096 with cond≥1e6 has zero
effective precision (per master-doc §6 rule, mirror of Rust gptq.rs
header comment).

Symbols here are torch-only (`tensor`-typed) so the GPTQ pipeline can
stay on GPU. Stand-alone unit tests for the pure-math kernels live in
`scripts/tests/test_gptq_algo.py` and exercise both CPU and CUDA tensors.

Function-by-function correspondence with Rust:

| Python                                | Rust                                       |
|---------------------------------------|--------------------------------------------|
| `gen_fwht_signs(seed, n)`             | `main.rs::gen_fwht_signs`                  |
| `fwht_256_inplace(x, s1, s2)`         | `main.rs::cpu_fwht_256` / `gptq.rs::fwht_256_inplace_f64` |
| `compute_awq_scales(in_sum2, alpha)`  | `main.rs::compute_awq_scales`              |
| `apply_awq_rescaling_h(H, s)`         | `gptq.rs::apply_awq_rescaling`             |
| `fwht_similarity_per_256_h(H, ...)`   | `gptq.rs::fwht_similarity_per_256`         |
| `symmetrize_in_place(H)`              | `gptq.rs::symmetrize_in_place`             |
| `apply_fwht_per_256_to_weights(W, …)` | `gptq.rs::apply_fwht_per_256_to_weights_f64` |
| `compute_frozen_block_grids(W)`       | `gptq.rs::compute_frozen_block_grids`      |
| `compute_damped_inv_cholesky_upper(H, perm, damp, max_mul)` | `gptq.rs::compute_damped_inv_cholesky_upper` |
| `weight_mode_actorder(diag)`          | `gptq.rs::weight_mode_actorder`            |
| `gptq_column_sequential(W, H, …)`     | `gptq.rs::gptq_column_sequential`          |

Frozen-grid layout (block index = `(row*K + col) // 256`) matches Rust's
`block_idx_for` row-major flat layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ─── FWHT-256 ─────────────────────────────────────────────────────────────
#
# Sign-PRNG: linear congruential generator that mirrors
# `main.rs::gen_fwht_signs` byte-for-byte (state = state * 1103515245 +
# 12345, mod 2^31, take bit 16, map {0→-1, 1→+1}). Used for both signs1
# (seed=42) and signs2 (seed=1042). Sign tensors live on the same device
# as the weight; emit them once per pipeline run, not per tensor.

def gen_fwht_signs(seed: int, n: int, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Generate the FWHT sign table for a given seed.

    Matches `main.rs::gen_fwht_signs` exactly (LCG with `1103515245`
    multiplier + `12345` increment, mask `0x7fffffff`, bit-16 → ±1).
    Used for both pre-multiply (seed=42) and post-multiply (seed=1042)
    sign vectors of the FWHT-256.
    """
    state = seed & 0xffffffff
    out = torch.empty(n, dtype=dtype, device=device)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7fffffff
        out[i] = 1.0 if ((state >> 16) & 1) == 1 else -1.0
    return out


def fwht_256_inplace(x: torch.Tensor, signs1: torch.Tensor, signs2: torch.Tensor) -> None:
    """In-place FWHT-256 on the last dim, expected length 256.

    Sign convention + 1/16 (= 1/sqrt(256)) scale match
    `main.rs::cpu_fwht_256` / `gptq.rs::fwht_256_inplace_f64`. Works on
    any leading-shape (the butterfly is broadcast over them).
    """
    assert x.shape[-1] == 256
    x.mul_(signs1)
    stride = 1
    while stride < 256:
        # Reshape last axis into [n_pairs, 2*stride] so butterfly halves
        # are addressable as views; works for any leading shape because
        # all earlier dims are preserved.
        view = x.reshape(*x.shape[:-1], -1, 2 * stride)
        a = view[..., :stride].clone()
        b = view[..., stride:].clone()
        view[..., :stride] = a + b
        view[..., stride:] = a - b
        stride <<= 1
    x.mul_(1.0 / 16.0)
    x.mul_(signs2)


def apply_fwht_per_256_to_weights(weights: torch.Tensor, signs1: torch.Tensor, signs2: torch.Tensor) -> None:
    """In-place per-256 FWHT on each row of the [M, K] weight matrix.

    Mirrors `gptq.rs::apply_fwht_per_256_to_weights_f64`. Requires
    `K % 256 == 0`. Reshapes to [M, K/256, 256] so the FWHT runs in a
    single fused kernel across all blocks of all rows.
    """
    m, k = weights.shape
    assert k % 256 == 0, f"K={k} must be divisible by 256"
    view = weights.view(m, k // 256, 256)
    fwht_256_inplace(view, signs1, signs2)


def fwht_similarity_per_256_h(h: torch.Tensor, signs1: torch.Tensor, signs2: torch.Tensor) -> None:
    """In-place per-256-block FWHT similarity transform on a K×K Hessian.

    Mirrors `gptq.rs::fwht_similarity_per_256` two-pass row-then-column
    formulation: first apply FWHT-256 to every 256-element column slice
    of every row (right-multiply by R^T block-by-block on the column
    axis), then apply FWHT-256 to every 256-element row slice of every
    column (left-multiply by R block-by-block on the row axis). Net
    effect: `H ← R · H · R^T` where `R` is the K×K block-diagonal
    FWHT operator.

    Result is symmetric in exact arithmetic but accumulates O(ε·K) FP
    drift across the two passes — the caller must follow with
    `symmetrize_in_place` before feeding it to Cholesky.
    """
    k = h.shape[0]
    assert h.shape == (k, k)
    assert k % 256 == 0

    # Stage 1: H ← H · R^T  — equivalent to FWHT every 256-element
    # column slice of each row. View [K, K/256, 256] then fwht over
    # the last dim.
    view = h.view(k, k // 256, 256)
    fwht_256_inplace(view, signs1, signs2)

    # Stage 2: H ← R · H — equivalent to FWHT every 256-element row
    # slice of each column. Transpose, repeat the column-FWHT, transpose
    # back. The transposed view is non-contiguous in memory but the
    # FWHT butterfly takes any tensor whose last dim is 256.
    h_t = h.t().contiguous()  # forces a copy to avoid the view-of-view issue
    view_t = h_t.view(k, k // 256, 256)
    fwht_256_inplace(view_t, signs1, signs2)
    h.copy_(h_t.t())


def symmetrize_in_place(h: torch.Tensor) -> None:
    """`H ← 0.5 * (H + H^T)`. Scrubs FP drift after FWHT similarity.

    Required between `fwht_similarity_per_256_h` and Cholesky — otherwise
    `gptq_column_sequential` reads `H_inv[j, kk]` for asymmetric (j, kk)
    pairs and OBS propagation corrupts. See Rust comment at
    `gptq.rs:643-650`.
    """
    h.add_(h.t().clone()).mul_(0.5)


def apply_awq_rescaling_h(h: torch.Tensor, awq_scales: torch.Tensor) -> None:
    """`H ← diag(1/s) · H · diag(1/s)`.

    Mirrors `gptq.rs::apply_awq_rescaling`. For non-AWQ tensors, pass
    `awq_scales = ones(K)` and the function is a no-op (multiplies by 1).
    """
    k = h.shape[0]
    assert h.shape == (k, k)
    assert awq_scales.shape == (k,)
    inv = 1.0 / awq_scales
    h.mul_(inv.unsqueeze(0)).mul_(inv.unsqueeze(1))


# ─── AWQ scales (geo-mean in log space) ───────────────────────────────────

def compute_awq_scales(in_sum2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute per-channel AWQ scales `s[j]` (geo-mean normalized to 1).

    Mirrors `main.rs::compute_awq_scales` exactly:
    `log(s_raw[j]) = (alpha/2) * log(max(in_sum2[j], 1e-12))`,
    then subtract `mean(log(s_raw))` and exponentiate.

    The `in_sum2` is `N_tok * RMS_act[j]²` as the imatrix stores it;
    the `N_tok^(alpha/2)` constant cancels under geo-mean normalization.
    The effective formula is `s[j] = RMS_act[j]^alpha`. Do NOT change
    this to `RMS^(alpha/2)` — would be a 2× exponent error (master-doc
    §6 rule and plan §4.1).
    """
    in_sum2_64 = in_sum2.to(torch.float64).clamp_min(1e-12)
    half_alpha = 0.5 * float(alpha)
    log_s_raw = half_alpha * in_sum2_64.log()
    mean_log = log_s_raw.mean()
    return (log_s_raw - mean_log).exp()


# ─── Cholesky-direct OBS (the OBS-fix from Rust commit 687aa2d0) ──────────

@dataclass
class CholeskyResult:
    """`U^T · U = (P^T (H + damp*I) P)^-1` invariant satisfied by `u`.

    `effective_damp` is the damping value that the adaptive cascade
    actually used to make Cholesky succeed (caller diagnostic).
    """
    u: torch.Tensor
    effective_damp: float


class CholeskyFailedError(RuntimeError):
    """Cholesky failed even at max damping. Caller falls back to plain
    MQ4 packing (matches Rust `quantize_mq4g256` fallback path)."""


def compute_damped_inv_cholesky_upper(
    h: torch.Tensor,
    perm: torch.Tensor | None,
    initial_damp: float,
    max_damp_multiplier: float,
) -> CholeskyResult:
    """Returns upper-tri `U` with `U^T · U = (P^T(H+λI)P)^-1`.

    This is the Frantar-Algorithm-1 form (master-doc rule, Rust commit
    687aa2d0). The earlier `L_H^{-T}` variant satisfies `U·U^T = H_inv`
    instead and breaks the Schur-complement submatrix property that OBS
    propagation depends on — produced GPTQ quality regressions at every
    model size tested.

    Algorithm:
      1. `H_eff = P^T H P` (or H if perm is None).
      2. damp = clamp(initial_damp, ε·diag_mean(H_eff)).
      3. Loop:
         a. `A = H_eff + damp*I`.
         b. `L, info = cholesky_ex(A)` (non-throwing).
         c. If `info==0`: break. Else: `damp *= 10`, clamp to `damp_cap`.
      4. `L_inv = solve_triangular(L, I, upper=False)` → lower-tri.
      5. `H_inv = L_inv^T @ L_inv` (symmetric K×K).
      6. `L_HI, info = cholesky_ex(H_inv)`.
      7. `U = L_HI^T` (upper-tri, `U^T·U = L_HI·L_HI^T = H_inv`).

    Uses `cholesky_ex` (non-throwing) so the damp-retry loop doesn't
    pay per-iter exception serialization on the CUDA stream — plan §4.6.
    """
    k = h.shape[0]
    assert h.shape == (k, k)

    if perm is not None:
        # P^T H P — permute rows and cols by `perm`. This realizes the
        # WEIGHT-mode actorder so U is upper-tri relative to the
        # processing order, not the storage order.
        h_eff = h[perm][:, perm].contiguous()
    else:
        h_eff = h.contiguous()

    diag_mean = h_eff.diagonal().mean().item()
    # `clamped_initial_damp` from Rust — snap to ε·diag_mean to prevent
    # damp=0 infinite-loop on singular matrices when the user passes 0.
    damp = max(initial_damp, torch.finfo(torch.float64).eps * max(diag_mean, 1.0))
    damp_cap = max_damp_multiplier * diag_mean

    # Memory budget at K=17408 FP64: each K×K is 2.43 GB. To fit on a
    # 16 GB consumer card we aggressively `del` intermediates and skip
    # the explicit `l_inv` materialization in favor of `cholesky_inverse`
    # (potri), which produces H_inv directly from L without materializing
    # the lower-triangular inverse. Saves ~2.4 GB peak per call.

    # `damped` is `h_eff + damp*I` — done in-place via diagonal.add_ on a
    # clone, avoiding the 2.4 GB `damp * eye_k` intermediate.
    effective_damp = damp
    l = None
    while True:
        damped = h_eff.clone()
        damped.diagonal().add_(damp)
        l_try, info = torch.linalg.cholesky_ex(damped, upper=False)
        del damped
        if int(info.item()) == 0:
            l = l_try
            effective_damp = damp
            break
        del l_try
        if damp >= damp_cap:
            raise CholeskyFailedError(
                f"Cholesky of K={k} Hessian failed even at damp={damp:.6e} "
                f"(diag mean={diag_mean:.6e}); skip GPTQ for this tensor"
            )
        damp = min(damp * 10.0, damp_cap)

    # We own h_eff (it's either a permuted copy of the caller's `h` or a
    # contiguous-no-permute clone). Free it now that Cholesky is done —
    # the caller's `h` is held independently outside our scope.
    del h_eff

    # H_inv = (L L^T)^-1 in one call (cuSOLVER's potri). Avoids the
    # explicit l_inv intermediate (K×K = 2.4 GB at K=17408).
    h_inv = torch.cholesky_inverse(l, upper=False)
    del l

    l_hi, info = torch.linalg.cholesky_ex(h_inv, upper=False)
    if int(info.item()) != 0:
        raise CholeskyFailedError(
            f"Second Cholesky on H_inv (K={k}) failed at effective_damp={effective_damp:.6e}; "
            "H_inv lost PSD due to FP drift in matmul"
        )
    del h_inv
    u = l_hi.T.contiguous()
    del l_hi
    return CholeskyResult(u=u, effective_damp=effective_damp)


# ─── Frozen per-256-block grids ──────────────────────────────────────────

def compute_frozen_block_grids(
    weights_flat: torch.Tensor,
    n_bits: int = 4,
) -> torch.Tensor:
    """Per-256-element block (scale, min_val) pairs for N-bit quantization.

    Input is the row-major flat `M*K`-length FP64 weight buffer (POST-
    FWHT and POST-AWQ-scale, per the pipeline). Output is shape
    `[n_blocks, 2]` where `n_blocks = M*K/256` and `[:, 0] = scale`,
    `[:, 1] = min_val`. `scale = (max - min) / (2^n_bits - 1)`;
    `scale = 1.0` when `range == 0` (constant block).

    `n_bits` choices currently supported by hipfire's runtime:
      4 → MQ4G256 (default, 16 levels, scale = range/15)
      3 → MQ3G256 (8 levels, scale = range/7) — note master-doc §5 warns
            uniform MQ3 may collapse; pair with AWQ pre-scale + GPTQ for
            best chance.

    Frozen pre-loop: matches `gptq.rs::compute_frozen_block_grids`. The
    layout is per-flat-block, NOT per-row-then-block — so block index
    for `(row, col)` in the M×K matrix is `(row*K + col) // 256`. Rust
    asserts this exact convention in `block_idx_for`. The grid array
    stays in this (un-permuted) layout through the whole GPTQ loop.
    """
    if n_bits not in (3, 4):
        raise ValueError(f"unsupported n_bits={n_bits}; only 3 and 4 are wired up")
    levels_minus_1 = (1 << n_bits) - 1  # 15 for 4-bit, 7 for 3-bit
    n = weights_flat.shape[0]
    assert n % 256 == 0, f"weight buffer length {n} must be divisible by 256"
    n_blocks = n // 256
    blocks = weights_flat.view(n_blocks, 256)
    min_vals = blocks.min(dim=1).values
    max_vals = blocks.max(dim=1).values
    ranges = max_vals - min_vals
    scales = torch.where(
        ranges > 0,
        ranges / float(levels_minus_1),
        torch.ones_like(ranges),
    )
    return torch.stack([scales, min_vals], dim=1)


def quantize_mq4_with_grid(
    w: torch.Tensor,
    scale: torch.Tensor,
    min_val: torch.Tensor,
    n_bits: int = 4,
) -> torch.Tensor:
    """Per-element N-bit quantize+dequant using a frozen grid.

    `q = clamp(round((w - min) / scale), 0, 2^n_bits - 1)`,
    `dequant = q * scale + min`. Element-wise; `w`, `scale`, `min_val`
    broadcast against each other. Matches `gptq.rs::quantize_mq4_element`
    + Rust's `+ 0.5).floor()` round-half-up convention.

    Despite the historical name, this function works for any 2-, 3-, or
    4-bit quant — the only difference is the clamp upper bound.
    """
    if n_bits not in (2, 3, 4):
        raise ValueError(f"unsupported n_bits={n_bits}")
    max_q = float((1 << n_bits) - 1)
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    q = torch.floor((w - min_val) / safe_scale + 0.5).clamp(0.0, max_q)
    return torch.where(scale != 0, q * scale + min_val, min_val)


# ─── WEIGHT-mode actorder ─────────────────────────────────────────────────

def weight_mode_actorder(h_diag: torch.Tensor) -> torch.Tensor:
    """Returns permutation indices ordered by `descending(diag(H))`.

    Stable sort to keep deterministic tie-breaking (Rust uses
    `partial_cmp` + stable `sort_by` for the same reason). The
    permutation is applied to BOTH `H` (similarity transform
    `P^T H P`) and the GPTQ processing order; the storage layout of `W`
    is unchanged (frozen grids and packed bytes index by original column).
    """
    return torch.argsort(h_diag, descending=True, stable=True)


def inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    """Inverse permutation: `inv[perm[i]] = i`. Used post-GPTQ to
    un-permute residual weights back into original-column order
    when needed (in our pipeline, frozen grids are kept un-permuted
    throughout, so this is mainly useful for sanity-check diagnostics)."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv
