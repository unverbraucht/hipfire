"""Per-tensor GPTQ pipeline: full quantize-time chain for one MQ4G256 tensor.

Mirrors `gptq.rs::gptq_pipeline_mq4g256` step-by-step, all in FP64 on
GPU (CUDA or HIP/ROCm). Inputs: BF16/F32 weight, FP32 unrotated Hessian (from HFHS sidecar),
AWQ scale vector (or `ones(K)` for non-AWQ tensors), FWHT signs.

Output: the **PipelineResult** dataclass holding the post-GPTQ weights
(in original column order, ready to write to `weights.safetensors`),
the per-256-block frozen grids, plus diagnostic stats (effective damp,
MSE vs original, clamp counts).

The pipeline matches the algorithm comment block in
`docs/plans/gptq_cuda.md` §2 exactly:
  1. AWQ-rescale H        H ← diag(1/s) H diag(1/s)
  2. FWHT-similarity H    H ← R H R^T per 256-block
  3. symmetrize H         H ← 0.5 (H + H^T)
  4. FWHT-rotate W        W ← per-row R W per 256-block
  5. frozen grids         G_k = (scale_k, min_val_k) from W
  6. actorder + Cholesky  P, U with U^T U = (P^T (H+λI) P)^-1
  7. column-sequential    GPTQ inner loop using frozen grids
  8. emit                 post-GPTQ W (un-permuted), G, s
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .algo import (
    apply_awq_rescaling_h,
    apply_fwht_per_256_to_weights,
    compute_damped_inv_cholesky_upper,
    compute_frozen_block_grids,
    fwht_similarity_per_256_h,
    quantize_mq4_with_grid,
    symmetrize_in_place,
    weight_mode_actorder,
)


@dataclass
class PipelineResult:
    """Output of `gptq_one_tensor`.

    `weights` is on the SAME device and stored as FP64 (cast down by
    the caller when writing the manifest). It's in ORIGINAL column
    order (un-permuted), in the AWQ-scaled + FWHT-rotated basis.

    `frozen_grids` has shape `[n_blocks, 2]` with `[:, 0]=scale` and
    `[:, 1]=min_val`, indexed in flat-block order matching the row-major
    `M*K`-length view (`block_idx = (row*K + col) // 256`).

    Diagnostics: `effective_damp` (damping that succeeded in Cholesky),
    `mse_vs_original` (mean squared error of the dequantized output
    against the original pre-GPTQ weight; outlier check at 10× median).
    `clamps` is `(below_count, above_count)` for the OBS-induced clamp
    diagnostic from Rust `quantize_mq4_element_with_clamp`.
    """
    weights: torch.Tensor             # [M, K] FP64, post-GPTQ, un-permuted
    frozen_grids: torch.Tensor        # [n_blocks, 2] FP64
    effective_damp: float
    mse_vs_original: float
    clamps: tuple[int, int]           # (below, above)


def gptq_one_tensor(
    weights_input: torch.Tensor,      # [M, K] FP32/BF16 input
    h_unrot: torch.Tensor,            # [K, K] FP64 Hessian, unrotated/unscaled
    awq_scales: torch.Tensor,         # [K] FP64; pass `ones(K)` for non-AWQ
    signs1: torch.Tensor,             # [256] FP64, seed=42
    signs2: torch.Tensor,             # [256] FP64, seed=1042
    *,
    initial_damp_ratio: float = 0.01,
    max_damp_multiplier: float = 1.0,
    n_bits: int = 4,
    name: str = "<unnamed>",
) -> PipelineResult:
    """One MQ4G256 tensor through the GPTQ pipeline.

    Returns `PipelineResult` on success. Raises `CholeskyFailedError`
    if even adaptive damping at the cap fails — caller is expected to
    fall back to a plain (non-GPTQ) MQ4 packing for that tensor, same
    as Rust at `main.rs:4430-4433`.

    All FP64; inputs are upcast on entry, output is FP64 (cast down at
    write-time by the manifest writer). All operations run on the
    weight tensor's device.

    `initial_damp_ratio` = 0.01 means damp = 0.01 * mean(diag(H_target))
    initially; doubles each Cholesky retry up to
    `max_damp_multiplier * mean(diag(H_target))`. Matches Rust's
    `gptq_initial_damp` + `gptq_max_damp_multiplier` defaults.
    """
    device = weights_input.device
    assert h_unrot.shape == (weights_input.shape[1], weights_input.shape[1]), \
        f"H shape {tuple(h_unrot.shape)} vs W K-dim {weights_input.shape[1]}"
    assert awq_scales.shape == (weights_input.shape[1],), \
        f"awq scales shape {tuple(awq_scales.shape)} vs W K-dim {weights_input.shape[1]}"
    assert signs1.shape == (256,) and signs2.shape == (256,)
    assert h_unrot.device == device, f"H on {h_unrot.device} but W on {device}"

    m, k = weights_input.shape

    # 1. AWQ-rescale H (no-op if awq_scales == ones).
    #
    # MEMORY: We CONSUME `h_unrot` — mutating it in place rather than
    # cloning saves a 2.4 GB allocation at K=17408. The orchestrator
    # is expected to `del h_gpu` immediately after gptq_one_tensor
    # returns; it doesn't re-use the Hessian. Document this in the
    # contract.
    apply_awq_rescaling_h(h_unrot, awq_scales)

    # 2. FWHT-similarity per 256-block
    fwht_similarity_per_256_h(h_unrot, signs1, signs2)

    # 3. Symmetrize — scrubs O(ε·K) drift from the FWHT row+col passes
    symmetrize_in_place(h_unrot)

    # 4. FWHT-rotate W per row (in place after cast to FP64)
    w_rot = weights_input.to(torch.float64).contiguous()
    # AWQ pre-scale is the caller's responsibility — it bakes into the
    # weights upstream (matching Rust's `apply_awq_prescale` which mutates
    # weights, then passes both to `gptq_pipeline_mq4g256`). The caller
    # for AWQ-eligible tensors should do `W *= s[None, :]` before calling
    # this function.
    apply_fwht_per_256_to_weights(w_rot, signs1, signs2)

    # 5. Frozen per-256-block grids — computed from POST-rotated W, frozen
    #    through the loop. Row-major flat block index: `(row*K + col)/256`.
    w_flat = w_rot.view(-1)
    frozen_grids = compute_frozen_block_grids(w_flat, n_bits=n_bits)

    # 6. WEIGHT-mode actorder + Cholesky-direct upper factor of H_inv.
    #
    # Permute h_unrot OUTSIDE compute_damped_inv_cholesky_upper so we
    # can immediately free the un-permuted version (saves 2.4 GB peak
    # at K=17408 — critical to fit on a 16 GB consumer card).
    h_diag = h_unrot.diagonal()
    initial_damp = initial_damp_ratio * h_diag.mean().item()
    perm = weight_mode_actorder(h_diag)
    del h_diag
    h_perm = h_unrot[perm][:, perm].contiguous()
    del h_unrot
    chol = compute_damped_inv_cholesky_upper(
        h_perm, perm=None, initial_damp=initial_damp,
        max_damp_multiplier=max_damp_multiplier,
    )
    u = chol.u  # [K, K] upper-tri, U^T U = (P^T (H+λI) P)^-1
    effective_damp = chol.effective_damp
    # `h_perm` was already consumed inside `compute_damped_inv_cholesky_upper`'s
    # internal `del h_eff` — but we still hold our caller-side reference.
    del h_perm

    # 7. Column-sequential GPTQ.
    #
    # Two state buffers per Rust:
    #   - `w_residual`: running residual, mutated by OBS propagation,
    #     READ as the source for the next column's quantize.
    #   - `w_out`: output buffer, gets the quantized-then-dequantized
    #     value for each processed column.
    # We keep both as [M, K] FP64, mutating slices.
    w_residual = w_rot.clone()
    w_out = w_rot.clone()  # carries quantized values; un-processed cols
                            # stay at their FWHT-rotated initial value
                            # until their step. After the loop, all cols
                            # are quantized.

    # Frozen-grid lookup tensors: per the flat block-index convention,
    # `grid_block_idx[row, col] = (row*K + col) // 256` reshapes to
    # `[M, K]` but is most cheaply computed per column inside the loop.
    # Pre-compute the `n_blocks_per_row = K/256` constant and use it.
    assert k % 256 == 0, f"K={k} must be divisible by 256 for MQ4G256"
    n_blocks_per_row = k // 256

    clamps_below = 0
    clamps_above = 0

    for step in range(k):
        j_orig = int(perm[step].item())
        u_ss = u[step, step].item()
        if u_ss <= 0.0:
            # Defensive: should not happen post-cholesky_ex success
            # but mirrors Rust's continue.
            continue

        # Block index for column j_orig across all M rows is
        # row * n_blocks_per_row + (j_orig // 256). Pre-compute once.
        col_block_in_row = j_orig // 256
        # Per-row block index → vectorize via arange:
        row_idx = torch.arange(m, device=device)
        block_idx_col = row_idx * n_blocks_per_row + col_block_in_row
        block_grid = frozen_grids[block_idx_col]   # [M, 2]
        scale_col = block_grid[:, 0]               # [M]
        min_col = block_grid[:, 1]                 # [M]

        # Phase A: quantize column j_orig from residual; compute err.
        w_col_residual = w_residual[:, j_orig]
        q = quantize_mq4_with_grid(w_col_residual, scale_col, min_col, n_bits=n_bits)
        # Clamp diagnostic — count pre-clamp grid indices outside [0, 15].
        safe_scale = torch.where(scale_col != 0, scale_col, torch.ones_like(scale_col))
        q_raw = torch.floor((w_col_residual - min_col) / safe_scale + 0.5)
        clamps_below += int((q_raw < 0).sum().item())
        clamps_above += int((q_raw > float((1 << n_bits) - 1)).sum().item())

        w_out[:, j_orig] = q
        err_col = (w_col_residual - q) / u_ss     # [M]

        # Phase B: OBS propagation. Update `w_residual[:, kk_orig]` for
        # all `next_step` > step. The propagation rows are `perm[step+1:]`
        # in PROCESSING order; the corresponding U entries are
        # `u[step, step+1:]` (a [K-step-1] row). Apply:
        #   w_residual[:, kk_orig] -= err_col[:, None] * u_sn
        # vectorized over both rows AND remaining columns.
        if step + 1 < k:
            next_perm = perm[step + 1:]                # [K - step - 1]
            u_row = u[step, step + 1:]                  # [K - step - 1]
            # outer product err_col [M, 1] * u_row [1, K-step-1]
            update = err_col.unsqueeze(1) * u_row.unsqueeze(0)
            # scatter-subtract into w_residual at columns `next_perm`
            w_residual[:, next_perm] -= update

    # 8. Diagnostics: MSE(post-GPTQ dequantized, original FWHT-rotated W).
    #    Tracks how aggressively GPTQ moved each tensor's quantized
    #    output away from a "no-GPTQ" pack. Outlier check at 10×
    #    median of MSE-per-tensor in the orchestrator (plan §5.3).
    mse = ((w_out - w_rot) ** 2).mean().item()

    return PipelineResult(
        weights=w_out,
        frozen_grids=frozen_grids,
        effective_damp=effective_damp,
        mse_vs_original=mse,
        clamps=(clamps_below, clamps_above),
    )
