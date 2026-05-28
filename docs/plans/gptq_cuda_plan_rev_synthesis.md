# GPTQ CUDA plan review synthesis — adjudication of three independent reviews

**Plan reviewed:** `docs/plans/gptq_cuda.md`
**Reviewers:** claude (`gptq_cuda_plan_rev_claude.md`), gemini (`gptq_cuda_plan_rev_gemini.md`), glm5 (`gptq_cuda_plan_rev_glm5.md`)
**Adjudicator:** claude, against source-of-truth code at HEAD `b709375c`
**Date:** 2026-05-15

Verdict legend: ✅ validated (code/doc confirms) · ❌ rejected (claim is wrong) · ➗ partial (some right, some wrong)

---

## Gemini findings

| # | Finding | Verdict | Evidence / note |
|---|---|---|---|
| G-C1 | FP64 mandatory for Cholesky + propagation | ✅ | `gptq.rs` is `Mat<f64>` throughout (138 f64 references, 28 explicit casts). My plan said nothing about precision. **New critical finding, incorporate.** |
| G-C2 | Actorder permutation omitted | ✅ | `gptq.rs:723 weight_mode_actorder`, used by `gptq_column_sequential` (line 769). Plan §2 has the inner loop as `for j in range(K)` with no permutation. **New critical finding, incorporate.** |
| G-C3 | Frozen-grid divergence | ✅ | Same as claude-C3. Overlap. |
| G-C4 | AWQ sidecar handoff broken — Rust skipping AWQ would skip writing the F16 `awq_scale.weight` sidecars the runtime needs to inverse-scale activations | ✅ | Confirmed: the `.hfq` reader pairs each MQ4G256 tensor with its `<name>.awq_scale.weight` sidecar (`hfq.rs:540, 549`); runtime dispatch keys on `awq_scale.is_some()`. If sidecars are missing, the kernel runs the non-AWQ activation path on AWQ-pre-scaled weights → catastrophic mismatch. **New critical finding, claude-C3 acknowledged the protocol gap but didn't name the runtime consequence. Incorporate.** |
| G-IG1 | Hessian re-symmetrization required after rotation | ✅ | `gptq.rs:508 symmetrize_in_place`, called at line 650 after FWHT similarity transform. Plan didn't mention. **Incorporate.** |
| G-IG2 | Damping schedule mirror | ✅ | Plan §2 mentions but vaguely. Tighten. |
| G-IG3 | Python K-loop overhead | ➗ | Valid concern but speculative. Note as performance caution, not blocker. |
| G-IG4 | RAM budget | ✅ | Same concern as claude-M4. Overlap. |

---

## GLM5 findings

| # | Sev | Finding | Verdict | Evidence / note |
|---|---|---|---|---|
| H1 | HIGH | RTX 5070 has 12 GB, not 16 GB | ➗ | User explicitly said "2× 16GB VRAM on 2 RTX5070" — defer to user. SKU may be RTX 5070 Ti (16 GB), or a regional variant. Per-tensor peak (~1.5 GB) fits in either 12 or 16 GB so the plan still works. **Fix: clarify with user, replace "16 GB" with the actual SKU, recompute headroom.** |
| H2 | HIGH | Hessian sidecar is ~6 GB, not 33 GB | ❌ | Actual file on disk: **33,822,883,298 bytes (33 GB)**. Docs say "~6 GB" (gptq.md, hessian_io.rs comment, gptq-hessian-format.md). Docs are stale; file size is authoritative. GLM5 was reading stale docs. **Plan was right; the companion doc is wrong. File a doc-fix issue.** |
| H3 | HIGH | Hessian entry format wrong in plan | ✅ | `hessian_io.rs:188-220` reads: `u32 name_len → utf8 name → u32 expert_idx → u32 K → u32 dtype_flag → payload`. Plan's `[u32 K, u32 reserved, f32×K²]` is fabricated. **Incorporate fix: replace inline summary with pointer to `gptq-hessian-format.md` §3.** |
| H4 | HIGH | AWQ exponent is `alpha`, not `alpha/2`, when expressed in terms of RMS_act | ✅ | Code uses `half_alpha * ln(in_sum2)` where in_sum2 = N_tok·RMS². After geo-mean normalization, effective `s = RMS^alpha`. Plan §1 says `s[j] = (RMS_act[j])^(alpha/2)` which is wrong when expressed in RMS terms; plan §2 pseudocode `s = (RMS_act ** 0.5)` is accidentally correct for alpha=0.5 but the formula text is misleading for general alpha. **Fix the formula text in §1.** |
| M1 | MED | FP64 vs FP32 precision unaddressed | ✅ | Same as G-C1. **Resolves under one rule: use `torch.float64` on CUDA for Cholesky + propagation. Document.** |
| M2 | MED | `scripts/gptq_9b_overnight.sh` doesn't exist | ✅ | The script is at `/tmp/gptq_9b_overnight.sh` only. Hand-back step §7 line 4 references the wrong path. **Fix: change to "create `scripts/gptq_9b_overnight.sh` in the cuda branch."** |
| M3 | MED | CUDA path creates NVIDIA-only ecosystem fork | ✅ | Real tradeoff. Plan should explicitly acknowledge: this machine (gfx1100, ROCm) cannot run the fast path; the RTX 5070 box is the only fast-quantize host. **Add an "ecosystem implications" section.** |
| M4 | MED | `compute_awq_scales` is in `main.rs`, not `gptq.rs` | ✅ | Confirmed `main.rs:2794`. Plan's "see `gptq.rs::compute_awq_scales` line 2805-2820" is wrong file. **Fix file reference.** |
| M5 | MED | `apply_gptq_column_sequential` doesn't exist | ✅ | Actual functions: `gptq_pipeline_mq4g256` (line 618) and `gptq_column_sequential` (line 769). Plan invented the name. **Fix.** |
| M6 | MED | `/tmp/` violates AGENTS.md rule 4 | ✅ | AGENTS.md: "Never store canonical bench prompts under `/tmp/`. /tmp gets wiped on reboot." 1h+ GPTQ output sits under the same spirit. **Fix: use `~/.hipfire/gptq-precomputed/` or `benchmarks/artifacts/gptq-cuda/`.** |
| M7 | MED | Companion doc has stale path to `hipfire-runtime/src/hessian_io.rs` | ✅ | File is in `crates/hipfire-quantize/src/`, not `hipfire-runtime/`. **Fix in `gptq-hessian-format.md`, separate from plan.** |
| L1 | LOW | "15h" vs documented ~14h | ✅ | Plan contradicts itself (§ TL;DR vs §3 hand-back). Normalize to "~14h". |
| L2 | LOW | AWQ whitelist omits `mlp.gate.weight` + `router.weight` | ✅ | `main.rs:2944-2946` adds both. Plan's narrative list groups them under "MoE router pattern" misleadingly. **Fix list.** |
| L3 | LOW | PyTorch ≥ 2.4 may not suffice for sm_120 | ✅ | Same as claude-M1. Bump requirement to ≥ 2.6 stable. |
| L4 | LOW | Timeline optimistic | ✅ | Same as claude-m4. 3-5 days realistic. |
| S1 | STRUCT | No tensor-name validation at Python→Rust boundary | ✅ | Real defect. Add explicit validation step in `--gptq-precomputed` flag: fail loud if expected tensor names don't all match the original BF16 model. |
| S2 | STRUCT | No GPU damping-retry strategy | ✅ | Use `torch.linalg.cholesky_ex` for non-throwing error returns. Cap retries. Document fallback to CPU if max_damp hit. |
| S3 | STRUCT | No per-tensor quality check | ✅ | Add per-tensor MSE(quant, ref) gate during Python loop. Reject the run if any tensor's MSE exceeds 10× the median. Catches localized failures KLD averages over. |
| S4 | STRUCT | Flag design `--gptq-precomputed` + `--awq-precomputed` fragile | ✅ | Replace with single `--precomputed-gptq-path <manifest>` taking a directory containing weights + AWQ scales + frozen grids. Mutually exclusive with `--gptq` / `--awq` / `--imatrix`. **Combine with claude-C3 fix.** |

---

## Overlap matrix (avoid double-counting in plan revision)

| Topic | claude | gemini | glm5 |
|---|---|---|---|
| Hessian-domain transform / frozen grids | C1, C3 | G-C3 | (covered indirectly via M1 + S4) |
| FP64 precision | — | G-C1 | M1 |
| Actorder permutation | — | G-C2 | — |
| AWQ sidecar runtime requirement | — | G-C4 | — |
| Hessian symmetrization | — | G-IG1 | — |
| Hessian file format details | — | — | H2, H3, M7 |
| AWQ exponent formula text | — | — | H4 |
| File/function reference errors | — | — | M4, M5 |
| Timeline / PyTorch version | M1, m4 | — | L3, L4 |
| Output path (`/tmp/`) | — | — | M6 |
| Flag design / handoff API | C3 | G-C4 (partial) | S4 |

Gemini contributed 4 critical math findings claude missed entirely (FP64, actorder, sidecar emission, Hessian symmetrization). GLM5 contributed 6 factual accuracy fixes (Hessian format, AWQ exponent, file/function references, /tmp violation) plus 4 structural design improvements (S1-S4). Claude's review focused on architecture-level scope/format gaps but missed the math details.

## New critical findings claude missed entirely

These must be in any plan revision:

1. **FP64 on CUDA for Cholesky + propagation** (G-C1, M1). `torch.linalg.cholesky_ex(H.double(), upper=False)` not `float32`.
2. **WEIGHT-mode actorder** (G-C2). Sort columns by `diag(H)` descending. Permutation `P` enters everything: `U^T·U = (P^T (H+λI) P)^-1`, column loop iterates in permuted order. `gptq.rs:723` is the reference.
3. **AWQ sidecar must be written** (G-C4). Python emits the `s` vector per AWQ-eligible tensor; Rust `--precomputed-gptq-path` writes them as `<name>.awq_scale.weight` F16 sidecars in the `.hfq`.
4. **Hessian symmetrization after FWHT similarity transform** (G-IG1). `H = 0.5 * (H + H.T)` to fix FP drift. `gptq.rs:508` is the reference.

## Factual corrections to plan text

- §1 line 17: VRAM per RTX 5070 — clarify with user (12 GB stock, 16 GB Ti)
- §1 line 19, §4 line 178: "33 GB Hessian sidecar" is correct (actual file size); the companion doc says ~6 GB but is stale
- §1 line 44: Hessian entry format — replace with pointer to `gptq-hessian-format.md` §3
- §1 line 121: `gptq.rs::compute_awq_scales line 2805` → `main.rs::compute_awq_scales line 2794`
- §1 line 48 (formula): `s[j] = (RMS_act[j])^alpha` (drop the `/2`)
- §1 line 170-171: AWQ whitelist — add `mlp.gate.weight`, `router.weight` explicitly
- §3 step 5 (`--gptq-precomputed`): collapse to single `--precomputed-gptq-path <dir>` with manifest containing weights + AWQ scales + frozen grids
- §3 step 6 + §5: `apply_gptq_column_sequential` → `gptq::gptq_pipeline_mq4g256`
- §4.4: paths — `/tmp/qwen3.5-9b-gptq-updated/` → `~/.hipfire/gptq-precomputed/` or `benchmarks/artifacts/gptq-cuda/`
- §7 step 3: "Update `gptq_9b_overnight.sh`" → "Create `scripts/gptq_9b_overnight.sh`"
- TL;DR / §3: "15h" → "~14h"
- §1 PyTorch version: ≥ 2.4 → ≥ 2.6 stable (or 2.5 nightly with TORCH_CUDA_ARCH_LIST="12.0")

## Recommendation

Revise `docs/plans/gptq_cuda.md` to incorporate the four new criticals (FP64, actorder, sidecar, symmetrization) + all factual corrections. Keep the architectural split (Python does math, Rust does packing) but redesign the handoff manifest per S4. Address claude-C1-C3 + gemini-C3 + glm5-S4 as a single consolidated "manifest format" section.

Open separately: doc-fix issue for `gptq-hessian-format.md` (size figure + hipfire-runtime vs hipfire-quantize path).
