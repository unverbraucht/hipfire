# Quant × workload × arch dispatch matrix (reference)

Date: 2026-05-06
Branch: `feat/gfx906-hfq6-hfq8-analysis` post-`d3a0575`.

A consolidated lookup table of which kernel each loader-producible
quant format dispatches to, on each target arch, for each forward-pass
workload. Reference document — not narrative. For the *why*, see
`2026-05-06-mq8-runtime-dispatch-audit.md` and plan §5.5–§5.8.

## How to read this

- **Quant** = `DType::*G256` produced by `hipfire-quantize` and consumed
  by `crates/hipfire-runtime/src/hfq.rs:417` (`load_weight_tensor_raw`).
- **Workload** = which forward-pass surface dispatches the GEMV/GEMM:
  - **B=1 GEMV** — AR decode single-token, `weight_gemv` in
    `crates/hipfire-runtime/src/llama.rs:530+`.
  - **Per-layer batched (B>1)** — prefill + DFlash verify, dense LA/FA
    bodies in `crates/hipfire-arch-qwen35/src/qwen35.rs:3946+`. Routes
    through `gemm_*_hfq{4,6}g256` family.
  - **MoE batched (B>1)** — MoE-LA/MoE-FA bodies at `qwen35.rs:4651, 4802`.
  - **MoE expert dispatch (B=1 + B>1)** — top-K indexed kernels at
    `qwen35.rs:1900+`. MQ4-only fast path; non-MQ4 falls through to
    per-expert `weight_gemv`.
  - **LM-head** — single GEMV/batched at end of forward; tied embedding
    or separate. Both dense and DFlash speculative paths.
  - **KV cache** — write + attention read. `weight_gemv` is irrelevant
    here; KV uses dedicated kernels.
- **Arch routing** = which `has_*` gating function decides:
  - `has_wave64_native(arch)` — gfx906, gfx908, gfx940-942 (Vega20 + CDNA1 + CDNA3)
  - `has_dot2_f32_f16(arch)` — gfx1011/1012/1030-1032/1100-1103/1150-1152/1200-1201
    (NB: **excludes gfx906** despite hardware support — see §5.6)
  - `has_wmma_f16(arch)` — gfx11* only
  - `has_wmma_f16_gfx12(arch)` — gfx12* only
  - `has_mmq_dp4a_or_wmma(arch)` — gfx906 + gfx1100-1103 + gfx1150-1152
- **Filename suffix conventions** (mostly self-documenting, with edge cases):

  | Suffix | Meaning | Routing |
  |---|---|---|
  | `.gfx1100.hip` | RDNA3-tuned variant | `has_wmma_f16` (or arch-specific dispatch path) |
  | `.gfx1030.{v1..v5}.hip` | RDNA2 sweep variants | `has_dot2_f32_f16` |
  | `.gfx12.hip` / `.gfx1201.hip` | RDNA4 | `has_wmma_f16_gfx12` |
  | `_wave64.hip` (no arch suffix) | wave64-native arches | `has_wave64_native` (gfx906/908/94x — **NOT gfx906-only** despite common assumption) |
  | `_wave64_dp4a.hip` | wave64 + sdot4 dp4a | `has_wave64_native`; primary target gfx906 |
  | `_mmq_gfx906_x{N}.hip` | gfx906-only MMQ redesign | `should_use_mmq && arch == "gfx906"` |
  | `_wmma.hip` (no `.gfx12` suffix) | RDNA3 WMMA | `has_wmma_f16` |
  | `_dot2.hip` | `v_dot2_f32_f16` | `has_dot2_f32_f16` |
  | `_fp16.hip` / `_fp16_wave64.hip` | FP16 packed math fallback | gfx1010/1013 (no dot2) or wave64-native |
  | (no suffix) | scalar wave32 / generic | universal fallback |

## 1. B=1 GEMV (AR decode single-token)

Source: `crates/hipfire-runtime/src/llama.rs` `weight_gemv()` (line 530+).
Dispatcher is dtype-aware; routes to a single shipped kernel per quant.

| Quant | Kernel | gfx906 | gfx1100 | gfx1201 | Status |
|---|---|---|---|---|---|
| MQ4G256 | `gemv_mq4g256` | ✓ | ✓ | ✓ | shipped |
| MQ6G256 | `gemv_mq6g256` | ✓ | ✓ | ✓ | shipped |
| MQ8G256 | `gemv_mq8g256_with_rotate` (post-`ee0fac6`) | ✓ | ✓ | ✓ | shipped (lm_head only in production) |
| MQ3G256 | `gemv_mq3g256_with_rotate` | ✓ | ✓ | ✓ | shipped |
| MQ2G256 | `gemv_mq2g256_with_rotate` | ✓ | ✓ | ✓ | shipped |
| HFQ4G256 | `gemv_hfq4g256` (+ `.gfx1030.v{N}.hip` and `.gfx1100.hip` variants per arch) | ✓ wave32 | ✓ tuned variant | ✓ tuned variant | shipped |
| HFQ4G128 | `gemv_hfq4g128` | ✓ | ✓ | ✓ | shipped |
| HFQ6G256 | `gemv_hfq6g256` (+ `.gfx1201.hip` variant for RDNA4) | ✓ wave32 | ✓ wave32 | ✓ tuned | shipped (no gfx906 wave64 — see §3.1.1 Phase A item 1) |
| HFQ3G256/G128 | `gemv_hfq3g256` (+ `.gfx1100.hip`) | ✓ wave32 | ✓ tuned | ✓ wave32 | shipped |
| HFQ2G256/G128 | `gemv_hfq2g256` | ✓ wave32 | ✓ wave32 | ✓ wave32 | shipped |
| Q8_0 | `gemv_q8_0_wide` | ✓ | ✓ | ✓ | shipped |
| F32 | `gemv_f32` | ✓ | ✓ | ✓ | shipped |

**No wiring gaps in B=1 path.** All 12 dtypes route correctly and all
target archs have at least a wave32-scalar-correct kernel. Per-arch
optimization gaps:

- **gfx906 has no wave64 GEMV for HFQ6/MQ6** — wave32 only; Phase A item 1.
- **gfx906 has no `.gfx906` suffix variants** — most kernels rely on the
  generic wave32 path or `_wave64.hip` (which is wave64-native, not
  gfx906-specific).

## 2. Per-layer batched fused GEMM (prefill + DFlash verify)

Source: `crates/hipfire-arch-qwen35/src/qwen35.rs:3946+` for dense
LA/FA bodies. **Dispatcher is NOT dtype-aware for non-rotated quants
beyond MQ4/MQ6/MQ3** — see audit findings.

### 2a. Dense LA/FA prefill (qkvza / qkv / gate_up / down / wo)

| Quant | gfx906 path | gfx1100 path | gfx1201 path | Wiring? |
|---|---|---|---|---|
| MQ4G256 | `gemm_qkvza_hfq4g256_wave64` (`fp16_wave64` for FP16 fast path; `wave64_dp4a` if dp4a-eligible) | `*_wmma` (RDNA3) | `*_wmma.gfx12` | ✓ all 8 sites |
| MQ6G256 | `gemm_qkvza_hfq6g256` (wave32 scalar — no wave64 sibling) | `*_wmma` | `*_wmma.gfx12` | ✓ all 8 sites |
| MQ3G256 | `gemm_qkvza_hfq3g256_wmma` (NB: WMMA, gfx11+ only — gfx906 falls through to scalar) | `*_wmma` | `*_wmma.gfx12` | ✓ all 8 sites |
| MQ8G256 | (none — falls through to `gemm_qkvza_hfq4g256_wave64` at HFQ4 stride) | (same) | (same) | **✗ silent corruption** — see audit dev-log |
| MQ2G256 | (none — falls through to HFQ4 stride at 72 vs 136 B/group) | (same) | (same) | **✗ silent corruption (latent — no .mq2 model deployed)** |
| HFQ4G256 | `gemm_qkvza_hfq4g256_wave64` | `_wmma` / `_dot2` | `_wmma.gfx12` | ✓ |
| HFQ6G256 | `gemm_qkvza_hfq6g256` (wave32) | `_wmma` / `_dot2` | `_wmma.gfx12` | ✓ |
| Q8_0 | rocblas FP16 shadow path (`rocblas_arch_eligible`) | rocblas | rocblas | ✓ |

**Corresponding `gemm_qkv_*` and `gemm_gate_up_*` families have
identical patterns** — every entry in the qkvza row applies to qkv
and gate_up too.

**`gemm_hfq6g256_residual` (per-layer wo + w_down batched, 6 call sites
across LA/FA)** — wave32 scalar on every arch; no wave64 sibling.
This is Phase A's missing 6th kernel (§5.8).

### 2b. MoE LA/FA prefill (qwen35.rs:4651, 4802)

| Quant | Wiring? | Trigger today? |
|---|---|---|
| MQ4G256 | ✓ | yes — `qwen3.6-35b-a3b.mq4` ships |
| MQ6G256 | ✓ | not deployed (no MoE+MQ6 model in registry yet) |
| **MQ3G256** | **✗ dropped from matcher** at qwen35.rs:4651 + 4802 | not deployed; would silently corrupt — **upstream issue #179** |
| MQ8G256 | ✗ | not deployed |
| MQ2G256 | ✗ | not deployed |

## 3. MoE expert dispatch (top-K indexed)

Source: `crates/hipfire-arch-qwen35/src/qwen35.rs:1900+`. Three sub-paths
gated on quant uniformity.

| Path | Predicate | Dispatched kernel | Quant constraint |
|---|---|---|---|
| GPU-top-K fast path | `k == 8 && gate_side_mq4 && routed_mq4 && routed_gate_up_mq4` | `gemv_hfq4g256_moe_gate_up_k8_indexed` + `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed` (+ `_wave64` siblings on wave64-native archs) | **all-MQ4** (router + shared + every routed expert) |
| CPU-top-K kernarg-fused | `k == 8 && routed_gate_up_mq4 && x_rot_local.is_some()` | `gemv_hfq4g256_moe_gate_up_k8` + `gemv_hfq4g256_moe_down_residual_scaled_k8` | MQ4 gate_up + (predicate doesn't check) down ← **gap** flagged in PR #147 comment |
| Per-expert fallback | else | `weight_gemv` per-expert (dtype-aware) | any quant; safe |

**Per-arch:** the `_wave64` sibling family runs on `has_wave64_native`
(gfx906/908/94x); the unsuffixed family is wave32 fallback. There is
**no MoE-indexed kernel for non-HFQ4/non-MQ4 weights** — the fast-path
predicates require MQ4 uniformity. Plan §3.1 Phase A item 5 specifies
5 missing MoE-indexed kernels for HFQ6/MQ6.

## 4. LM-head (final logits projection)

Two paths: dense forward (`weight_gemv` over the lm_head tensor, B=1)
and DFlash speculative-decode batched verify (`speculative.rs:2076+`).

### 4a. Dense lm_head (B=1 GEMV, single-token AR decode)

Falls through to the same B=1 GEMV table in §1. MQ8 is genuinely used
here as a *tied embedding* in production mq4-format models (not as a
per-layer weight). The path was the only MQ8 production use until
the `qwen3.5-9b.mq8` quantize-test on 2026-05-06.

### 4b. DFlash speculative-decode batched lm_head (B>1)

Source: `crates/hipfire-arch-qwen35/src/speculative.rs:2076` (`try_batched`).

| Quant | Batched path? | Fallback if not |
|---|---|---|
| Q8_0 | ✓ `gemm_q8_0_batched` | n/a |
| HFQ4G256 | ✓ `gemm_hfq4g256_batched_lmhead` | n/a |
| MQ4G256 | ✓ `rotate_x_mq_batched` + `gemm_hfq4g256_batched_lmhead` | n/a |
| MQ3G256 | ✓ `rotate_x_mq_batched` + `gemm_hfq3g256_batched_lmhead` | n/a |
| MQ6G256 | ✗ | falls through to unbatched per-row (perf miss) |
| MQ8G256 | ✗ | falls through to unbatched per-row (perf miss) |
| HFQ6G256 | ✗ | falls through (perf miss) |

Per-arch: this dispatch table is uniform across archs — the kernel
itself routes via the §2 matrix internally for the actual GEMM body.

## 5. KV cache write + attention

Workload-specific dedicated kernels. Not dtype-aware in the per-quant
sense — the KV mode (`q8` / `asym3` / etc.) determines the kernel,
not the weight quant.

| KV mode | Write kernel | Attention kernel |
|---|---|---|
| q8 | `kv_cache_write_q8_0` | `attention_q8_0_kv` (+ `_batched`) |
| asym3 | `kv_cache_write_asym3` | `attention_flash_asym3_tile` (+ `_batched`) |
| asym4 | `kv_cache_write_asym4` | `attention_flash_asym_reduce_batched` |
| asym2 | `kv_cache_write_asym2` | `attention_flash_asym2_tile` |

There's also `kv_cache_write_hfq8` + `attention_hfq8_kv` for the
**HFQ8 KV cache** path (dispatch.rs:10494, 10516) — separate from the
weight quants matrix. Quoting plan §1.1: HFQ8 *KV cache* and *attention KV*
are ✓ in the coverage table; HFQ8 weights are not deployed.

## 6. gfx906-specific kernels

Kernels with `_gfx906_` in the filename (gfx906-only redesigns vs the
generic wave64 path):

| Kernel | Workload | Notes |
|---|---|---|
| `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}` (8 entry points sharing `_body.cuh`) | MMQ batched prefill + DFlash verify | gfx906 redesign vs RDNA3 i8-WMMA MMQ; per-mmq_x X_STRIDE tuned (`docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`) |
| `gemm_hfq4g256_residual_mmq_gfx906_body.cuh` | (shared body for above) | — |

Plan §3.1 Phase C calls for HFQ6/MQ6 sibling MMQ family
(`_mmq_gfx906_x{N}` × {8..64}) — currently absent, ~5 sessions of work.

## 7. Cross-references

- `docs/plans/gfx906-mq6-mq8-port.md` v3.2.1 — full priority + planning
  context. §5.5 (kernel-source build status), §5.7 (matcher coverage),
  §5.8 (Phase A scope).
- `docs/perf-checkpoints/2026-05-06-mq8-runtime-dispatch-audit.md` —
  the deeper dive on the MQ8 silent-correctness gap that this matrix
  generalizes from.
- `crates/rdna-compute/src/dispatch.rs:115-225` — arch-gating functions
  (`has_dot2_f32_f16`, `has_wmma_f16`, `has_wave64_native`,
  `has_mmq_dp4a_or_wmma`, `should_use_mmq`).
- Upstream issue **#179** (Kaden-Schutt/hipfire) — MQ3 missing from
  MoE-batched matchers.
- Upstream PR **#147** comment — `use_kernarg_fused` predicate gap for
  mixed-precision MoE.

## Maintenance note

This matrix reflects state on `feat/gfx906-hfq6-hfq8-analysis` at
commit `d3a0575` (2026-05-06). It will drift as kernels are added /
removed / renamed. Re-run the audit (plan §5.4 part 2) to refresh:

```bash
# Part 1 — kernel source build verification
grep -rn '__builtin_amdgcn_\(sudot\|wmma\|s_wait_event\)' kernels/src/
# Part 2 — runtime-dispatch verification
grep -rn 'is_mq = matches!\|gpu_dtype.*matches!' crates/hipfire-arch-*/src/
```
