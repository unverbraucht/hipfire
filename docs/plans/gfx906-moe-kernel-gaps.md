# gfx906 MoE kernel audit — consolidated (GLM-5)

**Date:** 2026-05-08 (HFP4-context annotation added 2026-05-11)
**Sources:** GLM-5 + Claude + Gemini (3-way cross-validated; raw audits not preserved)
**Scope:** Cross-validated assessment of all Qwen 3.5 MoE kernel optimizations
for gfx906 (Vega 20 / MI50 / MI60).

## Status under the HFP4 format roadmap (annotation 2026-05-11)

This audit was written against the **HFQ4G256 / MQ4G256** kernel family
before HFP4 / MFP4 landed. With the format roadmap in
`docs/plans/qwen35-mq4-quality-gap.md` (Phase B' gfx906 kernel port), most
findings carry forward unchanged, two need reframing:

- **Gaps 1, 2, 5, 8 — survive format-transposed.** dp4a port for MoE
  GEMVs (Gap 1), MMQ port for MoE prefill (Gap 2), prefetch variant
  (Gap 5), 4-way fused preamble dp4a (Gap 8) — all are about kernel
  *shape* (dp4a vs scalar, MMQ tiling, prefetch pipelining), not
  element format. In an HFP4 world the target kernel name changes
  (`gemv_hfp4g32_moe_*_wave64_dp4a.hip` instead of
  `gemv_hfq4g256_moe_*_wave64_dp4a.hip`) but the work is identical.
  Phase B' should fold these gaps into the HFP4 port rather than
  closing them twice.

- **Gap 3 — survives, format-orthogonal.** Shared expert down wave64
  variant is a launch-bounds + LDS pattern fix; format-independent.

- **Gap 4 — partially obsolete.** `mq_rotate_x` and
  `fused_silu_mul_mq_rotate` are the **online rotation** kernels for
  MQ4. In an MFP4G32 world the rotation is **offline** (baked into
  the codes via `format_flags` rotation_kind `01`); the online
  rotation kernel is not on the MFP4 hot path. Wave64 port is still
  needed for legacy MQ4 models we keep supporting, but Phase B' does
  not need to close it.

- **Gap 6 — reframed.** "No HFQ6/MQ6/MQ3 MoE fast path" becomes "no
  HFP-family MoE fast path beyond HFP4G32" under the format roadmap.
  Per L5d in `qwen35-mq4-quality-gap.md`, per-tensor bit allocation
  via imatrix needs HFP3G32 / HFP6G32 / HFP8E4M3G32 variants. Each
  would need a MoE kernel — but defer until imatrix-driven bit
  allocation actually consumes them.

- **Gap 7 — survives.** Mixed-dtype fallback warning is trivial,
  format-orthogonal.

**Net effect:** ~6 of 8 gaps map 1:1 onto Phase B' work. The gfx906
kernel port (per qwen35-mq4-quality-gap.md §5 Phase B') should
explicitly include the dp4a / MMQ / prefetch / shared-expert-down
extensions identified here, not just a straight HFQ4→HFP4 port that
inherits today's MoE perf gap.

---

## Verdict: NOT fully optimized. Functional but leaves significant perf on the table.

The MoE path on gfx906 is **correct** (valid output, wave64 routing works,
top-K is GPU-only). However, it has fallen behind the dense path in
optimization investment. **8 gaps** identified across 3 independent audits;
none are correctness bugs.

---

## 1. Cross-audit finding reconciliation

### Findings agreed on by all 3 audits (VALIDATED)

| # | Finding | GLM-5 | Claude | Gemini |
|---|---------|-------|--------|--------|
| A | MoE routed GEMV kernels use wave64 but lack dp4a | Gap 1 | Gap 1 | Sec 2 |
| B | No MMQ port for MoE prefill | Gap 2 | Gap 2 | — |
| C | MQ3 + MoE is hard-refused at load time | Gap 4 | Gap 4 | — |
| D | Routing kernels (softmax + top-K) correctly stay wave32 | Sec 1 | Sec 2 | Sec 5 |
| E | GPU top-K fast path eliminates D2H sync | Sec 1 | Sec 2 | — |
| F | MQ4-only MoE kernel family (no HFQ6/MQ6/MQ3 variants) | Gap 4 | Gap 4 | — |

### Findings unique to one audit (evaluated below)

| Source | Finding | Verdict |
|--------|---------|---------|
| Claude | Gap 3: No prefetch variant for MoE GEMV | **VALIDATED** |
| Claude | Gap 5: Mixed-dtype fallback has no warning | **VALIDATED** |
| Gemini | Shared expert down (`sigmoid_scaled`) has no wave64 variant | **VALIDATED** (new) |
| Gemini | Rotation/fusion kernels (`mq_rotate_x`, `fused_silu_mul_mq_rotate`) have no wave64 variant | **VALIDATED** (new) |
| Gemini | Preamble `fused_qkvza_hfq4g256` is "fully optimized" on gfx906 | **PARTIALLY CORRECT** (see below) |
| Claude | References `gemv_hfq4g256_residual_wave64_dp4a.hip` as a file | **REJECTED** — file does not exist |

---

## 2. Rejected findings

### Claude: `gemv_hfq4g256_residual_wave64_dp4a.hip` reference

**Claude's claim** (line 41-42, 49): "The non-MoE sibling
`gemv_hfq4g256_residual_wave64_dp4a.hip` uses `__builtin_amdgcn_sdot4`...
port the dp4a inner loop from `gemv_hfq4g256_residual_wave64_dp4a.hip:78-120`"

**Verdict: REJECTED.** This file does not exist. Glob for
`*gemv_hfq4g256_residual*wave64*` returns only two files:
- `gemv_hfq4g256_residual_wave64.hip` (scalar FP, no dp4a)
- `gemv_hfq4g256_residual_wave64_prefetch.hip` (scalar FP + prefetch, no dp4a)

The dp4a GEMV path for gfx906 uses MMQ-streaming kernels
(`gemm_hfq4g256_residual_mmq_gfx906_x{8..64}.hip`) or fused dp4a kernels
(`fused_gate_up_hfq4g256_wave64_dp4a.hip`, etc.), not a standalone
`gemv_hfq4g256_residual_wave64_dp4a.hip`. The reference kernel for porting
dp4a into MoE should be `fused_gate_up_hfq4g256_wave64_dp4a.hip:138-156`.

### Gemini: Preamble is "fully optimized"

**Gemini's claim** (Sec 1): "Uses `fused_qkvza_hfq4g256_wave64_dp4a`.
This is the gold standard for gfx906 performance."

**Verdict: PARTIALLY CORRECT.** The preamble does use `fused_qkvza_hfq4g256`
with wave64 on gfx906. However:
- The `fused_qkvza_hfq4g256` kernel is HFQ4-only (not dp4a). There is no
  `fused_qkvza_hfq4g256_wave64_dp4a.hip` — the dp4a fused kernels exist only
  for `gate_up`, `qkv`, `qkvza` as **separate** kernels (not `fused_qkvza`).
- The `fused_qkvza_hfq4g256` wave64 kernel uses the same scalar FP inner loop
  as the MoE GEMV kernels. The dp4a optimization that exists for standalone
  `fused_gate_up_hfq4g256_wave64_dp4a` has not been folded into the 4-way
  fused variant.
- So the preamble is **wave64-optimized** but **not dp4a-optimized**. It's
  better than the routed expert GEMVs (which are also wave64 but scalar) only
  in that it fuses 4 GEMVs into one launch.

**Correction:** Gemini's description of the 4-way fused projection contents
is accurate — it fuses router + shared_expert_gate + shared.gate + shared.up
(`qwen35.rs:1817-1818`). The kernel name `fused_qkvza_hfq4g256` is correct.
But calling it "fully optimized" overstates the case.

---

## 3. Consolidated gap analysis (8 gaps, ranked by impact)

### Gap 1: No dp4a port for MoE routed GEMV kernels — MEDIUM IMPACT

**Agreement:** All 3 audits. Claude (Gap 1), GLM-5 (Gap 1), Gemini (Sec 2).

**Affected files (4):**
- `kernels/src/gemv_hfq4g256_moe_gate_up_indexed_wave64.hip`
- `kernels/src/gemv_hfq4g256_moe_down_indexed_wave64.hip`
- `kernels/src/gemv_hfq4g256_moe_gate_up_indexed_batched_wave64.hip`
- `kernels/src/gemv_hfq4g256_moe_down_indexed_batched_wave64.hip`

All four use scalar FP inner loops (`DOG` macro — 64 FP32 ops per uint32).
Non-MoE dense has 11 dp4a kernel files using `__builtin_amdgcn_sdot4`.
Zero `*moe*dp4a*` files exist anywhere in the codebase.

**Estimated impact:** ~2-3% overall A3B decode improvement on gfx906.
MoE spends ~25-30% of decode wall-clock in these kernels (8 experts × 2
GEMVs × 40 layers = 640 launches/token). Non-MoE dense saw +7.1% from
dp4a alone; MoE impact is diluted because the FFN is a fraction of total
wall-clock.

**Effort:** ~1 day. Mechanical port from
`fused_gate_up_hfq4g256_wave64_dp4a.hip:138-156` (NOT from the nonexistent
`gemv_hfq4g256_residual_wave64_dp4a.hip` cited by Claude). Activation
pre-quantization to Q8_1 needed once per MoE layer.

### Gap 2: No MMQ port for MoE prefill — HIGH IMPACT

**Agreement:** Claude (Gap 2) + GLM-5 (Gap 2). Gemini did not address.

**Affected:** batched MoE GEMV kernels used by `prefill_moe_ffn_body_batched`
(`qwen35.rs:3615`). Non-MoE prefill on gfx906 saw 1.15×-3.52× from MMQ.
MoE prefill uses scalar wave64 GEMV.

**Implementation:** Sort tokens by expert assignment, then MMQ per-expert
(standard approach — vLLM / TensorRT-LLM). Per-token expert indirection
breaks the contiguous batch assumption.

**Estimated impact:** +50-100% prefill at B >= 16.

**Effort:** 1-2 weeks.

### Gap 3: Shared expert down projection has no wave64 variant — MEDIUM IMPACT

**Source:** Gemini (Sec 3). **VALIDATED by code inspection.**

**Affected kernel:** `gemv_hfq4g256_residual_sigmoid_scaled_gpu`
(`kernels/src/gemv_hfq4g256_residual_scaled.hip:240`). Plus its batched
variant at line 356.

Dispatch at `dispatch.rs:6031, 6065, 6084, 6120` hardcodes `[32, 1, 1]` with
no `has_wave64_native` check. On gfx906 (native wave64), only the lower 32
lanes of each wave participate — 50% lane utilization.

This kernel runs once per MoE layer (40 launches/token) for the shared
expert down + sigmoid-gate + residual add. The **routed** expert down path
(`gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`) correctly uses wave64
via `has_wave64_native` dispatch — only the shared expert is missing.

**Why this matters:** The shared expert contributes to every token (it's
always active, unlike routed experts). At 40 layers × 1 launch/layer = 40
launches/token with 50% lane waste, this is a non-trivial gap.

**Effort:** ~half day. Straightforward wave64 port (2 rows/block, same
pattern as the routed-expert down wave64 kernel).

### Gap 4: Rotation/fusion kernels have no wave64 variant — LOW-MEDIUM IMPACT

**Source:** Gemini (Sec 4). **VALIDATED by code inspection.**

**Affected kernels (4):**
| Kernel | File | Dispatch |
|--------|------|----------|
| `mq_rotate_x` | `gemv_mq4g256.hip:63` | `dispatch.rs:2574` — `[32,1,1]`, no wave64 check |
| `mq_rotate_x_batched` | same source | `dispatch.rs:2619` — `[32,1,1]`, no wave64 check |
| `fused_silu_mul_mq_rotate` | `fused_silu_mul_mq_rotate.hip:19` | `dispatch.rs:2482` — `[32,1,1]`, no wave64 check |
| `fused_silu_mul_rotate_mq_batched` | same source | `dispatch.rs:2532` — `[32,1,1]`, no wave64 check |

Both kernels use `__launch_bounds__(32, 16)` and wave32 butterfly
patterns (`ds_swizzle` with stride masks on lower 5 bits of `tid`). On
gfx906, upper 32 wave slots execute no useful work.

These run in the MoE FFN decode path:
- `fused_silu_mul_rotate_mq_batched` at `qwen35.rs:1917` (routed experts)
- `fused_silu_mul_rotate_mq` at `qwen35.rs:1884` (shared expert)

**Impact:** Memory-bound kernels, so lane waste translates to scheduler
cycles and power rather than directly to bandwidth. Moderate cumulative
impact at 40 layers × 2-3 rotation launches/layer.

**Effort:** ~1-2 days. Requires redesigning the butterfly reduction to span
64 lanes or packing two groups per wave. Not as mechanical as the GEMV
wave64 ports.

### Gap 5: No prefetch variant for MoE GEMV — LOW-MEDIUM IMPACT

**Source:** Claude (Gap 3). **VALIDATED.**

Non-MoE has `gemv_hfq4g256_residual_wave64_prefetch.hip` and its HFQ6
counterpart. Non-MoE measurement: +4.8% decode. Same software-pipeline
trick applies to MoE because each expert's weight matrix is contiguous in
VRAM.

**Effort:** ~half day. Independent of Gap 1.

### Gap 6: HFQ6/MQ6/MQ3 MoE has zero fast path — LATENT

**Agreement:** Claude (Gap 4) + GLM-5 (Gap 4).

All 14 MoE kernel files are `hfq4g256` only. No `hfq6g256_moe_*`,
`mq6g256_moe_*`, or `mq3g256_moe_*` kernels exist. MQ3 + MoE is
hard-refused at load time. Currently not triggered (only
`qwen3.6-35b-a3b.mq4` ships).

**Effort:** Defer until HFQ6/MQ6/MQ3 MoE quants exist.

### Gap 7: Mixed-dtype fallback has no warning — LOW IMPACT

**Source:** Claude (Gap 5). **VALIDATED.**

When `gate_side_mq4 == false` (`qwen35.rs:1825-1831`), 4 separate
`weight_gemv()` calls replace fused `fused_qkvza_hfq4g256`. No `eprintln!`
warns the user. Dead code for all current models.

**Effort:** Trivial. Add `eprintln!` at slow-path entry.

### Gap 8: 4-way fused preamble (`fused_qkvza_hfq4g256`) has no dp4a variant — LOW-MEDIUM IMPACT

**Source:** New (derived from validating Gemini's "fully optimized" claim).

The 4-way fused kernel `fused_qkvza_hfq4g256` used for the MoE preamble
(router + shared_expert_gate + shared.gate + shared.up) is wave64 on gfx906
but uses scalar FP. Standalone 2-way `fused_gate_up_hfq4g256_wave64_dp4a`
and 3-way `fused_qkv_hfq4g256_wave64_dp4a` have dp4a variants, but no
4-way dp4a fused kernel exists for any architecture.

**Impact:** The preamble runs once per MoE layer (40 launches/token). Its
4 fused GEMVs are smaller than the routed expert GEMVs (router M=256,
shared_expert_gate M=1, shared gate M=512, shared up M=512 vs routed
gate_up M=1024). The scalar FP inner loop on these smaller matrices is
less bandwidth-bound than the routed path, so dp4a's activation-traffic
reduction would have proportionally smaller impact.

**Effort:** ~1-2 days. New kernel (4-way dp4a fused) or split the
preamble into separate 2-way + 2-way dp4a calls.

---

## 4. What IS optimized on gfx906 for MoE

| Feature | Status | Reference |
|---------|--------|-----------|
| LA4 fused QKVZA (HFQ4) | wave64 | `qwen35.rs:6371-6378` |
| LA4 fused QKVZA (HFQ6) | wave64 + dp4a | `qwen35.rs:6367-6370` |
| FA3 fused QKV (HFQ4) | wave64 | `qwen35.rs:6467-6474` |
| FA3 fused QKV (HFQ6) | wave64 + dp4a | `qwen35.rs:6463-6466` |
| Wave64 routing for MoE GEMVs | Correct (2 rows/block) | `dispatch.rs:6378-6385, 6443-6450` |
| GPU top-K (zero D2H) | Enabled for k==8 all-MQ4 | `qwen35.rs:1800-1871` |
| MMQ prefill (dense layers) | 8 tile-size gfx906 kernels | `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}.hip` |
| dp4a MMQ prefill (HFQ6 dense) | 9 tile-size gfx906 kernels | `gemm_hfq6g256_residual_mmq_gfx906_x{8..64}.hip` |
| Multi-GPU MoE attention | HFQ4 fused + HFQ6 dp4a fused | `qwen35.rs:6086-6125, 6213-6225` |
| MQ3 + MoE | Hard-refused (correct) | `daemon.rs:1165-1197` |

The optimization gap is concentrated in the **FFN body** (expert weight
GEMVs, shared expert down, rotation/fusion) — not in attention or routing.

---

## 5. Architecture dispatch summary

```
Model load
  └─ daemon.rs:1219-1224  arch_id=6 (MoE/A3B) recognized
     └─ daemon.rs:1162-1198  MQ3+MoE refused; MQ2+MoE refused
     └─ daemon.rs:1476-1481  DeltaNet state forced FP32 (Q8 drift mitigation)

Prefill eligibility (qwen35.rs:3320-3431)
  └─ moe_topk_ok = (k_top==8 && num_experts<=1024)
  └─ moe_ffn_all_mq4(&l.ffn) required for batched path
  └─ is_batchable_la() for attention (MQ3 only on gfx11+)

MoE FFN decode (qwen35.rs:1748-1988)
  ├─ gate_side_mq4 && routed_mq4?
  │  ├─ YES → fused_qkvza_hfq4g256 (4-way GEMV, wave64, SCALAR FP)  ← Gap 8
  │  │        + gpu_softmax + moe_topk_renorm_k8 (zero D2H)
  │  │        + fused_silu_mul_rotate_mq_batched [wave32]            ← Gap 4
  │  │        + gemv_hfq4g256_moe_gate_up_k8_indexed [wave64, SCALAR FP]  ← Gap 1
  │  │        + fused_silu_mul_rotate_mq [wave32]                    ← Gap 4
  │  │        + gemv_hfq4g256_moe_down_residual_scaled_k8_indexed [wave64] ← Gap 1
  │  │        + fused_silu_mul_rotate_mq [shared, wave32]            ← Gap 4
  │  │        + gemv_hfq4g256_residual_sigmoid_scaled_gpu [wave32]   ← Gap 3
  │  └─ NO  → 4× weight_gemv (per-expert loop, ~8× slower)          ← Gap 7

MoE FFN prefill (qwen35.rs:3615-3737)
  ├─ 4× gemm_hfq4g256 (router + shared expert) [MMQ on gfx906 if B>=8]
  ├─ moe_topk_renorm_k8_batched
  ├─ gemv_hfq4g256_moe_gate_up_k8_indexed_batched [wave64, SCALAR FP] ← Gap 1+2
  └─ gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched [wave64] ← Gap 1+2

Multi-GPU MoE (qwen35.rs:6030)
  ├─ Attention: fused_qkvza/qkv HFQ4 or HFQ6 dp4a (same as single-GPU)
  └─ FFN: moe_ffn_decode_with_scratch (same scalar path)
```

---

## 6. Compiled kernel cache

`kernels/compiled/gfx906/` is **empty** (0 files). All gfx906 MoE kernels
are JIT-compiled at runtime via `ensure_kernel()`. Other architectures
(gfx1010, gfx1030, gfx1100, gfx1200, gfx1201) have precompiled `.hsaco`
files. Adds cold-start latency on gfx906.

---

## 7. Recommendations

**Priority order for closing gaps:**

| Priority | Gap | Effort | Est. decode impact |
|----------|-----|--------|-------------------|
| 1 | Gap 3: Shared expert down wave64 | ~half day | +1-2% |
| 2 | Gap 1: MoE GEMV dp4a port | ~1 day | +2-3% |
| 3 | Gap 5: MoE GEMV prefetch | ~half day | +1-2% |
| 4 | Gap 4: Rotation/fusion wave64 | ~1-2 days | +0.5-1% |
| 5 | Gap 8: Preamble dp4a fused | ~1-2 days | +0.5-1% |
| 6 | Gap 2: MMQ prefill | 1-2 weeks | +50-100% prefill only |
| 7 | Gap 7: Mixed-dtype warning | trivial | n/a (correctness) |
| 8 | Gap 6: HFQ6/MQ6/MQ3 MoE | defer | n/a (latent) |

Gaps 1-5 together could yield ~5-9% overall A3B decode improvement on
gfx906. Gap 2 is the largest single opportunity but only affects prefill.

**Alternative:** If dense models are the priority (every recorded bench
exercises dense), these gaps are acceptable. The MoE path is functionally
correct on gfx906 — just not as optimized as the dense path.

---

## 8. Inter-audit disagreement log

| Topic | Claude | Gemini | GLM-5 | Resolution |
|-------|--------|--------|-------|------------|
| dp4a impact on MoE decode | +7-10% (header), +2-3% (body) | "highest potential perf win" | +2-3% | Claude header is misleading; body text (+2-3%) is correct. The +7-10% figure is the non-MoE dp4a win, not the MoE contribution. |
| Preamble optimization | Not separately analyzed | "Fully optimized" (gold standard) | Wave64 but not dp4a | Gemini overstated. Wave64-optimized yes, dp4a-optimized no. |
| Shared expert down | Not mentioned | Identified (wave32 gap) | Not mentioned | Gemini correct — confirmed by code inspection. |
| Rotation kernels | Not mentioned | Identified (wave32 gap) | Not mentioned | Gemini correct — confirmed by code inspection. |
| `gemv_hfq4g256_residual_wave64_dp4a.hip` | Referenced as existing file | Not mentioned | Not referenced | File does not exist. Correct reference is `fused_gate_up_hfq4g256_wave64_dp4a.hip`. |
