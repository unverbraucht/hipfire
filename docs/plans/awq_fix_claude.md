# AWQ runtime bug — diagnosis confirmed, fix scope expanded

**Date:** 2026-05-12
**Status:** Diagnosis agreed; fix at `awq_bug_hunt_glm5.md` is *partially correct* (catches 30 of 48 corrupting sidecars). Expanded fix below catches all 48.

---

## Where I agree with `awq_bug_hunt_glm5.md`

The root-cause analysis is correct:

1. **Quantizer applies AWQ universally** to every MQ4G256 tensor that has imatrix data (`hipfire-quantize/src/main.rs:4027-4044`). No tensor-type filter.
2. **Runtime AWQ inverse is only in `fused_rmsnorm_*_awq`** kernels. Tensors fed via `rotate_x_mq` (post-attention `wo`/`o_proj`) and `fused_silu_mul_rotate_mq` (post-silu `down_proj`/`w_down`) get the pre-scaled weights with no compensating divide. Verified at `llama.rs:877` (MQ4G256 wo arm) and `llama.rs:959` (MQ4G256 w_down arm).
3. **The α=0 discriminator (KLD 0.6721, matches mq4-base to 4 decimals)** is the smoking gun — with `s[j]=1.0 ∀j` the missing divide is a no-op, masking the bug.

I independently arrived at the same diagnosis via the discriminator. No disagreement on the root cause.

---

## Where the proposed fix at §"Option A" is incomplete

The proposed guard:

```rust
let awq_eligible = !name.contains("o_proj") && !name.contains("down_proj");
```

**Substring miss:** `"o_proj"` is NOT a substring of `"out_proj"`. Spell it out character-by-character:

```
"o_proj"   = o, _, p, r, o, j           (6 chars)
"out_proj" = o, u, t, _, p, r, o, j     (8 chars)
```

`"o_proj"` would need a consecutive `o, _` somewhere in `"out_proj"` — but the `o` and `_` in `out_proj` are separated by `u, t`. No match.

**Why this matters for Qwen3.5 specifically:** the hybrid linear_attn/full_attn architecture uses `out_proj` (linear_attn output projection) where most arches would use `o_proj`. Qwen3.5-9B has 24 linear_attn layers and 8 full_attn layers — so the linear_attn `out_proj` is the *majority* of attention output projections.

**Concrete evidence from the broken .hfq** (`qwen3.5-0.8b.mq4-awq-2026-05-12`):

| Sidecar pattern | Count | Caught by proposed `contains("o_proj") \|\| contains("down_proj")`? |
|---|---:|:---:|
| `.o_proj.weight` (full_attn) | 6 | ✓ |
| `.out_proj.weight` (linear_attn) | **18** | ✗ |
| `.down_proj.weight` (MLP) | 24 | ✓ |
| **Total corrupting sidecars** | **48** | **30 of 48 caught** |

Applying the proposed fix would leave the 18 `out_proj` sidecars corrupting the runtime output. KLD would drop substantially (probably 13.49 → ~1–3 nats) but not fully recover to a clean comparison against mq4-base (~0.67).

---

## My proposed fix — Option A2 (whitelist, fails closed)

Replace the loose substring check with an explicit whitelist of weights that *do* have an AWQ-aware runtime path. Anything not on the list defaults to NO AWQ (correct behavior — only RMSNorm-followed weights have the runtime divide).

```rust
// AWQ pre-scaling is mathematically valid ONLY for weights whose runtime
// path applies the inverse divide. Those are exactly the weights fed by
// `fused_rmsnorm_rotate_mq` / `_awq` (which dispatches to the AWQ kernel
// when awq_scale is present): the post-RMSNorm linear projections.
//
// Tensors NOT on this list (o_proj/wo/out_proj/down_proj/w_down) are fed
// by `rotate_x_mq` or `fused_silu_mul_rotate_mq`, which have no AWQ
// awareness. Pre-scaling those weights without a compensating runtime
// divide produces (W·s)·x ≠ W·x — broken output. We skip them entirely.
//
// Whitelist (vs blacklist) chosen for safety: any new tensor name in a
// future arch defaults to NO AWQ, which is the correct fail-closed
// behavior. Adding AWQ to a new projection requires confirming its
// runtime path uses an AWQ-aware kernel.
fn awq_eligible(name: &str) -> bool {
    // Full-attention pre-RMSNorm projections (HF naming + fused variants)
    name.ends_with("q_proj.weight")
        || name.ends_with("k_proj.weight")
        || name.ends_with("v_proj.weight")
        || name.ends_with("qkv_proj.weight")
        || name.ends_with("wqkv.weight")
        // MLP pre-RMSNorm projections (HF + hipfire naming)
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
        || name.ends_with("w_gate.weight")
        || name.ends_with("w_up.weight")
        // Linear-attention input projections (Qwen3.5 Gated-DeltaNet)
        // — note this is a `.contains()` rather than `.ends_with()` because
        // the suffix varies (in_proj_qkv/z/a/b). The substring is anchored
        // enough that no other tensor name should match.
        || name.contains(".in_proj_")
        // MoE router (post-RMSNorm gating logits)
        || name.ends_with("router.weight")
}
```

Then in the existing MQ4G256 branch:

```rust
let q = if let (Some(alpha), Some(im_weights))
    = (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
{
    if awq_eligible(name) {
        debug_assert_eq!(im_weights.len(), k_dim, ...);
        let scales = compute_awq_scales(im_weights, alpha);
        awq_sidecar_scales = Some(scales.clone());
        let m_dim = meta.shape[0];
        let mut scaled = f32_data.clone();
        awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
        quantize_mq4g256(&scaled, &signs1, &signs2)
    } else {
        // AWQ disabled for this weight — its runtime path has no inverse.
        // Plain MQ4G256; no sidecar emitted.
        quantize_mq4g256(&f32_data, &signs1, &signs2)
    }
} else {
    quantize_mq4g256(&f32_data, &signs1, &signs2)
};
```

### Why whitelist over blacklist

Compare:

| Approach | Behavior on new arch with new tensor name | Failure mode |
|---|---|---|
| Blacklist (e.g. `!contains("o_proj") && !contains("down_proj")`) | Defaults to APPLY AWQ | Silent corruption if new weight lacks AWQ-aware runtime path |
| Whitelist (proposed) | Defaults to SKIP AWQ | Suboptimal but correct quantization |

For a numerical-correctness lever, fail-closed is the only defensible default.

### Coverage check — Qwen3.5-0.8B AWQ .hfq

Running the whitelist against the 186 awq_scale sidecars in `~/.hipfire/models/qwen3.5-0.8b.mq4-awq-2026-05-12`:

| Bucket | Tensors | `awq_eligible()` returns |
|---|---|---|
| `q_proj`, `k_proj`, `v_proj` | (full_attn layers × 3) | true ✓ |
| `gate_proj`, `up_proj` | MLP per layer | true ✓ |
| `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` | linear_attn per layer | true ✓ |
| `out_proj` | linear_attn output | **false** ✓ (skip) |
| `o_proj` | full_attn output | **false** ✓ (skip) |
| `down_proj` | MLP output | **false** ✓ (skip) |

All 48 corrupting sidecars get dropped; all genuinely AWQ-amenable weights remain. Expected post-fix KLD: in the ballpark of mq4-base (0.67) with the literature-predicted ~15–20% reduction (so ~0.55–0.60 — though see below on the noise floor).

---

## Expected post-fix headline

The 0.8B AWQ delta (after this fix) is interpretable as:

```
delta_AWQ = KLD(mq4-awq, fixed)  −  KLD(mq4-base)
         ≈  0.55–0.60          −  0.6721         ≈ -0.07 to -0.12 nats
```

A ~10–18% reduction in KLD above the Q8 floor is consistent with literature lifts on Q4 AWQ. If the post-fix number lands materially better than that, suspect we accidentally caught a bug; if materially worse, AWQ is providing less benefit on hipfire's specific quantization layout than on AutoAWQ's INT4-symmetric format, which is plausible since hipfire uses asymmetric MQ4 + FWHT rotation — the calibration may be over-fitting to outliers that FWHT already mitigates.

Also worth noting: the Q8 floor of 0.4598 is engine-drift-dominated (see `project_engine_drift_floor_decomposition.md`), so the delta is read as a **fraction of the quantization-attributable gap** (0.6721 − 0.4598 = 0.2123). A -0.1 nat AWQ lift would close roughly half of that, which would be a strong Stage A result.

---

## Recommended Phase 3 sequence after this fix lands

1. **Land the whitelist guard** (Option A2) in `hipfire-quantize/src/main.rs`. ~10 lines + 1 small helper.
2. **Re-quantize 0.8B with `--awq --awq-alpha 0.5`.** Verify sidecar count drops from 186 → ~138 (186 − 18 out_proj − 24 down_proj − 6 o_proj = 138). Verify .hfq grep for `out_proj.awq_scale`, `o_proj.awq_scale`, `down_proj.awq_scale` returns 0.
3. **Run a single eval** (no full cohort needed) — just `eval_hipfire` on the fixed .hfq. ~15 min, gives KLD.
4. **Compare to baseline** (mq4-base 0.6721) and Q8 floor (0.4598). Land in `awq_fix_postfix_findings.md`.
5. **Decision tree:** if delta-above-Q8 closes ≥10% of the 0.21 nat gap → 9B confirmation worthwhile; else investigate α tuning or pivot to Stage B (GPTQ).

Total wall time to Stage A go/no-go: ~25 min from commit.

---

## Open follow-up (post-fix)

The whitelist intentionally drops AWQ on `o_proj` / `out_proj` / `down_proj`. Literature suggests AWQ provides additional benefit on these tensors too — when the runtime supports it. The right follow-up is **Option B from `awq_bug_hunt_glm5.md` §Fix**: add AWQ-aware variants of `rotate_x_mq` and `fused_silu_mul_rotate_mq` (4 new kernels + dispatcher wiring) so the skipped projections can also benefit. Defer until Stage A on the whitelisted subset is confirmed worthwhile.

---

## Postscript — second bug found + final measurement (2026-05-12 PM)

After the whitelist fix landed (commit `6711308d` + `ca759da6` for MoE completeness), the 0.8B re-eval gave KLD **9.5654** — still catastrophic but a 30% reduction. The proportional improvement matched the 26% of corrupting sidecars the whitelist removed (48 of 186), strongly suggesting another systemic issue affecting the remaining whitelisted weights too.

### Root cause #2 — the Qwen3.5 crate's private loader

`crates/hipfire-arch-qwen35/src/qwen35.rs:742` has its own `load_weight_tensor_raw` (and `load_weight_tensor` wrapper) which hardcoded `awq_scale: None` for every weight type. The Phase 2a infrastructure (`load_awq_scale` in `hipfire_runtime::hfq`) was simply never invoked from the Qwen3.5 forward path.

**Detection signal**: the AWQ kernel `fused_rmsnorm_mq_rotate_awq.hsaco` did NOT exist anywhere on disk. Compilation only happens when the dispatcher actually launches a kernel; since `if let Some(awq) = next_linear.awq_scale.as_ref()` was always `None`, the kernel was never invoked, and so never JIT-compiled. Verified by `find . -name "*awq*.hsaco"` returning nothing before the loader fix.

**Fix**: commit `0aa58185` adds a `load_awq_scale_for` helper to qwen35.rs (mirrors `hipfire_runtime::hfq::load_awq_scale`) and populates `WeightTensor.awq_scale` in the MQ4G256 arm of `load_weight_tensor`. After this landed, `fused_rmsnorm_mq_rotate_awq.hsaco` appeared on disk on the first eval run — proving the kernel was finally being invoked.

### Stage A measured results

**0.8B (dense, 24 layers):**

| Variant | KLD | KLD above Q8 floor | PPL |
|---|---:|---:|---:|
| q8f16 (engine floor) | 0.4598 | — | 30.996 |
| mq4-base | 0.6721 | +0.2123 | 36.594 |
| mq4-awq-loaderfix (α=0.5) | **0.6707** | **+0.2109** | **37.31** |

AWQ delta on 0.8B: −0.0014 nats KLD (−0.7% above-floor) — within noise. AWQ neither helps nor hurts at this scale.

**9B (hybrid, 32 layers = 8 full_attn + 24 linear_attn):**

| Variant | KLD | KLD above Q8 floor | PPL |
|---|---:|---:|---:|
| q8f16 (engine floor, 256ch) | 0.5735 | — | 13.383 |
| mq4-base (512ch) | 0.8165 | +0.2430 | 15.063 |
| **mq4-awq-loaderfix (α=0.5, 512ch)** | **0.7373** | **+0.1638** | **14.303** |

**AWQ delta on 9B: −0.0792 nats KLD (−32.6% reduction in quantization-attributable noise), −5.0% absolute PPL.**

That's *above* the literature's typical 15–25% lift on Q4 INT4 AWQ. The over-performance is plausibly because hipfire's FWHT rotation already mitigates dense outliers, leaving AWQ to focus on the sparse extreme outliers that survive rotation — where its outlier-preservation lever works best.

### Strategic confirmation

- AWQ is a real Phase A lever on dense+hybrid Qwen3.5 architectures at ≥9B scale.
- The 0.8B → 9B scale-dependence pattern matches the AWQ paper's prediction (outliers grow with scale, AWQ's value grows with outliers).
- Stage B (GPTQ on MQ4) and Stage C (MR-GPTQ on MFP4) are *unblocked* by this validation — the AWQ runtime path is correct, and the same Phase 2a/2b dispatch infrastructure will serve GPTQ.

### Cost summary of getting here

| Cost item | Wall time |
|---|---|
| Initial broken AWQ implementation discovery (whitelist + loader bug) | ~4h |
| Hypothesis testing + α=0 discriminator + sidecar inspection | ~1h |
| Loader fix code + build + verify | ~15 min |
| 9B re-quantize + dual eval (mq4-base + mq4-awq, 512ch each) | ~50 min |
| **Total cycle from "first bad cohort number" to "validated 9B Stage A win"** | **~6h** |

The infrastructure investment (`fused_rmsnorm_rotate_for_mq` AWQ dispatch, AWQ kernel, sidecar storage protocol) carries forward into Stage B/C with zero re-work — only the *calibration* algorithm changes between A and B.

### Open AWQ follow-up — `o_proj` / `out_proj` / `down_proj` (Option B from glm5 doc)

These tensors are skipped by the whitelist because their runtime paths (`rotate_x_mq`, `fused_silu_mul_rotate_mq`) have no AWQ inverse. Literature suggests another few percent KLD reduction is on the table from AWQ-ing them too. Cost: 4 new HIP kernels + dispatcher wiring. Deferred — confirm Stage B/C first; if AWQ + GPTQ + MR-GPTQ together leave a meaningful residual gap to UD-Q3_K_XL, revisit then.
