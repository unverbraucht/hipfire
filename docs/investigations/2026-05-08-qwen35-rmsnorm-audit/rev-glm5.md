# Qwen3.5/3.6/3-Next RMSNorm: validation of findings/qwen35-rmsnorm-bug.md

**Date:** 2026-05-09
**Reviewer:** glm-5 (automated audit, revised after empirical cross-check)
**Status: The REVISED findings document is wrong. The current hipfire code
(load_norm_weight with `+= 1.0`) is correct for ALL Qwen3.5+ norms, including
the final norm on dense models. The bug is in the opposite direction: the MoE
final norm should NOT have had `+= 1.0` removed.

---

## 1. Ground truth confirmed: Qwen3.5+ uses GemmaRMSNorm `(1 + w)`

All three reference engines agree:

| Engine | Qwen3.5 RMSNorm formula | Evidence |
|---|---|---|
| **HuggingFace transformers** | `x * (1.0 + self.weight.float())` | `modeling_qwen3_next.py` — `nn.Parameter(torch.zeros(dim))` init, `(1 + w)` at forward time |
| **vLLM** | `weight = self.weight.data.float() + 1.0`, then `rms_norm(x, weight, eps)` | `layernorm.py:355-401` — `GemmaRMSNorm.forward_native`. Qwen3.5 explicitly imports `GemmaRMSNorm as Qwen3_5RMSNorm` at `qwen3_5.py:40`. Used for ALL norms: `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, final `norm`. |
| **llama.cpp** | `rms_norm(x) * w` at inference, but `w` already has `+1` baked in during GGUF conversion | `convert_hf_to_gguf.py:4865-4866`: `Qwen3NextModel.modify_tensors` applies `data_torch = data_torch + 1` to ALL `*norm.weight` tensors (except `linear_attn.norm.weight`). `Qwen3_5TextModel` inherits through `Qwen3NextModel` on the MRO. |

**Critical detail:** In PyTorch, `state_dict()` saves raw parameter values. The
forward method's `(1 + w)` computation is NOT baked into the stored weights.
The checkpoint stores `w` (initialized to zero), and `(1 + w)` is computed at
inference time by every engine.

---

## 2. What the on-disk values actually are

I ran `/tmp/inspect_norms.py` against four `.hfq` files to confirm the
empirical numbers. These `.hfq` files store raw F16 norm weights copied from
HuggingFace safetensors without transformation (verified: `qwen35.rs:668-671`
reads qt=1 as F16, qt=2 as F32; no arithmetic is applied).

### Per-layer norms (mean ~0, consistent with `w` near init)

| Tensor | Model | mean | std | range |
|---|---|---|---|---|
| `layers.0.input_layernorm.weight` | 3.5-0.8B | +0.24 | 0.12 | -0.12 to +1.00 |
| `layers.0.input_layernorm.weight` | 3.5-9B | +0.03 | 0.04 | -0.06 to +0.30 |
| `layers.0.input_layernorm.weight` | 3.6-27B | -0.02 | 0.04 | -0.13 to +0.20 |
| `layers.0.input_layernorm.weight` | 3.6-A3B-MoE | +0.03 | 0.05 | -0.08 to +0.33 |
| `layers.11.q_norm.weight` | 3.5-0.8B | +0.63 | 0.21 | -1.02 to +1.13 |

### Final `norm.weight` (mean ~1.0+, still `w`, just trained to large values)

| Model | On-disk mean | What `+= 1.0` produces | What vLLM computes |
|---|---|---|---|
| Qwen3.5-0.8B (dense) | **+3.31** | +4.31 | +4.31 `(1 + 3.31)` |
| Qwen3.5-9B (dense) | **+1.14** | +2.14 | +2.14 `(1 + 1.14)` |
| Qwen3.6-27B (dense) | **+0.96** | +1.96 | +1.96 `(1 + 0.96)` |
| Qwen3.6-35B-A3B (MoE) | **+1.63** | +2.63 | +2.63 `(1 + 1.63)` |

**The on-disk values are `w` (deviation-from-zero).** They are NOT pre-shifted
`(1+w)` values. Proof:

1. vLLM loads `w` from checkpoint and adds `+1.0` at forward time. If the
   on-disk value were already `(1+w)`, vLLM would compute `(1+w) + 1 = 2+w`,
   which is wrong. vLLM works correctly, so the checkpoint stores `w`.
2. llama.cpp adds `+1` to ALL norm weights during GGUF conversion (including
   final norm, including MoE). If the on-disk value were already `(1+w)`,
   llama.cpp would double-apply the offset. llama.cpp works correctly.
3. The `GemmaRMSNorm` class initializes `self.weight` to `torch.zeros(dim)`.
   At init, `w = 0`, and the formula gives `x * (1+0) * rms = x * rms`.
   The parameter drifts from zero during training. A trained value of +3.31
   means the final norm learned to apply scale `(1 + 3.31) = 4.31`. This is
   architecturally normal — the final norm is closest to the loss and receives
   the most gradient pressure.

### DeltaNet `linear_attn.norm.weight` (mean ~1.0, plain RMSNorm, no `+1`)

| Model | mean | std |
|---|---|---|
| 3.5-0.8B layer 0 | +0.88 | 0.07 |
| 3.6-27B layer 60 | +1.02 | 0.07 |
| 3.6-A3B layer 0 | +0.88 | 0.05 |

These cluster around 1.0 — consistent with `RMSNormGated`'s
`nn.init.ones_()` initialization (standard RMSNorm, NOT GemmaRMSNorm).
Hipfire correctly loads these via `load_any_as_f32` without `+= 1.0`.

---

## 3. Chain of reasoning and where the revised findings went wrong

### My reasoning process (for the next agent to follow or critique)

**Step 1 — Establish ground truth from source code (3 independent engines).**
vLLM (`layernorm.py:372-374`), HF transformers (`modeling_qwen3_next.py`), and
llama.cpp (`convert_hf_to_gguf.py:4865`) all agree: Qwen3.5+ uses GemmaRMSNorm
with formula `x * (1 + w) * rms`, where `w` is initialized to zero and stored
as a PyTorch `nn.Parameter`. The `+1` is added at inference time, not stored
in the checkpoint. This is a hard fact from three independent codebases.

**Step 2 — Determine what `.hfq` files contain.**
The `.hfq` format stores norm weights as raw F16/F32 (`qt=1` or `qt=2`),
read directly from the HuggingFace safetensors with no transformation
(`qwen35.rs:668-671`). PyTorch `state_dict()` saves raw parameter values,
so `.hfq` contains `w`, not `(1+w)`. Confirmed by running
`/tmp/inspect_norms.py` against four model files and reproducing the exact
numbers in the revised findings doc.

**Step 3 — Trace the computation through each engine for a concrete value.**
Take 3.5-9B final norm mean = +1.14 (on-disk, this is `w`):
- vLLM: loads +1.14, adds +1.0 at forward → `x * 2.14 * rms` ✓
- llama.cpp: bakes +1.0 at conversion → stores 2.14, runtime `x * 2.14 * rms` ✓
- hipfire dense (`load_norm_weight`): loads +1.14, adds +1.0 → `x * 2.14 * rms` ✓
- hipfire MoE (`load_norm_weight_raw`): loads +1.14, no +1 → `x * 1.14 * rms` ✗
- proposed "fix" (`load_norm_weight_raw` for all): `x * 1.14 * rms` ✗

Three engines agree on 2.14. The proposed fix gives 1.14. The current dense
path gives 2.14. The current MoE path gives 1.14.

**Step 4 — The only way the revised doc could be right is if the on-disk
value is NOT `w` but `(1+w)`.** But this is refuted by Step 1: vLLM and
llama.cpp both apply `+1` to the on-disk value, and they produce correct
output. If the on-disk value were already `(1+w)`, both engines would
double-apply the offset and produce wrong output. They don't.

### Where the revised findings document went wrong (exact error)

The revised doc's author made one specific logical error, which then
propagated through the entire analysis:

**The error: conflating "the trained value is far from zero" with "the
storage convention is different."**

The argument structure in the revised doc (lines 62-75, 164-197) is:

1. Per-layer norms cluster near zero (mean ~0.03).
2. Final norm has mean ~+1.14 to +3.31 — far from zero.
3. Therefore the final norm uses a "different storage convention" — it
   stores `(1+w)` as a raw scale, while per-layer norms store `w`.
4. Therefore hipfire's `+= 1.0` baking "double-applies" the offset on
   the final norm.
5. Therefore the fix is to use `load_norm_weight_raw` (no `+= 1.0`)
   for the final norm.

The false step is **3**. The conclusion does not follow from the premises.
In `GemmaRMSNorm`, `w` is initialized to zero and the optimizer is free
to push it anywhere during training. A large trained `w` does not imply a
different storage convention — it just means the final norm learned a
large weight. This is architecturally expected: the layer closest to the
loss receives the most gradient pressure.

The correct inference from (1) and (2) is:

3'. Per-layer norms trained to small values (near zero init). Final norm
    trained to large values (far from zero init). Both are `w`. Both need
    `+= 1.0` at inference time. The storage convention is uniform.

The revised doc's own empirical data is correct — the numbers +0.96 to
+3.31 are real. The misinterpretation is in assigning semantic meaning
to the magnitude of the trained weights.

**Why the error was seductive:** The revised doc correctly observed that
the 0.8B model's final norm (mean +3.31) is very far from zero, and that
adding another +1.0 (to get 4.31) seems like it would "over-amplify."
Intuitively, a scale of 4.31 feels too large. But the GemmaRMSNorm
formula is `x * (1 + w) * rms`, and when `w = 3.31`, the correct scale
IS 4.31. The large value is the intended trained behavior, not a bug.

**A secondary error:** The revised doc asserts (line 13) that "this audit
started as 'the kernel is wrong, all norms over-amplify' and shrank
substantially after I dumped actual norm tensors." This framing implicitly
assumes that the initial ("all norms wrong") diagnosis was too broad, and
the empirical data narrowed it down. In reality, the initial diagnosis was
wrong in the opposite direction — it said the current code over-amplifies,
when the only actual bug is that the MoE final norm UNDER-amplifies. The
empirical data didn't narrow a correct diagnosis; it led to a different
incorrect diagnosis.

### Why I initially agreed with the revised doc (my own error)

In my first review (`qwen35-rmsnorm-bug-plan-rev-glm5.md` v1), I accepted
the revised doc's conclusion without tracing the computation through the
reference engines for a concrete value. I verified that vLLM uses
`GemmaRMSNorm` and that the on-disk values match the revised doc's
numbers, but I did not perform the critical Step 3 above: actually
computing what vLLM would do with the value +1.14. When I did, the
contradiction was obvious — vLLM adds +1 to +1.14 to get 2.14, so the
on-disk value cannot already be `(1+w)`.

The lesson: when an empirical finding (large weight values) conflicts with
a theoretical constraint (uniform storage convention from three independent
implementations), trace the actual computation with concrete numbers before
accepting the empirical interpretation.

---

## 4. The revised findings document is wrong

The revised `findings/qwen35-rmsnorm-bug.md` argues:

> "The final `norm.weight` is stored as a raw `(1+w)` scale, not as
> deviation-from-zero, for ALL four models."
>
> "The fix is ~3 lines: collapse the `num_experts` branch to always use
> `load_norm_weight_raw` for `norm.weight`."

This is incorrect. The final norm weight of +1.14 (9B) is `w`, not `(1+w)`.
Applying the proposed fix would produce:

- **Proposed:** `x * 1.14 * rms` (under-scaled by 47%)
- **Correct:** `x * 2.14 * rms` (matches vLLM and llama.cpp)
- **Current hipfire dense:** `x * 2.14 * rms` (already correct!)

The revised doc's "fix" would introduce a regression on dense models that
is LARGER than the bug it claims to fix. The current dense path (`+= 1.0`)
is correct.

---

## 5. The actual bug: MoE final norm is under-scaled

The current code:

```rust
let output_norm = if config.num_experts > 0 {
    load_norm_weight_raw(hfq, gpu, "norm.weight", &[config.dim])?  // x * 1.63 * rms
} else {
    load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?     // x * 2.63 * rms
};
```

For MoE (3.6-A3B, final norm mean +1.63):
- **Current hipfire:** `x * 1.63 * rms` (missing the `+1`)
- **Correct (vLLM/llama.cpp):** `x * 2.63 * rms`

The MoE final norm is under-scaled by ~38%. Commit `1e01c0b` introduced
this under-scaling to fix a ` Iterating` spiral. The spiral was likely
caused by a separate precision issue in the MoE path (expert routing
numerics, quantization interaction, etc.) that was masked by the lower
output magnitude — a fortuitous cancellation, not a correct fix.

The dense path is already correct. The per-layer paths are already correct.
Only the MoE final norm is wrong, and it's wrong in the direction of being
too small, not too large.

---

## 6. Recommended fix

**Option A (minimal, correct):** Flip the MoE branch to also use
`load_norm_weight` (with `+= 1.0`):

```rust
// Remove the num_experts fork entirely:
let output_norm = load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?;
```

This makes dense and MoE identical. Both apply `+= 1.0`, matching vLLM
and llama.cpp. Two locations (lines ~1230 and ~1459).

**Risk:** The MoE ` Iterating` spiral may return. If it does, the root
cause is NOT the norm convention — it's a separate precision bug in the
MoE expert routing or quantization path that needs its own investigation.

**Option B (original Option 1 from first audit — kernel patch):** Add a
`gemma_offset` parameter to the kernels and remove load-time `+= 1.0`
entirely. This is cleaner long-term (single source of truth, no
load-time mutation) but is a larger change with more surface area.

Both options produce identical numerical results. Option A is the safe
incremental fix; Option B is the architecturally clean fix.

---

## 7. Validation against the original (pre-revision) findings document

The original document (before the empirical norm-dump revision) argued:

> "ALL Qwen3.5+ norms store raw deviation-from-zero weights, so the
> `+= 1.0` baking is over-amplifying everywhere — switch to Option 1
> (kernel patch with `gemma_offset`)."

This was also wrong about the diagnosis ("over-amplifying everywhere")
but the prescription (Option 1: kernel patch) would produce correct
results because `gemma_offset=1.0` at kernel time is mathematically
equivalent to `+= 1.0` at load time. The original doc was right that
the kernel should apply `(1+w)`, wrong that the current load-time baking
is incorrect.

**Summary of correctness:**

| Claim | Original doc | Revised doc | Reality |
|---|---|---|---|
| Per-layer `+= 1.0` is correct | Wrong (said over-amplifying) | Correct | **Correct** |
| Dense final `+= 1.0` is correct | Wrong (said over-amplifying) | Wrong (said double-applying) | **Correct** |
| MoE final `+= 1.0` is correct | Wrong (said over-amplifying) | Wrong (said should be raw) | **Correct** (but spiral risk) |
| The `+1` belongs in the kernel | Correct (Option 1) | Wrong (said no kernel change) | Either location works |
| Dense-vs-MoE fork should be removed | Correct | Correct (but wrong direction) | **Correct** — both should use `+= 1.0` |

---

## 8. Files verified

| File | Lines checked | Status |
|---|---|---|
| `findings/qwen35-rmsnorm-bug.md` (revised) | All 376 lines | Core diagnosis wrong; empirical data correct but misinterpreted |
| `kernels/src/rmsnorm.hip` | All 26 lines | Plain `x * w * rms` confirmed |
| `kernels/src/fused_rmsnorm_mq_rotate.hip` | All 153 lines | Plain `x * w * rms` at line 79 confirmed |
| `crates/hipfire-arch-qwen35/src/qwen35.rs` | 659-692, 1215-1234, 1449-1463 | All load paths confirmed |
| `/local/git/vllm-gfx906-mobydick/vllm/model_executor/layers/layernorm.py` | 103-401 | `RMSNorm` (plain) and `GemmaRMSNorm` (`1+w`) confirmed |
| `/local/git/vllm-gfx906-mobydick/vllm/model_executor/models/qwen3_5.py` | 39-41, 172-176, 244 | Explicit `GemmaRMSNorm` for ALL norms including final |
| `/local/git/vllm-gfx906-mobydick/vllm/model_executor/models/qwen3.py` | 41 | Plain `RMSNorm` confirmed (Qwen3 ≠ Qwen3.5) |
| `/home/kread/git/llama.cpp/src/models/qwen35.cpp` | All 473 lines | Plain `build_norm` at inference, `+1` baked at conversion |
| `/home/kread/git/llama.cpp/convert_hf_to_gguf.py` | 4858-4866, 5524-5526 | `+1` baking for ALL Qwen3.5 norms (MoE and dense) |
| `/tmp/inspect_norms.py` | All 129 lines | Verified: reads raw F16 from `.hfq` with no transformation |
| Empirical dumps from 4 models | — | Confirmed: final norm means +0.96 to +3.31 are `w`, not `(1+w)` |
