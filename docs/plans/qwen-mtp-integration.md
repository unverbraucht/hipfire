# Qwen 3.5 / 3.6 MTP integration plan

**Status:** plan, no code yet. Research-only deliverable per user request 2026-05-09.
**Author:** Claude Opus 4.7
**Target:** Qwen3.5 dense (0.8B/4B/9B/27B) and Qwen3.6 dense (27B) with MTP heads.
**Out of scope (this plan):** Qwen3.5/3.6 MoE A3B, Qwen3-Next dense (we don't ship one), training-time MTP loss, tree speculation.

## Why this plan exists

Qwen3.5 and Qwen3.6 ship with native **multi-token prediction (MTP) heads** baked
into the released checkpoints — small auxiliary modules that, given the target
model's last-layer hidden state plus the just-sampled token, predict the *next*
token without rerunning the full model. vLLM and llama.cpp both support these
heads as a speculative-decoding "proposer." The user has asked: can hipfire add
the same?

This is a **research deliverable**: scope, architecture, risks, and a phased
implementation outline. No code is written.

## Hard prerequisite: MoE final-norm fix

This plan **assumes** the MoE final-norm fix from
`findings/qwen35-rmsnorm-bug.md` is landed first.

The findings doc went through three drafts; the corrected diagnosis is:
hipfire's dense path is correct, and the **MoE final norm is
under-scaled by ~38%** because `load_norm_weight_raw()` skips the `+1`
that vLLM/llama.cpp/HF transformers all apply. The fix collapses the
`if config.num_experts > 0` fork at `qwen35.rs:1253-1257` and
`qwen35.rs:1483-1485` to unconditionally use `load_norm_weight` (which
bakes `+= 1.0` at load time, equivalent to vLLM's runtime
`weight + 1.0`). See `qwen35-rmsnorm-convince-glm5.md` for the
concrete arithmetic trace that pins the diagnosis.

**Caveat:** the MoE under-scaling was originally introduced as a
workaround for a `<think>` infinite-spiral bug on Qwen3.6-A3B reasoning
prompts. Restoring the correct scale will likely re-expose that
spiral, which is a separate precision bug (probably in MoE expert
routing or quantization, not in the norm convention). The findings doc
recommends gating the workaround behind an env var
`HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1` so the spiral can be A/B-tested
independently. Since this plan is dense-only (MoE explicitly out of
scope), the spiral is not blocking for MTP work — but the prerequisite
fix still needs to land for code-hygiene reasons.

GemmaRMSNorm storage convention is uniform across the model family:
safetensors store raw `w` (init from zero), engines apply `(1 + w)` at
runtime. This applies to all GemmaRMSNorm tensors in the MTP head as
well:
- `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, and the per-decoder
  norms inside the MTP layer all need `+= 1.0` baking at load → use
  `load_norm_weight()`.
- The MTP head's final `norm.weight` also needs `+= 1.0` baking at
  load → use `load_norm_weight()`.

The Phase 3 NRMSE gate against vLLM logits will catch any deviation;
if a particular MTP norm tensor turns out to follow a different
convention (unlikely given the uniform storage in the rest of the
family), swap loaders and re-test.

The base-model fix is shippable as a standalone PR independent of MTP
work. Empirical verification: dense outputs should be unchanged
post-fix; MoE may regress (expected, addresses risk-1 in the findings).
Recommended sequencing: ship the prerequisite (with the env-var
fallback), re-baseline, then start Phase 1 of this plan.

## Trust framework: which reference for which question

vLLM is the cleanest end-to-end reference but it is **not** an unambiguous
gold standard. Sources have different strengths:

| Question | Trusted source | Why |
|---|---|---|
| MTP head architecture (layer count, ordering, fc shape, norm placement) | vLLM `qwen3_5_mtp.py` / `qwen3_next_mtp.py` | Direct nn.Module spec; matches HF transformers |
| Inference dispatch shape (sequential, K=1 head reused K times) | vLLM `llm_base_proposer.py:492-651` | Concrete dataflow with line numbers |
| RMSNorm formula `(1+w)` | HF transformers `modeling_qwen3_next.py` + vLLM `GemmaRMSNorm` | Two independent agreeing sources |
| GGUF / quantized weight layout (tensor names, FastMTP trim) | llama.cpp PR #20700, #22673; NodeNestor's notes | Quantization-specific; vLLM doesn't quantize |
| The `enorm`/`hnorm` "+1 offset" naming gotcha for naive ports | NodeNestor | Reverse-engineered failure case; documented |
| Head-count rationale, training loss | Qwen blog, DeepSeek-V3 paper | Authorial intent |
| Default acceptance / speculative depth | vLLM defaults + thc1006 benchmarks | thc1006 documents real-world behavior |
| **Performance projections on RDNA** | none — we benchmark | thc1006 saw 100% MoE regression on RTX 3090; vLLM saw +27% on H100/GB200; nothing trustworthy on RDNA |

**Specific contraindications to vLLM-as-gospel:**

1. **vLLM's MTP proposer is built on top of `EagleProposer` and
   `llm_base_proposer.py` (1824 lines)** — most of the load-bearing logic
   isn't in the two `qwen3_*_mtp.py` files. A "faithful port of vLLM's MTP
   files" undercounts the work substantially.
2. **vLLM's default acceptance numbers are H100/GB200-specific.** thc1006
   showed every spec-decode mode regressing on RTX 3090 for Qwen3.6-35B-A3B
   due to MoE expert-loading overhead. We expect dense models to behave
   better, but require empirical confirmation before claiming a win on
   RDNA3/4.
3. **vLLM uses `ColumnParallelLinear` / `ParallelLMHead` / `EagleProposer`**
   tensor-parallel infrastructure that hipfire does not have. The port has
   to bypass these, not reimplement them.

## Family confusion: what `qwen3_5_mtp` actually covers in vLLM

vLLM has **three** MTP files:

- `qwen3_5_mtp.py` — Qwen3.5 dense + Qwen3.5-MoE (via the `Qwen3_5MoeMTP`
  subclass at line 454). Config key for head count: `mtp_num_hidden_layers`.
  Used by Qwen3.5 9B, 27B, and the A3B MoE variants.
- `qwen3_next_mtp.py` — Qwen3-Next 80B-A3B (the original blog model). Config
  key: `num_nextn_predict_layers`. Architecturally identical to
  `qwen3_5_mtp.py` modulo class hierarchy (uses `Qwen3NextDecoderLayer` and
  `Qwen3NextRMSNorm`, both of which are aliases for the qwen3.5 variants).
- **No `qwen3_6_mtp.py`** — Qwen3.6 reuses the `qwen3_5_mtp.py` path. The
  `.6` checkpoints we have on disk
  (`/local/hipfire/qwen3.6-27b.mq4`,`qwen36-27b-dflash-mq4.hfq`,
  `qwen3.6-27b.mq4-lloyd`) are dense Qwen3.5-architecture models with
  reasoning-tuned weights and an MTP head. Same code path.

For our scope (dense 3.5/3.6, MoE/Next out of scope), `qwen3_5_mtp.py` minus
the MoE inheritance is the entire reference.

## What an MTP head is, concretely

Forward pass of one MTP step (line numbers from
`vllm/model_executor/models/qwen3_5_mtp.py`):

```
input:
  input_ids:       [B] — the token just sampled by the target (or last MTP step)
  positions:       [B] — positional index for this token
  hidden_states:   [B, hidden] — last hidden of target (first call)
                                 OR last MTP layer output (subsequent calls)

forward (line 123-159):
  1. inputs_embeds = self.embed_tokens(input_ids)        # [B, hidden]
  2. inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)  # GemmaRMSNorm
  3. hidden_states = self.pre_fc_norm_hidden(hidden_states)     # GemmaRMSNorm
  4. concat = cat([inputs_embeds, hidden_states], dim=-1)   # [B, 2*hidden]
  5. x = self.fc(concat)                                    # [B, hidden]
  6. (x, residual) = self.layers[k % num_mtp_layers](      # one decoder layer
         positions, x, residual=None)
  7. (x, _) = self.norm(x, residual)                       # GemmaRMSNorm

output: hidden_states [B, hidden]

then:
  logits = lm_head(x)    # shared with target — [B, vocab_size]
  draft_token = argmax(logits) (or sample)
```

For K speculative tokens, this whole chain runs K times. With the standard
`mtp_num_hidden_layers=1` (the default and what every released Qwen3.5 / 3.6
checkpoint ships), `self.layers` has exactly **one** `Qwen3_5DecoderLayer` —
**a single decoder layer is reused K times** as a sequential causal chain.

**Memory footprint of the MTP head** (Qwen3.5 27B: hidden=5120,
intermediate=27392, num_attention_heads=64, num_kv_heads=8, head_dim=128):

| Tensor | Shape | f16 bytes |
|---|---|---|
| `embed_tokens.weight` | shared with target | (no extra) |
| `lm_head.weight` | shared with target | (no extra) |
| `pre_fc_norm_embedding.weight` | [hidden] | 10 KB |
| `pre_fc_norm_hidden.weight` | [hidden] | 10 KB |
| `fc.weight` | [hidden, 2*hidden] | 100 MB |
| `layers.0.input_layernorm.weight` | [hidden] | 10 KB |
| `layers.0.self_attn.q_norm.weight` | [head_dim] | 256 B |
| `layers.0.self_attn.k_norm.weight` | [head_dim] | 256 B |
| `layers.0.self_attn.q_proj.weight` | [hidden, hidden] | 50 MB |
| `layers.0.self_attn.k_proj.weight` | [n_kv*head_dim, hidden] | 6.25 MB |
| `layers.0.self_attn.v_proj.weight` | [n_kv*head_dim, hidden] | 6.25 MB |
| `layers.0.self_attn.o_proj.weight` | [hidden, hidden] | 50 MB |
| `layers.0.post_attention_layernorm.weight` | [hidden] | 10 KB |
| `layers.0.mlp.gate_proj.weight` | [intermediate, hidden] | ~270 MB |
| `layers.0.mlp.up_proj.weight` | [intermediate, hidden] | ~270 MB |
| `layers.0.mlp.down_proj.weight` | [hidden, intermediate] | ~270 MB |
| `norm.weight` | [hidden] | 10 KB |
| **Total** | | **~1 GB f16** |

When quantized at the same level as the base model (MQ4G256), the MTP head
is roughly **~250 MB** — small enough that we can keep it resident in VRAM
alongside the target model on a 24 GB card. On the 7900 XTX (~24 GB) running
27B at MQ4 (~15 GB), there is comfortable room.

## Where the head weights live today

**Today: nowhere in any hipfire model file.** The hipfire quantizer
explicitly skips MTP weights at conversion time:

```rust
// crates/hipfire-quantize/src/main.rs:2520
if name.starts_with("mtp.") {
    skipped_params += n as u64;
    continue;
}
```

Every existing `.hfq` / `.mq*` Qwen3.5/3.6 file in `/local/hipfire/` was
quantized through this path. **None contain MTP weights.** The plan must
include either re-quantization of source HF checkpoints with MTP retained,
or a separate MTP-only quantizer pass that produces a sidecar file (or
appends to existing `.hfq`).

## Coexistence with existing spec-decode

Hipfire already has spec-decode infrastructure: DFlash, DDTree, PFlash, and
CASK scaffolding. From the audit:

- **DFlash:** small *separate* draft model + greedy verify. Per-step
  draft.forward() + target.forward() + argmax compare.
- **DDTree:** tree-of-candidates from same draft model with topk>1, custom
  tree-aware verify with linear-attention innovation replay.
- **PFlash:** prefill compression, not decode — orthogonal.
- **CASK:** scaffolding only.

**MTP is structurally most similar to DFlash:** sequential per-step proposer,
greedy/probabilistic verify of one drafted token at a time. The differences:

| Aspect | DFlash | MTP |
|---|---|---|
| Draft source | separate small model file | head inside the target model file |
| Per-step proposer cost | 1 full forward of draft | 1 fc + 1 decoder layer + 1 lm_head |
| KV cache for proposer | yes (separate cache for draft) | no — head is stateless |
| Recurrent state to snapshot/restore | DeltaNetSnapshot for draft | none |
| Verify path | greedy argmax compare | greedy argmax compare (same) |
| Bonus token on full accept | yes | yes (same logic) |

**Recommendation:** the verify path is reusable. The proposer path is not —
DFlash's proposer assumes a `ModelSlot` with its own KV cache and full
forward. MTP's proposer is a single sub-graph applied to target's existing
hidden state with no KV bookkeeping. We need a **new proposer**, not a new
verifier.

This implies a small refactoring opportunity: introduce a `Proposer` trait
that DFlash and MTP both implement. The audit flagged this as currently
absent (everything is `verify_dflash_block` / `spec_step_greedy` —
DFlash-specific function names with no abstraction layer). **The MVP does
not need this trait.** Just write `spec_step_mtp()` and `verify_mtp_block()`
parallel to the DFlash entries; refactor in a later PR.

## Decision: DFlash coexists, MTP is a new parallel mode

Per the user's "let research decide": run them as parallel modes selected
at runtime. Reasons:

1. **DFlash works today** for users who already have draft models cached
   (Qwen3.5-0.8B drafts for 9B/27B). Killing it to chase MTP is a regression.
2. **MTP requires re-quantization** of all 3.5/3.6 models to include the
   head weights. Not all users will want to refresh; legacy `.hfq` files
   with no MTP head should still work via the DFlash path.
3. **The verify infrastructure is shared.** Coexistence cost is low.

Selection mechanism: env var `HIPFIRE_SPEC_MODE={dflash,mtp,off}` and/or
config field, defaulting to current DFlash behavior unless an MTP-equipped
checkpoint is detected.

## Implementation phases

### Phase 0 — GemmaRMSNorm fix (prerequisite, separate PR)

Already scoped in `findings/qwen35-rmsnorm-bug.md`. Not part of this work
unless that fix is rejected; in which case revisit.

### Phase 1 — quantizer: stop dropping MTP weights, store as sidecar

**Goal:** produce `.hfq` files (or sidecar `.mtp.hfq` files) that contain
the MTP head weights, quantized to match the base model.

Subtasks:

1. **Detect** MTP-bearing checkpoints. The heuristic in vLLM is "config
   has `mtp_num_hidden_layers > 0`" or "checkpoint contains tensors named
   `mtp.*`". Either works; the second is more robust against malformed
   configs. Add a `--with-mtp` flag to the quantizer that flips the gate
   from "skip" to "include."
2. **Translate weight names.** Per `qwen3_5_mtp.py:441-451` the vLLM
   loader does:
   - `mtp.fc.*` → `model.fc.*`
   - `mtp.layers.0.*` → `model.layers.0.*`
   - `mtp.norm.weight` → `model.norm.weight`
   - `mtp.pre_fc_norm_*.weight` → `model.pre_fc_norm_*.weight`
   - `embed_tokens` and `lm_head` are filtered (shared with target).
   For hipfire, prefix the MTP head names with `mtp.` to keep them
   distinct from the target model's same-named tensors. Recommended
   storage names:
   - `mtp.fc.weight`
   - `mtp.input_layernorm.weight`, `mtp.q_norm.weight`,
     `mtp.k_norm.weight`, `mtp.post_attention_layernorm.weight`
   - `mtp.q_proj.weight`, `mtp.k_proj.weight`, `mtp.v_proj.weight`,
     `mtp.o_proj.weight`
   - `mtp.gate_proj.weight`, `mtp.up_proj.weight`, `mtp.down_proj.weight`
   - `mtp.pre_fc_norm_embedding.weight`,
     `mtp.pre_fc_norm_hidden.weight`
   - `mtp.norm.weight`
3. **Quantize** at the same level as the base model. The base model's
   K-map (per `kmap_resolve`) keeps norms at F16, large matmul weights at
   the configured base level (e.g. MQ4G256). MTP head should follow the
   same rules, since it is structurally a decoder layer + a fc + 3 norms.
4. **Sidecar vs. append.** Two options:
   - **A) Append to existing `.hfq`.** Backward-compatible because
     loaders skip unknown names. New `--with-mtp` runs produce files
     ~250 MB larger. No version bump. Loader detects MTP by tensor
     presence.
   - **B) Sidecar file `model.mtp.hfq`.** Smaller file delta, separable.
     Loader looks for `<model>.mtp.hfq` next to `<model>.hfq`.

   **Recommendation: A, append.** Single file is simpler for users.
   Filesize delta is small. Sidecars are a deployment / sync hazard
   (one-half-of-the-pair-missing). vLLM and llama.cpp both bundle.

5. **Skip the `+= 1.0` baking** for MTP norms — they go through the same
   GemmaRMSNorm kernel (post-Phase 0 fix), no load-time mutation needed.

Deliverable: `hipfire-quantize` accepts `--with-mtp` and produces enriched
`.hfq` files containing MTP head weights with prefix `mtp.*`.

### Phase 2 — engine: load and validate MTP weights

**Goal:** at model load, detect MTP presence and populate a new
`Qwen35MtpHead` struct alongside the existing `Qwen35Weights`. Verify
shapes against target config.

Subtasks:

1. **Add `Qwen35MtpHead` struct** in
   `crates/hipfire-arch-qwen35/src/qwen35.rs`:
   ```rust
   pub struct Qwen35MtpHead {
       pub pre_fc_norm_embedding: GpuTensor,  // [hidden]
       pub pre_fc_norm_hidden: GpuTensor,     // [hidden]
       pub fc: WeightTensor,                  // [hidden, 2*hidden]
       pub layer: FullAttnLayerWeights,       // reuse existing struct
       pub norm: GpuTensor,                   // [hidden]
   }
   ```
   Note: the MTP layer is structurally identical to a target full-attention
   layer (same q_proj/k_proj/v_proj/o_proj/gate/up/down + 3 norms) — reuse
   `FullAttnLayerWeights` to avoid duplication. The DeltaNet variant of MTP
   head does not exist in vLLM (MTP layer is always `layer_type="full_attention"`
   per `qwen3_5_mtp.py:102` — even on Qwen3-Next which has DeltaNet in
   the base).
2. **Add `mtp: Option<Qwen35MtpHead>` to `Qwen35Weights`** so the
   absence-of-MTP case is type-safe.
3. **Load path.** In the existing weight loader, after the main layer load
   loop, check for `mtp.fc.weight` presence; if found, load all MTP
   tensors. Otherwise leave `mtp = None`. Use the same
   `load_norm_weight` (post-Phase-0, that is the unified raw-load function)
   and `load_weight_tensor` helpers — no special quantization logic
   needed.
4. **Sanity log** at load time: "MTP head loaded (~XXX MB)" or "no MTP
   head present (DFlash spec-decode only)."
5. **Drop / freeing.** Add MTP tensors to the existing `Drop` impl that
   frees layer GPU memory.

Deliverable: a Qwen3.5/3.6 model file with MTP weights produces a
`Qwen35Weights { mtp: Some(_), .. }`; an old file produces `mtp: None`.
No forward pass changes yet.

### Phase 3 — engine: MTP head forward pass

**Goal:** implement `mtp_forward()` that, given target's last hidden, an
input token, and a position, produces logits.

Subtasks:

1. **New function** in `qwen35.rs`:
   ```rust
   pub fn mtp_forward(
       gpu: &mut Gpu,
       config: &Qwen35Config,
       mtp: &Qwen35MtpHead,
       target_hidden: &GpuTensor,   // [hidden] — from target's final norm output
       input_token: u32,
       position: usize,
       lm_head: &WeightTensor,      // shared with target
       embed_tokens: &GpuTensor,    // shared with target
       scratch: &mut MtpScratch,
   ) -> HipResult<GpuTensor /* [vocab] logits */>
   ```
   Steps mirror the vLLM forward at `qwen3_5_mtp.py:123-159`:
   1. Embed lookup → `inputs_embeds [hidden]`.
   2. `gpu.rmsnorm_f32(inputs_embeds, mtp.pre_fc_norm_embedding, ...,
      gemma_offset=1.0)` — kernel from Phase 0.
   3. `gpu.rmsnorm_f32(target_hidden, mtp.pre_fc_norm_hidden, ...,
      gemma_offset=1.0)`.
   4. Concat into `[2*hidden]` scratch buffer.
   5. GEMV with `mtp.fc` → `[hidden]`.
   6. **Run a full-attention layer forward** on `[hidden]` — but
      **without writing to KV cache.** This is the trickiest sub-step.
      The MTP layer reuses the *same* attention kernel as the base model,
      but its KV is ephemeral (one forward pass, then discarded). Two
      implementation options:
      - **6a)** Allocate a tiny ephemeral KV scratch buffer ([1, n_kv, head_dim]
        for keys + values), use it for the MTP layer's attention, and
        discard. Simplest. Cost: one extra VRAM alloc per call (or a
        persistent scratch in the spec-decode loop's preallocated set).
      - **6b)** Refactor `forward_layer_full_attn` to take an
        `Option<&mut KvCache>` and skip the KV write when None. More
        invasive; touches the hot path.
      **Recommendation: 6a.** Persistent scratch in `MtpScratch`.
   7. `gpu.rmsnorm_f32(layer_out, mtp.norm, ..., gemma_offset=1.0)`.
   8. GEMV with `lm_head` → `[vocab]` logits.

2. **`MtpScratch`** struct allocated once at engine startup:
   ```rust
   pub struct MtpScratch {
       pub embed_buf: GpuTensor,        // [hidden]
       pub hidden_norm_buf: GpuTensor,  // [hidden]
       pub concat_buf: GpuTensor,       // [2*hidden]
       pub fc_out_buf: GpuTensor,       // [hidden]
       pub layer_out_buf: GpuTensor,    // [hidden]
       pub final_norm_buf: GpuTensor,   // [hidden]
       pub logits_buf: GpuTensor,       // [vocab]
       pub k_scratch: GpuTensor,        // [1, n_kv, head_dim] — ephemeral KV
       pub v_scratch: GpuTensor,        // [1, n_kv, head_dim]
   }
   ```

3. **Numerical-correctness validation** (critical — this is where ports
   silently fail):
   - Add an example `crates/hipfire-arch-qwen35/examples/verify_mtp.rs`:
     load a Qwen3.5/3.6 model with MTP, run a small fixed-seed prompt
     through both target-only and target+MTP, dump
     `mtp_forward(target_hidden, draft_token, pos)` logits, compare to
     a reference (e.g. vLLM logits from the same checkpoint via
     `dump_logits`) with NRMSE.
   - Acceptable threshold: NRMSE ≤ 5e-3 (per `gemma4` arch-intake
     pattern in this repo). If NRMSE blows up to ≥1e-1, something is
     wrong — most likely the `+1` offset, weight remapping, or KV
     scratch shape.
   - **Failure mode to specifically test:** the NodeNestor "+1 offset"
     gotcha. If `pre_fc_norm_embedding` or `pre_fc_norm_hidden` were
     loaded WITHOUT applying the `(1+w)` formula, NRMSE will be small
     for the first token but cascade across the prompt. The Phase 0
     fix should already handle this, but verify here.

Deliverable: `mtp_forward()` produces logits whose NRMSE against vLLM's
MTP head logits is < 5e-3 on a held-out prompt set.

### Phase 4 — spec-decode: `spec_step_mtp` and `verify_mtp_block`

**Goal:** wire MTP into the speculative-decode loop, parallel to
`spec_step_greedy` (DFlash).

Subtasks:

1. **`spec_step_mtp` outer loop** in
   `crates/hipfire-arch-qwen35/src/speculative.rs`:
   ```rust
   pub fn spec_step_mtp(
       gpu: &mut Gpu,
       target: &mut ModelSlot,
       mtp: &Qwen35MtpHead,
       mtp_scratch: &mut MtpScratch,
       pos: usize,
       k: usize,  // num speculative tokens
       last_token: u32,
       last_target_hidden: &GpuTensor,
   ) -> HipResult<SpecStepResult> {
       // 1. Generate K draft tokens via MTP
       let mut drafts = Vec::with_capacity(k);
       let mut current_token = last_token;
       let mut current_hidden = last_target_hidden.clone();
       for i in 0..k {
           let logits = mtp_forward(
               gpu, ..., mtp, &current_hidden, current_token, pos+1+i,
               ..., mtp_scratch)?;
           let draft = gpu.argmax(&logits)?;
           drafts.push(draft);
           current_token = draft;
           // current_hidden = layer output from this MTP step (not target hidden)
           current_hidden = mtp_scratch.layer_out_buf.clone();
       }

       // 2. Run target on the K-token block — reuse existing block-verify path
       let target_logits = target.forward_block(&drafts, pos)?;

       // 3. Greedy compare — accept prefix where target argmax matches draft
       let accepted = compare_argmax(&target_logits, &drafts)?;

       // 4. Bonus token on full accept (target's argmax at position K)
       let bonus = if accepted.len() == k { Some(target_argmax_at_k) } else { None };

       Ok(SpecStepResult { accepted, drafts, bonus, ... })
   }
   ```
2. **`verify_mtp_block` reuses `verify_dflash_block`'s greedy path.** The
   verify operation is identical: target's logits at each drafted
   position, argmax compare, accept-prefix. Initially **copy-and-rename**
   `verify_dflash_block` to `verify_mtp_block`, then dedupe in a follow-up
   PR if the two stay structurally identical (likely they will, since the
   verify side has no proposer-specific assumptions).

3. **Probabilistic vs. greedy verify.** vLLM's MTP path goes through the
   probabilistic rejection sampler by default
   (`llm_base_proposer.py:837`: MTP is `not in ("eagle3", "dflash")` so
   it returns hidden states differently, but the rejection still uses
   the probabilistic sampler unless flagged otherwise). Hipfire's DFlash
   is greedy by default; matching that for MVP keeps complexity bounded.
   Add probabilistic later if quality requires.

4. **KV cache rollback on partial accept.** When verifier accepts only
   first j of K drafts, target's KV cache needs to drop entries
   `[j+1..K]`. Hipfire's existing DFlash path handles this; reuse the
   same logic. MTP's own ephemeral KV scratch is naturally discarded each
   call — no rollback bookkeeping needed for the proposer side.

5. **Mode dispatch.** Wire `HIPFIRE_SPEC_MODE` (or a config field) at
   the daemon's main decode loop:
   ```rust
   match spec_mode {
       SpecMode::Off       => greedy_decode(...),
       SpecMode::Dflash    => spec_step_greedy(...),  // existing path
       SpecMode::Mtp       => spec_step_mtp(...),     // new
   }
   ```
   Default: if MTP head is present and `HIPFIRE_SPEC_MODE` is unset,
   prefer MTP. If `HIPFIRE_SPEC_MODE=dflash`, use DFlash even if MTP is
   present. Off if explicitly off.

Deliverable: a daemon launched with MTP-enabled checkpoint produces
correct text on the canonical PEP-8 prompt; coherence-gate-dflash.sh
(repurposed for MTP) passes; τ measurement landed.

### Phase 5 — coherence + perf gates

**Goal:** prove MTP doesn't regress coherence and quantify its decode-tps
delta against DFlash and the no-spec baseline.

Subtasks:

1. **Adapt `scripts/coherence-gate-dflash.sh`** to accept
   `--mode={dflash,mtp}`. Both modes get the same three-tier attractor
   thresholds (first 128 / last 128 / full output). MTP must pass the
   same bar.
2. **Add MTP cases to `scripts/coherence-gate.sh`** for the dense
   battery. MoE A3B variants are out of scope per user direction.
3. **Re-baseline canonical bench.** Per CLAUDE.md the canonical 27B-3.5
   LRU code DFlash is 199 tok/s τ=10.36 with PEP-8 strict prompt +
   prompt_normalize=true. After Phase 0 fix this number changes. After
   Phase 4, MTP gets its own canonical: same prompt, same model, MTP
   mode, captured into CLAUDE.md or `docs/methodology/`.
4. **DFlash vs. MTP head-to-head** on the canonical prompt:
   - `--mode=dflash` (current draft-model path)
   - `--mode=mtp` (new path)
   - `--mode=off` (no spec-decode baseline)
   Report tok/s, τ, and acceptance-rate distribution. Fresh-process
   verification per `docs/methodology/perf-benchmarking.md`.
5. **Multi-GPU note.** Per CLAUDE.md target hardware is gfx906 / MI50
   (memory: this is the dev box). RDNA3 / RDNA4 (7900 XTX, 9070 XT,
   9070) are the deployment targets per user. MTP forward is structurally
   nothing the existing Qwen3.5 forward isn't already doing — same
   kernels (rmsnorm, gemv_*, attention_flash_*). No new kernel surface
   means no new arch-specific perf cliff. Re-bench on each arch as part
   of normal release ritual; this plan does not require new arch-specific
   work.

Deliverable: a results table in `docs/plans/qwen-mtp-results.md` (or
appended to this file) with per-arch tok/s for the three modes, plus
canonical-bench updates to CLAUDE.md.

## Risks, ranked by load-bearing-ness

### Risk 1: GemmaRMSNorm fix not landed → Phase 3 fails silently

If Phase 0 doesn't go in first, MTP forward will produce wrong logits
because the three head norms (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`,
final `norm`) all use GemmaRMSNorm. Wrong logits → wrong drafts → 0%
acceptance → MTP appears to be "implemented but worthless." This is a
silent failure mode; correctness gate would catch it (NRMSE), but if
someone skips the verify_mtp.rs example, the bug ships.

**Mitigation:** Phase 3 includes the NRMSE verification as a hard gate.
If the Phase 0 fix is reverted in any future session, Phase 3 catches
it before merge.

### Risk 2: KV scratch shape wrong on first attention call

The MTP layer's attention is full-attention (`layer_type="full_attention"`)
but with **a single token**. Most of hipfire's attention kernels
(`attention_dflash`, `attention_flash_*`) are written for batched decode
or prefill, not single-token-with-no-history. Verify which kernel handles
the case `seq_len=1, kv_len=1` cleanly. Likely candidates:
`attention_flash` (the basic single-token decode path) or
`attention_causal_batched` with batch=1.

**Mitigation:** Phase 3's NRMSE gate catches this. Include a sub-test
for this specific kernel call before wiring it into Phase 4.

### Risk 3: Single-MTP-layer-reused-K-times semantics misimplemented

vLLM's `current_step_idx = spec_step_idx % self.num_mtp_layers` (line 146)
with `num_mtp_layers=1` reduces to "always the same layer" — the K-th
call uses the same weights as the 0-th. **But** the **input** to the K-th
call is the **output** of the (K-1)-th call (the layer-output hidden,
not the target hidden). This sequential-recurrence semantics is easy to
get wrong by accident — a naive port might pass `target_hidden` to all
K calls and get away with it for K=1 but break for K>1.

**Mitigation:** the example pseudocode in Phase 4 step 1 spells out
`current_hidden = mtp_scratch.layer_out_buf.clone()` after each call.
NRMSE verification at K=2 and K=3 catches deviations.

### Risk 4: Performance regresses on RDNA3/4

thc1006 saw every spec-decode mode regress on RTX 3090 for 35B-A3B MoE.
We're staying away from MoE per user direction, but dense Qwen3.6-27B
on a 7900 XTX (memory-bandwidth-bound) might still see weaker speedup
than the 1.8× DeepSeek reports if the MTP head's fc + decoder layer
latency dominates the saved target-forward time.

**Sanity numbers (approximate, gfx1100):**

- 27B target forward (one decode step): ~6 ms (per CLAUDE.md τ context)
- MTP head forward (one fc + one decoder layer + one lm_head): roughly
  one-thirty-second the size, so ~0.2 ms? — but with kernel-launch
  overhead, more like 0.5–1.0 ms. K=4 means 2–4 ms of MTP work to save
  3 × 6 ms = 18 ms of target-forward (modulo acceptance rate).
- Even at 50% acceptance the math favors MTP: ~9 ms saved vs. 4 ms cost
  + 6 ms verify forward. Net positive.

**Mitigation:** Phase 5 measurement is the gate. If on RDNA3/4 we see
less than +10% over no-spec baseline, that's a real surprise and we
investigate. thc1006-style per-token expert overhead is a MoE
phenomenon; dense models should not exhibit it.

### Risk 5: Probabilistic-rejection-sampler gap

vLLM's MTP defaults to probabilistic rejection sampling (Leviathan et al.
acceptance). Hipfire's DFlash defaults to greedy argmax. **For greedy
decoding (temperature=0)** the two are identical. **For sampled decoding**
(temperature > 0) MTP without probabilistic rejection silently degrades —
greedy verify rejects every draft that doesn't match target's argmax
even when the draft was a perfectly valid sample.

**Mitigation:** ship MVP with greedy-only and document the limitation.
Add probabilistic in a follow-up. Most hipfire benchmarks are temp=0
anyway (per the canonical PEP-8 setup).

### Risk 6: 3.6 reasoning models with MTP ship a different head shape

The 3.6 lineup includes "thinking" models — fine-tuned for reasoning with
explicit `<think>...</think>` traces. We don't know if their MTP head is
identical to 3.5's, or trained against the thinking traces (which would
make it predict thinking-token continuations differently than text-token
continuations). vLLM treats them identically; this is the assumption.

**Mitigation:** verify_mtp.rs runs on at least one 3.5 model and one 3.6
model; if 3.6 NRMSE fails with the same code path, dig in.

## Out-of-scope (this plan)

- **Qwen3.5/3.6 MoE A3B MTP** — Per user direction (2026-05-09): MoE on
  gfx906 is a known regression risk (thc1006), and the MoE codepath has
  enough surface area (FusedMoE, expert routing, MTP-on-MoE class
  hierarchy) that bundling it with the dense MVP doubles complexity.
  Ship dense first, audit MoE separately.
- **Qwen3-Next** — We don't ship a Qwen3-Next checkpoint (no `qwen3-next-*`
  in `/local/hipfire/`). Architecturally identical to Qwen3.5 MTP modulo
  config-key naming. If a Next checkpoint is ever loaded, the same code
  path should work with a config alias.
- **Tree speculation (Medusa-style)** — MTP is fundamentally a linear
  sequential proposer. Tree-of-candidates would be a separate proposer
  on top of the MTP head; not in this plan.
- **Training-time MTP** — Pure inference. The MTP loss formulation
  (Qwen blog / DeepSeek-V3 paper) is not relevant to a port that consumes
  pre-trained heads.
- **FastMTP vocabulary trimming** (llama.cpp PR #20700, 248K → 32K) — A
  3.7× draft-pass speedup by trimming the lm_head projection to common
  tokens. Tempting but a separate optimization; scope creep for the
  baseline plan. Park it as a Phase 6 candidate.

## Decision points still owed to the user

The plan is research-only as instructed. Before any code, ask:

1. **Storage strategy** for MTP weights: append to `.hfq` (recommended)
   or sidecar `.mtp.hfq`?
2. **Default mode** when an MTP-equipped checkpoint loads: prefer MTP
   over DFlash automatically, or require explicit `HIPFIRE_SPEC_MODE=mtp`?
3. **3.6 thinking-model verification** — do we verify with a 3.6 thinking
   checkpoint or stick to 3.5 for MVP NRMSE?
4. **Probabilistic rejection** — ship MVP greedy-only, or include
   probabilistic from day one for users running with sampling?
5. **Where does Phase 0 (RMSNorm fix) sit in the queue?** This plan
   assumes it lands first. If it lands after MTP work begins, we
   double-pay for fixing tests.

## Files this plan would touch (eventually)

```
crates/hipfire-quantize/src/main.rs              # Phase 1: --with-mtp, weight retention
crates/hipfire-arch-qwen35/src/qwen35.rs         # Phase 2 + 3: MTP head struct, load, mtp_forward
crates/hipfire-arch-qwen35/src/speculative.rs    # Phase 4: spec_step_mtp, verify_mtp_block
crates/hipfire-arch-qwen35/examples/verify_mtp.rs  # Phase 3: NRMSE gate
crates/hipfire-runtime/src/lib.rs                # Phase 4: SpecMode enum + dispatch
crates/hipfire-runtime/examples/daemon.rs        # Phase 4: HIPFIRE_SPEC_MODE wiring
scripts/coherence-gate.sh                        # Phase 5: --mode=mtp variant
scripts/coherence-gate-dflash.sh                 # Phase 5: --mode=mtp variant
docs/methodology/perf-benchmarking.md            # Phase 5: MTP canonical bench
CLAUDE.md                                        # Phase 5: canonical bench update
```

## Source citations

- vLLM models: `vllm/model_executor/models/qwen3_5_mtp.py` (1-458),
  `qwen3_next_mtp.py` (1-298), `qwen3_5.py:40 import GemmaRMSNorm`,
  `qwen3_next.py:31 import GemmaRMSNorm`,
  `vllm/model_executor/layers/layernorm.py:356-394 GemmaRMSNorm`.
- vLLM spec-decode: `vllm/v1/spec_decode/llm_base_proposer.py:79-987`
  (proposer entry, num_speculative_tokens, rejection-sample dispatch),
  line 837 `method not in ("mtp", "draft_model", "dflash")` confirming
  shared verify path.
- HF transformers: `modeling_qwen3_next.py` (`output * (1.0 + self.weight.float())`).
- llama.cpp: PRs #20700 (FastMTP), #22673 (MTP support).
- thc1006 benchmarks:
  https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090,
  https://github.com/thc1006/qwen3.6-vllm-2x3090.
- NodeNestor llama.cpp injection notes:
  https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp.
- Hipfire current state: `findings/qwen35-rmsnorm-bug.md`,
  `crates/hipfire-arch-qwen35/src/qwen35.rs`,
  `crates/hipfire-quantize/src/main.rs:2520`.
- Qwen blog (rationale, not implementation):
  https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd
