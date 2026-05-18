# Configurable kmap pair + per-tensor lm_head quant — `--kmap-promote`, `--lm-head-format`

**Branch:** `feat/configurable-kmap-pair`.
**Primary target:** MQ3+MQ4 alternating dense quant with AWQ. The 4B-MQ3+AWQ+GPTQ result (mean KLD 0.197, p99 9.7, PPL 11.65) survived where uniform sub-4-bit PTQ usually collapses; the hypothesis is that at 9B/27B the same recipe + kmap promotion of a few precision-sensitive tensors to MQ4 produces a meaningfully cheaper Pareto point than uniform MQ4 with comparable quality.
**Companion configurability:** independent `lm_head` format selection (currently force-Q8 in `kmap_resolve_mode` rule 2).
**Vision encoder:** **explicitly deferred to a follow-up plan** — vision-tower runtime has no AWQ-aware kernel dispatch today (`hipfire-arch-qwen35-vl` calls `gemm_f16` only), and shipping a CLI flag that produces unloadable models is a foot-shoot. Vision quantization is the CUDA branch's Phase 3 (`gptq_lm_head_awq.md §3.3`); we add `--vision-format` in the same PR that lands their vision runtime.
**Targets:** gfx906, gfx1010–1102, gfx1100–1102, gfx1150–1152, gfx1200–1201 (per-arch fast-path coverage of each constituent format).
**Companion branch coordination:** `feat/mq-v2-quant-format-cuda` is concurrently developing AWQ on lm_head + vision. **This branch lands first and is the canonical CLI surface.** The CUDA branch then merges our work and implements the AWQ-aware runtime kernels for lm_head (their Phase 3); the env-var path they had on their branch (`HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ`) is replaced by our `--lm-head-format` flag in the merge.

*Plan folds three adversarial reviews: Claude self-review, GLM5, Gemini. Rev files dropped after fold-in per the memory rule.*

## Reference disambiguation: what exists where

A `kmap_resolve_mode` extension was prototyped on `feat/mq-v2-quant-format-cuda` (commits `4b0693d6` for lm_head, `2e06a649` for hessian-collector). **None of this is in `master` today.** When this plan references "the CUDA branch's safety check" or "the CUDA branch's awq_eligible modification", those refer to code on that branch that **must arrive in our PR via re-implementation** — either we write it fresh in this branch, or we wait for CUDA to merge first and rebase (current decision: we land first, so we write it fresh).

Specifically:
- `HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ`, `HIPFIRE_LM_HEAD_AWQ_UNSAFE` — env vars on CUDA branch only.
- `awq_eligible` whitelist (`main.rs::awq_eligible`) **does not include** `lm_head.weight` or `output.weight` in master (GLM5 §3, verified). Must be added in our Phase 1.
- `docs/plans/gptq_lm_head_awq.md` — on CUDA branch only; we mirror their tied-embedding safety contract.

All line numbers in this plan are anchored to function names (`fn kmap_resolve_mode`, `fn awq_eligible`, `fn batched_gemm_single_weight`) since line numbers drift between commits.

## Why

Three independent gaps in the current quantizer CLI:

1. **Promote target is hardcoded to MQ6** — `QuantLevel::Promote6` (`fn kmap_resolve_mode` returns this for promoted tensors; the dispatcher in `run_safetensors_pipeline` / `run_gguf_pipeline` maps it unconditionally to `quantize_mq6g256`). `--format mq3 --kmap-dense --kmap-mode 2` produces MQ3+MQ6, not MQ3+MQ4. No CLI path today produces an MQ3+MQ4 alternating quant.

2. **lm_head is hardcoded to Q8** — `fn kmap_resolve_mode` rule 2 force-promotes `lm_head.weight` / `output.weight` to Q8 regardless of `--format`. No way to pick e.g. MQ4 lm_head when the dense model is MQ3-base.

3. **awq_eligible whitelist excludes lm_head** — `fn awq_eligible` matches by suffix against `q_proj.weight`, `gate_proj.weight`, `o_proj.weight`, `down_proj.weight`, etc. `lm_head.weight` matches none. Even if rule 2 were relaxed, AWQ pre-scaling would silently not apply.

The 4B-MQ3+AWQ+GPTQ result demonstrates MQ3-base + AWQ is empirically viable. Cross-family mixed-format dispatch (`HFQ4+HFQ6 ↔ MQ4+MQ6`) was validated by the user post-#257 (works but at high bpw). The mechanism #257 added (`qkv_same_dtype` + `batched_gemm_single_weight`) is the same mechanism this plan extends: adding MQ3 to that helper's match arms is the only runtime change for the primary target.

## Phase 0 — research (mandatory; results gate the recommendation)

Smaller than the previous MQ2-Lloyd-led sweep because MQ3+MQ4 doesn't need kernel work — only runtime dispatch wiring. Tighter Pareto gate.

Deliverables:
1. **Reproducibility check (anchor)**: re-run `eval_hipfire` at q8 KV n=20 on `qwen3.5-9b.mq4-kmd2-q8conv1d`. Confirms post-#257 baseline (`KLD=0.155438, NLL=2.220, PPL=9.207`) reproduces before sweeping. Drift > 0 here means the gfx1151 environment changed and Phase 0 is unreliable until investigated.
2. **MQ3-uniform-AWQ at 9B + 27B**: `qwen3.5-{9b,27b}.mq3 --awq`, eval at q8 KV n=20. Sets the floor.
3. **MQ4-uniform-AWQ at 9B + 27B**: `qwen3.5-{9b,27b}.mq4 --awq`, eval at q8 KV n=20. Sets the ceiling.
4. **kmap-mode sweep at 27B**: three quants `--format mq3 --kmap-promote mq4 --awq --kmap-dense --kmap-mode {0,1,2}`. Eval at q8 KV n=20. Selection criterion: **lowest mean KLD with `NLL paired-t < -3` vs MQ3-uniform-AWQ** (statistically significant improvement; per master-doc §6 rule 9 paired-t is the primary NLL test). Ties broken by lowest p99 KLD. Record seeds: AWQ calibration uses the calibration set fixed in `main.rs::AWQ_ALPHA` (default α=0.55); no other randomness.
5. **27B n=512 finale** on the winning mode. The gating number.

**Pareto hard gate** for "recommend MQ3+MQ4 as a config":
- Mean KLD on the winning mode at n=512 ≤ uniform-MQ4-AWQ KLD + 0.05 (within 0.05 of the more-expensive uniform variant), **AND**
- Bpw saving ≥ 0.4 bpw vs uniform MQ4.

Outcomes:
| n=512 KLD vs uniform MQ4 | Bpw saving | Action |
|---|---|---|
| within +0.05 | ≥ 0.4 | **Recommended default config.** Ship CLI; flip docs to MQ3+MQ4-AWQ. |
| within +0.05 | < 0.4 | Ship CLI; document but don't recommend (uniform MQ4 dominates). |
| > +0.05 | ≥ 0.4 | Ship CLI as opt-in research config; don't ship as default. |
| > +0.05 | < 0.4 | Ship CLI as plumbing only; explicitly warn against the pair. |

**Total Phase 0 GPU time on gfx1151**: dominated by quantize + n=512 final, not eval iterations. Honest breakdown:
- 9B n=20 evals: 3 × ~5 min ≈ 15 min
- 27B n=20 evals: 3 + 3 = 6 × ~12 min ≈ 72 min
- 27B quantize: 6 quants × ~30 min ≈ 3 hours
- 27B n=512 final: ~5 hours (scaled from 9B's 104 min)
- **~9 hours total**, single overnight batch.

GPTQ track: if `feat/mq-v2-quant-format-cuda`'s GPTQ pipeline is runtime-validated at 27B by Phase 0 time, repeat (2-5) with GPTQ. **Phase 0's gate stays AWQ-only**; GPTQ is a strict improvement layered on top, recorded but not blocking.

Output: `docs/perf-checkpoints/<date>-mq3-mq4-awq-kmap-sweep.md` with the numbers + the Pareto-gate decision.

## Phase 1 — quantizer CLI (ships standalone)

### `QuantLevel` enum redesign (committed design)

Replace the current 4-variant enum:

```rust
enum QuantLevel {
    F16,
    Q8,
    Promote(GgufFormat),       // was: Promote6 — carries the promote target
    Override(GgufFormat),      // NEW — for lm_head when --lm-head-format != Q8
    Base,                      // use --format as-is
}
```

`Override` is taken first in the dispatcher's match chain (precedence below); `Promote` second. The dispatcher in `run_safetensors_pipeline` / `run_gguf_pipeline` matches on the carried `GgufFormat` and dispatches to the corresponding `quantize_*g256` function.

### Dispatcher precedence in `fn kmap_resolve_mode` (committed)

Explicit ordering after the rewrite:

```
1. Norms / 1D tensors          → F16
2. lm_head / output            → IF --lm-head-format != Q8 then Override(fmt)
                                  ELSE Q8
3. token_embd / embed_tokens   → Q8 (unchanged; embeddings are out of scope —
                                  see §"Embeddings are intentionally untouched")
4. MoE router                  → Q8
5. MoE expert (mode-gated)     → Promote(--kmap-promote)
6. Dense (mode-gated)          → Promote(--kmap-promote) per kmap-mode rules
7. default                     → Base
```

### 1a. `--kmap-promote <fmt>` — decouples promote target from MQ6

```
hipfire-quantize --format mq3 --kmap-promote mq4 --kmap-dense --kmap-mode 2 --awq
```

Default: `mq6` (preserves current behavior byte-for-byte — kmd2 stays kmd2).

**Explicit promote-pair allowlist** (replaces the original plan's vague "bit-width ≥ base" rule per GLM5 §8):

| Base | Allowed `--kmap-promote` targets |
|---|---|
| `mq2`, `mq2-lloyd` | `mq3`, `mq3-lloyd`, `mq4`, `mq6` |
| `mq3`, `mq3-lloyd` | `mq4`, `mq6` |
| `mq4` | `mq6` |
| `hfq4` | `hfq6` |
| `mfp4` | `mfp4` (no-op; no FP6 sibling) |
| Same value as base | always allowed (no-op promotion; useful for testing) |

Reject every other combination at parse time with a clear error citing this table. Cross-family promotions (e.g. MQ→HFQ, MQ→HFP) are not in the allowlist because the runtime mixed-format dispatch hasn't been validated across families (only same-rotation-family per the `batched_gemm_single_weight` docstring tightened in #257's last commit).

Wiring: `QuantLevel::Promote6` → `QuantLevel::Promote(GgufFormat)`. Every match arm currently matching `Promote6` (there are ~5 sites; ~47 `Promote6` references but most are simple usage) gets the carried format. The dispatcher in the quantize pipeline reads the carried format and dispatches to the right `quantize_*g256`.

### 1b. `--lm-head-format <fmt>` — picks lm_head target

```
hipfire-quantize --format mq3 --kmap-promote mq4 --lm-head-format mq4 --awq
```

Default: `q8` (preserves current behavior).

Accepted vocabulary: `q8`, `f16`, `mq4`, `mq6`, `mq3`, `hfq4`, `hfq6`, `mfp4`. **F16 is included** (per GLM5 §9 — useful for small models where lm_head is a negligible fraction of total parameters and zero quality loss is preferred).

**Required code changes that arrive in this PR (not "preserve verbatim" — write fresh)**:

(i) **Tied-embedding refusal**. Mirror the contract from CUDA branch commit `4b0693d6`:

```rust
let tied_embed = config.get("tie_word_embeddings")
    .or_else(|| config.get("text_config").and_then(|tc| tc.get("tie_word_embeddings")))
    .and_then(|v| v.as_bool())
    .unwrap_or(false);
if tied_embed && !matches!(lm_head_format, GgufFormat::Q8 | GgufFormat::F16) {
    abort("lm_head AWQ-scaling corrupts shared embed_tokens — refuse.");
}
```

Documented limitation (GLM5/Gemini §3): the `unwrap_or(false)` default catches the common case (`tie_word_embeddings: true|false` set in config). Models that omit the field but ARE tied silently default to "untied" and can corrupt. The CUDA branch accepts this risk; we inherit it. A pointer-alias scan beyond the config flag is **future hardening, out of scope here**.

(ii) **awq_eligible whitelist extension**. `fn awq_eligible` in master does **not** include `lm_head.weight` / `output.weight` (GLM5 §3, verified at `main.rs::awq_eligible`). Without this fix, `--lm-head-format mq4 --awq` produces an MQ4 lm_head **without** AWQ pre-scaling, which the runtime will read as if it were AWQ-scaled once the CUDA branch's runtime lands — silent corruption. The fix: add `lm_head.weight` and `output.weight` to the F1 suffix set, gated on `--lm-head-format != Q8 && --awq`.

(iii) **UNSAFE runtime gate**. Any non-Q8 non-F16 `--lm-head-format` requires `HIPFIRE_LM_HEAD_AWQ_UNSAFE=1` (env var introduced fresh in this PR; not yet in master). Without the gate set: refuse to quantize. The gate disappears when the CUDA branch's runtime-side AWQ-aware lm_head dispatch lands (their PR drops the gate requirement in lockstep with our env-var removal).

(iv) **CUDA env-var deprecation** (per user instruction). `HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ=1` continues to work as a deprecated alias for `--lm-head-format mq4` for one release cycle, with a `eprintln!("deprecation:...")` warning at startup when set. The CUDA branch owner has confirmed deprecation is acceptable; the env var is removed in the next release after merge.

**Quality-uncharacterized warning**: `--lm-head-format mq4` without `--awq` is allowed but emits a startup warning that lm_head quality at sub-Q8 without AWQ is uncharacterized. (GLM5 §9.)

### 1c. Vision encoder — **phased out of this PR**

Per the user instruction and Gemini §2. The vision tower's forward pass calls `gpu.gemm_f16(...)` only (`hipfire-arch-qwen35-vl`); shipping a `--vision-format` flag that produces models the runtime can't load is a silent-corruption trap regardless of UNSAFE gating.

**`--vision-format` lands when the CUDA branch's Phase 3 vision runtime lands**, in the same PR that adds the vision-tower AWQ-aware dispatch. Our hessian-collector / `awq_eligible` infrastructure is forward-compatible; the missing piece is the runtime kernel side.

Tracked in §"Future expansion".

### Tests

Unit (extend `mod tests` near the end of `main.rs`):
- New `Promote(GgufFormat)` paths for each (base, promote) pair in the allowlist.
- `Override(GgufFormat)` paths for each accepted `--lm-head-format` value.
- Promote-pair allowlist rejection: `--format mq6 --kmap-promote mq3` exits non-zero with the canonical error.
- Tied-embed refusal: `--lm-head-format mq4` on a config with `tie_word_embeddings: true` exits non-zero.
- UNSAFE gate sequencing: `--lm-head-format mq4 --awq` without `HIPFIRE_LM_HEAD_AWQ_UNSAFE=1` exits non-zero.
- `awq_eligible("lm_head.weight")` returns true when `--lm-head-format != Q8 && --awq`; false otherwise.
- CUDA env-var alias: `HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ=1` produces equivalent output to `--lm-head-format mq4` + emits the deprecation warning.

End-to-end:
- For each canonical configuration, quantize a tiny test model, decode via `dump_meta`, confirm per-tensor `quant_type` matches expectations.

**Phase 1 is shippable on its own** for the `--kmap-promote` flag and the F16-lm_head case. Non-Q8 non-F16 lm_head requires Phase 2 (no runtime change needed) AND the CUDA branch's runtime work to drop the UNSAFE gate.

## Phase 2 — runtime gap analysis + closures for MQ3+MQ4

### What MQ3+MQ4 alternating actually needs at runtime

Walking the **8 dispatch clusters** (GLM5 §5 was correct) in `qwen35.rs`, for `--format mq3 --kmap-promote mq4 --kmap-dense --kmap-mode 2` on Qwen3.5-9B/27B:

| Layer / role | weights | Promoted by mode 2? | Fused kernel uniform? | Runtime status |
|---|---|---|---|---|
| DN-attn QKVZA | all MQ3 | no | ✅ `gemm_qkvza_hfq3g256_wmma` | works |
| DN-attn wo | MQ3 | no | ✅ `gemm_hfq3g256_residual_wmma` | works |
| DN-FFN gate+up | MQ3 + MQ3 | no | ✅ `gemm_gate_up_hfq3g256_wmma` | works |
| DN-FFN w_down | MQ4 (promoted) | yes | ✅ `gemm_hfq4g256_residual` | works |
| FA-attn QKV | wq=MQ3, wk=MQ3, **wv=MQ4** | yes (v_proj) | ❌ mixed → fallback | **MQ3 arm needed in `batched_gemm_single_weight`** |
| FA-attn wo | MQ3 | no | ✅ works | works |
| FA-FFN gate+up | MQ3 + MQ3 | no | ✅ works | works |
| FA-FFN w_down | MQ4 (promoted) | yes | ✅ works | works |

**The only runtime gap for the primary target is the MQ3 arm in `batched_gemm_single_weight`.** Same shape as the MQ6 arm: pre-zero Y on the active stream, call `gemm_hfq3g256_residual_wmma`. ~25 LOC.

### Non-WMMA-arch reachability check (GLM5 §6)

`gemm_hfq3g256_residual_wmma` is WMMA-only (gfx11/gfx12). GLM5 §6 raised whether the new MQ3 arm could be reached on non-WMMA arches (gfx906, gfx10).

**Verified path**:
- `fn is_batchable_la` in `qwen35.rs` excludes `MQ3G256` on non-WMMA arches (the `mq3_uniform_with_wmma` predicate gates strictly on gfx11/12).
- When `is_batchable_la` returns false for any weight in the model, `forward_prefill_batch_with_pbs` takes the per-token decode fallback at the top of the function. This bypasses `forward_prefill_chunk` and therefore bypasses `batched_gemm_single_weight` entirely.
- Result: on gfx906/gfx10 with an MQ3+MQ4 model, the model loads, runs via per-token decode (gemv-based), is correct but slow. `batched_gemm_single_weight` is **not reached**.

**Defensive belt-and-suspenders**: the new MQ3 arm in `batched_gemm_single_weight` adds an arch check at entry — if the running arch doesn't have WMMA, return an error explaining that the caller should have routed through per-token decode. Costs nothing if unreachable; loud failure if a future refactor breaks the assumption. ~5 LOC.

### Latent gaps closed in this PR (defense-in-depth)

Two latent #249-class bugs that don't block MQ3+MQ4 today but are reachable through future kmap configurations:

- **DeltaNet QKVZA 4-way (Cluster 1)**: same stride-mismatch class if a future kmap promotes anything in `linear_attn.in_proj_*`. Add `qkvza_same_dtype` gate.
- **FFN gate+up 2-way (Clusters 3, 5)**: same class if future kmap promotes only one of `mlp.gate_proj` / `mlp.up_proj`. Add `gate_up_same_dtype` gate.

`gate_up_same_dtype` does not exist in the codebase today (GLM5 §4, verified by grep). `qkvza_same_dtype` likewise. Both are net-new code in this PR.

Per GLM5 §11, we add **explicit tests** for these gates in Phase 3's poison battery — synthesize a mixed-dtype layer through a unit test (no model needed) and assert the gate fires and the fallback dispatch runs. Without those tests the gates are bitrot risk.

### Llama-arch port — **mandatory** in this PR (Gemini §6)

`hipfire-runtime/src/llama.rs` has ~30 dispatch sites mirroring `qwen35.rs`. **`--kmap-dense` applies to GGUF/Llama models** (verified at `fn run_gguf_pipeline` in `main.rs` — the kmap gate doesn't exclude llama-arch). A user running `hipfire-quantize --format mq3 --kmap-promote mq4 --kmap-dense --kmap-mode 2` on a Llama GGUF gets a mixed-format model that the llama-runtime would NaN on (same #249 stride mismatch in the fused QKV).

This makes the Llama port mandatory — we can't ship `--kmap-promote` for GGUF without it. ~120 LOC including the `qkv_same_dtype` gate + `batched_gemm_single_weight` (lifted from qwen35.rs as a shared helper, or duplicated; see §"Refactor opportunity" below).

### qwen35-vl

`hipfire-arch-qwen35-vl` reuses the qwen35 layer dispatch for the language tower; the vision tower has its own forward path (using `gpu.gemm_f16` only). For VL models with non-vision kmap, the language tower goes through qwen35.rs's now-patched dispatch. Vision tower is out of scope (deferred to CUDA branch's Phase 3).

### Refactor opportunity (Gemini §5)

Gemini suggested extracting `kmap_resolve` into a `kmap.rs` module before adding flags. **Rejected as scope creep** — the right refactor is to extract `batched_gemm_single_weight` into a crate-level helper that qwen35.rs and llama.rs both consume (avoiding duplication for the mandatory Llama port). That's the actual code-deduplication win for this PR. Note as a Phase 2 sub-task.

### Phase 2 perf impact estimate (GLM5 §15.5)

From #249 analysis (memo on the merged PR): mixed-format fallback adds ~3 launches + 1 memset per FullAttention layer when triggered. On Qwen3.5-9B, 8/32 layers are FA; the fallback fires only when `qkv_same_dtype=false` (i.e. on every prefill of the affected layers).

| Metric | Cost |
|---|---|
| Launch overhead | +24 launches/forward ≈ +100µs at 60 tok/s prefill ≈ **0.6% wall** |
| wv memset + residual-read Y bandwidth | ~1 MB/layer × 8 layers × 256 GB/s = ~30µs ≈ **0.2% wall** |
| Total prefill slowdown for MQ3+MQ4 model | **~1% on 9B, similar on 27B** |
| Decode (per-token) | unchanged (uses `weight_gemv_prerotated` per weight) |
| Uniform-format models | unchanged (byte-exact original fused path) |

For comparison: the MQ4+MQ6 kmd2 model (#249) was measured at ~1-2% slowdown via the same mechanism. MQ3+MQ4 has identical structure (1 fused-kernel skip per FA layer + 1 memset+residual on `wv`), so the ~1% figure transfers.

### Phase 2 sequencing

A. **MQ3 arm in `batched_gemm_single_weight` + arch check** (~30 LOC). Run kmd2 regression + MQ3+MQ4 forward-pass check.

B. **Same-dtype gates on Clusters 1, 3, 5** (~110 LOC) + corresponding **unit tests for each gate firing** (~80 LOC, Phase 3 will cover).

C. **Llama port** (~120 LOC) — **mandatory**. Lifts `qkv_same_dtype` gate + `batched_gemm_single_weight` pattern from qwen35.rs.

D. Vision-encoder runtime — **out of scope** (CUDA Phase 3).

E. lm_head AWQ-aware runtime — **out of scope** (CUDA Phase 3). The UNSAFE gate in Phase 1 enforces.

## Phase 3 — poison testing + validation

### Poison test (new `crates/hipfire-runtime/examples/test_mixed_format_dispatch.rs`)

For each mixed-format combination reachable through the new code:
1. Allocate scratch buffers (`pbs.fa_q_full_batch`, etc.).
2. **Initialize all outputs to `f32::NAN`.**
3. Synthesize small `(m, k, batch)` tensors + weights; run the dispatch.
4. Download outputs; assert no NaN survives (every output element overwritten).

Cases covered:
- FA QKV mixed (MQ3+MQ4) → tests the MQ3 arm of `batched_gemm_single_weight`.
- DN QKVZA mixed (synthetic; doesn't occur in shipping models today) → tests the `qkvza_same_dtype` gate firing.
- FFN gate+up mixed (synthetic) → tests the `gate_up_same_dtype` gate.

Without the synthetic mixed-DN tests, the new same-dtype gates would have zero test coverage (GLM5 §11).

### Decode-coherence smoke (new `crates/hipfire-runtime/examples/smoke_mq3_mq4.rs`)

Per GLM5 §12 (no integration test today): load the Phase 0-winning MQ3+MQ4 quant, decode 100 tokens from a fixed prompt at greedy temperature, assert all 100 tokens are finite, and verify the output is non-degenerate (unique-token-ratio > 0.15 — same first-128 attractor threshold from `coherence-gate-dflash.sh`). Doesn't replace n=512 eval but catches gross runtime corruption.

### Acceptance criteria

1. `cargo check --workspace --features deltanet` clean.
2. Phase 0 report published with the per-mode KLD numbers + Pareto-gate decision.
3. `cargo test -p hipfire-quantize kmap` — unit tests for new `Promote(GgufFormat)`, `Override(GgufFormat)`, promote-pair allowlist rejection, tied-embed refusal, UNSAFE-gate sequencing, `awq_eligible` lm_head paths, CUDA env-var alias deprecation warning.
4. Poison tests pass (FA QKV mixed + the two synthetic same-dtype gates).
5. Decode-coherence smoke passes on the Phase 0-winning MQ3+MQ4 quant.
6. **kmd2 regression check**: `eval_hipfire` on `qwen3.5-9b.mq4-kmd2-q8conv1d` q8 KV n=20 byte-identical to post-#257 baseline (`0.155438 / 2.219963 / 9.2070`). Verified deterministic across runs in #257's eval.
7. **MQ3+MQ4 27B n=20** with Phase 0-winning mode: finite KLD/NLL/PPL.
8. Llama-arch QKV gate test — synthesize a mixed-dtype FA QKV on a llama-arch model harness, assert no NaN.

Coherence-gate (`scripts/coherence-gate.sh`) is **not** in this PR's acceptance set; its model matrix is uniform-format and doesn't exercise the new code path. A `qwen3.5-9b.mq3-kmap-mq4` row gets added in a follow-up.

## Risks

- **Phase 0 Pareto gate fails (n=512 KLD > uniform MQ4 + 0.05 OR bpw saving < 0.4)**: ship CLI as plumbing per the outcomes table. Don't recommend the pair as a default.
- **CUDA branch's runtime AWQ-aware lm_head dispatch doesn't ship within the next quarter**: per user, this is the CUDA branch owner's responsibility post-merge. Our Phase 1 produces files that error out at runtime (or are gated by UNSAFE) until that lands. If timeline slips significantly, escalate.
- **Llama-arch port introduces subtle behavior changes**: although the same-dtype gate is byte-exact-no-op for uniform models, ~120 LOC of dispatch refactor in a less-touched file is non-trivial. Mitigation: run any existing llama smoke test (we have `smoke_llama_prefill_batch.rs`) before/after to confirm no drift.
- **AWQ α=0.55 untuned for MQ3**: real risk. Phase 0 measures at default α; if KLD looks borderline, add an α∈{0.4, 0.5, 0.55, 0.6, 0.7} sweep on the winning kmap mode at n=20. Out of scope to do unconditionally.
- **Tied-embedding default-untied behavior**: a model that omits `tie_word_embeddings` from config but is actually tied corrupts silently with `--lm-head-format mq4 --awq`. Documented limitation; matches CUDA branch's contract. Exhaustive pointer-alias check is future hardening.
- **No new HIP kernel files**, but ~300 LOC of runtime dispatch wiring needs writing + testing. "No kernel work" is technically correct but misleading (GLM5 §15.4 — addressed by the §"Phase 2 perf impact estimate" + §"Phase 2 sequencing" tables).

## Out of scope (this PR)

- `--vision-format` and vision-tower AWQ — phased out, ships with CUDA branch's Phase 3 vision runtime.
- **MQ2-Lloyd kernel family + MQ2-Lloyd+MQ3-Lloyd alternating**: future expansion, separate plan (see §"Future expansion" below).
- AWQ-aware runtime kernel work for lm_head — CUDA branch's `gptq_lm_head_awq.md` §3.2.
- Three-or-more-level kmap (current model: base + one promoted level).
- AWQ α-tuning sweep for sub-4-bit (separate, after Phase 0 establishes the α=0.55 baseline).
- Exhaustive pointer-alias check for `tie_word_embeddings` (future hardening, beyond CUDA branch contract).
- `kmap.rs` module extraction from `main.rs` (Gemini §5; useful refactor, separate task).
- Coherence-gate row additions for the new pair (separate task).

## Effort estimate (revised per GLM5 §14)

- Phase 0 (research): ~9 hours GPU + 2 hours analysis. Single overnight batch.
- Phase 1 (quantizer CLI, `--kmap-promote` + `--lm-head-format`): **~2.5-3 days**, broken down:
  - `QuantLevel` enum redesign + `Override` variant threading through ~47 match arms: ~1 day
  - Three CLI flag parsers + promote-pair allowlist validation: ~0.5 day
  - Tied-embed safety check (written fresh) + UNSAFE gate logic + `awq_eligible` lm_head extension: ~0.5 day
  - CUDA env-var deprecated-alias + warning emission: ~0.25 day
  - Unit tests for ~10 CLI combinations + tied-embed-refusal + UNSAFE-gate sequencing: ~0.5 day
  - CUDA-branch owner coordination (sync on gate naming, deprecation timeline, end-to-end safety contract): ~0.25 day
- Phase 2 step A (MQ3 arm + arch check + kmd2 regression eval): ~0.75 day.
- Phase 2 step B (same-dtype gates on 3 clusters + their unit tests): ~1 day.
- Phase 2 step C (Llama port, **mandatory**): ~1 day for code + ~0.5 day for testing.
- Phase 3 (poison + decode-coherence smoke + acceptance harness): ~1 day.

**Total: ~5-6 working days end-to-end**, dominated by Phase 1's CLI surface (the three new flags + the QuantLevel dispatch chain rework) and Phase 2C's mandatory Llama port.

## File layout summary

| Path | Action | LOC delta |
|---|---|---:|
| `crates/hipfire-quantize/src/main.rs` | `Promote6` → `Promote(GgufFormat)` + new `Override(GgufFormat)`; `--kmap-promote`, `--lm-head-format` parsing; tied-embed + UNSAFE gates; `awq_eligible` lm_head extension; CUDA env-var deprecated-alias | +~250 / -~50 |
| `crates/hipfire-arch-qwen35/src/qwen35.rs` — `fn batched_gemm_single_weight` | add MQ3 arm + arch check | +~30 |
| `crates/hipfire-arch-qwen35/src/qwen35.rs` clusters 1, 3, 5 | `qkvza_same_dtype` + `gate_up_same_dtype` gates with fallback dispatch | +~110 |
| `crates/hipfire-runtime/src/llama.rs` | port `qkv_same_dtype` gate + shared helper for `batched_gemm_single_weight` | +~120 |
| `crates/hipfire-runtime/examples/test_mixed_format_dispatch.rs` | NEW poison + same-dtype-gate firing tests | +~180 |
| `crates/hipfire-runtime/examples/smoke_mq3_mq4.rs` | NEW decode-coherence smoke | +~80 |
| `docs/perf-checkpoints/<date>-mq3-mq4-awq-kmap-sweep.md` | Phase 0 report | +~100 |

**Total: ~770 LOC** — bulk in quantizer CLI + Llama port + tests.

## Embeddings are intentionally untouched

For clarity (per GLM5 §15.1, my own §8 in the previous review): `token_embd.weight` / `embed_tokens.weight` stays force-Q8. Embeddings are a row lookup, not a matmul — AWQ has no x to divide. The shared "rule 2" arm in `kmap_resolve_mode` gets split so the lm_head sub-arm consults `--lm-head-format` while the embed sub-arm stays at Q8 unconditionally. Sub-Q8 embeddings (e.g. K-means codebooks) are a separate compression direction and not part of this plan.

---

## Future expansion — MQ2-Lloyd-base + MQ3-Lloyd-promote alternating

Deferred from this PR; documented here for sequencing. Was the previous primary target before the MQ3+MQ4 pivot.

### Why it's a future target

MQ3+MQ4 (primary) gives a ~0.5 bpw bracket between uniform MQ3 and uniform MQ4. MQ2-Lloyd+MQ3-Lloyd would give a ~1 bpw bracket between uniform MQ2-Lloyd and uniform MQ3-Lloyd — a bigger compression ratio, but at much higher uncertainty (does Lloyd-MQ2 survive at 27B?) and much higher implementation cost (entire WMMA kernel family port).

### What's needed (preserved from earlier plan iterations)

1. **MQ2-Lloyd batched-prefill kernels** — port the MQ3-Lloyd WMMA family. 8 kernel files mirroring `gemm_qkv/qkvza/gate_up/residual_mq3g256_lloyd_wmma{,_mb4}` plus `fused_*` siblings for decode. Differs in two axes: codebook 8→4 entries, bits 3→2. Templatize over `<CODEBOOK_SIZE, BITS_PER_W>` so MQ3-Lloyd codegen stays byte-identical (`.hsaco` md5 check), per the `docs/plans/mq6_gemm.md` templatize approach.
2. **`is_batchable_la` extension** to accept `DType::MQ2G256Lloyd` on gfx11+ (mirror MQ3-Lloyd's arch list).
3. **MQ2-Lloyd arms** in all 8 dispatch clusters + `batched_gemm_single_weight`.
4. **27B Phase 0**: MQ2-Lloyd-uniform-AWQ-GPTQ baseline at 9B + 27B before kernel work. Hard gate: if 27B-MQ2-Lloyd-uniform collapses, abandon MQ2-Lloyd in any form.

Modeled on `docs/plans/mq3-lloyd-wmma-prefill.md`. Estimated ~3-5 working days for kernel work + ~1 day for dispatch wiring + Phase 0 GPU time. Total ~1000 LOC across kernel + dispatch + decode-path siblings.

### Sequencing

After MQ3+MQ4 lands and Phase 0 of the MQ2-Lloyd track confirms 27B viability. Strict superset of the MQ3+MQ4 PR's dispatch work; the same-dtype gates added here apply directly.

---

## Future expansion — `--vision-format` (phased out of primary PR)

When CUDA branch's Phase 3 vision-tower runtime lands:
1. Add `--vision-format <fmt>` to the CLI surface.
2. Extend tied-embed-style refusal check to vision-shared tensors (if any modern VL architecture shares vision encoder weights with anything else).
3. Mirror `HIPFIRE_VISION_AWQ_UNSAFE` gate semantics until they confirm a non-corruption recipe.
4. Vision-tower hessian collection is already supported on CUDA branch (commit `2e06a649`).

Companion PR; not standalone.

---

## Coordination with `feat/mq-v2-quant-format-cuda`

Our work lands first; their branch merges ours and implements the runtime-side AWQ-aware lm_head + vision dispatch. Touch-points:

| Surface | This PR | CUDA branch (post-merge) | Final state |
|---|---|---|---|
| `--lm-head-format <fmt>` CLI | added | inherited | canonical |
| `HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ` env var | deprecated-alias-with-warning | removed | gone |
| Tied-embed refusal check | written fresh | reused | canonical |
| `awq_eligible` lm_head extension | written fresh | reused | canonical |
| `HIPFIRE_LM_HEAD_AWQ_UNSAFE` gate | introduced | removed when runtime lands | gone after runtime PR |
| `--vision-format` CLI | **not added** (phased out) | added by them with runtime | canonical |
| `HIPFIRE_VISION_AWQ_UNSAFE` gate | **not introduced** here | introduced + removed in same PR | n/a from our side |
| Vision hessian-collector | reuse their suffix list (mirror `gptq_gpu_pkg/names.py` in Rust when we add `--vision-format`) | unchanged | canonical |
| GPTQ manifest format (`--precomputed-gptq-path`) | unchanged | extended for new formats | canonical |

When CUDA's Phase 3 runtime PR lands:
1. The `HIPFIRE_LM_HEAD_AWQ_UNSAFE` requirement drops; their PR includes the gate removal.
2. The default of `--lm-head-format` can flip from `q8` to `mq4` if their n=512 NLL paired-t shows a strict win (per their §1.2 acceptance).
3. `--vision-format` ships in the same PR.

Notify the CUDA-branch owner once Phase 1 is in flight so we don't duplicate gate-naming.
