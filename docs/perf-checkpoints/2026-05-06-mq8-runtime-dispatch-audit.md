# MQ8 runtime-dispatch audit — silent-correctness gap on gfx906

Date: 2026-05-06
Hardware: MI50 / gfx906 / HBM2 1 TB/s peak.
Branch: `feat/gfx906-hfq6-hfq8-analysis` at commits `ee0fac6` (kernel
fix) → `8063553` (plan v3.1 + audit).

## TL;DR

`gemv_mq8g256.hip` was the canary, not the bug. After `ee0fac6` made
the kernel buildable on gfx906, an end-to-end bench on
`qwen3.5-9b.mq8` produced 45.4 tok/s with deterministic per-token
timing — but the inference is invalid. The per-layer prefill batched
dispatch in `crates/hipfire-arch-qwen35/src/qwen35.rs` excludes
`MQ8G256` from all 14 `is_mq` matchers, so MQ8 weights silently fall
through to `gemm_qkvza_hfq4g256` etc. — kernels that read at
HFQ4-format byte stride (136 B/group) when MQ8 is 258 B/group. The
prefill produces corrupted DeltaNet state and KV cache; the gen loop
consumes that corrupted state through valid MQ8 GEMV calls and
emits nonsense tokens at the bench-measured speed.

**This is by design, not a regression.** MQ8 was originally shipped
(commit `246501a`, 2026-04-08) "targeting dp4a on gfx1100." The
`feat(engine): load MQ4/MQ8 tied lm_head embeddings` commit
(`bf0ba43`, 2026-04-13) explicitly states *"Not a current path —
standard mq4 mode of hipfire-quantize keeps embed_tokens as Q8 for
quality."* MQ8 has only ever been used as the **lm_head tied
embedding** in production — a single B=1 GEMV at the end of forward,
which `weight_gemv` in `llama.rs` does dispatch correctly. **No
production model has ever shipped MQ8 per-layer weights.**

## Discovery sequence

1. `docs/plans/gfx906-mq6-mq8-port.md` v3 (2026-05-06) framed MQ8 as
   priority-1 because "the dp4a-on-int8 inner loop is already
   shipped in `gemv_mq8g256.hip`."
2. Priority 0 baseline run on `qwen3.5-9b.mq8` (produced via
   `hipfire-quantize --format mq8`) failed to compile — the kernel
   used the RDNA3+-only `sudot4` builtin. Fixed in `ee0fac6`.
3. Re-bench succeeded: 45.4 tok/s, p50 21.46 ms/tok, 1.4% spread.
   Result was *suspiciously* clean — RDNA3-tested kernel running
   first-time on RDNA1 hardware shouldn't be that smooth.
4. Audit (5.5 in plan v3.1) confirmed all kernel sources build on
   gfx906. Did not test runtime dispatch wiring.
5. Followup audit (this document): grep `is_mq = matches!` in the
   arch crate. Found 14 matchers that exclude `MQ8G256`.

## The 14 missing matchers

`crates/hipfire-arch-qwen35/src/qwen35.rs`:

| Line | Surface | Matcher |
|---:|---|---|
| 3946 | LA prefill: wqkv | `MQ4 \| MQ6 \| MQ3` |
| 4118 | LA prefill: wo | same |
| 4147 | LA prefill: w_gate | same |
| 4194 | LA prefill: w_down | same |
| 4243 | FA prefill: wq | same |
| 4508 | FA prefill: wo | same |
| 4538 | FA prefill: w_gate | same |
| 4576 | FA prefill: w_down | same |
| 4651 | LA decode: wqkv | `MQ4 \| MQ6` (note: also drops MQ3) |
| (8 more) | … | … |

Every one of these falls through to a HFQ4-stride read on MQ8 input.

## What the bench actually measured

`bench_qwen35_mq4 qwen3.5-9b.mq8`:

| Phase | What happened |
|---|---|
| Prefill (32 tok, 3 runs) | `forward_prefill_batch` → 14 LA layers × wqkv/wz/w_beta/w_alpha + wo + w_gate + w_up + w_down each fall through to HFQ4-stride read of MQ8 weight memory. Output `dn_state` and `KvCache` contain garbage. |
| Warmup (5 tok untimed) | Each B=1 step calls `weight_gemv` (correct MQ8 dispatch), arithmetic is right per kernel, but reads from corrupted prior state. Produces nonsense tokens at correct GEMV speed. |
| Gen (50 tok timed) | Same as warmup but timed. **45.4 tok/s is real GPU work on bad inputs.** |

The 1.4% spread in per-token timing is consistent — every token does
the same MQ8 GEMV chain on the same shape of corrupted state, so
timing is deterministic even though logits are nonsense.

## Why this stayed undetected for ~1 month

- `hipfire-quantize --format mq8` was never used in shipped models;
  `--format mq4` produces files where MQ8 only appears as
  `embed_tokens` (the tied lm_head path).
- `coherence-gate.sh` test matrix has no `qwen3.5-9b.mq8` entry
  (`scripts/coherence-gate.sh:84-103` — mq4/mq3/mq6 only).
- The `gemv_mq8g256.hip` JIT-compile-time path was only exercised
  on gfx1100/gfx1201 where the kernel built fine; gfx906 never
  reached runtime dispatch because the build failed first (per
  `ee0fac6`).

The plan's v3 framing of MQ8 as "shipped reference" was wrong twice:
the kernel didn't compile on gfx906, **and** the runtime never wired
it into the per-layer dispatch surface anyway.

## Implications for plan §3.2 / §5

- §3.2's "no MMQ-streaming port required, Phase A item 4 covers
  batched directly" is now *more* load-bearing than v3 stated:
  the `gemm_*_mq8g256_*` kernels don't exist AND the runtime
  dispatch sites that would call them are absent. Phase A item 4
  is a correctness prerequisite, not an optimization.
- The "MQ8 first" priority ranking should be revisited. The
  "smaller scope" argument survives in nominal terms but the gap
  is wider than counted: implementing MQ8 batched means writing
  4-7 new kernels (qkvza, qkv, gate_up, residual, MoE-indexed
  variants × wave64) AND wiring 14 dispatch sites in the arch
  crate. MQ6 Phase A is comparatively *better-defined*: the
  wave32 path through bare `gemm_qkvza_hfq6g256` already works
  end-to-end; only kernel-level wave64 ports are needed.

## Decision

- **Drop MQ8 from Priority 0 baseline.** The bench number we have
  is meaningless; running it again wouldn't help.
- **Do not invest in MQ8 wiring without a deployed model.** The
  plan §3.2 should be tagged "deferred until a production model
  ships raw MQ8 per-layer weights, OR a measured advantage over
  MQ6 motivates the full wiring work."
- **Plan v3.2 errata**: append this audit's findings, restrict
  MQ8 scope to "lm_head-tier optimization, not primary weight
  format," reorder priorities so MQ6 Phase A leads.
- **Audit method gap**: §5.5 build-tested kernel SOURCES on
  gfx906 but not runtime DISPATCH wiring. Future audits need to
  grep the arch crate for `is_mq` / dtype-matchers and confirm
  every quant format the loader produces is handled at every
  per-layer call site.

## What we kept

- `ee0fac6` is still the correct fix — the `sudot4 → sdot4`
  substitution makes the kernel buildable everywhere, and the
  lm_head GEMV path needs the kernel to work for tied-embedding
  models. Without `ee0fac6`, even the existing production lm_head
  path on gfx906 with mq4-format models that have MQ8 embeddings
  would fail at JIT.
- Plan §5.5 audit (kernel-source build-test) remains valid as far
  as it goes; it just needed the runtime-dispatch sibling check.

## References

- `kernels/src/gemv_mq8g256.hip` — fixed kernel (commit `ee0fac6`)
- `crates/hipfire-arch-qwen35/src/qwen35.rs` lines 3946, 4118,
  4147, 4194, 4243, 4508, 4538, 4576, 4651, … — the 14 sites
- `crates/hipfire-runtime/src/llama.rs:601, 646, 680, 733` —
  `weight_gemv` MQ8G256 dispatch (works correctly, used by lm_head)
- `docs/plans/gfx906-mq6-mq8-port.md` v3.1 §3.2, §5 — the framing
  this audit invalidates
- Commits 246501a (MQ8 introduction), bf0ba43 (lm_head loader),
  ee0fac6 (sdot4 fix)
