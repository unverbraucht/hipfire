# AWQ on `lm_head` (and vision encoder) — reference + postmortem

**Status (2026-05-19): shipped on master** via the mega integration
PR #296 ("iterative AWQ/GPTQ plus Kevin PR stack"). The env-gated
prototype documented in earlier revisions of this doc has been
retired; the CLI surface is the supported entry point.

This doc now serves as:

1. **CLI reference** — how to actually request `lm_head` AWQ from
   either the Rust quantizer or the Python GPU GPTQ pipeline.
2. **Safety guardrails** — the hard-won correctness rules the
   implementation has to honor (tied-embed gate, imatrix coverage,
   F2 expansion).
3. **Postmortem** — what we learned during the env-gated prototype
   (4b0693d6 → 53e613fc) before the CLI landed on master.
4. **Vision encoder Phase 3** — still unshipped; design notes preserved.

Empirical anchor: **9B MQ4-AWQ-GPTQ-F2 + lm_head MQ4-AWQ** achieved
**KLD 0.0841** on n=256 q8-KV against the BF16 reference (A100 cloud
run 2026-05-18). The companion 27B run produced
`qwen3.6-27b.mq4-awq-gptq-f2-lmhead-a100.hfq` (md5 `b7317e70…`, 14 GB).

---

## 1. CLI surface (master)

### Rust quantizer (`hipfire-quantize`)

```text
--lm-head-format <fmt>     Required to put lm_head/output in any non-Q8 format.
                           Choices: q8 (default), f16, mq4-awq, mq3-awq, ...
--awq                      Enable AWQ pre-scaling (F2 default; --awq-scope f1
                           restricts to input-side only).
--awq-alpha <α>            AWQ exponent. Default 0.55 (per F2 sweep on
                           gfx906/gfx1100/gfx1151).
--awq-formula <name>       rms / variance / mean-abs. Default rms.
--imatrix <path>           Imatrix sidecar — must contain output.weight when
                           --lm-head-format selects an AWQ-bearing format.
```

`--lm-head-format mq{3,4}-awq` additionally requires
`HIPFIRE_LM_HEAD_AWQ_UNSAFE=1` in the environment. The gate exists
because a `.hfq` produced with `--lm-head-format mq4-awq` against a
runtime that doesn't dispatch the AWQ-aware lm_head kernel **silently
corrupts logits** — Phase 2 fixed this on master, but the gate stays
as belt-and-braces. Once the runtime kernel-allow-list
(`DType::supports_awq_sidecar`) catches up to every dtype, the gate
can be deleted.

**Deprecated env aliases (still honored, emit a deprecation warning):**

- `HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ=1` — was the env-gated entry point
  during the prototype. `main.rs:4036-4046` accepts it as an alias for
  `--lm-head-format mq4-awq` and prints a one-shot deprecation line.
  Will be removed once all internal scripts are off it.

### Python GPU pipeline (`scripts/gptq_gpu.py`)

```text
--lm-head-format {mq3-awq, mq4-awq}
                           Optional. When set, lm_head.weight /
                           output.weight join the eligible set and are
                           GPTQ-packed at --bits + AWQ-pre-scaled.
                           Must agree with --bits.
--bits {3, 4}              Per-tensor bit-width for the whole manifest
                           (matches the format in --lm-head-format).
--awq-f1-only              Restrict AWQ to F1 set. Default is F2.
--imatrix <path>           Imatrix GGUF; must cover output.weight when
                           --lm-head-format is set.
```

The Python pipeline produces a precomputed-gptq manifest consumed by
`hipfire-quantize --precomputed-gptq-path <dir>` — the Rust quantizer
emits the final `.hfq`, applying the same gates documented above.

### Combined workflow

```bash
# Stage A — Hessian collect (32 GB sidecar for 9B; multi-pass for 27B)
python scripts/collect_hessian.py \
    --model $HF_SNAPSHOT --output $HFHS_PATH \
    --corpus benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
    --n-sequences 128 --ctx-len 2048

# Stage B — imatrix (must include output.weight for lm_head AWQ)
cargo run --release --example imatrix_collect -- \
    --bf16-gguf $BF16_GGUF \
    --corpus benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
    --output $IMATRIX_PATH --process-output

# Stage C — GPTQ on GPU (CUDA or HIP/ROCm)
python scripts/gptq_gpu.py \
    --input $HF_SNAPSHOT --hessian $HFHS_PATH --imatrix $IMATRIX_PATH \
    --alpha 0.55 --bits 4 --lm-head-format mq4-awq \
    --output $MANIFEST_DIR

# Stage D — pack to .hfq
HIPFIRE_LM_HEAD_AWQ_UNSAFE=1 \
  hipfire-quantize --input $HF_SNAPSHOT --output $OUT_HFQ \
    --format mq4 --lm-head-format mq4-awq \
    --precomputed-gptq-path $MANIFEST_DIR
```

---

## 2. Safety guardrails

### 2.1 Tied-embedding hard-block

Tied-embedding models (Qwen 0.5B, Llama-3.2-1B, Mistral-7B-Instruct,
…) share the embedding matrix `embed_tokens.weight` with `lm_head`.
AWQ-prescaling `lm_head` then **corrupts the embedding lookup** —
`embed[token_id]` returns row × scale, producing token-position-
dependent garbage. Verified empirically on Qwen3-0.5B: KLD 0.67 →
13.5 (~20×) when this case was hit during prototyping.

The Rust quantizer reads `config.json:tie_word_embeddings` and
**hard-aborts at startup** if the field is missing or true while
`--lm-head-format mq{3,4}-awq` is selected. The check is intentionally
strict — early prototype `unwrap_or(false)` silently allowed missing
field, masking a real risk (hardened in dbcb050a). All Qwen3.5 / 3.6
models we've shipped against have `tie_word_embeddings: false`.

### 2.2 Imatrix `output.weight` coverage

AWQ on `lm_head` needs activation statistics for the final hidden
state. `imatrix_collect --process-output` opts into this — older
imatrices do NOT have the entry (substring grep for
`output.weight.in_sum2` is misleading because it also matches
`attn_output.weight.in_sum2` per-FA-layer attention projections).

The quantizer checks for the **specific** entry under both the direct
imatrix path AND the precomputed-gptq path (53e613fc fixed a silent-
skip bug in the Python eligibility check + Rust gate). If missing, it
aborts with a regenerate hint.

Post-2026-05-19, `benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.
imatrix.gguf` (5.17 MB) and `qwen3.6-27b-bf16.imatrix.gguf` (13.66 MB)
both carry the entry. The old 5.15 / 13.64 MB versions are pre-flag
and will fail the gate.

### 2.3 F2 expansion (output-side projections)

AWQ is by default applied to **both** input-side projections (F1: q,
k, v, qkv, gate, up, router) **and** output-side projections (F2:
o_proj, wo, out_proj, down_proj, w_down). F2 was justified by paired-
t NLL on 9B (master-doc §6 rule 9), and `--awq-scope f1` exists for
A/B comparison only.

`lm_head` semantically belongs to F1 (it's a projection whose
"activation" is the final hidden state, post-RMSNorm). Both the Rust
`awq_eligible` and Python `awq_eligible(..., include_lm_head=True)`
add it to the F1 set when the flag is on.

---

## 3. Math contract (runtime ↔ quant-time)

The runtime kernel for any AWQ-pre-scaled tensor multiplies the
**inverse** scale into the activation before the matmul:

```text
W' = W * s        (quant-time, per output channel)
x' = x / s        (runtime, per output channel)
y  = W' @ x'      ≡  W @ x   (mathematically equivalent)
```

For `lm_head`, the activation x is the post-RMSNorm final hidden
state, and the multiplier dimension is `vocab_size` (the output
dim of `lm_head`, which is `vocab_size × hidden`). Specifically:

- `awq_scale` shape: `[vocab_size]` F16, keyed `lm_head.weight.awq_scale`
- Geo-mean-normalized to 1.0 (so the quantized weights stay in their
  original dynamic range; only the per-channel relative scaling shifts).
- Runtime path: the post-norm `x` of shape `[batch, seq, hidden]` is
  matmul'd against `lm_head` of shape `[vocab, hidden]`; the
  per-output-channel inverse-scale is applied to `x` (broadcast along
  hidden) before the matmul, or to the output logits after — both are
  algebraically equivalent. The runtime allow-list
  (`DType::supports_awq_sidecar`) controls which dtype variants of
  the lm_head kernel actually consume the sidecar.

---

## 4. Postmortem — env-gated prototype era

Pre-mega-integration the work was env-gated:

| Commit | Effect |
|---|---|
| ccb9fb20 | Design plan + math contract (this doc, v1) |
| 4b0693d6 | `HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ` env + UNSAFE gate |
| 50077a4f | Imatrix-coverage gate (verify output.weight in_sum2) |
| dbcb050a | Tied-embed hard-block uses explicit match (no `unwrap_or(false)`) |
| 82d6b865 | MQ3 + `--precomputed-gptq-path` emits AWQ sidecars |
| 2e06a649 | `GPTQ_TARGET_SUFFIXES` covers lm_head + vision encoder |
| 53e613fc | Silent-skip fix: Python `is_mq4g256_eligible` + Rust precomp gate |

Master now exposes the same surface as `--lm-head-format` +
`--awq*` flags. The env aliases remain as deprecated entry points
to avoid breaking in-flight runbooks; expect them to be removed
once the existing scripts on cloud boxes get rewritten.

Two non-obvious failure modes the prototype era flushed out:

1. **Substring-grep false-pass on imatrix coverage.** Early checks
   used `grep "output.weight"` which matched the per-FA-layer
   `attn_output.weight.in_sum2` (8 hits on 9B, 16 on 27B) AND the
   real `output.weight.in_sum2` entry indistinguishably. Fix: count
   the **exact** key `output.weight.in_sum2` — the regenerated
   imatrices add exactly +1 such entry over the legacy versions
   (+16544 bytes for 9B; +20608 bytes for 27B — the new tensor
   record + GGUF metadata overhead).

2. **`is_mq4g256_eligible` hardcoded `lm_head.weight → False`.**
   When the prototype set the env, only the Rust side noticed;
   the Python GPU pipeline silently fell through to the
   "lm_head excluded" branch. 9B run completed without GPTQ-ing
   lm_head — output looked fine because the missing AWQ-scaled
   tensor stayed at Q8 (no harm, but also no benefit). Fix:
   Python's `is_mq4g256_eligible` now reads `include_lm_head` from
   the `--lm-head-format` CLI flag, matching the Rust gate.

---

## 5. Phase 3 — vision encoder (still unshipped)

Multimodal Qwen3.5/3.6 models have a vision tower under
`model.visual.blocks.<N>.{attn.qkv, attn.proj, mlp.linear_fc1,
mlp.linear_fc2}` plus a merger MLP. These tensors are eligible by
shape for MQ4G256 but currently aren't recognized by `awq_eligible`'s
suffix patterns.

**Quant-side** is the small change: add the vision suffix patterns
to `awq_eligible` (Rust + Python `names.py`) and confirm the Hessian
collector covers them (it already does — `2e06a649` extended
`GPTQ_TARGET_SUFFIXES` to include `qkv`, `proj`, `linear_fc1`,
`linear_fc2`).

**Runtime-side** is the work: the visual-tower attention + MLP
dispatch paths need their own AWQ-aware kernels (analogous to the
text-tower's `rotate_x_mq_awq` and `fused_silu_mul_mq_rotate_awq`),
plus loader plumbing so the AWQ sidecar reaches the visual layers.
Estimated 3-5 days of engineering. Defer until there's a concrete
multimodal eval need — current 27B/9B Phase 1+2 wins are on the
text tower.

The hard-block remains: **do not extend `awq_eligible` to vision
tensors quant-side until the runtime kernels exist.** Without runtime
support, a vision-AWQ `.hfq` would mis-process the visual stream
exactly as lm_head did during the Phase 1-only window.

---

## 6. Reference: empirical results

| Run | Model | Recipe | Wall (A100) | KLD (n=256, q8-KV) | Notes |
|---|---|---|---|---|---|
| 2026-05-18 | Qwen3.5-9B | MQ4 + AWQ-F2 α=0.55 + lm_head MQ4-AWQ | ~3h B+C+D | **0.0841** | Champion |
| 2026-05-18 | Qwen3.5-9B | MQ3 + AWQ-F2 α=0.55 + lm_head MQ4-AWQ | ~2h C+D | 0.197 | 4 GB .hfq |
| 2026-05-18 | Qwen3.6-27B | MQ4 + AWQ-F2 α=0.55 + lm_head MQ4-AWQ | ~7h B+C+D | not yet measured | 14 GB .hfq, md5 b7317e70 |
| 2026-05-15 | Qwen3.5-4B | MQ3 + AWQ-F2 α=0.55 | gfx1151 | 0.197 | First MQ3+AWQ validation |
| 2026-05-15 | Qwen3.5-4B | MQ4 (no AWQ) baseline | — | 0.197 | RTN MQ3 collapses (master-doc §5) |

Upstream Kaden's "sub-0.10 KLD" recipe on iterative-awq-gptq
(2026-05-18 investigation) reports 0.1257 on 9B — our 0.0841 is 33%
better. Open question: F1 vs F2 alpha-sweep bisection, deferred to
the next 9B experiment cycle.

---

## 7. Reference: pointers

- `crates/hipfire-quantize/src/main.rs` — CLI parsing for
  `--lm-head-format`, `--awq*`; tied-embed gate at startup; imatrix
  coverage check; deprecated-alias handling at line 4036.
- `crates/hipfire-quantize/src/precomputed_gptq.rs` — manifest reader
  for the GPU-pipeline output.
- `scripts/gptq_gpu.py` — GPU pipeline (CUDA or HIP/ROCm) with the
  `--lm-head-format` flag mirroring the Rust CLI.
- `scripts/gptq_gpu_pkg/names.py` — `awq_eligible(..., include_lm_head=...)`.
- `scripts/collect_hessian.py` — `GPTQ_TARGET_SUFFIXES` covers
  lm_head/output (text tower) and `qkv`/`proj`/`linear_fc{1,2}`
  (vision tower).
- `docs/plans/gptq_cuda.md` §13 — vast.ai cloud runbook with the CLI
  workflow above.
- `crates/hipfire-runtime/src/dtype.rs` — `DType::supports_awq_sidecar`
  allow-list (the runtime's safety net for unimplemented dtypes).
