# AWQ on `lm_head` (and vision encoder) — design plan

**Status**: not started. Companion to `gptq_cuda.md`. Created 2026-05-18,
revised same-day after adversarial review caught math errors + a
critical sequencing gap.

**Branch**: off `feat/mq-v2-quant-format-cuda` post-MQ3-fix (HEAD `82d6b86`).
PR back when validation passes on at least 0.8B + one larger model.

**First target**: Qwen3.5-0.8B. Parity gate before any larger run. The
small-model spike in §1.5 is a **prerequisite to committing engineering
time** to the full implementation.

**Blocked on**: nothing immediate, but should not start until the 27B
NLL paired-t baseline (Q8-head .hfq, md5 `c0f1b9874b…`) is recorded as
the anchor.

## Mission

Extend the AWQ + GPTQ + MQ4G256 pipeline to cover **`lm_head`** (the
final logits projection). Today `lm_head` on untied models is
force-promoted to Q8 by `kmap_resolve_mode` (`main.rs:2150-2167`), sits
outside the GPTQ math entirely, and never sees AWQ pre-scaling. On
Qwen3.6-27B that's a 1.27 GB tensor stuck at Q8 — moving it to
MQ4G256+AWQ+GPTQ drops it to ~675 MB (4.25 bpw effective, **not** the
2.1 bpw I claimed in v1 — MQ4G256 is 136 B per 256 weights, full math
in §4.0). Real savings: ~595 MB.

Also in scope as **Phase 3**: vision encoder weights
(`model.visual.blocks.<N>.attn.*`, `mlp.linear_fc*`) — currently
GPTQ-eligible by shape but NOT AWQ-eligible because the naming doesn't
match `awq_eligible`'s suffix patterns. Same fix shape; the heavy
lifting is the runtime AWQ-aware kernel dispatch, not the quant-time
changes. **HARD-BLOCKED on runtime kernel work** — see §3.3.

Explicitly **out of scope**: embeddings. They're a lookup, not a matmul.
`embed[token_id]` returns a row, never multiplies an activation. AWQ's
math has no x to divide. Separate compression path (e.g. kmeans
codebooks) if we ever want sub-Q8 embeddings.

## 1. Anchor + premise

### 1.1 Acceptance anchor

Current 27B baseline (just shipped via the V100 pipeline, see
`gptq_cuda.md` §13):

```
Q8-head:    qwen3.6-27b.mq4-awq-gptq-f2-q8head-v100.hfq   md5 c0f1b9874b…   15.0 GB
```

Run `eval_hipfire` n=512 q8-KV on this once for the anchor numbers
(KLD, NLL, PPL). Record in `kld-measurements-master.md` §1.1k or
similar. THIS is the baseline; anything this plan produces must be
acceptable against it.

### 1.2 Acceptance criterion (one-sided)

**Acceptance: NLL paired-t < 3** vs Q8-head anchor on per-chunk
n=512 q8-KV. Master-doc §6 rule 9: NLL paired-t is primary.

- **`paired-t < -3`**: variant B (AWQ on lm_head) significantly
  **better** than Q8 baseline → strict win, default-flip lm_head to
  MQ4+AWQ for new quants.
- **`|paired-t| ≤ 3`**: indistinguishable → acceptable. Save ~595 MB
  at no measurable quality cost; ship as opt-in default candidate.
- **`paired-t > 3`**: variant B significantly **worse** → fail. Keep
  Q8 lm_head default; flag opt-in for research-only via env var.

The v1 of this plan said "|t| < 3" which conflated those cases.
Acceptance is one-sided: only `t > 3` is a fail.

### 1.3 Prior art / why this might (or might not) work

The original AWQ paper (Lin et al. 2306.00978) benchmarks on
transformer-block FFN/attention. It does **not** make a claim about
lm_head specifically. The activation-statistics-based scaling argument
generalizes in principle (lm_head's input has channels with vastly
different magnitudes, same as a `down_proj` would), but:

- The M dimension on lm_head is **vocab size** (248320 on Qwen3.5/3.6).
  At that M, per-row quant grids span an order-of-magnitude more rows
  than transformer-block tensors (M ≈ hidden ≈ 5120). The grid-fitting
  math is still per-256-block on K, but row count affects total grid
  storage and may interact with the OBS propagation in ways untested
  at scale.
- An empirical 4-bit lm_head probe already exists in the codebase
  (`main.rs:4209-4232`, the `HIPFIRE_QUANTIZE_LM_HEAD_*` env gates —
  the "hypothesis #4" comment chain). The history suggests **4-bit
  lm_head without AWQ visibly degrades generation** on some prompts.
  The bet of this plan is that AWQ recovers what 4-bit alone gave up.

Prior probability this works **at acceptance threshold**: ~50-60%
(plausible from AWQ theory + analogous transformer-block evidence
within this codebase). Prior probability of a strict win
(`paired-t < -3`): ~25%.

### 1.4 Plan B if AWQ-on-lm_head doesn't help

If validation fails (`paired-t > 3`), the fallbacks in order of
preference:

1. **MQ6 lm_head** — env-gated path already wired
   (`HIPFIRE_QUANTIZE_LM_HEAD_MQ6=1`). 6-bit + per-block grid is a
   middle ground between MQ4 quality risk and Q8 storage. Promote6
   storage ≈ ~990 MB on 27B (~6 bpw equivalent). Saves ~280 MB vs Q8;
   easy ship if it passes the gate.
2. **Mixed precision lm_head** — top-K high-magnitude rows at F16,
   rest at MQ4G256+AWQ. Vocab size means a few hundred high-magnitude
   tokens dominate logits anyway; preserving those at full precision
   while quantizing the long tail could be the right shape. **Not yet
   designed**; would need code to identify high-mag rows and pack
   mixed.
3. **Just ship Q8 forever** — the boring answer. Live with the +600
   MB on 27B-class models.

### 1.5 Small-model spike (PREREQUISITE — do FIRST)

Before committing to the multi-day implementation in §3-§5:

1. Spend **~1 day on Qwen3.5-0.8B only**. The 0.8B Hessian sidecar
   already exists at
   `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.hessian.bin`.
   Patch `collect_hessian.py` to add lm_head, re-collect the 0.8B
   Hessian (~30 min). Patch quant + add the simplest possible runtime
   path (standalone elementwise divide between final norm and gemv,
   §3.2 option A). Run NLL paired-t on the 0.8B `.hfq`.

2. **Decision gate**: if 0.8B `paired-t > 3` → halt, fall through to
   §1.4 Plan B. If 0.8B `paired-t < 3` → proceed to full
   implementation on 9B + 27B.

This costs 1 day to derisk the 5-day plan. Skip it and you're betting
the larger work on a hypothesis we haven't tested.

## 2. Current state — what's in, what's missing

Mapping the existing scaffolding against what lm_head needs:

| Layer | Status | Gap |
|---|---|---|
| **Imatrix** | llama-imatrix dumps all matmuls — need to **verify** the 27B imatrix has `output.weight` | First action: `strings benchmarks/quality-baselines/refs/qwen3.6-27b-bf16.imatrix.gguf \| grep -E '(output\|lm_head)'` — see §4.1 |
| **Hessian** (`collect_hessian.py`) | `GPTQ_TARGET_SUFFIXES` excludes lm_head; line 117 comment explicitly: "lm_head, embed_tokens, top-level norms, vision encoder" | **Add** `"lm_head"` / `"output"` to `GPTQ_TARGET_SUFFIXES`, re-run Stage B |
| **AWQ whitelist (Python)** `gptq_gpu_pkg/names.py:awq_eligible` | F1 + F2; no `lm_head` / `output` | **Add** to F1 (lm_head is input-side semantically) |
| **AWQ whitelist (Rust)** `main.rs:awq_eligible` | Same; same omission | **Same** addition |
| **`kmap_resolve_mode` lm_head rule** `main.rs:2150-2167` | Hard-codes `Q8` for `lm_head` / `output.weight`. Env-gated diagnostics: `_F16=1`, `_Q8=1`, `_MQ6=1`. No MQ4+AWQ option | **Add** new mode (CLI flag or env), default behavior unchanged |
| **Precomputed-gptq-path dispatch** `main.rs:4447+` (just patched for MQ3 — commit `82d6b86`) | Handles AWQ sidecar emission for MQ4G256 / MQ3G256 weights | **No code change** once `kmap_resolve` allows lm_head to flow through — verify with test |
| **Runtime weight loading** `hipfire-runtime/src/hfq.rs:540` | `load_awq_scale()` only called for `DType::MQ4G256` / `MQ6G256` | **No code change** — fires automatically once lm_head is stored as MQ4G256 |
| **Runtime AWQ-aware kernels** `kernels/src/*_mq_rotate_awq.hip` | 3 kernels cover transformer-block input/output sides. **None target final-norm → lm_head** | **New dispatch** at lm_head call sites — see §3 |
| **Runtime lm_head call sites** `llama.rs:1336, 1597, 2414, 2526, 2549, 2671, 2800` (per current grep) | Existing pattern: `rmsnorm_f32` then `weight_gemv`. No AWQ divide between them | **Insert** AWQ-aware path before gemv at all sites |
| **Tied-embed detection** | None | **New** — see §3.4 |

## 3. Design

### 3.1 Phase 1 — quant-time wiring (opt-in)

The MQ3 commit `82d6b86` taught us: extending an existing dispatch
gate to a new format is one diff. Same shape here:

1. **`collect_hessian.py`**: extend `GPTQ_TARGET_SUFFIXES`:
   ```python
   GPTQ_TARGET_SUFFIXES = (
       # ... existing ...
       # lm_head (HF dense + multimodal naming, GGUF twin)
       "lm_head", "output",
   )
   ```
   This is a substring of the LAST component of the module name, so
   `"lm_head"` matches `"model.lm_head"` and `"model.language_model.lm_head"`,
   while `"output"` matches the GGUF flat-naming `"output"`. Verify
   with a unit test before re-running Stage B.

2. **`awq_eligible` (Python + Rust)**: add to F1:
   ```python
   # Python
   or safetensors_name.endswith("lm_head.weight") \
   or safetensors_name == "output.weight"
   ```
   ```rust
   // Rust mirror
   || name.ends_with("lm_head.weight")
   || name == "output.weight"
   ```

3. **`kmap_resolve_mode`** (`main.rs:2152`): add a new env gate
   alongside the existing `_F16` / `_Q8` / `_MQ6` ones:
   ```rust
   } else if (name.contains("lm_head") || name.ends_with("output.weight"))
       && std::env::var("HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ").ok().as_deref() == Some("1")
   {
       // Fall through to base MQ4 + AWQ pipeline — only safe when
       // (a) source model is NOT tied-embedding AND (b) runtime has
       // the AWQ-aware lm_head dispatch (Phase 2). Quantizer asserts
       // (a); runtime detection enforces (b) — see §3.4.
       return QuantLevel::Base;   // (the actual base level, not Q8)
   }
   ```

4. **Same gate in the precomputed-gptq path** (`main.rs:4209+`),
   matching the existing `_F16` / `_Q8` / `_MQ6` chain.

5. **Tied-embed assertion** (runtime check at quantize start) — see §3.4.

### 3.2 Phase 2 — runtime: AWQ-aware lm_head dispatch (BLOCKING)

**Phase 1 alone produces an unusable `.hfq`.** If the runtime doesn't
apply `x / s` between the final norm and the lm_head gemv, the
AWQ-pre-scaled weights produce `(W·s)·x ≠ W·x` — the exact corruption
master-doc §6 rule 5 warns about (KLD 0.67 → 13.5 on 0.8B Qwen3.5
measured when this guard was missing). Phase 1 and Phase 2 **must
ship as a single PR**.

Safety guard: **gate the Phase 1 quantizer flag behind a second env
var until Phase 2 lands**:

```rust
if env::var("HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ").is_ok()
    && env::var("HIPFIRE_LM_HEAD_AWQ_UNSAFE").as_deref() != Ok("1")
{
    eprintln!("error: HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ requires \
              HIPFIRE_LM_HEAD_AWQ_UNSAFE=1 until runtime support lands. \
              Setting only the first env produces .hfq files that the \
              current runtime cannot consume correctly.");
    std::process::exit(2);
}
```

Drop the `_UNSAFE=1` requirement in the same commit that lands Phase 2.

**Dispatch design — single option, no flip-flop**:

Reuse `kernels/src/rotate_x_mq_awq.hip` (already exists for o_proj /
out_proj input prep). It does FWHT-256 rotation + AWQ divide on x.
For lm_head:

```rust
// Pseudocode at each of the ~7 lm_head call sites in llama.rs:
gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, eps)?;
if let Some(awq) = weights.output.awq_scale.as_ref() {
    gpu.rotate_x_mq_awq(&scratch.tmp, awq, &scratch.tmp_rotated)?;
    weight_gemv(gpu, &weights.output, &scratch.tmp_rotated, &scratch.logits)?;
} else {
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
}
```

(The `rotate_x_mq_awq` call subsumes the rotation that the MQ4G256
gemv kernel currently expects on its input — for Q8 storage there's
no rotation, hence the branch.)

Performance: this is one extra kernel launch (FWHT-256 + divide on
hidden=5120) per decoded token. Sub-microsecond on 5070 Ti.
Optimization to a fused `fused_final_rmsnorm_rotate_awq` kernel is a
follow-up if profiling shows it on the critical path; not needed for
correctness.

Refactor: rather than copy-paste the if/else at 7 sites, introduce a
helper `weight_gemv_with_optional_awq` or similar that encapsulates the
branch. Call sites become one line each.

### 3.3 Phase 3 — vision encoder AWQ (FOLLOW-UP, HARD-BLOCKED)

Vision tensors currently get GPTQ but no AWQ because their naming
(`attn.qkv.weight`, `mlp.linear_fc1.weight`) doesn't match the
whitelist. Adding them to `awq_eligible` looks like a one-line diff
per side.

**DO NOT DO THIS without the runtime kernel work first.** This is the
exact pattern from `awq_fix_claude.md`: pre-Stage-2 of the AWQ history,
output-side projections (o_proj / down_proj) got added to the
whitelist before their AWQ-aware kernels existed, producing
catastrophic logit corruption on 0.8B (KLD 0.67 → 13.5). The F2 fix
that landed `rotate_x_mq_awq.hip` and `fused_silu_mul_mq_rotate_awq.hip`
was the proper sequence: kernels first, whitelist second.

For vision Phase 3:

1. **Audit** the vision-tower forward path in `hipfire-runtime`.
   Identify which kernels run the activation prep for each vision
   linear layer.
2. **Implement** the AWQ-aware kernel variant(s) for each pattern.
   This may be reusable from existing language-model kernels if the
   activation flow matches (RMSNorm → rotate → gemv pattern).
3. **Validate** with 0.8B-equivalent vision-tower probe — generate
   per-tensor activation magnitudes, check whether AWQ scales make
   sense (geo-mean normalized, magnitude variance bounded).
4. **Then and only then** patch `awq_eligible` to include vision
   tensor names.

Estimated effort for vision Phase 3 alone: **3-5 days** depending on
how much the vision-tower kernel set diverges from the language model.
Defer until Phases 1+2 ship and Phase 1 validates positively.

### 3.4 Tied-embed detection (mandatory safety)

Some models tie `lm_head` to `embed_tokens` (the same physical tensor
serves both). **Confirmed mid-Qwen-3.5 family variance**:

| Model | Tied? |
|---|:---:|
| Qwen3.5-0.8B | unknown — check |
| Qwen3.5-4B | **YES** (verified via Stage D output — only one `[vocab, hidden]` tensor; total params 4.2B not 4.85B) |
| Qwen3.5-9B | unknown — check |
| Qwen3.6-27B | NO (separate tensor, verified via Q8-head quant) |

For a tied-embed model, AWQ-pre-scaling `lm_head` would ALSO scale
the embedding lookup: `embed[token_id]` would return `s · row` instead
of `row`. This corrupts every transformer-block input.

**Quantizer-side detection** (cheap, deterministic):

```rust
// Before applying AWQ to lm_head, check the safetensors index for
// duplicate tensor entries pointing to the same data range.
fn is_tied_embedding(safetensors_index: &SafetensorsIndex) -> bool {
    let lm_head = safetensors_index.find_tensor("lm_head.weight")
        .or_else(|| safetensors_index.find_tensor("output.weight"));
    let embed = safetensors_index.find_tensor("embed_tokens.weight")
        .or_else(|| safetensors_index.find_tensor("model.embed_tokens.weight"));
    match (lm_head, embed) {
        (Some(lh), Some(emb)) => lh.data_offset == emb.data_offset,
        (None, Some(_)) => true,   // lm_head missing → must be tied
        _ => false,
    }
}
```

If detected, **abort** with a clear error when
`HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ=1`:

```
error: model has tied embeddings (lm_head shares storage with embed_tokens).
       AWQ-pre-scaling lm_head would corrupt the embedding lookup. Refusing
       to proceed. To force, you'd need to untie first (separate physical
       tensor) — out of scope for this flag.
```

Also fall through gracefully when the env is NOT set — the existing Q8
default already handles tied-embed correctly (embed_tokens is Q8 by
its own rule, lm_head inherits as the same tensor).

## 4. Math + sharp edges

### 4.0 Storage math (corrected from v1)

Per Qwen3.6-27B (vocab 248320, hidden 5120; lm_head shape [248320, 5120]):

| Format | Bytes per weight | Total | vs BF16 |
|---|---:|---:|---:|
| BF16 | 2 | 2.54 GB | 1× |
| Q8_F16 (current default) | 1 + scale overhead | ~1.27 GB | 0.5× |
| **MQ4G256 (this plan)** | 136/256 = 0.531 | **~675 MB** | **0.27×** |
| AWQ sidecar (additional) | 2 × K | 10.2 KB | negligible |

Savings vs Q8 baseline: **~595 MB**. (v1 of this plan claimed
"~330 MB" / "~950 MB savings" — both wrong by ~2×; corrected here.)

Whole-`.hfq` projection for 27B post-AWQ-lm_head: **~14.0-14.1 GB**
(starting from the Q8-head 15.0 GB minus ~595 MB plus 10 KB sidecar).

### 4.1 Imatrix coverage verification — HARD BLOCK (implemented)

**v1 of this plan suggested a `strings | grep` check; that was wrong.**
GGUF stores tensor names in length-prefixed binary fields, so plain
substring search collides with `attn_output.weight.in_sum2` (FA output
projection per layer) and produces a false-pass for `output.weight`.

Empirical audit on 2026-05-18 found **all four local imatrix files
have ZERO lm_head/output coverage**, despite `grep "output.weight"`
returning non-zero counts:

| Imatrix | `.in_sum2` entries | lm_head entries | attn_output entries |
|---|---:|---:|---:|
| 0.8B | 186 | 0 | 6 |
| 4B | 248 | 0 | 8 |
| 9B | 248 | 0 | 8 |
| 27B | 496 | 0 | 16 |

**The right structural check** is parsing the GGUF tensor name list:

```python
import struct
with open(path, 'rb') as f:
    data = f.read()
# ... parse header → tensor names ...
has_output = any(n == "output.weight.in_sum2" for n, _ in names)
```

Or — preferable — **let the quantizer enforce it at startup**.
Implemented in `main.rs` (lines ~3920+) alongside the tied-embed and
UNSAFE-gate aborts: when `HIPFIRE_QUANTIZE_LM_HEAD_MQ4_AWQ=1` is set,
the quantizer aborts (exit 2) if the loaded imatrix doesn't carry
`output.weight`. Error message points the operator at
`imatrix_collect --process-output` for regeneration.

**Cause of the missing coverage**: `llama-imatrix`'s default skips
lm_head/output to save calibration time (it's a vocab-scale tensor —
expensive to accumulate). The hipfire wrapper at
`crates/hipfire-runtime/examples/imatrix_collect.rs` line 95+ exposes
`--process-output` to opt in. None of the shipped imatrices in
`benchmarks/quality-baselines/refs/` were generated with it.

**Fix for the cloud-box A100 run**:

```bash
cargo run --release --example imatrix_collect -- \
    --bf16-gguf <model.bf16.gguf> \
    --corpus benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
    --output <new.imatrix.gguf> \
    --process-output
```

Wall: ~30-60 min on A100 80 GB for 9B-class; ~2h for 27B. Cheap vs
Stage B's 3-4h, and it's a one-time fixed cost — the new imatrix file
ships back to the repo and replaces the existing one for all future
runs.

### 4.2 lm_head Hessian: high-vocab, K=hidden

For Qwen3.6-27B: lm_head Hessian is `K² × 8 B = 5120² × 8 = 210 MB`
FP64 accumulator. Negligible vs the existing 118 GB sidecar. Add one
hook in `collect_hessian.py`, single layer (no `.layers.<N>.` infix
so it lands in the "always-included" bucket).

**Cost to recollect**: per §6 effort table.

### 4.3 lm_head FWHT-256 + frozen-grids math

```
Shape: [M=248320, K=5120]
FWHT-256 per row: 20 blocks per row × 248320 rows = 4.97M blocks
Frozen-grid storage: 4.97M × 4 B (F16 scale + min) = ~20 MB
GPTQ packing: 4.97M blocks × 136 B (MQ4G256 block) = 675 MB
```

No special handling needed; same as transformer-block MQ4G256 just
with bigger M.

### 4.4 Q8 default exists for a strong reason

The kmap rule at `main.rs:2150` putting lm_head at Q8 is not just a
default — the comment chain in the env-gate region (lines 4209-4232)
references *"the 4-bit lm_head (default for dense MQ4 without
--kmap-dense)"* and treats `HIPFIRE_QUANTIZE_LM_HEAD_Q8=1` as a
**recovery** flag. Background: 4-bit lm_head without AWQ broke
generation coherence on some prompts during the engine-drift-floor
investigation (Phase 1c follow-up, May 2026). That's stronger than
"lossy" — it's a known coherence failure mode.

This plan's bet is that AWQ closes the gap that broke coherence.
If it doesn't, §1.4 Plan B is the answer.

### 4.5 lm_head naming variants

| Format | Name |
|---|---|
| HF safetensors, multimodal Qwen3.6 | `model.language_model.lm_head.weight` |
| HF safetensors, dense Qwen3.5 | `lm_head.weight` |
| GGUF | `output.weight` |

`names.py:56-57` already maps HF ↔ GGUF. The AWQ whitelist addition
must cover all three substrings/endings.

### 4.6 Other layers worth considering (per user)

The user asked about embeddings + vision. Verdicts:

- **Embeddings**: AWQ doesn't apply (lookup, not matmul). Out of
  scope. If sub-Q8 embed compression is wanted later, a separate plan
  (kmeans codebooks / rank-decomposed embedding) is the right shape.
- **Vision encoder**: in scope as Phase 3, hard-blocked on runtime
  kernel work (§3.3). Same payoff structure as lm_head (transformer-
  style attention/MLP, currently missed by whitelist naming).
- **Conv1d (DeltaNet `linear_attn.conv1d.weight`)**: already Q8 by
  the `q8_conv1d_default`. Conv1d AWQ semantics are unusual (kernel
  dim is small, "activation" is 4-step window). Skip.
- **Routers (`mlp.gate.weight`)**: already AWQ-eligible per F1. No
  action.
- **MTP head** (`mtp.layers.0.*`): already covered by suffix patterns
  in `awq_eligible`. No action.

## 5. Validation

### 5.1 0.8B parity gate (THE small-model spike from §1.5)

Same protocol as `gptq_cuda.md` §5.1. Quantize Qwen3.5-0.8B in two
variants:

- **Variant A (anchor)**: current Q8 lm_head default
- **Variant B (this plan)**: MQ4+AWQ+GPTQ lm_head, opt-in flag set

Compare:

1. **Numerical reproducibility check**: run gptq_gpu.py twice with same
   seed on lm_head. The post-OBS BF16 bytes should be bit-identical.
   Catches non-determinism bugs.
2. **`coherence-gate.sh`** on variant B `.hfq`. Hard fail: any
   structural attractor per `coherence-gate-dflash.sh` Tier 1+2
   thresholds.
3. **`eval_hipfire` n=512 q8-KV** on both. **Acceptance: `paired-t < 3`**
   (per §1.2).
4. **Token-level diff probe** at temperature=0: run 5-10 prompts on
   both variants, diff the first 200 tokens. < 5% position
   disagreement = fine. > 20% = red flag (coherence-gate may not
   catch subtle vocabulary drift).

If 0.8B fails: halt. Fall to §1.4 Plan B. Don't sink 27B time.

### 5.2 27B endpoint (only after 0.8B passes)

27B already has the 118 GB Hessian sidecar but it excludes lm_head.
Two paths:

**Path A — full Stage B re-run with lm_head added** (~5-10h on V100):
new sidecar replaces old. Wasteful but simple.

**Path B — targeted Hessian extension** (~30 min if implemented):
new `--only-tensor lm_head` flag on `collect_hessian.py` produces a
single-tensor Hessian, then merge into the existing sidecar via a
new `collect_hessian_merge.py` helper.

Path B is **new code that doesn't exist**. Implementing it is ~3-4h.
For one-off 27B run, Path A may be cheaper than the implement-Path-B
cost. **Recommend Path A for first 27B run; implement Path B as a
followup once we know the lm_head extension is going to be a
repeated operation across model sizes.**

Stage C re-run for the new lm_head tensor only: ~1 min (single tensor).
Stage D re-pack with `--lm-head-format mq4-awq` flag (or env): ~5 min
on V100.

Validation: same as 0.8B + coherence-gate-dflash for spec-decode
correctness (master-doc §6 covers the DFlash gate).

### 5.3 Per-tensor MSE outlier — track lm_head separately

lm_head's M is so much larger than any transformer-block tensor that
its MSE distribution may not be comparable. Track it as a separate
diagnostic line in Stage C output. Flag-loud if its MSE is more than
3× the median of LM-body tensors (NOT 10× — the threshold from
§5.3 of gptq_cuda.md was per transformer-block tensor variance; lm_head
deserves tighter scrutiny because its M-dim heterogeneity is novel).

## 6. Implementation effort (realistic)

v1 of this plan had wildly optimistic estimates (Day 1: 3h). Corrected:

### Day 1 — quant-time wiring + small-model Stage B (8-10h)

1. Patches per §3.1 (~3h)
2. Patch `collect_hessian.py` for `--only-tensor` if pursuing Path B
   for 27B (skip if going Path A) — **doesn't include the actual
   Stage B run time**
3. Run 0.8B Stage B with extended `GPTQ_TARGET_SUFFIXES` (~30 min)
4. Run 0.8B Stage C with new lm_head Hessian (~15 min)
5. Run 0.8B Stage D + smoke test that .hfq has the
   `lm_head.awq_scale.weight` sidecar (~5 min)
6. Add tied-embed detection per §3.4 (~2h)
7. Smoke fail on Qwen3.5-4B (verified tied) to confirm the abort
   works (~30 min)

### Day 2 — runtime dispatch (4-6h)

8. Audit `weight_gemv` call sites for lm_head — 7 sites in `llama.rs`
9. Implement `weight_gemv_with_optional_awq` helper or equivalent
10. Per-site refactor — 30 min × 7 ≈ 3.5h with testing
11. 0.8B end-to-end smoke (logits reasonable, no NaN/Inf)

### Day 3 — 0.8B validation gate (4-6h)

12. NLL paired-t vs Q8 anchor on 0.8B (~1h harness setup + 1h run)
13. Coherence-gate + token-level diff probe (~30 min)
14. Decision gate: pass → continue; fail → halt, write up Plan B
    decision in `kld-measurements-master.md`

### Day 4-5 — 27B run + validation (8-12h wall, mostly Stage B)

15. 27B Stage B re-run with lm_head added (~5-10h on V100 — depends
    on whether V100 is still up; vast.ai may need fresh instance)
16. 27B Stage C re-run for lm_head only (~1 min)
17. 27B Stage D re-pack with `--lm-head-format mq4-awq` (~5 min on V100)
18. 27B coherence-gate + NLL paired-t + DFlash gate (~2h)
19. Document results

### Day 6 — buffer + writeup (~4h)

Either consumed by debugging, or used for the Path B (`--only-tensor`)
implementation if multiple model sizes need lm_head extension.

**Total realistic: 5-7 days.** v1 of this plan said "3-5 days"; that
was contingent on Stage B re-run cost being free, which it isn't.

## 7. Risks

1. **Hessian for lm_head is near-singular** (high-M, low-K relative
   to vocab). Damping may need to be larger than the 0.01 default.
   Mitigation: track damp values per-tensor in Stage C; if lm_head
   needs damp >> 0.1, the Hessian is too ill-conditioned and we
   should fall to Path B (MQ6 lm_head).

2. **AWQ M-side asymmetry**: AWQ paper's evidence is on M ≈ hidden,
   not M ≈ vocab. At vocab-M, a few high-magnitude rows (rare tokens?)
   may dominate quant error. **Per-row MSE check** during Stage C —
   if more than 0.1% of rows have MSE > 10× median, AWQ alone may
   not be sufficient and we'd want the mixed-precision Plan B
   variant.

3. **Phase 1/2 ship together — release coordination risk.** A
   developer running quantize in isolation could produce a `.hfq`
   that triggers the runtime AWQ path on a runtime build that lacks
   it. **The `HIPFIRE_LM_HEAD_AWQ_UNSAFE=1` gate in §3.2 prevents
   accidental shipping.** Drop only in the runtime PR.

4. **Coherence-gate masks subtle drift.** Coherence-gate looks for
   attractors/structural failures. AWQ-on-lm_head could shift
   top-K probabilities without breaking coherence — same fluent
   output, different token choices vs Q8 baseline. The
   token-level-diff probe in §5.1.4 is the additional guardrail.

5. **Tied-embed detection edge cases.** If a future model uses an
   alternative tying scheme (e.g. transposed shared linear layer,
   or partial tying), the detection in §3.4 may miss it.
   Mitigation: log the tied-detection decision at quantize start
   (visible diagnostic), and add a `--force-untied` flag for
   override-with-warning if needed.

6. **Effort estimate is still optimistic.** Day 4-5 assumes the V100
   is up and reachable. If it isn't, +1-2 days for vast.ai instance
   bring-up.

## 8. What NOT to do

- **Do NOT make MQ4-AWQ lm_head the default** until at least 0.8B +
  9B + 27B all pass §5 validation independently. Q8 is the safe
  default.
- **Do NOT touch the embedding Q8 rule.** AWQ doesn't apply; the
  separate-compression-strategy investigation is its own plan.
- **Do NOT skip the small-model spike in §1.5.** That's the
  derisking gate.
- **Do NOT regenerate the full 118 GB 27B Hessian for the lm_head
  add** unless Path B isn't ready. Path A is wasteful but reliable;
  Path B is efficient but requires new code.
- **Do NOT extend `awq_eligible` to vision tensors without Phase 3
  runtime kernel work.** This is the exact failure mode of pre-F2
  output-side AWQ (`awq_fix_claude.md`).
- **Do NOT extend `awq_eligible` to lm_head without the
  `HIPFIRE_LM_HEAD_AWQ_UNSAFE=1` gate active**, until Phase 2 ships.

## 9. References

| File / Commit | What |
|---|---|
| `main.rs:2150-2167` | lm_head Q8 force-promotion + diagnostic env gates |
| `main.rs:4209-4232` | Precomputed-gptq-path lm_head env gates ("hypothesis #4" chain) |
| `main.rs:awq_eligible` | F1+F2 whitelist (Rust source-of-truth) |
| `scripts/gptq_gpu_pkg/names.py:awq_eligible` | F1+F2 whitelist (Python mirror) |
| `scripts/collect_hessian.py:GPTQ_TARGET_SUFFIXES` | Tuple excluding lm_head from Hessian collection |
| `kernels/src/rotate_x_mq_awq.hip` | The kernel we reuse between final norm and lm_head gemv (Phase 2) |
| `crates/hipfire-runtime/src/llama.rs` | The 7 lm_head dispatch call sites |
| `crates/hipfire-runtime/src/hfq.rs:540` | Where `load_awq_scale()` fires for MQ4G256 — automatic once lm_head storage is MQ4G256 |
| `docs/plans/gptq_cuda.md` §1.2, §2, §4.1-4.5, §5.1-5.2 | Math + sharp edges + validation protocol |
| `docs/plans/awq_fix_claude.md` | The pre-F2 KLD-blowup precedent that justifies §3.2's hard "ship together" rule |
| Commit `82d6b86` | MQ3 precomputed-gptq gate fix — same shape of bug we're avoiding here |
| Commit `9ca8d900` | F2 expansion — adding `rotate_x_mq_awq.hip` + `fused_silu_mul_mq_rotate_awq.hip` to make output-side AWQ safe. Vision Phase 3 should mirror this sequence |
