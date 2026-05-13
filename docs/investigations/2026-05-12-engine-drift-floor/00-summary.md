# Engine-Drift Floor Investigation on Qwen3.5 (2026-05-12)

## Background

The hipfire eval pipeline measures KLD between hipfire's logits and a BF16 reference produced by `llama-perplexity` on `wikitext2-1024s-2048ctx.txt`. On Qwen3.5-9B with Q8 weights ("near-lossless quantization" — measured 2.63e-8 mean MSE, 113× lower than MFP4), eval was reporting a slice-mean KLD of **~0.57 nats** against the BF16 reference.

Literature Q8-vs-FP16 KLD is ~0.001–0.005 nats. 0.57 is "different model" territory. So the 0.57 was not Q8 quantization noise; it was *engine-vs-engine* drift between hipfire and llama.cpp. The investigation in this directory localizes the contributions, identifies one real bug (already fixed in PR #241), and rules out four other candidate hypotheses.

## Tooling shipped in this PR

### Hipfire-side dumps (env-gated, zero-overhead when unset)

* **`crates/hipfire-runtime/examples/dump_qwen35_hidden_states.rs`** — new example binary. Reuses the existing `HiddenStateRingBuffer` from spec-decode infra to dump every layer's post-residual output for every position in one chunk. Output is HFHS-v1 binary format (header + `[n_layers, n_pos, hidden_dim] f32` body).

* **`crates/hipfire-arch-qwen35/src/qwen35.rs`** — two helper functions + env-gated call sites inside `forward_scratch_layers`:
  * `HIPFIRE_DUMP_DN_INPUTS=<path>` + `HIPFIRE_DUMP_DN_LAYER=<idx>` (default 4) → dump Q/K/V/α/β at the input to `gated_delta_net_*` for the target LinearAttention layer.
  * `HIPFIRE_DUMP_A_RAW=<path>` + `HIPFIRE_DUMP_DN_LAYER=<idx>` → dump the post-w_alpha-projection `a` value BEFORE `fused_sigmoid_alpha_gate` fires (i.e., the input to the softplus/A_log/dt_bias transform).

### HF transformers oracle scripts

All Python, expect a `.venv` with `torch`, `transformers`, `safetensors`, `datasets`, `numpy`. Each script writes a binary that aligns with a hipfire dump for offline comparison.

* `scripts/dump_hf_hidden_states.py` — per-layer hidden-state oracle. Captures `output_hidden_states=True` from a forward; uses a `register_forward_pre_hook` on `model.model.norm` to capture the pre-final-norm last-layer output (transformers' `hidden_states[n_layers]` is POST-norm and that mismatches hipfire's HiddenStateRingBuffer capture point).
* `scripts/dump_hf_dn_inputs.py` — monkey-patches `chunk_gated_delta_rule` to capture its call args, then replays the in-rule `l2norm + 1/sqrt(D) scale` on Q/K so the output aligns with hipfire's post-`fused_qk_l2_norm_scale` state. Same record layout as `dump_dn_inputs`.

### Comparators

* `scripts/compare_hidden_states.py` — per-layer cosine + relative-L2 across all positions.
* `scripts/compare_layer_positions.py` — for a single layer, bucket the per-position drift across the sequence. Distinguishes recurrent state accumulation (monotonic growth) from constant input-amplification.
* `scripts/compare_dn_inputs.py` — per-tensor (Q/K/V/α/β) compare between hipfire and HF dn-input dumps.
* `scripts/cross_engine_check.py` — sanity check that the BF16 reference itself is consistent across engines. Extracts chunk-0 tokens from a `.kldref` (built by `llama-perplexity`), runs the same tokens through HF transformers, computes per-position top-K KL between llama.cpp's stored distribution and HF's.

### Quantize env-gates

In `crates/hipfire-quantize/src/main.rs`:

* `HIPFIRE_QUANTIZE_LM_HEAD_F16=1` — force `lm_head.weight` / `output.weight` to `QuantLevel::F16` regardless of K-map state.
* `HIPFIRE_QUANTIZE_LA_F16=1` — force all `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.weight` to `QuantLevel::F16`.

Both default off. Diagnostic only — they bloat the `.hfq` by the F16-vs-Q8 size delta on the affected tensors and are not load-bearing for any normal workflow.

### Probe-only kernels

* `kernels/src/rmsnorm.hip` — adds `rmsnorm_f32_f64acc` variant (fp64 accumulator in the parallel-reduction sum-of-squares + fp64 rsqrt). Dispatched by `rmsnorm_batched` when `HIPFIRE_RMSNORM_F64=1`.
* `kernels/src/gated_delta_net_f64acc.hip` (new) — same algorithm as `gated_delta_net_f32` but the per-token state tile is held in fp64 (4 KB → 8 KB shared memory). Dispatched by `gated_delta_net_f32` when `HIPFIRE_DELTANET_F64=1`.

## Findings (Q3.5-0.8B, per-token kv-q8, n=20 chunks)

### Confirmed bug — RoPE convention mismatch (FIXED in PR #241)

HF `transformers/models/qwen3_5/modeling_qwen3_5.py:573-579` uses `rotate_half` — pairs (i, i + n_rot/2), HALF-SPLIT convention. Plain Qwen3, Qwen2, llama.cpp, and vLLM all use the same convention. Hipfire's `kernels/src/rope_partial_interleaved.hip` used pairs (2i, 2i+1) — INTERLEAVED — and hipfire-quantize does NOT permute Q/K weights at quantize time. Every FullAttention layer applied the wrong rotation pairing to weights stored in half-split layout.

Fix: kernel sibling `rope_partial_halfsplit.hip` rotates the correct pairs. Default flipped to halfsplit in PR #241; legacy interleaved retained behind `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1`.

| Metric | Interleaved (BUG) | Halfsplit (FIX) | Change |
|---|---:|---:|---:|
| 20-chunk KLD | 0.4945 | 0.0806 | **-83.7%** |
| PPL | 33.20 | 18.56 | **-44%** |

PPL 18.56 is within ~5% of plain Qwen3-0.6B's intrinsic PPL (19.5) on the same slice. After the RoPE fix the floor is dominated by distributed pipeline imprecision (see below), not a single bug.

### Ruled out — KV cache quantization (probe c.1)

Re-ran the per-layer dump with `--kv-mode fp32` (no KV quantization), halfsplit default. Per-layer rel_L2 profile is essentially identical to the q8-KV run — layer 4 rel_L2 differs by only -0.011. KV-cache quant adds ~0.02 nats of cumulative mid-stack drift but does NOT explain the layer-4 LA jump.

### Ruled out — recurrent state precision (probe c.2)

Re-ran with `HIPFIRE_DELTANET_F64=1`. Output is byte-identical to the fp32-state run. Layer 4 rel_L2 still 0.140 mean, still grows from 0.064 (pos 0) to 0.188 (pos 1664-1792). fp32 accumulation has enough precision for the recurrence; promoting to fp64 changes nothing.

### Ruled out — QK-norm accumulation precision (probe)

Re-ran with `HIPFIRE_RMSNORM_F64=1`. Layer 4 rel_L2 = 0.1873 byte-identical to the fp32-acc run. With n=256 elements summed in fp32 the parallel-reduction precision was never plausibly the source; this probe confirms.

### Ruled out — Q8 weight quantization noise on LA layers

Re-ran with `HIPFIRE_QUANTIZE_LA_F16=1` (forces all `linear_attn.*` projection weights to F16 storage instead of Q8). Layer 4 rel_L2 = 0.128 vs Q8 baseline 0.140 — **a -0.012 shift only**. If quantization-noise amplification were the source, F16 weights (~16× less per-element noise than Q8) should have dropped the layer-4 drift to ≲ 0.02. They did not.

### Ruled out — lm_head Q8 vs F16 precision

Re-quantized with `HIPFIRE_QUANTIZE_LM_HEAD_F16=1`. Per-token kv-asym3 n=20 KLD: 0.6006 vs Q8-lm baseline 0.6004. Identical within numerical noise (per-sequence KLDs match to 4 decimal places). lm_head precision contributes < 0.001 nats.

### Ruled out — DeltaNet loader convention (sign / log / unit mismatch on A_log or dt_bias)

`A_log` and `dt_bias` safetensors values match byte-exactly between hipfire's `.hfq` and HF's safetensors (max abs diff 0). `in_proj_a` weight Q8 dequant matches HF BF16 to mean diff 2e-6. Per-head constants are correct.

### Final attribution — distributed pipeline imprecision

Direct chain-of-bias measurement at LinearAttention layer 4:

| Stage | HF mean | hipfire mean | delta |
|---|---:|---:|---:|
| Layer 3 output (= layer 4 residual input) | -0.001736 | -0.001700 | +3.6e-5 / dim |
| post-attn_norm × in_proj_a (raw `a`) | -2.093 | -2.054 | +0.039 |
| post `softplus(a+dt_bias) * (-exp(A_log))` (final α) | -0.332 | -0.344 | -0.012 |

The 3.6e-5 per-dim residual-stream bias is inherited from 4+ prior layers of slightly-imprecise compute. It gets amplified by ~24× through the layer-internal RMSNorm scale-gain, ~3× through softplus in its non-linear region, and accumulated over 2048 positions by the DeltaNet recurrence. Net result: ~0.04 nats of drift per LA layer at the layer output, and a final-output floor of ~0.08 nats.

**No single localizable bug** beyond the RoPE convention mismatch already fixed in PR #241. Each individual kernel is mathematically clean; small per-kernel deviations (accumulator order, fused-vs-separate-op rounding, gain stacking through norm weights) collectively produce a small per-dim residual-stream bias that compounds layer-by-layer.

## Recommendation

Accept the **post-RoPE-fix 0.08-nat floor** as the engine-vs-engine numerical-pipeline cost on Qwen3.5. Pushing below 0.08 would require a multi-day kernel-by-kernel accumulator audit. Marginal value is ~0.07 nats KLD reduction on a model whose post-fix floor is already within ~10× of plain Qwen3's intrinsic floor (0.0098).

Re-evaluate if a sensitive downstream test (PPL on a code corpus, HumanEval pass-rate at low temp, etc.) demonstrates the 0.08 actually matters in practice.

## Reproducer

```bash
# Per-layer dump (hipfire)
HIPFIRE_KV_MODE=q8 ./target/release/examples/dump_qwen35_hidden_states \
    --model ~/.hipfire/models/qwen3.5-0.8b.q8f16 \
    --ref   benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.kldref.bin \
    --chunk 0 --kv-mode q8 \
    --out   /tmp/q3.5-0.8b-hipfire-hidden-chunk0.bin

# HF oracle
.venv/bin/python3 scripts/dump_hf_hidden_states.py \
    --model <Qwen3.5-0.8B-safetensors-dir> \
    --ref   benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.kldref.bin \
    --chunk 0 \
    --out   /tmp/q3.5-0.8b-hf-hidden-chunk0.bin

# Per-layer compare
python3 scripts/compare_hidden_states.py \
    --hf      /tmp/q3.5-0.8b-hf-hidden-chunk0.bin \
    --hipfire /tmp/q3.5-0.8b-hipfire-hidden-chunk0.bin
```

For DN-input bisect on a chosen LA layer (default 4):

```bash
HIPFIRE_DUMP_DN_INPUTS=/tmp/dn-hipfire.bin HIPFIRE_DUMP_DN_LAYER=4 \
    ./target/release/examples/dump_qwen35_hidden_states \
        --model <hfq> --ref <kldref> --chunk 0 --kv-mode q8 \
        --out /tmp/discard-hidden.bin

.venv/bin/python3 scripts/dump_hf_dn_inputs.py \
    --model <safetensors-dir> --ref <kldref> --chunk 0 --layer 4 \
    --out /tmp/dn-hf.bin

python3 scripts/compare_dn_inputs.py --hipfire /tmp/dn-hipfire.bin --hf /tmp/dn-hf.bin
```
