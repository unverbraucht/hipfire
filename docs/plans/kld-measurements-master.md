# KLD Measurements — Master Table

**Status:** Living document. Append new measurements; do not delete.
**Last updated:** 2026-05-13
**Owner:** hipfire eval
**Authoritative methodology:** [`issue-113-quant-quality-eval.md`](issue-113-quant-quality-eval.md)

This doc is the single place to look for every KLD/PPL number we've measured against the BF16 reference. Pull-out tables in other docs (`qwen35-mq4-quality-gap.md` §1.5, `awq_hipfire.md`, individual cohort `result-table.md` files) are derived views; this is the source-of-truth catalog.

---

## 0. Reading conventions

- **All hipfire rows are POST-RoPE-fix** (commit `1805d820` flipped halfsplit RoPE to default; 2026-05-12 evening) unless explicitly marked `PRE-FIX`. Pre-fix hipfire numbers are ~3–5× inflated due to the interleaved-vs-halfsplit RoPE pair-convention bug — see [`project_engine_drift_floor_decomposition.md`](../../memory/project_engine_drift_floor_decomposition.md).
- **Reference distribution:** `qwen3.5-{0.8b,9b}-bf16.kldref.bin` (built via `build_kld_ref` from llama.cpp `llama-perplexity --kl-divergence-base` over the BF16 GGUF). Slice: wikitext2-1024s-2048ctx (md5 `83b0205a`, 1175 chunks; quick-slice runs use `--max-chunks 512`).
- **KV mode is load-bearing for cross-engine comparison.** llama.cpp GGUF baselines were measured with its default FP16 KV cache (lossless). Hipfire defaults to `asym3` KV (lossy, ~0.07-nat penalty on 0.8B mq4-base; see §4). When comparing across engines, only compare rows with similar KV lossiness, OR explicitly compute the above-floor lift.
- **Above-floor KLD** = raw KLD − engine's Q8-weight floor (in matching KV mode). Surfaces the format/calibration cost stripped of engine drift + KV-quant cost.
- **Scoring mode:** all hipfire rows use `prefill` (canonical, see §5.3 of issue-113). GGUF rows use llama.cpp's native (`gguf` mode, not directly comparable to per-token).
- **Top-K=256 caveat:** all KLD values are lower bounds on true full-vocab KLD. Cross-variant ordering is reliable; absolute magnitudes are conservative. See issue-113 §2.

---

## 1. Qwen3.5-9B

### 1.1 Hipfire (post-RoPE-fix, gfx1100, KV=asym3)

Source: `benchmarks/quality-baselines/results/2026-05-12-cohort-post-rope-fix-9b/result-table.md`

| Variant | bpw | KLD (CI) | p99 | PPL | Above-floor |
|---|---:|---|---:|---:|---:|
| q8f16 | 8.50 | **0.1459** (0.1383–0.1541) | 13.576 | 9.795 | (floor) |
| mq4-base | 4.25 | 0.3376 (0.3263–0.3494) | 18.194 | 9.116 | 0.1917 |
| **mq4-awq** | 4.25 | **0.2800** (0.2697–0.2910) | 17.537 | 9.271 | **0.1341** |
| hfp4 | 4.50 | 0.4594 (0.4475–0.4720) | 19.279 | 11.511 | 0.3135 |
| mfp4 | 4.50 | 0.4653 (0.4535–0.4782) | 18.278 | 11.138 | 0.3194 |
| hfp4-l4-l5c | ~5.0 | 0.3836 (0.3722–0.3959) | 18.665 | 10.299 | 0.2377 |
| mfp4-l4-l5c | ~5.0 | 0.7783 (0.7625–0.7951) | 21.199 | 12.571 | 0.6324 |

### 1.2 llama.cpp GGUF anchors (gfx1151, default FP16 KV)

Source: `benchmarks/quality-baselines/results/2026-05-10/per-seq/qwen3.5-9b.gguf-*__gfx1151.kldseq`

| Variant | bpw | KLD | PPL | Above-floor |
|---|---:|---:|---:|---:|
| Q8_0 | 8.50 | **0.0163** | 9.31 | (floor) |
| UD-Q6_K_XL | ~6.7 | 0.0213 | 9.14 | 0.0050 |
| Q6_K | 6.56 | 0.0250 | 9.31 | 0.0087 |
| UD-Q5_K_XL | ~5.5 | 0.0408 | 9.27 | 0.0245 |
| **UD-Q4_K_XL** | 5.32 | **0.0670** | 9.34 | 0.0507 |
| Q4_K_M | 5.07 | 0.1249 | 8.70 | 0.1086 |
| **UD-Q3_K_XL** | ~4.50 | **0.1411** | 8.67 | 0.1248 |

### 1.3 9B cross-engine summary

- **Engine-drift floor gap (Q8 vs Q8):** hipfire 0.146 vs llama.cpp 0.016 → **+0.13 nats hipfire-side**. Mostly from asym3 KV + cumulative pipeline imprecision per the floor decomposition; see §4 for KV-mode normalization data.
- **Bpw-matched 4-bit comparison (hipfire mq4-awq @ 4.25 vs llama.cpp UD-Q3_K_XL @ ~4.50):** above-floor 0.134 vs 0.125 → **~7% behind despite 0.25 bpw less**. PPL: 9.27 vs 8.67 → +6.9% worse PPL at matched bpw. Down from ~53% above-floor gap on mq4-base (no AWQ).
- AWQ closed roughly 30% of mq4's above-floor cost: 0.192 → 0.134 (= −30%).

---

## 2. Qwen3.5-0.8B

### 2.1 Hipfire (post-RoPE-fix, gfx1100, KV=asym3)

Source: `benchmarks/quality-baselines/results/2026-05-13-cohort-post-rope-fix-0.8b/result-table.md`

| Variant | bpw | KLD (CI) | p99 | PPL | Above-floor |
|---|---:|---|---:|---:|---:|
| q8f16 | 8.50 | **0.1256** (0.1234–0.1280) | 1.458 | 19.633 | (floor) |
| mq4-base | 4.25 | 0.3341 (0.3308–0.3374) | 2.707 | 23.895 | 0.2085 |
| **mq4-awq** | 4.25 | **0.3000** (0.2971–0.3029) | 2.515 | 23.383 | **0.1744** |
| hfp4-l4-l5c | ~5.0 | 0.7890 (0.7785–0.7999) | 5.671 | 43.407 | 0.6634 |
| mfp4-l4-l5c | ~5.0 | 0.7524 (0.7445–0.7603) | 5.259 | 38.829 | 0.6268 |

### 2.2 Hipfire q8-KV probes (gfx1100, KV=q8) — apples-to-apples-er with GGUFs

Source: `benchmarks/quality-baselines/results/2026-05-13-cohort-post-rope-fix-0.8b/per-variant/*__kv-q8.kldseq`

| Variant | KLD (CI) | p99 | PPL | Δ vs asym3 |
|---|---|---:|---:|---:|
| mq4-base | 0.2675 (0.2650–0.2699) | 2.366 | 22.088 | −0.0666 (−20%) |
| mq4-awq | _running 2026-05-13 10:40_ | | | |
| q8f16 | not measured yet (deferred — kernel-perf work in flight) | | | |

### 2.3 llama.cpp GGUF anchors

No 0.8B GGUF anchor data measured yet against `qwen3.5-0.8b-bf16.kldref.bin`. A Q4_K_M GGUF run with q8 KV cache is in flight (user reported 2026-05-13 10:40, ETA ~10 min).

| Variant | KV mode | KLD | PPL | Status |
|---|---|---:|---:|---|
| Q4_K_M | q8 | _pending_ | | external run, expected 2026-05-13 ~10:50 |

### 2.4 0.8B cross-engine summary

Pending the in-flight Q4_K_M GGUF run. Once it lands, fill in §1.3-style analysis here.

---

## 3. Status of other model sizes

| Model | Hipfire status | GGUF status |
|---|---|---|
| Qwen3.5-4B | not yet quantized; queued (task #7) | not measured |
| Qwen3.5-A3B | not yet quantized; queued (task #8) | not measured |
| Qwen3.6-27B | BF16 ref dumped on gfx1151; not on gfx1100 (memory entry) | not measured |
| Qwen2.5-7B | kldref built; cohort historical only | not measured |

---

## 4. KV-cache-mode contribution

### 4.1 Measurements

| Model | Variant | asym3 KLD | q8 KLD | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| 0.8B | mq4-base | 0.3341 | 0.2675 | −0.0666 | −20% |
| 0.8B | mq4-awq | 0.3000 | _running_ | | |
| 0.8B | q8f16 | 0.1256 | _deferred_ | | |
| 9B | (any) | (asym3 only so far) | _not measured_ | | |

### 4.2 What this means for cross-engine comparison

- The original "asym3 vs Q8 = ~0.20 nats" estimate from the pre-RoPE-fix floor decomposition appears to overstate the penalty at 0.8B post-fix. On mq4-base 0.8B, the actual delta is **~0.07 nats**, i.e. about **20% of the raw KLD**.
- This means roughly 20% of the hipfire-vs-llama.cpp KLD gap at 4-bit is the KV-mode-choice confounder, **not** format quality.
- Post-fix the residual engine drift after subtracting both RoPE bug and KV asym3 noise is ~0.07-nat (cumulative pipeline imprecision over 24 layers) plus ~0.01 (Q8 weight + cross-engine inherent). See [floor decomposition memory](../../memory/project_engine_drift_floor_decomposition.md).
- **Recommendation:** when publishing hipfire-vs-llama.cpp tables, either (a) state both KV modes side-by-side, or (b) normalize via above-floor with the matching-KV-mode floor. Don't claim apples-to-apples without one of these.

### 4.3 Open KV-mode measurements to fill in

- [ ] 0.8B q8f16 + q8 KV (engine-drift floor under matching KV mode) — deferred until kernel-perf work finishes
- [ ] 0.8B mq4-awq + q8 KV — running now
- [ ] 9B q8f16 + q8 KV (engine-drift floor at scale)
- [ ] 9B mq4-base + q8 KV
- [ ] 9B mq4-awq + q8 KV

A four-row 9B strip (q8f16, mq4-base, mq4-awq @ q8 KV + one Q4_K_M GGUF @ q8 KV) would give us the cleanest apples-to-apples picture and is the highest-leverage next measurement after the 0.8B Q4_K_M GGUF lands.

---

## 5. Key findings catalog

- **2026-05-12 — RoPE fix flips defaults.** Halfsplit (HF convention) becomes default; interleaved relegated to `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1`. Drops 0.8B Q8 floor from ~0.49 to ~0.13. Commit `1805d820`.
- **2026-05-12 — AWQ on MQ4 (Stage A) closes 30% of 9B above-floor gap.** mq4-base 0.192 above-floor → mq4-awq 0.134 above-floor. AWQ implementation per `crates/hipfire-quantize/src/main.rs` AWQ block; calibration via `awq_collect.py`.
- **2026-05-13 — Engine-drift floor investigation closed.** No further single-localizable bugs in the post-RoPE-fix residual ~0.08 nats. Decomposed as 0.07 (cumulative pipeline imprecision, 24 layers) + 0.01 (Q8 weight + cross-engine inherent). Refuted GLM-5's F16-LA-weights claim along the way. See memory entry.
- **2026-05-13 — Cohort humaneval/smoke bug fixed.** `HIPFIRE_DEFAULT_MODEL` → `HIPFIRE_MODEL` in `scripts/bench_humaneval_completion.sh` and `scripts/quant_cohort.sh`. KLD/PPL/MSE data unaffected (eval_hipfire path doesn't use the daemon).
- **2026-05-13 — asym3 KV vs q8 KV delta is smaller than pre-fix estimate.** Empirically 0.07 nats at 0.8B mq4-base, not the 0.20 implied by the pre-RoPE-fix decomposition.

---

## 6. Methodology rules for adding rows

1. **Always disclose KV mode** (`asym3` / `q8`) in every row. No "default" hand-waving.
2. **Always disclose scoring mode** (`prefill` / `per-token` / `gguf`). Mixing is forbidden by issue-113 §5.3.
3. **Always link the source `.kldseq` file path** or the cohort directory it lives in.
4. **Bootstrap 95% CI** (10,000 resamples on per-seq means) — both `quant_cohort.sh` and the standalone q8-KV probe scripts emit this.
5. **For "above-floor" math:** subtract the Q8-weight floor measured under the SAME KV mode. Cross-KV-mode floor subtraction is meaningless.
6. **Flag PRE-FIX (pre-2026-05-12 RoPE) rows explicitly** if they need to be cited; otherwise omit them.
