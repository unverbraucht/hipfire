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
| mq4-awq | **0.2531** (0.2506–0.2556) | 2.305 | 22.149 | −0.0469 (−16%) |
| q8f16 | not measured yet (deferred — kernel-perf work in flight) | | | |

### 2.3 llama.cpp GGUF anchors (0.8B)

Run via `eval_gguf` against `qwen3.5-0.8b-bf16.kldref.bin`. **Full slice (1175 chunks), not the 512-chunk quick-slice used by the hipfire 0.8B rows.** The slice-mean KLD usually shifts <2% between 512 and 1175 chunks (wikitext slice converges fast), but flagged here so it can't be forgotten.

| Variant | bpw | KV mode | KLD | PPL | Chunks | Source |
|---|---:|---|---:|---:|---:|---|
| Q4_K_M | ~5.07 | q8 | **0.0351** | 17.334 | 1175 (full) | user-reported 2026-05-13 |

Q8_0 / UD-* anchors at 0.8B not yet measured.

### 2.4 0.8B cross-engine summary (after Q4_K_M anchor)

At matched 4-bit + KV mode (q8 vs q8):

| Engine | Variant | bpw | KLD | PPL | Gap to llama.cpp Q4_K_M |
|---|---|---:|---:|---:|---:|
| **llama.cpp** | Q4_K_M | 5.07 | **0.0351** | **17.33** | (anchor) |
| hipfire | mq4-base | 4.25 | 0.2675 | 22.088 | +0.232 nats KLD, +27.5% PPL at −0.82 bpw |
| hipfire | **mq4-awq** | 4.25 | **0.2531** | **22.149** | **+0.218 nats KLD, +27.8% PPL** at −0.82 bpw |

**AWQ uplift is KV-mode-dependent.** At asym3 KV, AWQ improved 0.8B mq4 by −10.2% KLD; at q8 KV it's only −5.4%. Interpretation: AWQ partially compensates for KV-rotation noise (asym3 K rotation precision interacts with per-channel outliers); when the KV cache is less lossy, AWQ has less to clean up. The 9B picture (where AWQ closed −30% above-floor at asym3) may show similar shrinkage if re-measured at q8 KV — worth flagging for the Stage A retrospective.

**PPL is roughly unchanged by AWQ on this slice.** mq4-base PPL 22.088, mq4-awq PPL 22.149 — within slice noise. KLD improvement comes from tail-distribution matching, not top-1 probability, which is exactly what AWQ targets (outlier preservation in heavy channels). Consistent with §2 of issue-113 ("PPL collapses the full output distribution... KLD surfaces tail").

The 0.8B picture is **substantially worse than 9B** for hipfire's MQ4
relative to llama.cpp:

- **9B mq4-awq vs UD-Q3_K_XL @ matched bpw:** +6.9% PPL gap
- **0.8B mq4-base vs Q4_K_M @ near-matched bpw (−0.82 bpw):** +27.5% PPL gap
  → mq4-awq will narrow this but unlikely to close even half of it
  given the 9B AWQ improvement was 30% above-floor.

Two interpretations (need data to disambiguate):

1. **Small-model penalty on the MQ4 format.** Smaller models concentrate more information per parameter; per-256 single-FP32-scale (no sub-32 scales, no weighted-LS fit) loses more relative precision on dense small-model weights. The structural format-level disadvantage that's 7% at 9B might be 25%+ at 0.8B.
2. **Engine-drift contribution doesn't scale uniformly.** The 0.8B q8f16 asym3 KLD is 0.126 (close to 9B's 0.146); the 4-bit gap inflates because the format-level cost compounds with engine-drift differently across scales.

The clean disambiguation needs:
- 0.8B q8f16 + q8 KV (to give the matching-mode floor → strip engine drift cleanly)
- 0.8B Q8_0 GGUF at q8 KV (to give llama.cpp's matching-mode floor)
- 0.8B GGUF UD-Q3_K_XL or UD-Q4_K_XL at q8 KV (for a bpw-matched 4-bit peer instead of Q4_K_M's 5.07 bpw)

**Methodological note: slice mismatch (full vs quick).** The hipfire 0.8B rows use `--max-chunks 512`; the Q4_K_M GGUF was full-slice. Re-running mq4-base + mq4-awq on full slice (~30 min wall each on this gfx1100) would close the apples-to-apples residual; that's the cleanest cross-engine comparison we can produce post-RoPE-fix without waiting for the full anchor set.

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

## 4A. Open format-coverage measurement gaps

The current cohort runs only cover MQ4 / HFP4 / MFP4 family. Sub-4-bit
formats and the Lloyd codebook variants have NOT been measured against
the post-RoPE-fix BF16 reference. Per CLAUDE.md and the format roadmap,
several of these are theorized to deliver meaningful quality lifts and
need to be benched before strategic decisions get cemented.

### 4A.1 Sub-4-bit and Lloyd variants — unmeasured

| Format | bpw | Status on disk | Gated by | KLD against current ref? |
|---|---:|---|---|---|
| MQ3G256 (uniform 3-bit) | 3.25 | `~/.hipfire/models/qwen3.5-9b.mq3` exists | — | **NOT MEASURED** |
| MQ3G256-Lloyd (3-bit + per-block Lloyd-Max 8-entry FP16 codebook) | 3.50 | `~/.hipfire/models/qwen3.5-9b.mq3-lloyd` exists | — | **NOT MEASURED** |
| MQ4G256-Lloyd (4-bit + Lloyd codebook, prefill kernel) | ~4.5 | not quantized | PR [#197](https://github.com/Kaden-Schutt/hipfire/pull/197) (`feat/issue-182-mq4-lloyd`, open) | **NOT MEASURED** |
| MQ2G256-Lloyd (2-bit + Lloyd codebook) | ~2.5 | check `~/.hipfire/models/` | — | **NOT MEASURED** |

### 4A.2 Lloyd-transform uplift — investigation note

The Lloyd-Max per-block codebook (used in mq3-lloyd / mq4-lloyd /
mq2-lloyd) replaces the uniform 4/8/16-codepoint grid with K
data-driven centroids fit to each post-FWHT block's value distribution.
Theory: for post-FWHT weights with heavy-tailed kurtosis, a uniform
grid wastes codepoints on tail bins while under-resolving the dense
center. Lloyd codebooks should claw back precision proportional to
how non-uniform the post-FWHT distribution is.

**Expected uplift to measure (against current post-RoPE-fix ref):**
1. **MQ3 → MQ3-Lloyd at 9B.** Lloyd at 3-bit is theorized to be the
   bigger lever (MQ3 uniform is on the cliff of where 3-bit becomes
   unusable; codebook adaptation may rescue it). Direct A/B since both
   .hfq files already exist on disk.
2. **MQ4 → MQ4-Lloyd at 9B.** Gated on PR #197 merging or being
   checked out temporarily. Smaller theoretical uplift than MQ3-Lloyd
   (MQ4 uniform is already in the well-behaved zone) but worth a
   measurement to know the order of magnitude before committing
   format/kernel work.
3. **Stacking interaction with AWQ.** AWQ pre-scaling reshapes the
   per-channel value distribution before FWHT; whether Lloyd codebooks
   trained on post-AWQ post-FWHT blocks deliver the same uplift as on
   non-AWQ blocks is open. The Stage A → Stage B → Stage C roadmap
   currently treats AWQ + GPTQ + Lloyd as orthogonal levers; measuring
   them as additive vs interfering is part of the design validation.

**Highest-leverage measurement order:**
1. mq3 + mq3-lloyd at 9B (no PR gating, no quantize required, ~30 min wall)
2. mq3 + mq3-lloyd at 0.8B (need to quantize first; ~10 min quantize + ~15 min eval)
3. mq4-lloyd at 9B (gated on PR #197 — see below)
4. mq4-lloyd + AWQ stack at 9B (Phase A Step 5c follow-up)

### 4A.3 How to measure mq4-lloyd (gated on PR #197)

PR `Kaden-Schutt/hipfire#197` (`feat/issue-182-mq4-lloyd`, OPEN) ships
the MQ4-Lloyd WMMA prefill kernels. To measure mq4-lloyd KLD without
merging:

```bash
# Save current branch
git checkout -b backup-feat-mq-v2 feat/mq-v2-quant-format

# Fetch + check out the PR
gh pr checkout 197 -R Kaden-Schutt/hipfire

# Quantize a candidate
cargo run --release -p hipfire-quantize -- \
    --hf <bf16-path> --format mq4-lloyd \
    --out ~/.hipfire/models/qwen3.5-9b.mq4-lloyd

# Eval against the existing ref
./target/release/examples/eval_hipfire \
    --model ~/.hipfire/models/qwen3.5-9b.mq4-lloyd \
    --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
    --output benchmarks/quality-baselines/results/<dated>/qwen35-9b-mq4-lloyd__gfx1100.kldseq \
    --kv-mode asym3 --scoring-mode prefill --max-chunks 512

# Reduce + copy results back to feat/mq-v2-quant-format
# (the .kldseq file is engine-version-agnostic; row can be appended
#  to the master doc without re-running on the main branch)

# Restore working branch
git checkout feat/mq-v2-quant-format
```

The `.kldseq` artifact + reduced KLD/PPL values are portable across
engine versions (the slice is the slice; the reference is the
reference). Result row goes into §1 of this doc; flag the engine
SHA in the row's "Notes" field (e.g. "engine=PR#197@<sha>").

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
