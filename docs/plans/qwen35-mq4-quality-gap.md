# Hipfire quantization: where we are, what HFP4/MFP4 added, what's still missing vs Unsloth Dynamic, and the format roadmap

**Status:** consolidated design doc — supersedes prior revs of this file and `mfp4-vs-unsloth-dynamic-gap.md`.
**Date:** 2026-05-11
**Audience:** hipfire engineers settling on the next-years quantization format.
**Goal:** answer the single question — *"what's the current problem, and how do we get from here to a 4-bit format that beats Unsloth Dynamic 2.0 GGUFs in quality, on RDNA, across model architectures spanning Qwen2.5-VL through Gemma4?"*

The doc has three movements: (1) the journey MQ4 → HFP4 → MFP4 and the empirical noise wins per lever, (2) what's still missing relative to imatrix-calibrated dynamic GGUFs, with concrete per-lever quality estimates, (3) how the HFP4 wire format extends — without breaking changes — to absorb the missing levers, plus future-proofing for archs we haven't shipped yet.

---

## 1. The journey so far (per-lever noise wins)

### 1.1 Hipfire quantization-quality levers — taxonomy

Any 4-bit quant format trades five levers against bits-per-weight and dequant cost:

| Lever | What it controls | Implementation in MQ4 | Implementation in MFP4 | Implementation in Q4_K_M | Implementation in UD-Q4_K_XL |
|---|---|---|---|---|---|
| **L1 — element format** | Codepoint allocation (8 / 16 codes, uniform vs log-spaced) | Linear INT4 + zero-point | E2M1 FP4 (log-spaced; 8 signed magnitudes) | Linear INT4 + zero-point | Linear INT4 + zero-point |
| **L2 — scale granularity** | How many scales per N elements | 1 FP32 scale + 1 FP32 min per 256 | 1 UE8M0 per 32 + 1 FP16 per row | 1 super-FP16 + 1 super-FP16 min + 8×(6-bit scale, 6-bit min) per 256 | Same as Q4_K_M |
| **L3 — rotation** | Outlier dispersion within scale scope | FWHT-256 (offline) | FWHT-256 (offline; MFP4 variant) | none | none |
| **L4 — scale fitting** | Per-block scale chosen by min-max vs weighted LS | min-max | min-max | weighted LS over 20 candidates (`make_qkx2_quants`) | Same as Q4_K_M |
| **L5 — bit allocation** | Per-tensor / per-layer mixed precision | Hardcoded K-map rules | Same K-map | none (uniform) | imatrix-driven per-tensor selection of Q3/Q4/Q5/Q6 |

We measured the L1/L2/L3 combination empirically on Qwen3.5-9B / Qwen3.6-A3B BF16 safetensors:

| Format | Effective bpw | Per-weight MSE (median) | Notes |
|---|---|---|---|
| Q4_0 (per-32 INT4, no rotation) | 4.5 | 9.5e-7 – 1.6e-6 | Reference for L2 only. |
| MQ4G256 (per-256 INT4 + FWHT-256) | 4.25 | 1.5e-6 – 2.9e-6 | L3 alone < L2 alone. **~2× MSE vs Q4_0.** |
| Q4_K_M (per-32 INT4 + super-scales + weighted LS) | 4.5 | not measured here | Reference for L2+L4. |
| **MFP4G32** (per-32 UE8M0 + FP16 row + E2M1 + FWHT-256) | 4.5 | **5.8e-7** (PR #225 commit) | L1+L2+L3. **~3–5× MSE vs MQ4.** |

The MFP4 commit message reports 5.8e-7 mean quantization error on Qwen3.5-9B, which is **5–30× better than MQ4** across the four tensors in our analysis. That single format change (PRs #224 + #225) captured three levers simultaneously.

### 1.2 What MQ4 had — and what MQ4 was missing

MQ4G256 was the first hipfire-native quantization format. Its strengths and weaknesses, restated as the lever taxonomy:

| Lever | MQ4 status | Why it underperformed |
|---|---|---|
| **L1 element format** | Linear INT4 | LLM weights are roughly log-normal post-norm; uniform spacing wastes codepoints on the dynamic-range tails. |
| **L2 scale granularity** | One scale per 256 | A single scale per 256-element block is **forced to span the largest sub-range** in the block. With per-32 sub-blocks, 8 separately-fit ranges capture cross-sub-block variation at ~3 dB SNR advantage. |
| **L3 rotation** | FWHT-256 (within-block) | Helps but bounded: ~3 dB SNR within a block, **antagonistic with L2** (FWHT equalizes within-block range that per-32 scales would exploit). |
| **L4 scale fitting** | naive `(max - min) / 15` | No weighted LS, no candidate search. ~1.2–1.5× MSE worse than `make_qkx2_quants`. |
| **L5 bit allocation** | Uniform with hardcoded K-map promotions | Promotes router/embeddings/output to Q8 and edge-FFN to MQ6, but **not data-driven** — no activation importance. |

The empirical 2× MSE gap vs Q4_0 is **L2 alone**. The 6.2× KLD gap vs unsloth UD-Q3_K_XL is the **combined L2+L4+L5** gap, with L5 being the dominant explainer at that level of disparity.

### 1.3 What HFP4/MFP4 closed (and what it kept)

HFP4G32 (PR #224) and MFP4G32 (PR #225) ship with the following design choices, which align almost exactly with the levers above:

**Closed:**
- **L1 — element format → E2M1.** Eight signed magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, log-spaced. Matches the OCP MXFP4 wire format. Better fit to log-normal weight distributions than linear INT4. **Estimated MSE win vs MQ4 L1: ~1.3–1.6×** (data-dependent, larger on tensors with heavy tails).
- **L2 — scale granularity → per-32 UE8M0 + FP16 row scale.** UE8M0 is an 8-bit power-of-2 exponent (`v_ldexp_f32` is free on RDNA). FP16 row scale carries cross-block outlier compensation. **Estimated MSE win vs MQ4 L2: ~2×** (the empirical Q4_0 gap, since L2 granularity is now equivalent).
- **L3 — rotation → optional FWHT-256 via `format_flags` rotation_kind bits.** MFP4 = HFP4 + rotation; HFP4 alone has no rotation. The kernel can read both because the rotation is offline (baked into the codes). **Estimated win: 1.0–1.3× on top of L1+L2** — smaller marginal benefit because L1+L2 already capture much of what L3 used to compensate for.

Multiplied: **L1 × L2 × L3 ≈ 2.6–4.2×.** The commit message's "5–30×" is the high end of that range plus what we don't yet have a clean lever for (the FP16 row scale captures a long-tail outlier behavior that none of the integer formats handle).

**Kept (i.e. not addressed by HFP4/MFP4):**
- **L4 — scale fitting.** HFP4's per-block UE8M0 is chosen by `ceil(log2(block_max / 6.0))` — a deterministic min-max-based rule, not weighted LS. The FP16 row scale uses `max_abs(row) / 6.0` — also min-max. **No change vs MQ4 here.**
- **L5 — bit allocation.** Same K-map rules from MQ4 (norms/embeddings/output/router/edge promotions). No activation-driven importance. **No change vs MQ4 here.**

So HFP4/MFP4 captured L1+L2+L3 — call this the **"per-weight format quality"** group of levers — but left L4+L5 — the **"per-tensor and data-driven"** group — entirely untouched. That's where Unsloth Dynamic 2.0 lives.

### 1.4 Note on RDNA hardware acceleration

HFP4's RDNA-optimal choices (`v_ldexp_f32` for UE8M0 dequant, native FP8 WMMA on gfx1201, V_PERMLANE16 cross-lane broadcasts, VOPD dual-issue) are **not why HFP4 has lower MSE than MQ4**. Those choices make HFP4 *fast* on RDNA3/4. The MSE win comes from L1+L2+L3 — which would also be faster on Hopper / Blackwell, just not as fast as a hardware-native MXFP4 / NVFP4 there.

On gfx906 (our dev box), there is **no specialized HFP4 kernel** — only the generic wave32-oriented path. HFP4/MFP4 on gfx906 is correctness-only; bench numbers from this hardware are meaningful for math but not for production tok/s.

---

## 2. What's still missing vs imatrix-calibrated dynamic GGUFs

### 2.1 The three Unsloth levers

Stripped of marketing, "Unsloth Dynamic 2.0" is three independent levers stacked on top of Q4_K_M:

**(A) Q4_K_M per-tensor format quality** (= our L1+L2+L4):
- Per-32 sub-block 6-bit linear scales + super-FP16 scale-of-scales
- 6-bit per-sub-block mins (asymmetric)
- Weighted least-squares scale search (`make_qkx2_quants`, 20 candidates)
- No rotation

**(B) Per-tensor format mixing** (= our L5, mechanism only):
- `llama-quantize --tensor-type "regex=quant"` — regex selects tensors; per-match quant override
- Pre-set flags for embeddings (`--token-embedding-type`), output (`--output-tensor-type`)
- Used to demote `attn_q/k` on alternating layers to Q3_K and promote `attn_v`/`ffn_down` to Q5_K/Q6_K

**(C) imatrix activation-aware calibration** (= our L5, *data-driven decision*):
- Forward-pass a calibration corpus (Unsloth uses Calibration_v3/v5, >1.5M tokens of hand-curated data) through the BF16 model
- Per linear-layer input dimension, record `Σ_token (x[token, i])²` — squared activation
- During quantization: weighted LS becomes `min Σ_i act²[i] · (w[i] - q[i])²` instead of `min Σ_i (w[i] - q[i])²`
- Per-tensor statistics: `Σ(Act²)`, `% Active`, `Entropy`, `ZD Score` (Liu et al. 2024), `CosSim` to previous layer
- The decision *which tensors get Q3 vs Q4 vs Q5 vs Q6* uses these statistics, not hardcoded rules

### 2.2 Per-lever quality estimates relative to MFP4 baseline

These are estimated improvements **on top of MFP4G32** — they are not additive with the MFP4 wins above.

| Lever | What it adds | Estimated PPL Δ at fixed bpw | Estimated KLD-mean Δ | Effort | Notes |
|---|---|---|---|---|---|
| **L4a — weighted LS for UE8M0** | Replace `ceil(log2(block_max/6))` with a small candidate search over `block_e ∈ {e_ideal-1, e_ideal, e_ideal+1}` + brute-force re-rounding | −3% to −7% | −10% to −20% | ~1 week | Pure quantizer-side. Format unchanged. Easy win — the current UE8M0 chooser is intentionally simple. |
| **L4b — weighted LS for FP16 row scale** | Solve for `row_scale_a` minimizing post-block-quantization MSE instead of `max_abs/6.0` | −2% to −5% | −5% to −15% | ~3 days | Pure quantizer-side. Format unchanged. Probably should ship with L4a. |
| **L5a — `--tensor-type regex=quant` CLI** | Generalize the hardcoded K-map rules to a CLI/config-driven regex matcher; ship K-map presets as default configs | 0% (mechanism only) | 0% (mechanism only) | ~3 days | Unblocks A/B experiments without rebuilding the quantizer. Format unchanged. |
| **L5b — imatrix collection** | Forward-pass corpus through BF16 model; dump `Σ act²` per linear-layer input dim to sidecar `.imatrix` file | 0% (collection only) | 0% (collection only) | ~1 week | New `imatrix_collect` example; ~1M-token corpus (wikitext-103-test + humaneval prompts + slice of multi-turn chat). |
| **L5c — activation-weighted LS in quantizer** | Use the collected imatrix to switch L4's LS objective to importance-weighted | −5% to −15% | **−40% to −70%** | ~2 weeks | The dominant lever. Median MSE may move ~1.2× — but **p99 MSE on hot input dimensions** moves dramatically, which is what KLD measures and what attractor-style coherence failures see. |
| **L5d — imatrix-driven per-tensor bit allocation** | Use per-tensor `Σ(Act²)` / `% Active` / ZD score to pick HFP4G32 / HFP6G32 / Q8 per tensor instead of hardcoded K-map | −3% to −10% | −15% to −30% | ~2 weeks (after L5b/c) | Replaces K-map rule 4/5 with data-driven promotion. Bit budget held constant. |
| **L5e — sweep `Q3_K`-equivalent demotion** | Once L5d is in place, allow demoting low-importance tensors below 4-bit (HFP3 variants in the HFP family, or fall back to HFP4 at coarser group size) | +0 to +5% (quality-neutral at lower bpw) | 0% | ~1 month | Requires HFP3 kernel work. Defer until L5b/c/d are in. |

Multiplied through, L4a/b + L5b/c on top of MFP4G32 should land at:
- **PPL within 1–3% of Q4_K_M + imatrix at the same bpw**
- **KLD-mean within 1.2–1.5× of Unsloth UD-Q4_K_XL** (vs current 6.2× of UD-Q3_K_XL — i.e. expected to *beat* UD-Q3_K_XL outright and approach UD-Q4_K_XL parity)

The rotation lever (L3) is the one place hipfire **should beat** Unsloth at equal bits-per-weight on the *same* calibration corpus: FWHT-256 provides ~1.0–1.3× MSE on top of an otherwise-equivalent format, and that win compounds with everything in §2.2.

### 2.3 Why imatrix is the dominant lever

It's worth explaining *why* L5c specifically dominates, because it's not obvious.

A per-tensor MSE of 1e-6 sounds tiny — but it's not uniformly distributed across input dimensions. The same MQ4-quantized router has:
- ~95% of input dims at MSE near the median (~1e-6)
- ~3–5% of input dims at MSE 10–100× the median (the "hot" dims that carry most of the activation energy on real prompts)

KLD measures the divergence of the *output logit distribution* from the FP16 reference. The output is dominated by hot dims. A quant that's median-good but p99-bad is exactly what produces:
- **MQ4 mean KLD 0.876 vs UD-Q3_K_XL mean KLD 0.141** (our Phase 10 data) — at lower bits but with imatrix
- The block-attractor spiral we saw in Phase 11 of `qwen35-moe-coherence-investigation.md`

imatrix-weighted LS specifically targets the p99: it puts more quantization budget on the dims the corpus actually exercises, and accepts higher error on dormant dims. The mean MSE only moves modestly; the p99 KLD moves a lot.

This is also why MFP4 didn't close the A3B spiral in Phase 11 — MFP4 improves median MSE (per-weight), but doesn't differentially protect hot input dimensions. The spiral lives at the p99 cliff.

### 2.4 Where we sit today vs Unsloth Dynamic at equal bpw

| Aspect | Hipfire (MFP4G32 + K-map alternating) | Unsloth UD-Q4_K_XL | Gap |
|---|---|---|---|
| Element format | E2M1 (log-spaced) | INT4 linear | **+** (we win L1) |
| Per-32 scales | UE8M0 + FP16 row | super-FP16 + 8×6-bit | wash |
| Rotation | FWHT-256 | none | **+** (we win L3) |
| Scale fitting | min-max | weighted LS | **−** (they win L4) |
| Per-tensor mixing | hardcoded K-map | regex / imatrix-driven | **−−** (they win L5) |
| Calibration | none | 1.5M-token corpus | **−−** (they win L5) |

We are roughly 2 levers behind. Closing the 2 levers is **2–4 weeks of focused engineering, no new fundamental research.** The pieces (imatrix collection, weighted LS, per-tensor selection) are all 2024-era public techniques.

---

## 3. Format roadmap: extending HFP4 to absorb the missing levers

The whole question is whether we have to invent a new quant family — or whether HFP4 already has the extension points. Reviewing the HFP4 wire format (see `docs/quant-formats/hfp4.md` for the spec):

### 3.1 What HFP4 reserved for future use

The per-row header is 16 bytes:

```
+0  : f16  row_scale_a       // primary FP16 second-level scale
+2  : f16  row_scale_b       // dual-output scale (fused gate+up, qkv)
+4  : u16  block_count       // K/32
+6  : u8   format_flags      // bit 0: rotation present
                              // bit 1: row_scale_b used
                              // bits 2-3: rotation_kind (00..11)
                              // bits 4-7: reserved
+7  : u8   reserved
+8  : u32  reserved          // future: D_diag pointer offset (joint-D smoothing)
+12 : u32  reserved          // future use
```

Specifically:
- **`format_flags` bits 4-7 (4 bits)** — 16 future flags
- **`format_flags` bits 2-3 rotation kind** — 4 rotation modes; `01` shipped (offline FWHT-256), `10`/`11` reserved (online block-diag-128, HadaCore-16)
- **`reserved` u8 at +7** — 256 future flag values
- **`reserved` u32 at +8** — explicitly earmarked for joint-D smoothing (a per-channel pre-multiplier vector that lives in a sidecar buffer, pointed at by this offset)
- **`reserved` u32 at +12** — fully unallocated
- **Quant-type IDs 21–29** reserved (currently using 21=HFP4G32, 24=MFP4G32; IDs 22, 23, 25–29 reserved for ablations and v2/v3 variants)

### 3.2 How each missing lever maps onto the existing format

| Lever | Format change required | Wire-format break? |
|---|---|---|
| **L4a/b — weighted LS for UE8M0 + row scale** | **none.** Pure quantizer-side. The kernel reads the same bytes; only the chosen `block_e` / `row_scale_a` values change. Existing HFP4 files coexist with weighted-LS-fit HFP4 files at identical IDs. | **No.** |
| **L5a — regex CLI for per-tensor format** | Per-tensor metadata: which quant_type was chosen. Already present in `.hfq` tensor index (`quant_type: u8` per tensor). The CLI is pure quantizer-side. | **No.** |
| **L5b — imatrix collection** | Sidecar `.imatrix` file. Not part of `.hfq` at all — used only at quantize time. | **No.** |
| **L5c — activation-weighted LS** | Pure quantizer-side again. The format that comes out is byte-compatible with HFP4G32 / MFP4G32; the *content* differs (better-fit scales/codes). | **No.** |
| **L5d — per-tensor bit allocation** | Already supported via per-tensor `quant_type`. We just need more HFP family members (see below). | **No.** |
| **L5e — HFP3 / HFP4@G64 / HFP6 variants** | Each new variant gets one of the reserved quant_type IDs 22-29. The byte layout is essentially the same family with different element format / group size / row-scale dimensions. | **No** if we use reserved IDs; otherwise yes. |

**Conclusion: every Unsloth-tier lever fits inside the existing HFP4 wire format without a single breaking change.** L4 + L5a–c are pure quantizer-side improvements that produce byte-compatible HFP4G32 files. L5d–e use already-reserved quant_type IDs.

### 3.3 Extension proposals for HFP family

To round out the family for Unsloth-equivalent per-tensor bit allocation:

| Proposed | ID | Spec | Use case |
|---|---|---|---|
| HFP3G32 | new (e.g. 30) | E2M0 (no mantissa; 4 magnitudes) or 3-bit linear + UE8M0 + FP16 row | Demotion target for low-importance tensors per L5d |
| HFP6G32 | new (e.g. 31) | INT6 + UE8M0 + FP16 row, packs as 6-bit nibbles | Promotion target for hot tensors |
| HFP4G64 | 23 (reserved) | E2M1 + UE8M0 g64 + FP16 row | Coarser scale granularity for embedding / lm_head where row-scale dominates |
| HFP4G16 | 22 (reserved) | E2M1 + UE8M0 g16 + FP16 row | Finer scale granularity for outlier-heavy tensors |
| HFP8E4M3G32 | 27 (reserved) | E4M3 + UE8M0 g32 | Q8-equivalent for routers, embeddings, lm_head |
| **HFP4G32W** | new (e.g. 32) | HFP4G32 + per-block 8-bit weight `w_block` byte in payload (signals importance to a hypothetical importance-aware dequant) | **Speculative** — not needed for L4/L5 closure. Listed for completeness. |

L5d's "per-tensor bit allocation" using just the existing ID set (HFP3G32, HFP4G32, HFP4G16/G64, HFP6G32, HFP8E4M3G32) gives ~5 effective bit-width slots — more than Unsloth's Q3/Q4/Q5/Q6/Q8 mix.

### 3.4 Rotation extension — already partially designed

The `rotation_kind` bits in `format_flags` reserved three future modes:
- **`10` — online block-diag-128** (AMD's recipe; requires Stiefel-manifold-calibrated rotations and fused-rotation kernels for QKV/QKVZA/gate_up — v3 roadmap)
- **`11` — HadaCore-16** (16×16 Hadamard via WMMA fragments; gfx1201 only — research)
- (`00` = none, `01` = offline FWHT — both shipped)

If the imatrix work in §2.2 reveals that calibrated rotations (QuaRot-style optimal Hadamards instead of fixed seeds 42/1042) are a meaningful additional lever, that fits into rotation_kind `10` with an `R` matrix stored per layer in a sidecar. **No wire-format break.**

### 3.5 What can't be done inside HFP4

There are two structural limits worth naming:

1. **Block sub-32 elements.** The per-block `block_e` byte is 1 byte / 32 elements = 0.25 bpw overhead at g=32. Going to g=16 doubles that to 0.5 bpw — substantial. HFP4G16 ships as an ablation slot, but for a "go finer than per-32" format we'd want a separate spec.
2. **Non-row-shaped weights.** The 16-byte row header assumes row-major matmul weights. Conv weights or batched embeddings shaped `[batch, in, out]` would either consume a row header per (batch, in) plane — wasteful — or need a different wire format. **This is the most likely future incompatibility.** Not relevant for transformer LLMs.

Neither is a near-term blocker.

---

## 4. Future-proofing for archs we haven't shipped yet

Hipfire supports Qwen3.x, Qwen3.x-VL, Llama families today. The user asked specifically about Gemma (newer than Qwen3 in design) and Qwen2.5-VL (older). The format's per-arch concerns are orthogonal to L1–L5; they're about *what gets quantized, where it lives, and how the engine matches tensors to handlers.*

### 4.1 What changes across architectures

| Concern | Qwen3.x (current) | Qwen2.5-VL (older) | Gemma 2/3/4 (newer) | Other (Llama, Mistral, …) |
|---|---|---|---|---|
| Norm type | RMSNorm (`w * rms`) | RMSNorm | **GemmaRMSNorm** (`(1+w) * rms`) | RMSNorm |
| Norm storage | bare `w` | bare `w` | **bare `w` or pre-baked `(1+w)`** depending on source | bare `w` |
| Attention | GQA + DeltaNet hybrid | GQA + vision encoder | GQA, sometimes softcapping | GQA / MQA |
| MoE | Yes (A3B/A22B) | No | No (so far) | Some (Mixtral) |
| Tokenizer | tiktoken-style BPE | Same | SentencePiece | Mixed |
| Vision encoder | ViT in qwen35-vl | ViT (different patch size, different fusion) | none (Gemma1) / native multimodal (Gemma 3+) | Llama-vision pipelines |
| Tensor names | `model.language_model.layers.X.…` | `model.layers.X.…` or `transformer.h.X.…` | `model.layers.X.…` with different qkv shape conventions | mixed |
| Special tensors | router, shared_expert_gate | none | per-layer logit softcap scalar | none |

The HFP4 wire format is **architecture-agnostic at the per-tensor level** — a row of floats, quantized, stored. The architectural complexity lives in **(a) tensor naming**, **(b) per-tensor classification rules** (currently the K-map), and **(c) loader code that wires tensors to model parts.**

### 4.2 What needs to be future-proofed in the format

Three concrete extensions to keep the `.hfq` format from accumulating arch-specific cruft:

**(i) Per-tensor metadata sidecar in `.hfq`.** Currently per-tensor info is `(name, quant_type, shape, group_size, data_offset, data_size)`. Adding a free-form `metadata_json` per tensor — analogous to the file-level metadata — would let arch-specific loaders carry hints without burdening the format:
- Gemma's `(1+w)` baking flag for norms
- Softcap scalars stored alongside attention weights
- Activation-importance summary (`Σ act² mean/p99`) used by the imatrix-driven loader to pick KV-cache precision

**(ii) Tensor-class normalization.** Currently `kmap_resolve_mode` in `crates/hipfire-quantize/src/main.rs` greps on substrings like `"down_proj"`, `"v_proj"`, `"mlp.experts."`. This breaks the moment a HuggingFace export changes the naming convention (Qwen2.5-VL uses `transformer.h.X.attn.c_attn` for fused QKV; Gemma uses `model.layers.X.self_attn.qkv_proj`). The robust fix:

- Define a **canonical tensor-class enum** in `crates/hipfire-quantize/src/tensor_class.rs`:
  ```
  pub enum TensorClass {
      EmbedTokens, LmHead, OutputNorm,
      AttnQ, AttnK, AttnV, AttnQKVFused, AttnO,
      FfnGate, FfnUp, FfnDown,
      MoeRouter, MoeSharedExpertGate, MoeExpertGate, MoeExpertUp, MoeExpertDown,
      LayerNorm, RmsNorm, GemmaRmsNorm,
      VisionPatchEmbed, VisionAttn, VisionFfn, VisionMerger,
      SoftcapScalar, RouterLogitsBias,
      Unknown,
  }
  ```
- Per-arch classifier function: `fn classify(name: &str, arch: ModelArch) -> TensorClass`. Currently this logic is implicit in K-map substring matches and engine-side loader code; explicit centralization is the future-proofing.
- K-map rules and imatrix-driven bit allocation work on `TensorClass`, not on raw names. Adding a new arch becomes "add a new `classify` arm" — no format change, no quantizer logic change.

**(iii) Per-arch reserved fields in HFP4 row header.** The reserved u32 at offset +12 in the HFP4 row header can hold a per-tensor opaque token; the engine's arch-specific loader interprets it. Examples:
- Gemma: pre-quantized `(1+w)` or bare `w` flag for norm rows
- Vision: spatial dimension hints for non-1D-row weight shapes
- MoE: expert-index-in-stack for the 3D expert tensors we use today

The format already has the bytes; we just need a convention.

### 4.3 Adding Gemma and Qwen2.5-VL: concrete shopping list

What it actually takes to add these arches, given the format roadmap above:

**Gemma 3/4 (newer, simpler-than-MoE):**
- Add `ModelArch::Gemma3` to `crates/hipfire-runtime/src/arch.rs`
- Implement `GemmaRMSNorm` correctly (`(1+w) * rms`) — this is **already correct in hipfire** since PR #228 fixed the MoE final-norm convention; same code path applies
- Tensor-class classifier handles HF's `model.layers.X.self_attn.qkv_proj` (fused QKV)
- Softcap scalar (Gemma 2 has it; Gemma 3 dropped it) gets a per-arch metadata blob via §4.2(i)
- No HFP4 format change needed

**Qwen2.5-VL (older, more naming inconsistency):**
- Add `ModelArch::Qwen25VL` to arch enum
- Tensor-class classifier handles `transformer.h.X.attn.c_attn` (HF's older fused QKV name)
- Vision tower: same `ViT` patterns as qwen35-vl but with different patch size; quantize as a separate sub-graph (the K-map already excludes `visual.` tensors from MoE rules)
- No HFP4 format change needed

**Estimated effort for each:** 1 week of arch crate work + 2 days of quantizer K-map / tensor-class additions. The format is ready.

---

## 5. The plan — concrete sequencing

This is the ordered roadmap that gets us from where we are (MFP4 ships, A3B spiral exposed, 6.2× KLD gap to Unsloth) to *beating* Unsloth Dynamic 2.0 at the same bpw on RDNA.

### Phase A — Quantizer quality, no format change (4–6 weeks)

Order matters because each step's bench harness validates the next.

1. **(week 1)** L4a — Weighted-LS UE8M0 chooser. Land as quantizer-only patch; existing HFP4 files unchanged byte-format. Bench: MSE on Qwen3.5-9B BF16 reference tensors should drop ~5–7%.
2. **(week 1.5)** L4b — Weighted-LS FP16 row scale. Bench: combined MSE drops 8–12%.
3. **(week 2)** L5a — `--tensor-type "regex=quant"` CLI. Mechanism only; no quality change. Ships K-map presets as default `.kmap` configs for Qwen3.5, Qwen3.6, Llama3, Gemma2/3.
4. **(week 3)** L5b — `imatrix_collect` example. New `crates/hipfire-runtime/examples/imatrix_collect.rs` runs forward pass on calibration corpus and dumps `Σ act²` per linear-layer input dim to sidecar `.imatrix.bin`. Calibration corpus: wikitext-103-test (300k tok) + humaneval prompts (50k tok) + slice of HF datasets multi-turn chat (650k tok) ≈ 1M tokens.
5. **(week 4)** L5c — Activation-weighted LS in MFP4G32 quantize path. **The dominant lever.** Bench: KLD-mean on Qwen3.5-9B with calibrated MFP4 should drop from 0.876 (current) to **0.20–0.30** (within 1.5× of UD-Q4_K_XL).
6. **(week 5)** L5d — imatrix-driven per-tensor bit allocation. Use per-tensor `Σ(Act²)` and ZD score to pick HFP4G32 / HFP6G32 / Q8 per tensor; bit budget held constant. Bench: KLD-mean drops another 15–30%.
7. **(week 6)** Smoke test: re-run train-pursuit reasoning on calibrated MFP4 A3B. **Strong prediction** (low confidence — hedged because Phase 11 already falsified one quant-quality hypothesis): the A3B spiral resolves or moves to a different attractor signature.

**Expected end state of Phase A:** KLD-mean on Qwen3.5-9B between **0.10 and 0.20** — competitive with or beating UD-Q4_K_XL. PPL within 1–2% of Q4_K_M + imatrix at the same bpw. **All without a wire-format change.**

### Phase B — Format-family fillout (3–4 weeks, can run in parallel with Phase A)

8. HFP3G32 (qt=30) — E2M0 + UE8M0 + FP16 row. Quantizer + correctness-anchor kernel. Bench: 3.0 bpw with median MSE ~3× of HFP4G32, but on dormant tensors (low ZD score) the model-aggregate KLD impact is negligible.
9. HFP6G32 (qt=31) — INT6 + UE8M0 + FP16 row. Promotion target for hot tensors.
10. HFP8E4M3G32 (qt=27) — Q8-equivalent at 8 bpw. Replaces hardcoded "router/embedding/lm_head → Q8" with a real HFP-family member that has the same row header and dequant convention.

### Phase C — Arch coverage (2–3 weeks, can run in parallel with Phase B)

11. Tensor-class enum + per-arch classifier (§4.2(ii)). Refactor K-map + engine loaders to consume `TensorClass` instead of substring greps.
12. Gemma 3/4 arch crate. GemmaRMSNorm is already correct; just need arch glue + tensor-class classifier + tokenizer wiring.
13. Qwen2.5-VL arch crate. Older naming convention handled in classifier.

### Phase D — Validation as canonical bench (1 week)

14. Standing benchmark: extend `scripts/bench_quant_quality.sh` to run the full MSE + KLD + smoke-test triple under three configurations (uncalibrated MFP4, calibrated MFP4, calibrated MFP4 + per-tensor bit allocation) on three reference models (Qwen3.5-9B, Qwen3.6-A3B, a Gemma3 family member). Lock as the regression bar.
15. Update CLAUDE.md: any quantization-format change MUST regress this triple.

---

## 6. Closing — what this means strategically

The question we set out to answer: *"what's the current problem, and how do we get from where we are to a 4-bit format that beats Unsloth Dynamic GGUFs in quality, given our rotation lead and modern element format?"*

**The current problem:** the rotation + modern element format (L1+L2+L3) we have **is genuinely a quality win** — MFP4 measurably beats MQ4 by 3–5× per-weight MSE, and at lever-for-lever parity should beat unrotated Q4_K_M by ~1.0–1.3× MSE. But Unsloth's lead comes from a different lever entirely: activation-aware calibration (L4 weighted LS + L5b/c/d imatrix). Without calibration, even a theoretically-better format underperforms a calibrated worse format on what users actually measure (KLD, instruction-following, reasoning coherence).

**The path:** Phase A (4–6 weeks) closes the calibration gap inside the existing HFP4 wire format. The reserved bits in the row header, the configurable rotation_kind, and the reserved quant_type IDs were designed exactly for this kind of staged extension — we don't need a new format family, we need to use the one we have.

**The strategic posture:** **HFP4 is the format we should commit to for the next several years.** It is:
- Architecturally extensible (4 rotation modes, 4 reserved IDs, 16 reserved metadata flags, 1 reserved sidecar pointer)
- RDNA-optimal (UE8M0 → free `v_ldexp`; FP16 row → cheap WMMA epilogue; E2M1 → wire-compatible with future MXFP4 silicon)
- Arch-portable (zero per-arch fields in the per-tensor format; arch concerns ride in tensor-class classifier + per-tensor metadata sidecar)
- Quantization-quality-extensible (all of L4 + L5 closes inside the existing wire format)

If Phase A delivers what the lever-by-lever math predicts, hipfire ships an MFP4 + calibrated quantizer that **beats UD-Q4_K_XL on KLD at equal bpw, while also being 70–75% of theoretical FP16 WMMA throughput on RDNA3+**. That's a quality lead AND a perf lead, on a wire format we own.

The risk is Phase 11's lesson: the A3B `<think>` spiral was not closed by reducing per-weight noise. If after Phase A the spiral still fires, the answer is not more quantization work; it's the runtime-side investigation noted in `docs/plans/qwen35-moe-coherence-investigation.md` §"Phase 11 engine-pass implications" (sampler intervention, vLLM FP16-router contract, period-N block-attractor detection). Quantization closes the quality gap; runtime closes the residual coherence risk.

---

## References

### In-tree
- `docs/quant-formats/hfp4.md` — HFP4 wire format spec (reservations, taxonomy, kernel targets)
- `docs/plans/qwen35-moe-coherence-investigation.md` — parent investigation; Phase 11 falsifies "format quality → spiral" link
- `crates/hipfire-quantize/src/main.rs` — quantizer (MQ4/MQ6/HFP4/MFP4 functions, K-map rules)
- `crates/hipfire-runtime/src/hfq.rs` — `.hfq` file format reader
- `crates/hipfire-runtime/examples/quant_quality_mse.rs` — per-tensor MSE harness
- `crates/hipfire-runtime/examples/compare_hfq.rs` — tensor-by-tensor NRMSE diff
- `scripts/bench_quant_quality.sh` — combined MSE + smoke triple

### External
- OCP Microscaling Formats (MX) v1.0 — element format reference for E2M1
- AMD ROCm Blog "High-Accuracy MXFP4, MXFP6" — UE8M0 + FP16 row design
- AMD ROCm Blog "Advanced MXFP4 with Online Rotation" — block-diag-128 rotation (HFP4 rotation_kind `10`)
- NVIDIA NVFP4 announcement — E4M3 scale + FP32 tensor scale reference
- llama.cpp `tools/imatrix/README.md` — calibration methodology (PR #4861)
- llama.cpp `tools/quantize/README.md` — `--tensor-type` regex, `--imatrix` flags
- Liu et al. 2024, "Layer-Wise Quantization" (arXiv 2406.17415) — ZD score per-tensor importance metric
- QuaRot (arXiv 2404.00456) — full-hidden-dim rotation + GPTQ calibration
- SpinQuant (arXiv 2405.16406) — Stiefel-manifold rotation optimization
- Unsloth Dynamic 2.0 — marketing page (`https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs`); algorithmic details derive from underlying llama.cpp tools, not the page itself
