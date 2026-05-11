# MR-GPTQ as calibration alternative to imatrix for MFP4

**Date:** 2026-05-12
**Trigger:** User research request after the fivetide PPL/KLD exchange.
**Reference paper:** Castro et al., *"Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization"* — arXiv:2509.23202v3
**Reference implementations:** llm-compressor `examples/transform/quip_example.py`; vLLM PR #22486 (online Hadamard rotations, closed unmerged)
**Status:** Research note. Recommendation conditional on Phase A bench data.

## TL;DR

MR-GPTQ ("Micro-Rotated GPTQ") is a recent (2025) calibration recipe that **directly targets the FWHT+FP4 incompatibility fivetide identified**. It replaces the imatrix `Σ act²` weighted-LS recipe with a different calibration story — GPTQ's sequential weight update + block-wise Hadamard rotation + MSE-optimized scale grid + static activation reordering — and reports +3.8% accuracy recovery on MXFP4 (Llama-3.1-8B-Instruct W4A4) over standard GPTQ.

**For hipfire's purposes:** MR-GPTQ is structurally a better fit for our existing MFP4G32 wire format than imatrix is, because (a) the block-size matches our g=32, (b) the rotation is offline-fused-into-weights (same as our MFP4G32 `format_flags` rotation_kind `01`), (c) it's the *only* published method that explicitly reports closing the MXFP4-vs-INT4 accuracy gap that fivetide's PPL data exposed. **But** the implementation cost is substantially higher than imatrix (GPTQ's per-layer Hessian inverse Cholesky factorization is heavyweight) and the gfx906 acceleration question is non-trivial — MR-GPTQ assumes B200/RTX5090-class kernels.

Concrete recommendation: **add MR-GPTQ as a candidate in Phase A Step 5 (L5c), in addition to imatrix-weighted LS, not in place of it.** The harness work (Step 0/0.5) is unchanged; the calibration *recipe* becomes one more variable to bench. If MR-GPTQ-calibrated MFP4 measures materially better KLD/PPL than imatrix-calibrated MFP4 on Qwen3.5-9B + Qwen3.6-A3B, lock it as the format's calibration story. If not, imatrix is cheaper and equally effective.

## What MR-GPTQ actually is

The paper's framing: "FP4 is not an automatic upgrade over INT4." MXFP4 and NVFP4 have hardware decode advantages on B200 / RTX5090, but **vanilla GPTQ on FP4 loses 5–10 accuracy points vs INT4** in their measurements — exactly the kind of result fivetide's PPL data on Qwen3.5 suggested (modulo their KLD reversal on 0.8B).

MR-GPTQ has three components, all layered on top of standard GPTQ:

### 1. Block-wise Hadamard transforms (the rotation)

- Block size `k` is a power of 2; `k ∈ {16, 32, 64, 128}` tested
- **For MXFP4 (group size 32), they use a matching 32×32 Hadamard block** — *not* the 256-element FWHT hipfire's MFP4 uses
- Applied as `Q(W·H_k) · Q(X·H_k)ᵀ` — both weights and activations rotated, but the H_k is *fused into the weights* offline and applied *online to activations* via a small fast kernel
- Block-diagonal Hadamard (not a single dense rotation across the full hidden dim) — this is the key to making the online activation rotation cheap

This is **structurally the same trick hipfire's MFP4 uses**, just at finer block granularity and with a different basis. Our MFP4G32 uses FWHT-256 (a 256-element Walsh-Hadamard transform) offline-fused into weights, and `mq_rotate_x` applies it online to activations. MR-GPTQ would substitute Hadamard-32 for FWHT-256, matching the block-quant group size.

### 2. MSE-optimized scale grid (the per-block scale chooser)

Instead of `block_scale = max_abs(block) / 6.0`, MR-GPTQ does **alternating optimization over block scales and per-tensor scale** before GPTQ. Specifically:
- Sweep candidate block scales
- For each candidate, compute the MSE of the round-trip quantization
- Pick the block scale that minimizes MSE — but in a *joint* optimization with the per-tensor (in our case, per-row FP16) scale

This is exactly Phase A's L4a + L4b work — weighted-LS UE8M0 chooser + weighted-LS FP16 row scale. MR-GPTQ's framing is that the two scales should be co-optimized rather than serially refined. The implementation cost is bounded — it's a small candidate search per block, not a Hessian computation.

### 3. Static activation reordering (the column shuffle)

The novel piece I hadn't seen elsewhere: **before GPTQ's sequential update**, columns of the weight matrix are reordered based on a static analysis of activation statistics from the calibration corpus. After GPTQ finishes, the columns are shuffled back so the on-disk layout preserves the group structure (32-element groups for MXFP4). The intuition: GPTQ processes columns sequentially and uses earlier columns' quantization errors to compensate later columns. Reordering puts the "easiest" columns first so the cumulative error budget is spent where it matters least.

Order: scales & grid computation → column shuffle → standard GPTQ → reverse-shuffle to restore layout. The reordering itself is the new contribution; the GPTQ inner loop is unchanged.

## Calibration corpus & cost

- **Source:** FineWeb dataset (not wikitext; more representative of modern web text)
- **Size:** 1024 calibration sequences for most experiments
- **QAT experiments:** 10% of Tülu 3 instructions (~93k samples) when going further

This is **substantially smaller than Unsloth's 1.5M tokens** but consistent with the GPTQ literature (GPTQ traditionally needs only ~128–1024 sequences because of the Hessian-based propagation).

## Quality numbers (the central case)

Llama-3.1-8B-Instruct, W4A4 (weights and activations both quantized), average task recovery vs FP16:

| Format | Method | Avg. Recovery % |
|---|---|---|
| NVFP4 | GPTQ | 95.92 |
| NVFP4 | MR-GPTQ | 96.08 |
| MXFP4 | GPTQ | 89.47 |
| **MXFP4** | **MR-GPTQ** | **93.31** |

**The MXFP4 gap to recovery is the relevant data point for us** — MXFP4 is OCP E2M1, byte-identical to hipfire's HFP4/MFP4 element format. The vanilla-GPTQ-MXFP4 result (89.47%) is consistent with fivetide's "MFP4 loses on PPL" framing; MR-GPTQ closing 3.8 pp of that gap is exactly the bridge they're claiming.

**Caveat:** the paper measures W4A4 (activations also quantized). Hipfire's MFP4G32 is W4A16 — weights quantized, activations FP16. MR-GPTQ's NVFP4 numbers (95.92 → 96.08) show *much smaller* MR-GPTQ benefit in a higher-activation-precision setting. The W4A16 regime hipfire actually ships in is probably between the two — MR-GPTQ helps, but not by 3.8 pp.

## How MR-GPTQ compares to imatrix (the original Phase A L5c plan)

Both are calibration-driven, but they're solving different problems:

| Property | imatrix (llama.cpp / Unsloth) | MR-GPTQ |
|---|---|---|
| What it computes | Per-input-dim `Σ act²` (importance scores) | Per-layer Hessian inverse, used for sequential weight update |
| Use of calibration corpus | Forward-pass to record activations | Forward-pass to compute Hessian |
| Quantizer change | Weighted-LS scale fit (every block solves a weighted `min Σ w_i (w-q)²`) | Sequential column-by-column weight update via GPTQ — each column's quantization error is propagated into later columns' weights |
| Cost | O(1) per block (just weighted LS); calibration is a single forward pass | O(K²) per layer Cholesky + O(K·G) sequential update — heavy by comparison |
| Rotation interaction | Doesn't address rotation directly | Hadamard-fused-into-weights is a co-design |
| Per-tensor selection | Optional second pass using `Σ(Act²)` and ZD score | Not directly; could be added as a per-layer "include in GPTQ" mask |
| Quality story | Empirically validated on Q4_K_M family; gives Unsloth's KLD wins | Reports closing MXFP4-vs-INT4 gap; smaller body of validation work |
| Hardware implication | Trivial — runtime path unchanged | Online Hadamard rotation requires a fast kernel; on B200 they use QuTLASS fused kernels |

**The big structural difference:** imatrix tells the quantizer *which input dimensions matter*; MR-GPTQ tells it *how to allocate quantization error across all dimensions given that all of them matter*. Imatrix is a re-weighting; GPTQ is a re-allocation. They could plausibly compose (use imatrix to select tensors for promotion, then use MR-GPTQ to quantize each promoted tensor) but the paper doesn't claim that combination.

## Implementation-side observations

### llm-compressor `quip_example.py`

The reference QuIP implementation uses `transform_block_size=128, transform_type="hadamard"` and `rotations=["v", "u"]`. It's offline-fused-into-weights (not GPTQ — just rotation + RTN quant). **No calibration data** — it's a "datafree" pipeline. This is a useful simpler reference but is **not** what we want — it shows the rotation infrastructure, not the GPTQ infrastructure.

For MR-GPTQ specifically, we'd want llm-compressor's `GPTQModifier` + a custom `MicroRotateModifier` that implements the three components above. The paper claims the QuTLASS kernels are public for B200; the calibration-side code (offline weight modification) is presumably in llm-compressor or being upstreamed.

### vLLM PR #22486 (closed, unmerged)

vLLM PR #22486 adds *runtime* support for online Hadamard rotations in their CompressedTensors loader. The PR was closed without merge but provides the contract a calibration-side tool would need to produce. Important detail: their `SharedWeightParameter` registry assumes the H_k matrix is stored as a separate tensor in the checkpoint and applied at inference via a fused kernel. **Hipfire's MFP4G32 already fuses the rotation into the codes** (the FWHT-256 transform happens at quant time and dequant produces rotated values; `mq_rotate_x` applies the matching transform to activations at runtime). So the storage-format question is partially answered for us.

### What changes in hipfire's MFP4G32 wire format if we adopt MR-GPTQ

**Nothing fundamental.** The `format_flags` bits 2-3 (rotation_kind) already reserve four modes:
- `00` = none (HFP4G32)
- `01` = offline FWHT (current MFP4G32)
- `10` = online block-diag-128 (reserved for v3 — the AMD ROCm blog recipe)
- `11` = online HadaCore-16 (reserved)

We'd add a new rotation_kind `10` interpretation: **offline block-Hadamard-32 with MR-GPTQ-calibrated codes**. The block-Hadamard-32 is what the paper actually uses for MXFP4; the FWHT-256 we currently use is a different choice. The wire format accommodates both with one flag bit and one new quant_type ID — say `MFP4G32-MR` at qt=29 (currently reserved for `MFP4G32R`, the online-rotation variant; we can rename or use one of the other reserved slots).

**The dequant kernel needs updating** because online activation rotation differs. Currently `mq_rotate_x` applies a 256-element FWHT to x. For MR-GPTQ-MXFP4 we'd need a 32-element block-Hadamard applied to x. **This is a significantly smaller kernel than FWHT-256** — 32-lane reduction, fits trivially in a wave32 (or half a wave64 on gfx906). The online-rotation kernel cost drops, not rises.

## What this changes about Phase A

The original Phase A Step 5 (L5c) was "activation-weighted LS in MFP4G32 quantize path." MR-GPTQ is a third option in that step. The bench harness (Steps 0 + 0.5 from the 2026-05-12 update) is unchanged — the metric we measure is still {MSE, KLD, PPL, HumanEval} per format variant.

Updated Step 5 framing:

> **Step 5 (week 5) — L5c: Calibration recipes. Bench three candidates.**
>
> Run all three against all three baseline formats:
> 1. **imatrix-weighted LS** (the original L5c plan; cheap, well-understood)
> 2. **MR-GPTQ with offline block-Hadamard-32** (new — requires GPTQ-side infrastructure plus possibly a new rotation_kind in the wire format)
> 3. **Combined: imatrix-driven per-tensor selection + MR-GPTQ per-tensor calibration** (speculative; the paper doesn't claim this combination)
>
> Decision criterion: KLD-mean and HumanEval-pass@1 on Qwen3.5-9B + Qwen3.6-A3B, against the Step 0.5 baseline reference table. Whichever recipe measures best on the metrics users actually care about, lock it in.

The +1 week budget for Phase A turns into +2–3 weeks if we add MR-GPTQ — the GPTQ Hessian computation is non-trivial to implement correctly and the block-Hadamard-32 kernel needs to be written. Probably worth it if the bench shows imatrix-alone insufficient; not worth it pre-emptively.

## gfx906 viability — the harder question

This is the real follow-up. MR-GPTQ assumes B200-class hardware for both calibration and inference. Hipfire's dev box is gfx906/MI50 — no FP8, no WMMA, dp4a only for INT8. Two separate viability questions:

### Calibration-time viability on gfx906

**Likely OK.** GPTQ calibration is offline — it runs once per model on whatever hardware you have. Hessian inverse Cholesky for a 4096-dim layer is ~100 MB of FP32 in working memory, no special ISA features needed. The forward pass to collect activation Hessians is just FP16 matmul, which gfx906 does fine.

The cost concern is wall-time:
- Standard GPTQ on Llama-3-8B is ~30 min on an A100, ~1 hour on a 4090
- gfx906/MI50 is roughly 0.3× A100 FP16 throughput — so ~100 min for 8B
- For Qwen3.5-9B: ~2 hours, tolerable
- For Qwen3.6-A3B (35B params, 8 active): the calibration sees all parameters not just active ones; ~6–8 hours, painful but tractable

**This is a one-time cost per model.** Not a daily concern. The CPU-side allocator/orchestration cost is the bigger constraint — GPTQ needs ~30 GB of host RAM for a 35B model's Hessian buffers. Manageable.

### Inference-time viability on gfx906 (the harder question)

The MR-GPTQ inference path requires:
1. **Online activation rotation** — block-Hadamard-32 applied to x before each linear layer
2. **MXFP4 dequant** — already designed for gfx906 in the format roadmap §1.4 via LUT-decode + dp4a
3. **Per-block UE8M0 scale** — free via `v_ldexp_f32` (already in our roadmap)

The new requirement is (1). Let me work through whether this is fast on gfx906.

**Hadamard-32 on gfx906 wave64:**
- Input: 32-element FP16 vector per group
- Transform: 5-level butterfly (32 = 2⁵), each level is one add + one subtract
- ISA: gfx906 has `v_pk_add_f16` and `v_pk_sub_f16` (packed FP16 ops). Each can do 2 ops per cycle per lane.
- Wave64 fits 32 elements into half the lanes — wasteful, but cheap to do twice (process two groups per wave64 simultaneously)
- 5 butterfly levels × 1 instruction pair each = 10 instructions per 32-element group
- LDS-side: `ds_swizzle_b32` provides cross-lane shuffles for the butterfly partner indices
- Estimated cost per group: ~5 ns on gfx906 (10 ops × 8 lanes wide effective × 60 CUs at 1.7 GHz)

**Per-token activation rotation cost** (Qwen3.5-9B: hidden 5120, groups of 32 → 160 groups per token per layer × 7 linear layers per layer-block × 36 layer-blocks):
- 160 × 7 × 36 = 40,320 Hadamard-32 invocations per token
- At ~5 ns per group: 200 μs per token
- Decode throughput at ~50 tok/s = 20 ms per token → activation rotation is **1% overhead**

**Compare to our current MFP4G32 FWHT-256 online rotation:**
- FWHT-256 is an 8-level butterfly on 256 elements — 4× the depth, 8× the data
- LDS dependencies across full wave (128 elements per FWHT-256 stride at peak)
- Empirically MFP4G32 decode on gfx906 ran at ~21 tok/s in our Phase 11 test (slower than HFQ4G256's ~24 tok/s, but the kernel is wave32-tier launch bounds)

**The block-Hadamard-32 rotation is structurally cheaper than FWHT-256 on gfx906** — fewer butterfly stages, no cross-wave dependencies, smaller LDS footprint, can pack two groups per wave64. If anything, switching from FWHT-256 to block-Hadamard-32 should be a *perf win* on gfx906, not a cost.

The dp4a inner loop for MXFP4-via-LUT is identical to what the format roadmap §1.4 already plans (kvalues_mxfp4 = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}). The 95%-of-llama.cpp HFQ4 mmq kernel ports 1:1, per the existing analysis.

### Verdict on gfx906 viability

**Yes, MR-GPTQ-MFP4 is realistically accelerable on gfx906.** Specifically:

1. The online activation rotation (block-Hadamard-32) is *cheaper* than the current MFP4 FWHT-256 — fewer butterfly stages, no cross-wave deps, can pack two groups per wave64
2. The dp4a dequant inner loop is byte-identical to HFQ4G256's, which already runs at 95% of llama.cpp on gfx906
3. The per-block UE8M0 scale is one free `v_ldexp_f32` op (same as already planned)
4. The calibration is offline, one-time, ~2 hours per dense 9B / ~6–8 hours per A3B — tolerable

Projected gfx906 perf for MR-GPTQ-MFP4 vs current MFP4G32:
- Prefill (pp512): probably similar to HFQ4G256 — 90–95% of llama.cpp baseline
- Decode: slightly faster than current MFP4 (smaller online rotation) — possibly matches HFQ4's ~24 tok/s

The main caveat is that I have no measurement-supported claim on accuracy regression from FWHT-256 → Hadamard-32 rotation switch. The paper measures on B200 where the kernel cost difference is irrelevant. We'd need to bench both.

## Open questions for measurement

If we decide to invest in MR-GPTQ (decision point: after Phase A Step 0/0.5 numbers establish the baseline), the bench should answer:

1. **Does MR-GPTQ close the MXFP4 gap on Qwen3.5-9B?** Paper claims +3.8 pp on Llama-3.1-8B W4A4. We'd run W4A16 (closer to our regime) and use KLD instead of average task recovery.

2. **Does it stack with imatrix-driven per-tensor bit allocation (L5d)?** The two are theoretically compatible — imatrix picks per-tensor target precision, MR-GPTQ does the per-block calibration for whichever tensors are 4-bit. The paper doesn't claim this combination but doesn't argue against it.

3. **What's the right rotation block size for Qwen3.5/3.6 weight statistics?** The paper tests k ∈ {16, 32, 64, 128} and uses 32 for MXFP4. Hipfire's empirical post-FWHT kurtosis at 256-block scope is ~2.82 (per fivetide). At 32-block scope it'll be lower (less averaging) — possibly closer to Gaussian, where Hadamard's variance-reduction property is exactly what FP4 needs.

4. **Does MR-GPTQ help A3B specifically?** Phase 11 showed pure format-quality work didn't close the A3B spiral. MR-GPTQ is a stronger calibration than imatrix; if any quant-side recipe could shift the spiral, MR-GPTQ is the most likely candidate. Worth a smoke test.

5. **Does the FineWeb calibration corpus generalize to Qwen-trained models?** The paper validates on Llama-3 family. Qwen has different tokenizer + different pre-training distribution. Possible calibration mismatch; would need cross-bench against a Qwen-tuned calibration corpus.

## Recommendation

**Add MR-GPTQ to Phase A Step 5 (L5c) as a third candidate** alongside imatrix-weighted LS and the do-nothing baseline. Don't pre-commit to it — let the Step 0.5 baseline measurements + Step 5 multi-baseline runs drive the decision. The cost-benefit:

- **If imatrix-calibrated MFP4 measures within 5% of UD-Q4_K_XL on KLD**: ship imatrix, defer MR-GPTQ (not worth the implementation cost)
- **If imatrix-calibrated MFP4 measures more than 10% behind UD-Q4_K_XL**: implement MR-GPTQ as the dominant fix (the paper's +3.8 pp accuracy recovery directly addresses this case)
- **If neither measures usefully** (i.e. the format-quality lever is structurally insufficient for MFP4): switch to calibrated MQ4G256 (path A from the rebuttal doc) and treat MFP4 as a hardware-accelerated-arch-only path

**Phase A timeline update:** if MR-GPTQ becomes the path, add 2–3 weeks for:
1. Port a small GPTQ implementation into `hipfire-quantize` (probably ~600 LOC; reference llm-compressor's `GPTQModifier`)
2. Add MSE-optimized scale grid (folds in with L4a/L4b work)
3. Add static activation reordering pass
4. Add `mq_rotate_x_block_hadamard_32` runtime kernel (small and faster than the FWHT-256 kernel it replaces)
5. New quant_type ID `MFP4G32-MR` at qt=29; backward-compatible (existing MFP4G32 files unchanged)

**gfx906-specific:** the block-Hadamard-32 online rotation should be *faster* than the current FWHT-256 path, so adopting MR-GPTQ is not a gfx906-perf regression risk. It might be a small win.

## References

- Castro et al., "Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization", arXiv:2509.23202v3 — the MR-GPTQ paper
- llm-compressor `examples/transform/quip_example.py` — QuIP-style offline Hadamard rotation reference (datafree pipeline; not GPTQ)
- vLLM PR #22486 — runtime online-Hadamard support for CompressedTensors (closed unmerged; provides the inference-side contract)
- QuTLASS — fused B200/RTX5090 kernels for online block-Hadamard (referenced by the paper but not directly inspected here)
- `docs/plans/qwen35-mq4-quality-gap.md` — format roadmap; §3.4 already reserves `format_flags` bits for new rotation kinds and §3.3 reserves quant_type IDs that MR-GPTQ variants could occupy
- `docs/plans/hfp4-fivetide-rebuttal-perspective.md` — the empirical context this investigation responds to
- `docs/quant-formats/hfp4.md` — wire format spec; the reserved `rotation_kind` bits 2–3 accommodate block-Hadamard-32 with no breaking change
- `docs/plans/issue-113-quant-quality-eval.md` (on `chore/113-quant-eval-plan` branch) — the eval harness that would bench MR-GPTQ vs imatrix vs baseline
