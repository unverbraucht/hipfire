# Adversarial review: `docs/plans/gfx906-hfq6-hfq8-port.md`

**Reviewer:** glm-5-turbo (opencode agent)
**Date:** 2026-05-06
**Original plan:** `docs/plans/gfx906-hfq6-hfq8-port.md` (419 lines, draft)
**Verdict:** Technically grounded in most areas, but contains several
assertions that are wrong, underspecified, or silent on real risk.

---

## 1. Factual errors and mischaracterizations

### 1.1 HFQ6 fused GEMVs: "none" is wrong — but the plan's *intent* is right

§3.1 coverage-gap table says "Fused gate_up / none", "Fused qkv / none",
"Fused qkvza / none". The `fused_*_hfq6g256.hip` single-row decode
kernels do not exist on disk — confirmed. However, the plan *does*
acknowledge (§3.1.1 item 3) that the batched `gemm_*_hfq6g256.hip`
family exists (17 files, 3,588 lines total across all HFQ6 kernel
variants). The table conflates two different things: single-row fused
decode GEMVs (truly absent) and batched fused GEMMs (present, 5
families × {base, fp16, dot2, wmma, wmma_gfx12}). A reader scanning
the table would conclude HFQ6 has zero batched GEMM support, which is
false — the batched `gemm_gate_up_hfq6g256`, `gemm_qkv_hfq6g256`, and
`gemm_qkvza_hfq6g256` families all exist and work on gfx906 today.

**Impact:** the effort estimates for "AR-only coverage" (§3.1.1 Phase A)
overstate the gap. The plan proposes writing `fused_gate_up_hfq6g256_wave64.hip`
etc. as ~1 day of work, but doesn't acknowledge that the *batched*
counterparts already exist and handle the prefill path. Only the
single-row *decode* fused GEMVs are missing.

### 1.2 HFQ8 on-disk format: the dequant formula is unsigned, not signed

§3.2.1 states HFQ8 dequant is `(int8 + 128) * sc + zp`. The actual kernel
(`gemv_hfq8g256.hip:30`) computes `(scale * (float)data[i] + zero) * x[i]`
where `data` is `unsigned char*`. The weights are **unsigned** 8-bit values
in `[0, 255]`, not signed int8 in `[-128, 127]`. The formula is
`scale * q + zero` with `q ∈ [0, 255]` — no `+128` shift exists anywhere
in the codebase for HFQ8.

This error doesn't change the plan's dp4a analysis (the conclusion that
dp4a doesn't help HFQ8 is correct for different reasons), but it
undermines credibility and suggests the plan was written from memory
rather than from the source.

### 1.3 "HFQ8 has only one GEMV kernel" understates existing coverage

The plan says HFQ8 has "only one GEMV kernel." This is true for the
linear algebra path (just `gemv_hfq8g256.hip`), but omits two
infrastructure kernels that exist and work:
- `kv_cache_write_hfq8.hip` (48 lines) — KV cache writeout for HFQ8
- `attention_hfq8_kv.hip` (88 lines) — attention with HFQ8 KV cache

These mean HFQ8 can actually run end-to-end on gfx906 for AR decode at
B=1 today — the attention and KV paths are wired. The plan makes it
sound like HFQ8 is barely functional.

---

## 2. Underspecified technical risks

### 2.1 wave64 HFQ6 port: the `__shfl_down` reduction is half-wave only

Both `gemv_hfq6g256.hip:59-60` and `gemv_hfq8g256.hip:40-41` use
`__shfl_down(acc, offset)` with `offset` from 16 to 1 — a 32-lane
warp reduction. A wave64 kernel (block=[64,1,1]) needs either:
- Two separate half-wave reductions + a cross-half add, or
- A lane remapping to use `__shfl_down_sync` with width=64 (not
  available on gfx906 — `v_shr_i32_b32` with wider mask requires
  CDNA2+).

The plan says the wave64 GEMV is "a direct copy of `gemv_hfq4g256_wave64`'s
structure" and estimates ½ day. But it doesn't document the reduction
strategy. PR #158's wave64 kernels use the "two warp_id halves" pattern,
but those are HFQ4 kernels with 4-bit weights (4 bytes per thread). The
HFQ6 kernel has 6 bytes per thread — the VGPR pressure for staging two
rows' worth of weights while maintaining the two-half reduction is
different and needs explicit analysis. The VGPR budget on gfx906 is 256
dwords/SIMD; the prefetch variant already uses 41-44 (per the HFQ4
prefetch kernel's comment). HFQ6's wider per-thread footprint may push
occupancy down.

**Missing from the plan:** a VGPR occupancy estimate for the wave64
HFQ6 variants, especially the prefetch variant.

### 2.2 HFQ6 dp4a: the unpack cost is understated

§3.1.2 estimates "+15-25% lift over Phase A's FP wave64 path" from dp4a
and calls the unpack overhead "~12 instructions per quad vs HFQ4's ~6."
This is a factor-of-2 increase in unpack arithmetic but only a 15-25%
net lift estimate. The plan doesn't quantify the actual instruction
budget. On gfx906, the dp4a inner loop for HFQ4 does 2 `sdot4` calls
per quad. For HFQ6 with the "decode to int8 then dp4a" strategy, you
need:
- 6 byte loads → 8 int8 values (12 shifts+ORs+ANDs, as stated)
- 2 `sdot4` calls (4 lanes each)
- Total: ~14 instructions per quad

The FP wave64 path does:
- 6 byte loads → 8 float values (12 shifts+ORs+ANDs for unpack)
- 8 FP multiply-adds
- Total: ~20 instructions per quad

So dp4a saves 6 FMA instructions at the cost of... nothing extra (the
unpack is the same). The plan's "net +20-30% per-call lift" claim is
reasonable but the "per-call lift over Phase A's FP wave64" estimate of
+15-25% ignores register pressure — dp4a on gfx906 uses the VALU, same
as the FP path, and the compiler may not be able to overlap the unpack
with the dp4a issue port. The plan should call this out explicitly.

### 2.3 MMQ HFQ6: the 200-byte group stride breaks the existing LDS layout

§3.1.3 notes "per-thread byte count differs — 6 bytes/thread for HFQ6
vs 4 for HFQ4" and says "the same X_STRIDE / bank-conflict diagnostic
from PR #158 will need to be redone." This is correct but dramatically
understates the risk. The HFQ4 MMQ body uses a specific LDS tiling
derived from the 136-byte group stride. The HFQ6 group is 200 bytes.
The X_STRIDE tuning sweep for HFQ4 took 4 days (per the plan's own
citation). The 200-byte stride changes:
- LDS allocation per tile (200 vs 136 bytes per K-group)
- Bank conflict patterns (stride mod 64 changes)
- The streaming-128-K window alignment (128 K-elements × 200/256
  bytes/group = 100 bytes per group per window, not a round number)

The plan estimates 2-3 sessions for the full MMQ port. Given the
HFQ4 MMQ took the majority of PR #158's effort and the HFQ6 variant
has a structurally different group size, 2-3 sessions is aggressive.
The PR #158 LDS bank-conflict issue alone consumed 4 days.

### 2.4 No mention of the `mmq_screen` path needing HFQ6-awareness

§4 correctly notes the `mmq_screen` mechanism needs per-quant threshold
tuning. But the current `mmq_screen_weight()` method (dispatch.rs:1263)
dispatches the screening reference computation to `gemm_hfq4g256_residual_mmq_gfx906`
hardcoded for HFQ4. For HFQ6 screening, the reference path and MMQ path
both need HFQ6 variants. The plan doesn't address this dispatch plumbing.

---

## 3. Missing scope items

### 3.1 HFQ6 `gemv_hfq6g256.gfx1201.hip` exists but is unmentioned

There's a gfx1201-specific HFQ6 GEMV kernel (100 lines) that handles
RDNA4 layout differences. The plan focuses exclusively on gfx906 but
doesn't note that any wave64 port would need to consider whether the
gfx1201 variant also needs a wave64 counterpart. If the wave64 work
is "mechanical mirroring" as claimed, this is low risk — but it should
be noted.

### 3.2 HFQ6 `gemm_*_wmma_k2` variant — does the wave64 port affect WMMA paths?

The HFQ6 kernel family includes `gemm_hfq6g256_residual_wmma_k2.hip`
and `gemm_qkvza_hfq6g256_wmma.hip` (plus `.gfx12.hip` variants). The
plan's wave64 port targets gfx906 specifically, but the WMMA paths are
the gfx11/gfx12 fast paths. No conflict — but the plan should
explicitly state "this work is gfx906-only; gfx11/gfx12 WMMA paths
are unaffected" to prevent a future agent from trying to port wave64
to RDNA3 where it's unnecessary.

### 3.3 MQ6 rotation integration is handwaved

§3.1.4 says the MQ6 rotate pass "produces the rotated x that feeds the
GEMV" and that "the activation Q8_1 quantize must happen after the
rotate." This is stated as a requirement but with no analysis of the
actual pipeline wiring. The current `gemv_mq6g256_with_rotate` dispatch
(call at llama.rs:581) passes the rotate as a separate kernel call
*before* the GEMV. For the dp4a path, the quantize-x kernel
(`quantize_q8_1_mmq_ds4`) needs the rotated activations — but the
rotate kernel outputs FP32 activations. The plan doesn't describe the
data flow: rotate → FP32 → Q8_1 quantize → dp4a GEMM. This is
load-bearing for Phase B and Phase C.

### 3.4 Coherence gate applicability is not discussed

Per CLAUDE.md, "any change to kernels, quant formats, dispatch, fusion,
rotation, rmsnorm, or the forward pass MUST pass coherence-gate.sh."
The plan proposes 10+ new kernel files and dispatch changes but doesn't
mention coherence gate validation at all. Each new kernel variant
needs to pass before merge. The effort estimates should include
coherence gate runtime.

### 3.5 No performance baselines for HFQ6/HFQ8 on gfx906

The plan estimates "+30-50% AR decode lift" from wave64 ports but
provides no current baseline tok/s numbers for HFQ6 or HFQ8 on gfx906.
PR #158 had extensive perf checkpoints (the two 943-line and 539-line
documents in `docs/perf-checkpoints/`). Without a current baseline,
the "+30-50%" estimate is unreproducible. Per AGENTS.md §5: "Don't
claim a perf win without ≥3 fresh-process runs, prompt md5, binary md5."

---

## 4. Structural issues

### 4.1 "Analysis-only" framing limits utility

The plan explicitly says "analysis-only" and "implementation is not in
scope." But it then provides session-level effort estimates (§1 table:
"0.5 session", "2-3 sessions") and priority ordering (§5). These are
implementation commitments dressed as analysis. An analysis doc should
either commit to implementation estimates or not include them.

### 4.2 The priority ordering lacks workload evidence

§5 recommends priorities based on "per-session cost vs expected lift"
but admits "there's measured production demand" is the gating condition.
This is circular: the plan can't measure demand without baselines
(§3.5), and can't justify priorities without demand data. The plan
should either:
- Include a baseline measurement step as priority 0, or
- Remove the priority ordering and present it as a flat gap analysis.

### 4.3 HFQ3 deferred without justification

§3.3 defers HFQ3 to a separate plan. But AGENTS.md §2A states MQ3 is
"production on gfx11" and "gfx906... MQ3 weights still load and run
via per-token GEMV fallback." If HFQ3/MQ3 is production on gfx11 and
has the same wave32-only gap on gfx906, why is HFQ6/HFQ8 the priority?
The plan should explicitly rank HFQ3 vs HFQ6/HFQ8 based on workload
prevalence, not just say "out of scope."

### 4.4 The `gfx906-dot8-port.md` reference is a dead link

§4 references `docs/plans/gfx906-dot8-port.md` as "(closed)." The file
has been deleted. Its conclusions are summarized inline, which is good,
but a reader trying to follow the reference chain hits a dead end. The
plan should either inline the relevant conclusions more completely or
note the file was deleted and where its conclusions were archived.

---

## 5. What the plan gets right

- The dp4a non-applicability to HFQ8 is correctly analyzed (§2.2, §3.2.1).
  The instruction-level reasoning is sound.
- The HFQ6 on-disk format description (200 bytes/group, 4 weights per
  3 bytes) matches the actual kernel code exactly.
- The identification of wave32-on-wave64 hardware as the primary
  throughput loss is correct and well-motivated by PR #158's data.
- The Q8_1 activation reuse claim (§2.3) is correct — the existing
  `block_q8_1_mmq` format works for any int8-dequantized weight path.
- The MMQ complexity acknowledgment (§3.1.3 risks) is honest about the
  LDS bank-conflict and threshold-tuning history.

---

## 6. Summary of recommended changes

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1 | Medium | §3.1 table says "Fused none" but batched GEMMs exist | Clarify table distinguishes fused decode (absent) vs batched prefill (present) |
| 2 | Medium | §3.2.1 HFQ8 dequant formula wrong (`int8 + 128` doesn't exist) | Fix to `scale * q + zero` with `q ∈ [0, 255]` unsigned |
| 3 | Medium | "Only one kernel" omits KV/attention infrastructure | Acknowledge `kv_cache_write_hfq8` and `attention_hfq8_kv` |
| 4 | High | No VGPR occupancy estimate for wave64 HFQ6/HFQ8 | Add per-variant VGPR budget before claiming "½ day" |
| 5 | High | MMQ effort estimate (2-3 sessions) aggressive given HFQ4 history | Re-estimate 4-6 sessions or scope to a minimal mmq_x subset |
| 6 | Medium | `mmq_screen` dispatch needs HFQ6-aware plumbing | Add as explicit scope item |
| 7 | Low | No coherence gate validation plan | Add per-phase gate requirement |
| 8 | Low | No performance baselines | Add baseline measurement as priority 0 |
| 9 | Low | Dead link to `gfx906-dot8-port.md` | Inline or note deletion |
| 10 | Low | HFQ3 priority vs HFQ6/HFQ8 unexplained | Add explicit justification for ordering |
