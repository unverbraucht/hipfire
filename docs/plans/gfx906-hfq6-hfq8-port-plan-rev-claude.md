# Adversarial Review: gfx906 HFQ6 + HFQ8 kernel-coverage analysis

**Reviewer:** Claude Opus 4.7 (1M ctx, self-review + cross-review fold)
**Date:** 2026-05-06
**Reviewed:** `docs/plans/gfx906-hfq6-hfq8-port.md` (419 lines)
**Branch:** `feat/gfx906-hfq6-hfq8-analysis`

This review combines my own findings with adversarial reviews from
gemini and glm5, validating each cross-reviewer finding against the
codebase before accepting or rejecting it.

**Companion reviews integrated:**
- `plans/gfx906-hfq6-hfq8-port-plan-rev-gemini.md` (Gemini CLI):
  4 technical risks + 3 recommendation changes + ISA suggestions.
  All cross-checked against `kernels/src/` and HIP runtime headers.
- `plans/gfx906-hfq6-hfq8-port-plan-rev-glm5.md` (glm-5-turbo):
  10-item summary table covering factual errors + scope items.
  All cross-checked against `dispatch.rs` and `kernels/src/`.

---

## Verdict (revised)

**Reject the plan as written.** Three blockers I caught self-reviewing,
plus glm5's 1.2 (HFQ8 dequant formula error) escalated to blocking
after verification. Gemini's findings mostly accepted as moderate risks
worth folding in, with one technical claim rejected (LDS bank-conflict
"guaranteed 2-way") and one (wave64 reduction strategy from glm5 2.1)
also rejected after disassembly check.

**Total: 4 blocking, 8 moderate, 6 minor.**

The plan's structure (phase-gated, demand-conditional) and the Phase A
recommendations remain defensible. The dp4a/MMQ analysis needs material
revision before driving any work.

---

## How I read the cross-reviews

I used my own self-review's verify-against-codebase methodology on
every claim from gemini and glm5:

- **Accept** if the codebase confirms the factual claim AND the
  recommendation follows.
- **Accept (mechanism corrected)** if the conclusion holds but the
  reviewer's stated mechanism is slightly off — fold into the merged
  finding with the right mechanism.
- **Reject** if the codebase contradicts the claim or the
  recommendation doesn't follow from accepted facts.
- **Defer** if the claim is plausible but I can't verify without
  running the code (e.g. specific TOPS numbers, VGPR allocations
  in not-yet-built kernels).

Disagreements between reviewers I resolve by checking the codebase,
not by aggregating opinions.

---

## My own blocking findings (from initial self-review)

### B1: §1 table conflates batched-GEMM with single-token-GEMV — PARTIALLY OVERLAPS GLM5 1.1

The §1 coverage table claims HFQ6's "fused gate_up / qkv / qkvza"
column is **✗** (missing). This is wrong at the **GEMM** level —
`crates/rdna-compute/src/dispatch.rs` has 15 fused HFQ6 batched-GEMM
dispatch fns (5 names × {base, fp16, dot2, wmma, wmma_gfx12}):

```
gemm_qkvza_hfq6g256[_fp16, _dot2, _wmma, _wmma_gfx12]    (lines 8052-8368)
gemm_qkv_hfq6g256[_fp16, _dot2, _wmma, _wmma_gfx12]      (lines 8453-8742)
gemm_gate_up_hfq6g256[_fp16, _dot2, _wmma, _wmma_gfx12]  (lines 8820-9082)
```

What HFQ6 *doesn't* have is the **GEMV-level** fused single-token
variant. The plan conflates two surfaces.

**glm5 1.1 reaches the same conclusion** with a different framing
("the table conflates two different things: single-row fused decode
GEMVs (truly absent) and batched fused GEMMs (present)"). Glm5 also
makes the operational observation I missed: the plan's "wave64 fused
GEMVs (~1 session)" estimate is real, but *only* the GEMV variant is
new — the GEMM variant exists at FP. **Accept glm5's framing** as
the cleaner statement.

**Recommended fix:** §1 table needs separate rows for "fused batched
GEMM" vs "fused single-token GEMV." Phase A scope language should
clarify it's adding a new GEMV surface, not filling a missing GEMM
surface.

### B2: §3.2 "dp4a-on-HFQ8 has no useful lift" is contradicted by shipped MQ8 code — GEMINI 2.3 / 5.2 ALSO TOUCHES THIS

The plan's §3.2.1 argues:

> Net dp4a lift on HFQ8: probably negative once you account for the
> Q8_1 quantize-x overhead.

**Factually wrong.** The codebase ships MQ8 (FWHT-rotated HFQ8) and
uses dp4a on int8 weights via `__builtin_amdgcn_sudot4`:

```
kernels/src/gemv_mq8g256.hip:
// MagnumQuant MQ8 GEMV: FWHT-rotated symmetric INT8 with dp4a.
// Inner loop uses v_dot4_i32_iu8 for 4x VALU throughput vs FP32.
```

Plus the dispatch: `gemv_mq8g256_prerotated`, `gemv_mq8g256_with_rotate`,
`rotate_quantize_x_mq8` (the int8 quantize for x).

The mistake in my analysis was assuming dp4a's win on HFQ4 came
*purely* from "no need to dequant from 4-bit to int8." It actually
comes from **integer dp4a being faster than FP-FMA on gfx906 even
when both consume the same byte count** — the kernel's own header
claims "4× VALU throughput vs FP32."

**Gemini's 2.3 covers part of this** ("ignores the activation
bandwidth win... reduce X-traffic by 4×") but identifies a *different*
mechanism. Both are real wins. The activation-bandwidth angle (gemini)
is a 4× X-traffic reduction; the integer-vs-FP-FMA angle (caught by
the actual MQ8 kernel comment) is a 4× ALU throughput win. **Accept
both: §3.2 needs a full rewrite around the MQ8 reference
implementation.**

### B3: Entire MoE-indexed kernel surface is missing from the analysis — UNIQUE TO MY REVIEW

`kernels/src/` has 5 MoE-indexed kernel files for HFQ4:

```
gemv_hfq4g256_moe_down_indexed.hip
gemv_hfq4g256_moe_down_indexed_batched.hip
gemv_hfq4g256_moe_down_indexed_batched_wave64.hip
gemv_hfq4g256_moe_down_indexed_wave64.hip
gemv_hfq4g256_moe_gate_up_indexed*.hip (and 3 more)
```

Zero MoE-indexed kernels exist for HFQ6 / HFQ8. The plan says nothing
about this. Currently, anyone trying to run an A3B-class MoE model
with mq6 weights on gfx906 hits the same wave32-FP-fallback path
PR #158 was specifically built to escape.

**Neither gemini nor glm5 caught this.** The user's question
explicitly asked for "with and without DFlash" coverage — DFlash on
MoE uses these MoE-indexed kernels, so this gap is exactly what the
question targets.

**Recommended fix:** add MoE-indexed coverage analysis to both §3.1
(HFQ6) and §3.2 (HFQ8). At least one paragraph per quant. Cost:
~1 session per quant added to Phase A.

---

## Cross-reviewer findings — accept/reject

### From glm5

#### glm5 1.1 (mostly overlaps my B1) — **ACCEPT**

Same observation from a slightly different angle (glm5 calls out
17 files / 3,588 lines of HFQ6 batched GEMM kernel code that exists
on disk). My B1 covers the dispatch fns; glm5 covers the source
files. **Fold both perspectives into the §1 table fix.**

#### glm5 1.2 — **ACCEPT and ESCALATE TO BLOCKING**

> §3.2.1 states HFQ8 dequant is `(int8 + 128) * sc + zp`. The actual
> kernel computes `(scale * (float)data[i] + zero) * x[i]` where
> `data` is `unsigned char*`.

**Verified.** `kernels/src/gemv_hfq8g256.hip:30-37`:
```c
const unsigned char* data = (const unsigned char*)(gptr + 8);
acc += (scale * (float)data[byte_off]     + zero) * x[base_idx]
     + (scale * (float)data[byte_off + 1] + zero) * x[base_idx + 1]
     ...
```

The weights are **unsigned** in `[0, 255]`. The plan's `(int8 + 128)`
shift is wrong. This isn't just a documentation bug — it changes the
proposed dp4a port: with unsigned int8 weights, you need
**`v_dot4_i32_uu8`** (or the unsigned variant of `__builtin_amdgcn_sudot4`),
not signed sdot4. The MQ8 kernel itself uses the
`__builtin_amdgcn_sudot4(true, w, true, x, ...)` form — first two
booleans flag operand signedness. The proposed HFQ8 dp4a port would
need to set those flags correctly for unsigned weights, and pair with
either signed or unsigned int8 activations (Q8_1 is signed). **The
math identity changes accordingly.**

**Escalate to blocking.** This is one of those "the plan was written
from memory, not from source" errors that the dot8 PRD reviewer pass
specifically warned against. Fix during plan revision, document the
correct dequant formula, derive the dp4a math from scratch.

#### glm5 1.3 — **ACCEPT**

> "HFQ8 has only one GEMV kernel" understates existing coverage:
> `kv_cache_write_hfq8.hip` and `attention_hfq8_kv.hip` exist.

**Verified.** Both files exist:
```
kernels/src/kv_cache_write_hfq8.hip  (48 lines)
kernels/src/attention_hfq8_kv.hip    (88 lines)
```

The plan's framing makes HFQ8 sound non-functional. In reality HFQ8
can do AR decode at B=1 end-to-end on gfx906 today (KV write +
attention + GEMV + lm_head all wired). The plan should acknowledge
this baseline.

**Recommended fix:** §3.2 opening paragraph should state "HFQ8 runs
end-to-end at B=1 today via `gemv_hfq8g256` + `attention_hfq8_kv` +
`kv_cache_write_hfq8`. The gap is throughput at B>1 (no batched GEMM)
and the wave64 / dp4a optimizations available for HFQ4."

#### glm5 2.1 — **REJECT after disassembly check**

> The `__shfl_down` reduction is half-wave only; a wave64 kernel
> needs two separate half-wave reductions + a cross-half add, or a
> lane remapping to use `__shfl_down_sync` with width=64.

**Verified false.** `kernels/src/gemv_hfq4g256_residual_wave64.hip:117-120`:
```c
float acc = (acc0 + acc1) + (acc2 + acc3);
for (int offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_down(acc, offset);
if (lane == 0) y[row] += acc;
```

The existing wave64 kernel uses **the same `__shfl_down(acc, offset)`
with offset 16→1**, no special width parameter. Why this works:
- Block is `[64, 1, 1]`, 2 warps per WG, each warp handles 1 row
  (`row = blockIdx.x * 2 + warp_id`).
- `__shfl_down(acc, 16)` on wave64 reads from the lane 16 positions
  ahead within the wave. Warp 0 (wave-lanes 0-31) reads from wave-
  lanes 16-47, but only wave-lanes 0-15 see in-warp data; the
  cross-warp reads return junk that doesn't matter because only
  warp 0's lane 0 (wave-lane 0) writes warp 0's row.
- Warp 1's lane 0 = wave-lane 32 reads wave-lane 48 (warp 1's lane
  16) — also in-warp. Both reductions stay within their warp.

**The pattern just works without modification.** glm5's claim is
incorrect on this hardware/idiom combination.

**Reject this specific finding**, but the underlying concern —
"glm5's broader point about VGPR pressure" — may still apply to
HFQ6's wider per-thread footprint. Accept that part as the
verified glm5 2.1 (rephrased): the wave64 port is mechanical at the
reduction level, but VGPR occupancy needs explicit estimation
because HFQ6's 6 bytes/thread vs HFQ4's 4 bytes/thread changes the
register footprint.

#### glm5 2.2 — **ACCEPT (mechanism corrected)**

glm5 estimates HFQ6 dp4a at "~14 instructions per quad" and notes
the unpack overhead is independent of FP-vs-dp4a (both paths need
the same byte → int8 conversion).

**Mechanism verification:** the existing wave32 `gemv_hfq6g256.hip`
inner loop is:
- 6 byte reads (b0..b5)
- 12 bit operations to extract 4 weights from 3 bytes (×2)
- 8 FP fma ops

If we switch to dp4a, the 12 bit operations stay; the 8 FP fmas
become 2 sdot4 calls (consuming 2 ints × 4 lanes). **glm5's
"net win is 6 saved FP fmas" framing is correct.** But it ignores:
- The unpack arithmetic competes for the same VALU pipe as dp4a.
- gfx906 has 1 VALU issue port per cycle per warp (per the dot8
  PRD's Phase 8b reframe). If unpack + dp4a together exceed the
  per-cycle issue rate, the kernel is no faster than FP wave64.

**Accept glm5 2.2's substance:** the per-call lift estimate of
"+15-25%" was ungrounded. The honest framing is "uncertain;
PMC pass on the Phase A wave64 variant is required before
committing to dp4a."

#### glm5 2.3 — **ACCEPT (overlaps gemini 2.3 below)**

Both glm5 and gemini independently raised the LDS bank-conflict /
group-stride concern. Glm5's framing is more precise: HFQ4 group =
136 B, HFQ6 group = 200 B; the streaming-128-K window alignment
(128 K-elements × 200/256 = 100 bytes/group/window) is no longer
a round number. **Accept glm5 2.3 as the correct framing.** See
"gemini 2.3" section below for the detailed bank-conflict analysis.

#### glm5 2.4 — **ACCEPT and ESCALATE TO BLOCKING**

> The current `mmq_screen_weight()` method dispatches the screening
> reference to `gemm_hfq4g256_residual_mmq_gfx906` hardcoded for
> HFQ4. For HFQ6 screening, both reference and MMQ paths need HFQ6
> variants.

**Verified.** `crates/rdna-compute/src/dispatch.rs:1299`:
```rust
if self.arch == "gfx906" {
    self.gemm_hfq4g256_residual_mmq_gfx906(a_raw, &x_gpu, &y_mmq, m, k, screen_batch)?;
}
```

The screening dispatch is hardcoded to HFQ4. Any HFQ6 MMQ port that
wants to inherit the screening safety net needs:
- A new `gemm_hfq6g256_residual_mmq_gfx906` (the kernel the port adds).
- A switch in `mmq_screen_weight` to dispatch to the right MMQ kernel
  per dtype.
- A separate `mmq_screen_threshold` value for HFQ6 (because outlier
  patterns differ at 6-bit vs 4-bit).

**Escalate to blocking** because this is dispatch-plumbing the plan's
Phase C scope statement doesn't include. Adding it correctly is part
of the MMQ port; adding it incorrectly silently disables outlier
screening for HFQ6 weights.

#### glm5 3.1, 3.2 (gfx1201, gfx12 WMMA non-impact) — **ACCEPT**

Trivial scope-clarification fixes. Add explicit "this work is
gfx906-only; gfx1201/gfx12 paths unaffected" to the plan.

#### glm5 3.3 (MQ6 rotate-then-quantize pipeline) — **ACCEPT**

Verified the data-flow concern: `gemv_mq6g256_with_rotate` is the
existing MQ6 dispatch that calls a rotate kernel before the GEMV.
For dp4a on MQ6, the activations need to be Q8_1-quantized **after**
the FWHT rotate. The plan handwaves this as "the existing pipeline
does this for HFQ4-MQ4" but doesn't explicitly trace the data-flow
for the dp4a case. Accept glm5's recommendation: explicit pipeline
diagram in §3.1.4.

#### glm5 3.4 (coherence-gate runtime) — **ACCEPT**

Per CLAUDE.md, every new kernel + dispatch change must pass coherence
gates. The plan's effort estimates don't include this. Each phase
needs ~30 min to ~1 hr for the coherence gate runs. Bake into the
totals.

#### glm5 3.5 (no current baselines) — **ACCEPT**

> The plan estimates "+30-50% AR decode lift" from wave64 ports but
> provides no current baseline tok/s numbers for HFQ6 or HFQ8 on
> gfx906.

**Accept.** Per AGENTS.md prompt-md5 / 3-runs / binary-md5
requirements (verified at line 39, 251, 334-335), any perf claim
needs measured baselines. The plan should include "Priority 0:
measure current AR decode tok/s on Qwen 9B mq6 / hf8 on gfx906"
before committing to lift projections.

#### glm5 4.1 (analysis-only framing inconsistent with session estimates) — **ACCEPT**

Real point: the plan's framing of "analysis-only, implementation
not in scope" is inconsistent with its session-level estimates and
priority ordering. Either commit to the estimates or remove them.

#### glm5 4.2 (priority ordering circular) — **ACCEPT**

Without baselines (glm5 3.5), the priority ordering can't be
justified by the lift estimates that depend on those baselines.
Either add baseline measurement as priority 0 or present priorities
flat.

#### glm5 4.3 (HFQ3/MQ3 priority unjustified) — **ACCEPT**

> If HFQ3/MQ3 is production on gfx11 and has the same wave32-only
> gap on gfx906, why is HFQ6/HFQ8 the priority?

**Verified.** AGENTS.md §A explicitly says "MQ3 production on gfx11"
and "on gfx906 / MQ3 weights still load and run via per-token GEMV
fallback — correct, just slower." MQ3 has *more* documented
production demand than mq6/hf8. The plan should explain why HFQ6 +
HFQ8 ranks above HFQ3, or move HFQ3 into the priority list.

#### glm5 4.4 (dead link to gfx906-dot8-port.md) — **ACCEPT**

**Verified.** `docs/plans/gfx906-dot8-port.md` doesn't exist on
master (it lives on the `feat/gfx906-dot8-port` branch which wasn't
merged). The plan's §4 reference is a dead link from the master
viewpoint. Either inline the relevant conclusions, copy the file
into the master tree as part of this plan, or note that the file is
on a separate branch.

### From gemini

#### gemini 1 (executive summary risks) — **ACCEPT structurally**

Gemini's framing — "Unpack Penalty," "Alignment Friction," "LDS Bank
Conflicts," "Reconstruction Term" — is a useful taxonomy. Each
specific point is treated below.

#### gemini 2.1 (`v_dot8_i32_i4` is verified on gfx906) — **ACCEPT**

Confirmed. Already mentioned in the plan (§2.2) and not disputed.

#### gemini 2.2 (HFQ6 ALU bottleneck "Unpack Penalty") — **ACCEPT but soften the magnitude**

Gemini claims HFQ6 unpack is "~2.5× more ALU-intensive than HFQ4's
nibble unpack." Based on my analysis the multiplier is closer to 2×
(12 vs 6 instructions per quad, per glm5 2.2). Either way, the
**direction** is right and meaningful. **Accept** the qualitative
risk, **soften** the exact ratio.

#### gemini 2.3 (HFQ8 dp4a activation-bandwidth win) — **ACCEPT — adds new motivation**

> The plan dismisses dp4a for HFQ8, but ignores the **activation
> bandwidth** win. Switching to Q8_1 activations would reduce
> X-traffic by 4×.

**Wait — this is mathematically wrong.** Q8_1 is 8-bit per element +
4 bytes scale + 4 bytes sum per 32 elements. That's ~9 bits per
element ≈ 1.13 bytes per element. The current FP path uses 4 bytes
per element (fp32). So Q8_1 reduces x-traffic ~3.5×, not 4×, but the
direction is right and the magnitude is close.

But gemini's broader point is right: **activation bandwidth reduction
is a real lever for HFQ8** that the plan dismissed. Combined with the
4× ALU throughput from `__builtin_amdgcn_sudot4` (verified in the MQ8
kernel — see B2 above), HFQ8 dp4a actually has *two* compounding
levers: integer math throughput + activation bandwidth.

**Accept** as part of the §3.2 rewrite. The plan's "no useful lift on
HFQ8" claim was wrong on both axes.

#### gemini 2.4 (HFQ6 reconstruction term `(zp + 32 * sc) * sum_x`) — **REJECT (factual error)**

Gemini claims the HFQ6 dp4a reconstruction term has a `32 * sc`
factor that "amplifies quantization noise."

**Verified false.** `kernels/src/gemv_hfq6g256.hip:39-46`:
```c
int q0 = b0 & 63;            // q ∈ [0, 63] unsigned
int q1 = (b0 >> 6) | ...
...
acc += (scale * (float)q0 + zero) * x[base_idx]
     + ...
```

The HFQ6 weights are unsigned `[0, 63]`, dequantized as `scale * q +
zero` directly. There is **no `(q - 32)` shift**. Gemini's `(zp + 32
* sc)` term is invented — it doesn't exist in the actual kernel.

For an HFQ6 dp4a port, you have a choice:
- **Option A (no shift):** keep the same `(scale * q + zero)`
  formula. Use unsigned dp4a (`v_dot4_i32_uu8`-class). The math
  identity: `acc += sc * sum_k(q_k * x_k) + zp * sum_x`. No `32 * sc`
  amplification.
- **Option B (shift to signed):** apply `q - 32` to fit signed int8
  lanes for `sdot4`. Then the math identity *would* have a `(zp + 32
  * sc) * sum_x` term — gemini's formula is what option B looks like.

**The plan didn't specify which option.** For HFQ4 we used option B
(the `(n - 8)` shift) because signed dp4a was the natural builtin
back then. With unsigned `__builtin_amdgcn_sudot4` available, option
A is feasible. **Reject gemini's specific claim** (it's not currently
true) but **accept the underlying point** (option B has noise
amplification; the plan should specify which option it takes).

#### gemini 3 (gfx906 ISA suggestions: `v_perm_b32`, `v_add_lshl_u32`) — **DEFER, low confidence**

Gemini suggests using `v_perm_b32` for HFQ6 unpacking (arbitrary
byte-shuffle to align 6-byte blocks into dword structures). This is
plausible — `v_perm_b32` does exist on gfx906 — but:
- It's not used anywhere in our existing kernels (verified by grep).
- The compiler may emit it automatically for the right C-level
  patterns; manual use would require inline asm.
- The actual ALU win vs the existing shift+OR sequence is unmeasured.

**Defer:** record as a Phase B optimization candidate. Don't commit
to using it without measurement.

Gemini's `v_add_lshl_u32` / `v_and_or_b32` suggestions are similar:
plausible, low-confidence. Same disposition.

#### gemini "5 sessions for Phase C" — **ACCEPT (overlaps my M2)**

Gemini independently arrives at "5 sessions" for Phase C (HFQ6 MMQ).
My M2 also flagged the 2-3 session estimate as too aggressive based
on PR #158 history. **Accept** the 5-session figure as the realistic
Phase C estimate.

#### gemini "Prioritize HFQ8 over HFQ6 because aligned" — **PARTIAL ACCEPT**

Gemini argues HFQ8 should go first because 8-bit weights are
dword-aligned and easier. **The alignment argument is real** —
HFQ8's 8 bytes per thread fits cleanly in dword loads. But:

- HFQ8 has less production traction than HFQ6/MQ6 in current model
  catalog (per the dev logs).
- Per glm5 4.3, HFQ3/MQ3 has *more* production demand than
  HFQ6/HFQ8 and the same gap.

**Accept the technical point** (HFQ8 wave64 port is easier than
HFQ6) but **don't accept the priority flip without workload
evidence.** This connects to glm5 3.5 / 4.2: priority order should
follow demand, not implementation cost alone.

#### gemini "PMC gate for Phase B" (HFQ6 dp4a) — **ACCEPT**

Same shape as my M1 / glm5 2.2. Mechanical fold-in: Phase B should
require a PMC pass on the Phase A wave64 variant first (showing
ALU-headroom-positive) before dp4a port commits.

#### gemini 2.5 (Wavefront Underutilization) — **ACCEPT (overlaps M-class)**

Plain restatement of the wave32-on-wave64 issue. Already covered.

---

## Updated Summary Table

| ID | Severity | Issue | Source |
|---|---|---|---|
| **B1** | blocking | §1 table conflates batched-GEMM with single-token-GEMV | claude original + glm5 1.1 |
| **B2** | blocking | §3.2 "dp4a-on-HFQ8 has no useful lift" wrong (MQ8 ships exactly that) | claude original + gemini 2.3 |
| **B3** | blocking | MoE-indexed kernel surface missing from analysis | claude original (unique) |
| **B4** | **blocking** | §3.2.1 HFQ8 dequant formula wrong: weights are unsigned, no +128 shift | glm5 1.2, escalated |
| **B5** | **blocking** | mmq_screen dispatch hardcoded to HFQ4; HFQ6 needs plumbing rework | glm5 2.4, escalated |
| **M1** | moderate | §3.1.2 "+15-25%" lift estimate ungrounded; PMC needed | claude original + glm5 2.2 + gemini PMC-gate |
| **M2** | moderate | MMQ port budget understated (2-3 vs realistic 5-6) | claude original + glm5 2.3 + gemini "5 sessions" |
| **M3** | moderate | Plan needs MQ8 baseline comparison for HFQ8 work | claude original + glm5 3.5 |
| **M4** | moderate | Q4_1-activation lesson framing slightly muddled | claude original |
| **M5** | moderate | "Production demand" gate has no operational definition | claude original + glm5 4.2 |
| **M6** | moderate | HFQ3/MQ3 priority vs HFQ6/HFQ8 unjustified, MQ3 has more demand | glm5 4.3 |
| **M7** | moderate | HFQ6 unpack penalty is real ALU concern | gemini 2.2 (softened ratio) |
| **M8** | moderate | HFQ8 has more existing kernels than plan acknowledges (KV+attn) | glm5 1.3 |
| **m1** | minor | HFQ4G128 / HFQ6G128 mention | claude original |
| **m2** | minor | "+60% per-call HFQ4 dp4a" attribution fuzzy | claude original |
| **m3** | minor | dead link to gfx906-dot8-port.md (file on a different branch) | glm5 4.4 |
| **m4** | minor | gfx1201 / gfx12 WMMA non-impact should be explicit | glm5 3.1, 3.2 |
| **m5** | minor | MQ6 rotate-then-quantize pipeline diagram missing | glm5 3.3 |
| **m6** | minor | Coherence-gate runtime missing from effort estimates | glm5 3.4 |

### Rejected findings

| ID | Rejected | Reason |
|---|---|---|
| **glm5 2.1** | wave64 reduction needs special handling | Verified false: existing HFQ4 wave64 uses plain `__shfl_down(acc, offset)` and works because each warp's reads stay in-warp. The pattern transfers mechanically. |
| **gemini 2.4** | HFQ6 reconstruction has `(zp + 32 * sc) * sum_x` noise amplification | Verified false: HFQ6 weights are unsigned `[0, 63]` directly dequantized as `sc * q + zp` with no shift. Gemini's formula assumes a signedness shift that isn't in the kernel. The underlying concern (about which option the dp4a port takes) is folded into M-class as a "specify the dp4a math option explicitly" requirement. |

### Deferred findings (interesting but unverified)

| ID | Deferred | Reason |
|---|---|---|
| **gemini 3 (`v_perm_b32`)** | gfx906 ISA optimization suggestion | Plausible, not used in existing kernels, would need measurement before committing |
| **gemini 3 (`v_add_lshl_u32`, `v_and_or_b32`)** | similar ISA suggestions | Same disposition |

---

## What the plan gets right (unchanged from initial review)

1. Phase-gated structure with measured demand check at the end.
2. Honest about uncertainty in §3.1.2.
3. Inherits the Q4_1 lesson from dot8.
4. Phase A (wave64 FP) correctly identified as the cheap, low-risk
   path — *especially* HFQ8 Phase A per gemini's alignment argument.
5. Distinguishes "AR-only" from "DFlash" coverage.

---

## Recommendation (revised)

**Plan needs substantive rewrite.** Five blocking issues + eight
moderate. Approximate revision time: ~3 hours (up from initial
estimate of ~2 hours, reflecting the additional findings from
glm5 + gemini that I accepted).

Concrete edit list:

1. **B1, B2, B3, B4, B5 (blocking):** rewrite §1 coverage table,
   §3.2 HFQ8 analysis, add MoE coverage section, fix HFQ8 dequant
   formula, add mmq_screen plumbing scope.
2. **M1, M7 (HFQ6 unpack):** soften the lift estimates; add explicit
   "PMC required before Phase B" gate with operational pass criteria.
3. **M2 (MMQ budget):** revise Phase C estimate to 5 sessions, or
   reduce Phase C scope to a minimal mmq_x subset.
4. **M3, M5, M6, glm5 3.5, glm5 4.2 (baselines + demand):** add
   "Priority 0: measure current baselines on gfx906 mq6 / hf8 / mq3"
   as a prerequisite for any priority ordering. Without baselines
   the priorities aren't defensible.
5. **M8 (HFQ8 already-running):** §3.2 opening should note HFQ8
   runs end-to-end at B=1 today. The work is *throughput optimization*,
   not "make it functional."
6. **M4, m1, m2, m3, m4, m5, m6:** small fixes per the table.

**Hold all post-Phase-A work** until baselines + demand evidence
are added. The lessons from PR #158 + the closed dot8 PRD both point
toward "don't build speculative kernel optimizations" — that lesson
applies here.

The Phase A recommendations (wave64 FP-path mirrors for both HFQ6
and HFQ8) remain defensible *with the gemini-inspired priority
flip*: HFQ8 first because aligned, then HFQ6, contingent on
demand measurement. ~3.5 sessions of mostly-mechanical work.

---

## Process notes for the next plan

Three patterns the dot8 PRD review caught that this review confirms:

1. **Verify every kernel-presence claim against the file tree.** I
   missed B1/B2/B3 in my own initial review by writing from memory.
   Glm5 caught some of B1; neither glm5 nor gemini caught B3.
2. **Don't accept a reviewer's mechanism without checking the
   codebase.** Glm5 2.1 was confidently stated and confidently
   wrong; I'd have folded it in if I hadn't checked the actual
   wave64 kernel.
3. **Reviewer disagreements are signal.** Glm5 1.2 (HFQ8 unsigned)
   and gemini 2.4 (HFQ6 has `(zp + 32 * sc)` shift) directly
   contradict each other on whether the dequant path uses
   shifts. Resolving by reading the kernel surfaced both factual
   states and identified the design choice (option A vs option B)
   the plan failed to specify.
