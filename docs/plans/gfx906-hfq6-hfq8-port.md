# gfx906 HFQ6 + HFQ8 — kernel-coverage analysis

**Status:** Draft (2026-05-06). Branch: master.
**Hardware:** AMD Instinct MI50 (gfx906, Vega 20)
**Predecessor:** PR #158 (gfx906 HFQ4 dp4a + AR-decode optimizations,
merged as `afb84bd`).
**See also:**
- `docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`
  (follow-up #9 from that dev log: "Port the prefetch + dp4a levers
  to HFQ3 / HFQ6 / MQ3 / MQ6").
- `docs/plans/gfx906-dot8-port.md` (closed: Q4_1 activations don't
  preserve coherence; the same accuracy concern applies here).

This document is **analysis-only**. It maps what's missing for HFQ6
and HFQ8 to reach the same kernel-coverage level we have for HFQ4
post-PR-158, separately considering the AR-only and DFlash workloads.
Implementation is not in scope.

---

## 1. Executive summary

| Quant | Bits | Group bytes | GEMV | residual GEMV | fused GEMV (gate_up/qkv/qkvza) | wave64 variant | dp4a/MMQ batched | Notes |
|---|---:|---:|:---:|:---:|:---:|:---:|:---:|---|
| HFQ4 | 4 | 136 / 256 | ✓ | ✓ + prefetch | ✓ + dp4a | ✓ | ✓ MMQ + dot4a (PR #158) | flagship; full coverage on gfx906 |
| HFQ6 | 6 | 200 / 256 | ✓ | ✓ | ✗ | ✗ | ✗ | wave32 only; FP path |
| HFQ8 | 8 | 264 / 256 | ✓ | ✗ | ✗ | ✗ | ✗ | only one kernel total |
| HFQ3 | 3 | 104 / 256 | ✓ | ✓ | ✗ | ✗ | ✗ | wave32 only; FP path |
| HFQ2 | 2 | 72 / 256 | ✓ | ✗ | ✗ | ✗ | ✗ | minimal coverage |

**Bottom line:**

- **HFQ6** has FP-path coverage at the GEMV + residual-GEMV layer
  (wave32) but **no wave64-native variant and no dp4a/MMQ batched
  path**. AR-only decode works today on gfx906 via the wave32 GEMV
  (half-throughput on wave64-native hardware). DFlash verify-pass is
  *broken* — there's no batched HFQ6 GEMM that gfx906 can use; it
  would dispatch to `gemm_hfq6g256_residual_fp16` (FP16 path, wave32,
  no wave64-native equivalent).
- **HFQ8** has *only* a single GEMV kernel. **No residual variant, no
  fused variants, no batched GEMM.** AR decode at B=1 works; anything
  else falls through unsupported paths.
- The **dp4a / dot4a / wave64 / prefetch / MMQ** kernel family from
  PR #158 is **HFQ4-only** and not extensible to HFQ6/HFQ8 without
  significant new work, because the math identity that made dp4a
  efficient (`(n - 8)` shift mapping unsigned 4-bit nibbles into
  signed int8 lanes) doesn't apply to 6-bit or 8-bit weights.

**Estimated work to bring HFQ6 + HFQ8 to feature parity with HFQ4 on
gfx906:**

| Surface | HFQ6 | HFQ8 | Combined effort |
|---|---:|---:|---:|
| wave64 GEMV (AR decode B=1) | 0.5 session | 0.5 session | small |
| wave64 residual GEMV (FFN-down + WO) | 0.5 session | 0.5 session | small |
| Fused gate_up / qkv / qkvza GEMVs | 1 session | 1 session | medium |
| Batched GEMM wave64 (LM-head + verify) | 1 session | 0.5 session | medium |
| **AR-only complete coverage** | **2 sessions** | **2 sessions** | medium |
| MMQ-equivalent dp4a path (DFlash verify) | 2-3 sessions | not viable* | large |
| **DFlash full coverage** | **5 sessions** | (FP-only, ~1) | large |

\* HFQ8 is already 8-bit; "MMQ via Q8_1" is essentially the same
weight precision. The dp4a lever doesn't reduce instruction count
on int8×int8 (it's already 4× per instruction); the only available
lift is `v_dot8_i32_i4` which would require lossy weight repack.
Realistically HFQ8 + DFlash means the FP-fallback path for the
verify pass — same situation as PR #158 had pre-cutover-fix.

---

## 2. Reading the existing HFQ4 wins as a template

PR #158 shipped **5 kernel-level lifts** for gfx906 + HFQ4. To
generalize to HFQ6 / HFQ8, each lift has to be ported with awareness
of the per-quant differences:

### 2.1 The five HFQ4 levers and their quant-dependence

| Lever | What it does | Generalizable to HFQ6? | Generalizable to HFQ8? |
|---|---|---|---|
| **wave64 GEMV (1-row-per-warp half-wave split)** | block=[64,1,1] with 2 rows per WG; halves grid count vs wave32 | yes — purely a topology change, no quant-format dependence | yes |
| **ILP-prefetch in residual GEMV** (commit `3ef127d`) | software-pipelined per-quad weight prefetch | yes — 4-quad-per-iter pattern works at any nibble width | yes (8-byte-per-thread pattern) |
| **dp4a substitution** in fused GEMVs (gate_up, qkv, qkvza) | `v_dot4_i32_i8` instead of FP-FMA on dequantized nibbles | **partial** (see §3.1: nibble width mismatches int8 lanes) | **no useful lift** (already int8) |
| **dp4a-MMQ batched GEMM** (the `_mmq_gfx906_x{8..64}` family) | Q8_1 activations × HFQ4 weights via dp4a | **partial** (same 6-bit / int8 mismatch) | **no useful lift** |
| **LM-head dp4a port** (commit `cdcd43d`) | dp4a applied to `gemm_hfq4g256_wave64` for the batched output projection | partial / no | no useful lift |

### 2.2 Why the dp4a class is HFQ4-specific

The HFQ4 dp4a kernels work because:

1. A 4-bit unsigned nibble `n ∈ [0, 15]` maps cleanly to a signed
   int8 lane via `(n - 8) ∈ [-8, 7]`. The +8 offset folds into the
   reconstruction term `zp_eff = zp + 8 * sc`, which is then accounted
   for via the per-block `sum_x` reduction.
2. `v_dot4_i32_i8` packs 4 int8 lanes per int32 register. Each lane
   holds one 4-bit weight (after the shift). 32 lanes (= one half-wave)
   process 4 ints × 8 lanes = 32 K-elements per dp4a call.
3. The Q8_1 activation format already exists in stock llama.cpp's
   prefill MMQ; the gfx906 quantize kernel was already in the tree.

For **HFQ6**:
- A 6-bit weight `q ∈ [0, 63]` doesn't map to int8 lanes one-per-byte;
  4 of them pack across 3 bytes. Two natural strategies:
  - **Decode to int8 then dp4a** — works correctly, gives ~half
    the dp4a-on-int4 benefit because we still have to shift each
    6-bit value into an int8 lane. Net per-call lift over FP-FMA:
    likely +20-30% (vs HFQ4's +60%).
  - **Hand-pack into int8 lanes with bias correction** — keeps
    the 6-bit lanes and uses a 6-bit-aware reconstruction. More
    complex, marginally faster. Probably not worth it.
- The Q8_1 activation format is unchanged; the *weight* unpacking
  is the new work.

For **HFQ8**:
- Weights are already int8. dp4a gives **no per-instruction speedup**
  — `(n - 128) → signed int8` is a no-op equivalent; `v_dot4_i32_i8`
  consumes the same byte count. The lever doesn't apply.
- The only available throughput improvement on gfx906 for HFQ8 +
  Q8_1 activations would be to use `v_dot4_i32_i8` directly with
  Q8_1 activations — but that's the same path stock MMQ already
  takes for Q8_0 weights. Worth porting (~0.5 session) for the
  batched GEMM, but the lift is purely "cleaner kernel" not
  "faster math."

### 2.3 The activation-side question (Q8_1 reuse)

PR #158 reuses the existing `block_q8_1_mmq` activation format across
all 5 HFQ4 dp4a kernels (one quantize-x kernel feeds all of them). For
HFQ6 the same scratch can be reused — int8 activations work for HFQ6
weights regardless of the unpack strategy. **No new activation format
is needed for HFQ6 dp4a.**

For HFQ8 the activation format is already what we'd want.

---

## 3. Per-quant porting plan

### 3.1 HFQ6 — the realistic target

**On-disk format:** 200 bytes per 256-element group = 8 bytes header
(fp32 scale + fp32 zero-point) + 192 bytes packed (4 weights per 3
bytes × 64 groups of 4).

**Coverage gap (gfx906 specifically):**

| Path | HFQ6 today | HFQ4 reference | Gap |
|---|---|---|---|
| Plain GEMV | wave32 (`gemv_hfq6g256.hip`) | wave64-native | needs wave64 variant |
| Residual GEMV | wave32 (`gemv_hfq6g256_residual.hip`) | wave64 + ILP-prefetch (`3ef127d`) | needs wave64 + prefetch |
| Fused gate_up | none | wave64 + dp4a (cd75833) | needs wave64; dp4a optional |
| Fused qkv | none | wave64 + dp4a (7cff629) | needs wave64; dp4a optional |
| Fused qkvza | none | wave64 + dp4a | needs wave64; dp4a optional |
| Batched GEMM (LM-head + verify) | `gemm_hfq6g256_residual` (wave32 FP) | wave64 + dp4a (cdcd43d) | needs wave64 + dp4a; MMQ optional |
| MMQ batched (DFlash verify hot path) | none | `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}` (PR #158 redesign) | needs full MMQ port — biggest gap |

#### 3.1.1 Phase A: AR-only coverage (~2 sessions)

Three small wave64 ports that mirror PR #158's wave64 work without
the dp4a complexity:

1. `gemv_hfq6g256_wave64.hip` (block=[64,1,1], 2 rows/WG via warp
   split). Direct copy of `gemv_hfq4g256_wave64`'s structure with
   the 6-bit unpack from the existing `gemv_hfq6g256.hip`.
   Per-thread workload: 8 weights = 6 bytes (already correctly
   handled by the existing wave32 unpack). ~½ day.
2. `gemv_hfq6g256_residual_wave64.hip` + ILP-prefetch variant.
   Mirror of `gemv_hfq4g256_residual_wave64_prefetch.hip`. 4-quad
   interleave + software pipeline applied to the 6-bit unpack.
   ~½ day.
3. `fused_gate_up_hfq6g256_wave64.hip`, `fused_qkv_*`,
   `fused_qkvza_*` — three more kernels mirroring the PR #158
   wave64 fused FP path. ~1 day.

**Expected lift on AR decode 9B mq6:** the current path uses wave32
GEMVs, which the predecessor's Phase 1 audit noted run at half
throughput on wave64-native hardware. Converting to wave64 should
recover that — empirically PR #158's wave64 GEMVs delivered 1.5-2×
over the wave32 baseline. **Estimated lift: +30-50% AR decode**
on Qwen 9B mq6 / 27B mq6.

No correctness risk: the math is the FP path that mq6 already uses.

#### 3.1.2 Phase B: dp4a port for fused GEMVs (~1 session, optional)

Apply the dp4a substitution to the three fused GEMVs introduced in
Phase A. The 6-bit weight unpacks to int8 lanes; per-thread workload
is 8 weights = 8 int8s = 2 ints. The dp4a inner loop is 2 calls per
quad (vs HFQ4's 1) because each int holds 4 lanes' worth of 6-bit
weights. Estimated per-call lift over Phase A's FP wave64 path:
+15-25% on memory-bound kernels (smaller than HFQ4's +30-50% because
the unpack overhead is larger).

**Activation format:** reuse `block_q8_1_mmq` from PR #158. No new
quantize kernel needed.

**Risk:** the unpack from 6 bytes → 2 int8 packs requires more
arithmetic per K-element than the HFQ4 unpack. The compiler will
need to emit ~12 instructions per quad vs HFQ4's ~6. May leave
ALU-bound ceiling unchanged or even regress on memory-bound kernels.
**Worth a per-kernel PMC pass before wide deployment** (same
methodology as the predecessor decode investigation).

**End-to-end estimated lift:** +5-10% AR decode on top of Phase A,
*if* PMC shows the kernels are still ALU-headroom-positive after
the wave64 port.

#### 3.1.3 Phase C: HFQ6 MMQ batched path (~2-3 sessions)

This is the biggest gap. The current `gemm_hfq6g256_residual_*` is
wave32 + FP only. For DFlash verify-pass on Qwen 27B mq6, this is
the kernel that fires for ~57% of decode time (per PR #158 Phase
14's MMQ share for HFQ4).

The port mirrors the PR #158 redesign:
- nwarps=4, block=(64, 4, 1), `__launch_bounds__(256, 2)`
- runtime-dispatched `mmq_x` ∈ {8, 16, 24, 32, 40, 48, 56, 64}
- 24 entry symbols (8 mmq_x × {bounds-checked, _full_add,
  _full_set}) sharing a templated body
- 128-K window streaming (4 syncs/HFQ6-group)
- per-mmq_x X_STRIDE tuning sweep

**Differences from HFQ4 MMQ:**
- HFQ6 group is 200 B (vs HFQ4's 136 B). The streaming-128-K
  pattern still applies (HFQ6 group = 256 K-elements = 2 Q8_1
  blocks).
- Weight unpack reads 192 packed bytes per group (vs 128 for HFQ4).
  Each thread handles 8 weights = 6 bytes (existing unpack from
  `gemv_hfq6g256.hip`); the unpack inside the streaming loader
  needs to decode 4-of-3-bytes → 4 int8 lanes.
- The same X_STRIDE / bank-conflict diagnostic from PR #158 will
  need to be redone (LDS layout differs because per-thread byte
  count differs — 6 bytes/thread for HFQ6 vs 4 for HFQ4).

**Risks (carried forward from PR #158):**
- mmq_screen_threshold tuning per-quant. HFQ4's 0.50 default was
  empirically PMC-tuned; HFQ6 will need its own sweep.
- LDS bank-conflict pattern at stride 32 — caught the PR #158
  redesign for 4 days. Will likely recur for HFQ6 with the
  different per-thread byte count.
- Real-data NRMSE test at the gfx906 MMQ correctness threshold
  (PR #158 used 0.30%). HFQ6 has more weight precision (6 bits vs
  4) so should pass more easily, but still needs validation.

**Estimated lift:** matching PR #158's HFQ4 MMQ result on Qwen 9B
prefill (5× over wave32 baseline). For 27B mq6 + DFlash, this
unlocks the same +90% DFlash speedup PR #158 delivered for
27B mq4.

**Conditional value:** this phase only matters if there's a real
production workload using mq6 + DFlash. Per the predecessor dev log:
"mq6 typically used for higher-quality smaller models" — mq6 +
27B is unusual. Prefer to *measure* the workload demand before
committing to Phase C.

#### 3.1.4 mq6 (FWHT-rotated HFQ6) considerations

`mq6` is HFQ6 weights with FWHT rotation (per `MagnumQuant`). It
routes through `gemv_mq6g256_with_rotate` for AR decode and the same
wave32 FP path for batched. The wave64 / dp4a / MMQ ports above
benefit mq6 too, with one extra step:

- The MQ rotate pass (`fused_silu_mul_mq_rotate` / `mq_rotate_x`)
  produces the rotated x that feeds the GEMV. For dp4a paths,
  the activation Q8_1 quantize must happen **after** the rotate
  (the current pipeline already does this for HFQ4-MQ4).

No format change needed — mq6 is just HFQ6 with a pre-applied
weight rotation.

### 3.2 HFQ8 — limited gain available

**On-disk format:** 264 bytes per 256-element group = 8 bytes header
+ 256 bytes int8 weights.

**Coverage gap:**

| Path | HFQ8 today | Gap |
|---|---|---|
| Plain GEMV | wave32 (`gemv_hfq8g256.hip`) | needs wave64 variant |
| Residual GEMV | none | needs both wave32 and wave64 |
| Fused gate_up / qkv / qkvza | none | needs wave64 |
| Batched GEMM | none | needs wave64 + (optionally dp4a) |
| MMQ batched | n/a | dp4a doesn't reduce ALU count for int8 weights |

#### 3.2.1 Why dp4a/MMQ doesn't help HFQ8

The HFQ4 dp4a port worked because:
- Weights were 4-bit, dequantized to int8 lanes for free via shift.
- `v_dot4_i32_i8` consumed 4 int8 weights per cycle, matching the
  4-K-element-per-thread pattern.

For HFQ8 weights, they're already int8. There's no shift-into-int8
step to fuse; `v_dot4_i32_i8` consumes the *same* bytes from HBM as
the FP path does (one int8 per K-element). The only lever the dp4a
family provides is the **integer math** itself replacing FP-FMA.

But:
- gfx906's FP32 multiply throughput is comparable to dp4a per cycle
  (~2.6 TFLOPS vs ~22 TOPS dp4a, scaled by per-instruction width).
- dp4a's win on HFQ4 came from "no need to dequant to FP first."
  For HFQ8 the dequant is already trivial: `(int8 + 128) * sc + zp`
  is one fma per element if scale/zp are wave-uniform.

**Net dp4a lift on HFQ8: probably negative** once you account for
the Q8_1 quantize-x overhead (which has to happen for activations,
just like the HFQ4 case).

#### 3.2.2 Where HFQ8 *does* benefit from PR #158's work

The non-dp4a wins still apply:

1. **wave64 GEMV** — 1.5-2× over wave32. ~½ day. Worth doing.
2. **ILP-prefetch on residual** — first need a residual variant
   (which doesn't exist yet for HFQ8); then the prefetch transform
   is mechanical.
3. **wave64 batched GEMM** — straight FP path, mirror of
   `gemm_hfq4g256_wave64.hip`. ~½ day. **No dp4a variant**, just
   FP int8-dequant-then-FMA.

**Estimated AR decode lift on Qwen 9B (hypothetical) hf8:** +30-50%
just from the wave64 ports.

#### 3.2.3 HFQ8 + DFlash

The DFlash verify pass would currently dispatch to whatever batched
GEMM exists for HFQ8, which is **nothing**. It would have to fall
through to a generic FP path or fail.

**Current realistic state:** HFQ8 + DFlash isn't a supported
configuration on gfx906. To make it work:
- Phase 1: add wave64 batched GEMM (`gemm_hfq8g256_wave64.hip`,
  `gemm_hfq8g256_residual_wave64.hip`). ~1 session.
- This gives FP path performance — competitive with stock
  llama.cpp's Q8_0 path on gfx906.
- No further dp4a-class wins available without weight repack.

### 3.3 HFQ3 — out of scope for this analysis

HFQ3 (3-bit) was flagged in the predecessor follow-up #9. Different
quant-format issues than HFQ6/HFQ8 (3-bit packs awkwardly into bytes;
8 weights per 3 bytes via 24-bit pack). **Distinct work plan; not
covered here.** Document only as "same diagnostic-first methodology
as HFQ6/HFQ8 should apply."

---

## 4. Coherence-validation note (carried from dot8 work)

The closed `gfx906-dot8-port.md` PRD established that **int4
activations are not viable for transformer inference on these
models** (Q4_1 NRMSE 18× Q8_1, geometric floor at ~9-12% worst-block
even with asymmetric quant + smaller groups). That conclusion is
load-bearing for HFQ6 / HFQ8 work too:

- The activation format for any HFQ6/HFQ8 dp4a variant must be
  **Q8_1** (the existing format), not Q4_1.
- This is the reason the dp4a-on-HFQ8 lift is small: we'd be
  using Q8_1 activations × int8 weights, which is the same as the
  existing FP path's bandwidth profile, just integer math instead
  of FP-FMA.
- Any future HFQ6/HFQ8 port must inherit PR #158's `mmq_screen`
  + `mmq_screen_threshold` mechanism for outlier-row rejection;
  the threshold will need its own per-quant sweep.

---

## 5. Recommended priority order

Based on per-session cost vs expected lift, and assuming production
workloads use HFQ6 / mq6 more than HFQ8:

| Priority | Phase | Cost | Expected lift | Risk |
|---:|---|---:|---:|---|
| 1 | HFQ6 Phase A (wave64 GEMV + residual + fused, AR-only) | 2 sessions | +30-50% AR decode 9B mq6 | low — FP-path mirror |
| 2 | HFQ8 Phase A (wave64 GEMV + residual + batched FP, AR + minimal DFlash) | 1.5 sessions | +30-50% AR decode hf8 | low — FP-path mirror |
| 3 | HFQ6 Phase B (dp4a port for fused GEMVs, AR optimization) | 1 session | +5-10% on top of Phase A | medium — needs PMC validation per kernel |
| 4 | HFQ6 Phase C (MMQ batched, DFlash verify) | 2-3 sessions | up to +90% on Qwen 27B mq6 DFlash *if anyone uses that combo* | high — rerun the LDS bank-conflict diagnostic |

**Recommendation: do priority 1 and 2 if/when there's measured
production demand for mq6/hf8 on gfx906.** Priority 3 is
diminishing-returns optimization; priority 4 only if 27B mq6 + DFlash
becomes a real workload.

The cost-effectiveness picture: priorities 1+2 deliver the bulk of
the easy lift in ~3.5 sessions of mostly-mechanical wave64 ports
that mirror existing HFQ4 kernels. Priorities 3+4 are real
engineering with PMC/LDS-tuning risk that should be gated on a
specific workload demand.

---

## 6. What's not blocked by this analysis

The `mq3` / `MQ3G256` path (3-bit MagnumQuant) routes through the
HFQ3 family which has the same wave32-only situation as HFQ6. The
same Phase-A-style wave64 ports apply, with HFQ3's tighter packing
(8 weights per 3 bytes) requiring slightly different thread-byte
arithmetic. Out of scope for this document; flagged here so it
isn't lost.

---

## 7. References

- `docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`
  §"Phase 13" (LM-head dp4a port — analogue for HFQ6 batched GEMM)
- `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`
  (the MMQ redesign that priority 4 would mirror)
- `docs/plans/gfx906-dot8-port.md` (closed; relevant for the
  activation-format decision)
- PR #158 (`afb84bd` on master) — the HFQ4 reference implementation
- HFQ4 reference kernels:
  - `kernels/src/gemv_hfq4g256_residual_wave64_prefetch.hip` — the
    ILP-prefetch pattern to mirror
  - `kernels/src/fused_gate_up_hfq4g256_wave64_dp4a.hip` — the
    dp4a-on-fused-GEMV pattern
  - `kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh` — the
    MMQ kernel body to adapt for HFQ6
