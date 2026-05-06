# gfx906 HFQ6 + HFQ8 — kernel-coverage analysis

**Status:** Draft v2 (2026-05-06, post-three-reviewer pass).
Branch: `feat/gfx906-hfq6-hfq8-analysis`.
**Hardware:** AMD Instinct MI50 (gfx906, Vega 20)
**Predecessor:** PR #158 (gfx906 HFQ4 dp4a + AR-decode optimizations,
merged as `afb84bd`).
**Reviews integrated:** `gfx906-hfq6-hfq8-port-plan-rev-{claude,gemini,glm5}.md`
(co-located in this directory). v1 had 5 blocking errors caught by
adversarial review; v2 corrects the factual claims and reorders
priorities by realistic implementation cost + measurement gates.

This document is **analysis-only**. It maps what's missing for HFQ6
and HFQ8 to reach the same kernel-coverage level we have for HFQ4
post-PR-158, separately considering AR-only and DFlash workloads.
Implementation is gated on baseline measurement (Priority 0) and
demonstrated workload demand.

---

## 1. Executive summary

### 1.1 Coverage table

Two distinct surfaces — **batched GEMM** (prefill + DFlash verify, B>1)
and **single-token GEMV** (AR decode at B=1) — must be tracked
separately. v1 conflated them in a single "fused" column.

| Quant | Bits | Group bytes | GEMV (B=1) | residual GEMV (B=1) | fused GEMV (B=1: gate_up/qkv/qkvza) | wave64 GEMV variant | dp4a/MMQ batched | KV cache | attn KV |
|---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HFQ4** | 4 | 136 | ✓ | ✓ + prefetch | ✓ + dp4a | ✓ | ✓ MMQ + dot4a (PR #158) | ✓ | ✓ |
| **HFQ6** | 6 | 200 | ✓ (wave32) | ✓ (wave32) | ✗ | ✗ | ✗ | (n/a) | (n/a) |
| **HFQ8** | 8 | 264 | ✓ (wave32) | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| HFQ3 | 3 | 104 | ✓ | ✓ | ✗ | ✗ | ✗ | (n/a) | (n/a) |
| HFQ2 | 2 | 72 | ✓ | ✗ | ✗ | ✗ | ✗ | (n/a) | (n/a) |

| Quant | batched GEMM (B>1: gate_up/qkv/qkvza/residual) | wave64 batched GEMM | dp4a batched | MoE-indexed |
|---|:---:|:---:|:---:|:---:|
| **HFQ4** | ✓ (multiple variants) | ✓ | ✓ MMQ | ✓ wave64 |
| **HFQ6** | ✓ (base / fp16 / dot2 / wmma / wmma_gfx12 — 15 dispatch fns total) | ✗ | ✗ | ✗ |
| **HFQ8** | ✗ | ✗ | ✗ | ✗ |

**Bottom line:**

- **HFQ6 has full FP-path coverage at the *batched GEMM* level for
  prefill / DFlash verify** (5 fused families × {base, fp16, dot2,
  wmma, wmma_gfx12}). The gap on gfx906 is everything else: no
  wave64 variant, no single-token fused GEMVs, no dp4a/MMQ batched.
- **HFQ8 runs end-to-end at B=1 today on gfx906** via
  `gemv_hfq8g256` + `attention_hfq8_kv` + `kv_cache_write_hfq8`.
  The gap is throughput at B>1 (no batched GEMM at all) and the
  wave64 / dp4a optimizations available for HFQ4.
- **MoE-indexed kernel coverage is HFQ4-only**. Five MoE kernel
  files exist for HFQ4 (down + gate_up, indexed + batched
  variants). Zero exist for HFQ6 or HFQ8. A3B-class models with
  mq6 weights on gfx906 fall through to wave32 FP fallbacks.

### 1.2 Lever availability at a glance

| Lever | HFQ4 | HFQ6 | HFQ8 | Why |
|---|:---:|:---:|:---:|---|
| wave64 topology (1.5–2× over wave32 on wave64 native HW) | ✓ shipped | applicable | applicable | mechanical port; no quant dependence |
| ILP-prefetch in residual GEMV | ✓ shipped | applicable | applicable | mechanical port; per-thread byte count differs (HFQ6=6, HFQ8=8) but pattern transfers |
| dp4a (int8×int8 via `__builtin_amdgcn_sudot4`) | ✓ shipped | applicable (int8 unpack from 6-bit) | **shipped as MQ8**, see §3.2 | gfx906 has the instruction; works for any int8-dequantizable weight |
| dot8 (`v_dot8_i32_i4`, int4×int4) | ✓ HFQ4 native | **NO — would require lossy 6→4 repack** | **NO — would require lossy 8→4 repack** | hardware is int4×int4 only, no mixed-precision; see §2.4 |
| MFMA / WMMA (CDNA2+ / RDNA3+) | n/a on gfx906 | n/a on gfx906 | n/a on gfx906 | hardware not available |

The dot8 lever **does not apply** to HFQ6 or HFQ8 — see §2.4 for the
full reasoning.

### 1.3 Estimated effort to feature parity (revised v2)

| Surface | HFQ6 | HFQ8 | Gemini-revised |
|---|---:|---:|---|
| wave64 GEMV (AR decode B=1) | ~½ session | ~½ session | HFQ8 trivial (aligned); HFQ6 needs split-load handling |
| wave64 residual GEMV + ILP-prefetch | ~½ session | ~½ session | HFQ8 first |
| Single-token fused GEMVs (gate_up / qkv / qkvza) | ~1 session | ~1 session | new GEMV-level surface |
| Wave64 batched GEMM | ~1 session | ~1 session | HFQ8 has no batched at all |
| MoE-indexed kernels (5 files per quant) | ~1 session | ~1 session | A3B / MoE workload coverage |
| **AR-only complete coverage** | **~3 sessions** | **~2.5 sessions** | both gemini- and glm5-validated |
| dp4a port for fused GEMVs (HFQ6 only — HFQ8 already done as MQ8) | ~1 session | n/a (use MQ8) | gemini PMC-gate before commit |
| MMQ-equivalent dp4a path (DFlash verify) | **5 sessions** (was 2-3 in v1) | **not viable** (no dot8 lever; dp4a-batched ≅ MQ8) | gemini + glm5 + PR-158-history-aligned |

**v2 caveat: every lift estimate above is gated on baseline measurement
(Priority 0).** v1's `+30-50% AR decode` claim was unbacked by gfx906
measurements of mq6/hf8. The revised numbers in §3 are
order-of-magnitude estimates pending Priority 0 results.

---

## 2. Reading the existing HFQ4 wins as a template

PR #158 shipped 5 kernel-level lifts for gfx906 + HFQ4. To generalize
to HFQ6 / HFQ8, each lift has quant-format dependence to track.

### 2.1 The five HFQ4 levers and their quant-dependence

| Lever | What it does | HFQ6 portability | HFQ8 portability |
|---|---|---|---|
| **wave64 GEMV (1-row-per-warp half-wave split)** | block=[64,1,1] with 2 rows per WG; halves grid count vs wave32 | yes — pure topology; no quant dependence | yes |
| **ILP-prefetch in residual GEMV** (commit `3ef127d`) | software-pipelined per-quad weight prefetch | yes — pattern transfers; VGPR cost scales with per-thread byte count | yes |
| **dp4a substitution** in fused GEMVs | `v_dot4_i32_i8` (signed) or `__builtin_amdgcn_sudot4` (signed/unsigned flagged) instead of FP-FMA on dequantized weights | yes — 6-bit unpacks to int8 lanes via shifts; uses unsigned 6-bit weights | **already shipped as MQ8** |
| **dp4a-MMQ batched GEMM** (the `_mmq_gfx906_x{8..64}` family) | Q8_1 activations × HFQ4 weights via dp4a + LDS streaming | yes (with new LDS layout for 200-byte group); see §3.1.4 | not directly applicable; MQ8 already provides the equivalent at B>1 single-token |
| **LM-head dp4a port** (`cdcd43d`) | dp4a applied to `gemm_hfq4g256_wave64` for batched output projection | yes — same structural mirror | yes (mirror on int8 weights) |

### 2.2 Why the dp4a class is HFQ4-friendly

The HFQ4 dp4a kernels work because:

1. A 4-bit unsigned nibble `n ∈ [0, 15]` maps cleanly to a signed
   int8 lane via `(n - 8) ∈ [-8, 7]`. The +8 offset folds into the
   reconstruction term `zp_eff = zp + 8 * sc`, accounted for via the
   per-block `sum_x` reduction.
2. `v_dot4_i32_i8` (or its signed/unsigned flagged variant
   `__builtin_amdgcn_sudot4`) packs 4 int8 lanes per int32. Each
   lane holds one 4-bit weight after the shift. 32 lanes (one
   half-wave) process 4 ints × 8 lanes = 32 K-elements per dp4a call.
3. The Q8_1 activation format (`block_q8_1_mmq`) was already in the
   tree from stock llama.cpp's prefill MMQ; the gfx906 quantize-x
   kernel was reused.

For **HFQ6** (`q ∈ [0, 63]` unsigned, 4 weights per 3 bytes):
- Two natural dp4a strategies:
  - **Option A (no shift, unsigned):** keep `acc = sc * q + zp` with
    `q ∈ [0, 63]`. Use `__builtin_amdgcn_sudot4` with `unsigned`
    flag set on the weight side (Q8_1 activations are signed, so the
    flag combination is `(unsigned, signed)`). Math identity:
    `acc += sc * sum_k(q_k · x_int8_k) + zp · sum_x_fp32`. No
    noise-amplifying shift term.
  - **Option B (shift to signed):** apply `q - 32` to fit signed
    int8 lanes for `sdot4`. Then `zp_eff = zp + 32 * sc` and the
    `(zp + 32 * sc) * sum_x` reconstruction term has more
    sensitivity to `sum_x` quantization noise than the equivalent
    HFQ4 `(zp + 8 * sc)` term (4× larger coefficient).

  **Decision: option A.** The unsigned dp4a builtin exists; option B
  has no advantage. Document this in the kernel comments.
- Q8_1 activation format unchanged; the *weight* unpacking is the
  new work.

For **HFQ8** (unsigned `[0, 255]`, 8 weights = 8 bytes per thread):
- Already int8. dp4a gives a real 4× ALU throughput win vs FP-FMA per
  the gfx906 spec — and **this is what MQ8 already ships** via
  `__builtin_amdgcn_sudot4(true, w, true, x, ...)` (both unsigned).
  See `kernels/src/gemv_mq8g256.hip` for the reference implementation.
- For *un-rotated* HFQ8 weights (no FWHT), the same kernel pattern
  applies with the only difference being the absence of the rotate
  step on the activation side.

### 2.3 The activation-side question (Q8_1 reuse)

PR #158 reuses `block_q8_1_mmq` across all 5 HFQ4 dp4a kernels (one
quantize-x kernel feeds all). For HFQ6 the same scratch can be reused
— int8 activations work for HFQ6 weights regardless of the unpack
strategy. **No new activation format needed for HFQ6 dp4a.**

For HFQ8 the activation format is already what we'd want.

### 2.4 dot8 (`v_dot8_i32_i4`) — explicitly ruled out for HFQ6 + HFQ8

`v_dot8_i32_i4` is gfx906's 8-way **int4 × int4 → int32** dot product
(both operands packed as 8 int4 lanes per int32). Verified in
`/opt/rocm/include/hip/amd_detail/math_fwd.h:66`:

```c
int __ockl_sdot8(int, int, int, bool);          // signed 4-bit × 4-bit
unsigned int __ockl_udot8(unsigned, unsigned, unsigned, bool);  // unsigned
```

**There is no mixed-precision dot8.** The hardware does int4×int4
only.

For HFQ6 and HFQ8, this means dot8 *cannot be used* without lossy
weight repack:

- **HFQ6 + dot8:** would require repacking 6-bit weights to 4-bit
  (throw away 2 bits per weight). After repacking, the weights are
  effectively HFQ4. **Equivalent to "use HFQ4 instead of HFQ6."** Not
  a separate lever.
- **HFQ8 + dot8:** would require repacking 8-bit weights to 4-bit
  (throw away 4 bits per weight). Same problem, worse loss.
  Equivalent to "use HFQ4 instead of HFQ8."
- **Activations at int4 (Q4_1) for dot8 use:** the closed
  `gfx906-dot8-port.md` PRD (on the `feat/gfx906-dot8-port` branch
  — not in master) measured Q4_1 activations at 18× Q8_1 NRMSE
  (worst-block 16% on Qwen 9B activations). Asymmetric quant + smaller
  groups + stochastic rounding all failed to clear a 5% gate. The
  geometric floor is ~9-12% worst-block; not viable for transformer
  inference.

**Conclusion: dot8 is HFQ4-territory only.** For HFQ6 / HFQ8, the
ceiling per-instruction throughput on gfx906 is `dp4a` /
`__builtin_amdgcn_sudot4` at 4× FP-FMA. No further per-instruction
throughput is reachable on Vega 20 hardware (MFMA is CDNA1+).

---

## 3. Per-quant porting plan

### 3.1 HFQ6

**On-disk format:** 200 bytes per 256-element group = 8 bytes header
(fp32 scale + fp32 zero-point) + 192 bytes packed (4 weights per 3
bytes × 64 groups of 4). Weights are unsigned `q ∈ [0, 63]`. Dequant
formula: `acc += (sc * q + zp) * x_k` directly — no signed shift.

**What HFQ6 has on gfx906 today:**

| Path | HFQ6 today | HFQ4 reference | Gap |
|---|---|---|---|
| Plain GEMV (B=1) | wave32 (`gemv_hfq6g256.hip`) | wave64-native | needs wave64 variant |
| Residual GEMV (B=1) | wave32 (`gemv_hfq6g256_residual.hip`) | wave64 + ILP-prefetch | needs wave64 + prefetch |
| **Batched fused GEMM (B>1)** | **15 dispatch fns: gate_up + qkv + qkvza × {base, fp16, dot2, wmma, wmma_gfx12}** | wave64 + dp4a | base ✓; needs wave64; dp4a optional |
| Single-token fused GEMV (B=1) | none | wave64 + dp4a (cd75833) | needs new GEMV-level surface; dp4a optional |
| Batched GEMM (LM-head + verify) | `gemm_hfq6g256_residual` (wave32 FP) | wave64 + dp4a (cdcd43d) | needs wave64; dp4a or MMQ optional |
| MMQ batched (DFlash verify hot path) | none | `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}` (PR #158) | needs full MMQ port — biggest gap |
| MoE-indexed kernels | none | 5 HFQ4 kernels (down + gate_up, indexed + batched + wave64) | needs 5 HFQ6 kernels for A3B-class workloads |

#### 3.1.1 Phase A: AR-only coverage (~3 sessions, no dp4a)

Five wave64 ports that mirror PR #158's wave64-FP work:

1. `gemv_hfq6g256_wave64.hip` (block=[64,1,1], 2 rows/WG via warp
   split). Direct copy of `gemv_hfq4g256_wave64`'s structure with
   the 6-bit unpack from the existing `gemv_hfq6g256.hip`.
   Per-thread workload: 8 weights = 6 bytes. ~½ session.
   - **Reduction:** existing wave64 HFQ4 kernels use plain
     `__shfl_down(acc, offset)` with `offset` 16→1; this works on
     wave64 because each warp's reads stay in-warp (verified in
     `gemv_hfq4g256_residual_wave64.hip`). No special handling needed.
   - **VGPR estimate:** unknown; HFQ6's 6 bytes/thread is wider than
     HFQ4's 4. Phase A entry-gate: extract VGPR via
     `clang-offload-bundler --type=o --unbundle | llvm-readelf
     --notes` after first build. If > 96 VGPR, occupancy concern;
     adjust before continuing.
2. `gemv_hfq6g256_residual_wave64.hip` + ILP-prefetch variant.
   Mirror of `gemv_hfq4g256_residual_wave64_prefetch.hip`. ~½ session.
3. `fused_gate_up_hfq6g256_wave64.hip`, `fused_qkv_*`,
   `fused_qkvza_*` — three new GEMV-level kernels (these don't exist
   today; the `gemm_*_hfq6g256` family covers the *batched* case
   only). ~1 session.
4. `gemm_hfq6g256_wave64.hip` for the LM-head batched GEMM. ~½ session.
5. MoE-indexed: `gemv_hfq6g256_moe_down_indexed_wave64.hip` etc. (5
   files mirroring the HFQ4 family). ~1 session.

**Expected lift:** TBD by Priority 0 baseline. Direction: positive
from wave32 → wave64 transition (1.5-2× was empirically observed for
HFQ4 in PR #158); HFQ6 should be in the same ballpark with possible
penalty from the wider per-thread footprint.

No correctness risk: math is the FP path that mq6 already uses.

#### 3.1.2 Phase B: dp4a port for fused GEMVs (~1 session, optional, PMC-gated)

Apply the dp4a substitution to the GEMV-level fused kernels from
Phase A. The 6-bit weight unpacks to int8 lanes via shifts; per-thread
workload is 8 weights = 8 int8 = 2 ints. Inner-loop arithmetic:

| Path | Bit ops | dot/FMA | Total |
|---|---:|---:|---:|
| Phase A FP wave64 | 12 (unpack) | 8 FMA | ~20 |
| Phase B dp4a | 12 (unpack) | 2 sdot4 | ~14 |

Net win is "saves 6 FMA instructions; unpack arithmetic is the same."

**Activation format:** reuse `block_q8_1_mmq` from PR #158. The
unsigned-weight, signed-activation combination uses
`__builtin_amdgcn_sudot4(false, w, true, x, ...)` (first bool flags
unsigned weights).

**Math identity (option A from §2.2):**

```
acc += sc * sum_k(q_k · x_int8_k) + zp * sum_x_fp32
       ^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
       2× sudot4 per quad           same as Q8_1 dp4a
```

`zp_eff = zp` (no shift). No noise amplification beyond Q8_1's
existing geometry.

**PMC entry-gate (per gemini's recommendation, accepted in review
M1):** before committing to Phase B, run a PMC pass on the Phase A
wave64 variant. If VALUBusy < 50% on the production hot kernel, the
Phase B dp4a substitution may be net-negative (the unpack arithmetic
already saturates the VALU pipe; saving 6 FMA doesn't help). If
VALUBusy ≥ 60%, the dp4a port is likely positive; commit to Phase B.

**Estimated lift:** uncertain. Range from -10% (memory-bound
regression) to +20% (best case for ALU-headroom-positive kernels).
Don't commit lift estimates without measurement.

**`mmq_screen` plumbing:** none required for Phase B (mmq_screen is
only used on the MMQ batched path in Phase C).

#### 3.1.3 Phase C: HFQ6 MMQ batched (~5 sessions, gated)

This is the biggest gap. The current `gemm_hfq6g256_residual_*` is
wave32 + FP only. For DFlash verify-pass on Qwen 27B mq6, this is
the kernel that fires for ~57% of decode time (per PR #158 Phase 14's
MMQ share for HFQ4).

The port mirrors PR #158's redesign:
- nwarps=4, block=(64, 4, 1), `__launch_bounds__(256, 2)`
- runtime-dispatched `mmq_x` ∈ {8, 16, 24, 32, 40, 48, 56, 64}
- 24 entry symbols (8 mmq_x × {bounds-checked, _full_add, _full_set})
  sharing a templated body
- 128-K window streaming (4 syncs/HFQ6-group)
- per-mmq_x X_STRIDE tuning sweep (full re-derivation; HFQ4 strides
  don't transfer)

**Differences from HFQ4 MMQ:**
- HFQ6 group is 200 B (vs HFQ4's 136 B). The streaming-128-K pattern
  still applies (HFQ6 group = 256 K-elements = 2 Q8_1 blocks).
- Weight unpack reads 192 packed bytes per group. Each thread handles
  8 weights = 6 bytes (existing unpack from `gemv_hfq6g256.hip`); the
  unpack inside the streaming loader needs to decode 4-of-3-bytes →
  4 int8 lanes.
- The 6-byte per-thread stride **does not directly cause LDS bank
  conflicts** — actual loads are dword-aligned anyway (the existing
  HFQ6 GEMV reads byte-by-byte and the compiler emits dword loads).
  But the 200-byte group stride changes the LDS allocation per tile
  and changes bank-conflict patterns vs HFQ4's 136-byte group. The
  per-mmq_x stride sweep that took PR #158 4 days will need to be
  redone from scratch.

**Risks (PR #158 history):**
- mmq_screen_threshold tuning per-quant. HFQ4's 0.50 default was
  empirically PMC-tuned. HFQ6 has more weight precision (6 bits vs
  4) so should pass screening more easily, but still needs validation.
- LDS bank-conflict diagnostic cost (PR #158 spent 4 days on it).
- Real-data NRMSE test at the existing 0.30% threshold.

**`mmq_screen` plumbing rework (per glm5 2.4 / B5):** the current
`mmq_screen_weight()` at `dispatch.rs:1263` dispatches the screening
reference computation to `gemm_hfq4g256_residual_mmq_gfx906`
hardcoded for HFQ4. For HFQ6 screening, both reference and MMQ paths
need HFQ6 variants. Add a switch on dtype before the screen
dispatches. **This is part of Phase C scope, not optional.**

**Estimated lift:** if Phase C lands cleanly, similar to PR #158's
HFQ4 MMQ result (5× over wave32 baseline on prefill). For 27B mq6 +
DFlash, this would unlock the same +90% DFlash speedup PR #158
delivered for 27B mq4 — *if* mq6 + 27B + DFlash is a real workload.

**Conditional value:** Phase C only matters with measured production
demand for mq6 + DFlash. Per dev-log notes, "mq6 typically used for
higher-quality smaller models" — mq6 + 27B + DFlash is unusual. **Do
not start Phase C until Priority 0 baselines + workload-demand check
demonstrate the need.**

#### 3.1.4 mq6 (FWHT-rotated HFQ6) data-flow detail

`mq6` is HFQ6 weights with FWHT rotation. It routes through
`gemv_mq6g256_with_rotate` for AR decode and the same wave32 FP path
for batched. The wave64 / dp4a / MMQ ports above benefit mq6 too,
with one extra step:

For dp4a paths, the activation Q8_1 quantize must happen **after**
the rotate. The pipeline is:

```
input fp32 x                         (per layer)
        ↓
  [rotate kernel]                    (fused with rmsnorm via fused_rmsnorm_rotate_mq)
        ↓
  rotated fp32 x_rot
        ↓
  [quantize_q8_1_mmq_ds4]            (NEW: needs to consume x_rot,
                                      not raw x; same as HFQ4-MQ4)
        ↓
  Q8_1 x_q8_scratch
        ↓
  [gemv_*_dp4a kernel]               (consumes x_q8_scratch + HFQ6 weights)
```

The current HFQ4-MQ4 pipeline already handles this correctly — the
quantize-x dispatch reads from the rotated buffer. **For HFQ6-MQ6,
the same dispatch path applies; no new wiring needed at the
runtime layer.** Kernel-level changes are exactly Phase A/B/C above.

#### 3.1.5 ISA opportunities (deferred)

Gemini's review suggested gfx906-specific ISA (`v_perm_b32` for
arbitrary byte shuffle, `v_add_lshl_u32` for fused mask+shift). These
are plausible — gfx906 supports them — but no current kernel uses
them, and the actual ALU win vs the existing shift+OR sequence is
unmeasured.

**Defer:** record as Phase B/C optimization candidates. Don't commit
to using them without measurement.

### 3.2 HFQ8 — most of the win is shipping as MQ8

**Critical correction from v1:** v1 said "dp4a-on-HFQ8 has no useful
lift." This was wrong. The codebase ships **MQ8** (FWHT-rotated
HFQ8) using exactly the dp4a-on-int8-weights pattern. From
`kernels/src/gemv_mq8g256.hip`:

```c
// MagnumQuant MQ8 GEMV: FWHT-rotated symmetric INT8 with dp4a.
// Inner loop uses v_dot4_i32_iu8 for 4x VALU throughput vs FP32.
//
// Weight format per group (258 bytes for 256 elements):
//   [0:2]   f16 scale
//   [2:258] int8[256] quantized FWHT-rotated weights
```

The `__builtin_amdgcn_sudot4(true, wp, true, xp, ...)` call (both
booleans `true` → both operands signed int8) delivers the 4× ALU
throughput PR #158 also exploited for HFQ4. **The lever exists for
HFQ8; it just ships under the MQ8 name.**

#### 3.2.1 On-disk format and runtime status

**HFQ8 weights:** unsigned `q ∈ [0, 255]`, 8 weights = 8 bytes per
thread (dword-aligned). Group is 264 bytes (8-byte header + 256
unsigned bytes).

Dequant formula: `acc += (sc * q + zp) * x_k` directly. **No signed
shift, no `+128` offset** — q is unsigned, treated as `q ∈ [0, 255]`
in the FP path.

**HFQ8 runs end-to-end on gfx906 at B=1 today** via:

| Kernel | Path | Status |
|---|---|---|
| `gemv_hfq8g256` | linear-algebra B=1 | ✓ wave32 FP |
| `kv_cache_write_hfq8` | KV cache write | ✓ |
| `attention_hfq8_kv` | attention with HFQ8 KV cache | ✓ |

The plan's framing of "HFQ8 is barely functional" was wrong. The
work is **throughput optimization at B>1**, not "make HFQ8
functional."

#### 3.2.2 Coverage gap on gfx906

| Path | HFQ8 today | Gap |
|---|---|---|
| Plain GEMV (B=1) | wave32 (`gemv_hfq8g256.hip`) | needs wave64 variant |
| Residual GEMV (B=1) | none | needs both wave32 and wave64 |
| Single-token fused GEMV (B=1) | none | needs wave64 |
| **Batched GEMM (B>1)** | **none** | needs wave64 + dp4a |
| MoE-indexed | none | needs 5 HFQ8 kernels |

#### 3.2.3 Lever map for HFQ8

| Lever | Status | Notes |
|---|---|---|
| wave64 GEMV (1.5–2× over wave32) | not shipped for HFQ8 | mechanical port; HFQ8 is 8-byte-aligned (dword-friendly), strictly easier than HFQ6 |
| ILP-prefetch in residual | not applicable yet | needs residual variant first; then mechanical |
| dp4a on int8 weights × Q8_1 activations | **shipped as MQ8** | use MQ8 if FWHT-rotated weights are acceptable; mirror to HFQ8 if raw int8 weights needed |
| dp4a-MMQ batched (Q8_1 × int8 weights) | not shipped | feasible (mirror of MQ8 at B>1) but ~½ session of work |
| dot8 (`v_dot8_i32_i4`) | NOT applicable (would require lossy 8→4 weight repack) | see §2.4 |
| LM-head dp4a port | not shipped | mechanical mirror of `gemm_hfq4g256_wave64_dp4a` |

#### 3.2.4 Phase A: wave64 + batched GEMM (~2.5 sessions)

Five small ports:

1. `gemv_hfq8g256_wave64.hip` — wave64 mirror of the existing
   wave32. Trivial because 8-byte alignment plays nicely with dword
   loads. ~¼ session.
2. `gemv_hfq8g256_residual.hip` + `gemv_hfq8g256_residual_wave64.hip`
   — new residual surface (doesn't exist today). ~½ session.
3. `gemm_hfq8g256_wave64.hip` — batched GEMM (mirror of
   `gemm_hfq4g256_wave64.hip`). FP path. ~½ session.
4. `gemm_hfq8g256_wave64_dp4a.hip` — dp4a variant of (3). Mirror
   the MQ8 inner loop on un-rotated weights. ~½ session.
5. MoE-indexed: 5 files. ~½ session.

#### 3.2.5 No HFQ8 MMQ port

Q8_1 × int8 weights is structurally the same as MQ8 at B=1. Adapting
the MQ8 kernel for batched B>1 is simpler than a new MMQ port — it's
just batched dp4a, no LDS streaming required (the activations are
already int8 and small enough that they fit in registers per-batch).
**Expected work: covered by Phase A item 4 above.** No separate
"Phase C" for HFQ8.

#### 3.2.6 HFQ8 + DFlash

DFlash verify-pass on a hypothetical 27B hf8 target would currently
have **no batched GEMM kernel to dispatch to**. After Phase A item 3
+ 4, the dp4a-batched path covers it. The verify-pass behaves like
MQ8 verify (which works today via the existing batch=1 kernel called
N times — slow, but functional).

**Production-relevance check:** HFQ8 is rare. Most 8-bit production
deployments use MQ8 (rotated) or Q8_0 (stock llama.cpp). The HFQ8
work matters only if a specific user shipped raw HFQ8 weights without
the FWHT rotation.

### 3.3 HFQ3 / MQ3 — out of scope but flagged

Per AGENTS.md §A: MQ3 is production on gfx11/gfx12, and on gfx906
"MQ3 weights still load and run via per-token GEMV fallback —
correct, just slower." MQ3 has *more* documented production demand
than mq6/hf8 on gfx906.

**The plan's recommended priority list (§5) considers MQ3 alongside
HFQ6/HFQ8 in the demand check.** A separate HFQ3-specific plan
would mirror this document but is not in scope here.

---

## 4. Coherence-validation note (carried from dot8 work)

The closed `gfx906-dot8-port.md` PRD (on the `feat/gfx906-dot8-port`
branch — not in master) established that **int4 activations are not
viable for transformer inference on these models**. Q4_1 NRMSE 18×
Q8_1; geometric floor at ~9-12% worst-block even with asymmetric
quant + smaller groups. That conclusion is load-bearing for
HFQ6/HFQ8 work too:

- The activation format for any HFQ6/HFQ8 dp4a variant must be
  **Q8_1** (the existing format), not Q4_1. (Weight quant choice is
  independent.)
- Any future HFQ6/HFQ8 MMQ port must inherit PR #158's `mmq_screen`
  + `mmq_screen_threshold` mechanism for outlier-row rejection.

**The dot8 lever explicitly does not apply to HFQ6 or HFQ8** — see
§2.4.

---

## 5. Recommended priority order (revised v2)

### 5.0 Priority 0: baseline measurement (~½ session, prerequisite)

**Before any kernel work**, run the canonical AR decode + DFlash
benches on existing mq6 / hf8 / mq3 / mq8 paths on gfx906 with
3-run deterministic medians per AGENTS.md prompt-md5 / binary-md5
requirements:

- Qwen 9B mq6 AR decode tok/s
- Qwen 9B mq3 AR decode tok/s
- Qwen 9B mq4 AR decode tok/s (sanity baseline against PR #158
  numbers)
- Qwen 9B mq8 AR decode tok/s (the dp4a reference for HFQ8 work)
- DFlash 27B mq6 humaneval-0 tok/s (if model exists)

Record absolute tok/s + the comparison matrix. **All lift
estimates below are placeholders pending Priority 0.**

### 5.1 Priority list (post-Priority-0, demand-conditional)

| Priority | Phase | Cost | Expected lift | Risk | Demand gate |
|---:|---|---:|---|---|---|
| 1 | HFQ8 Phase A (wave64 + batched GEMM + dp4a + MoE) | ~2.5 sessions | TBD by P0 | low (HFQ8 is dword-aligned; MQ8 is the reference) | needed if any production deployment uses raw HFQ8 (vs MQ8) |
| 2 | HFQ6 Phase A (wave64 GEMV + residual + fused + batched + MoE, FP-only) | ~3 sessions | TBD by P0 | low — mirror of HFQ4 wave64 work | needed if mq6 has measured production demand |
| 3 | HFQ6 Phase B (dp4a port for fused GEMVs, AR optimization) | ~1 session | TBD; PMC-gated | medium — needs PMC validation | only if Phase A's HFQ6 kernels show ALU headroom |
| 4 | HFQ6 Phase C (MMQ batched, DFlash verify) | **5 sessions** | TBD; up to +90% on Qwen 27B mq6 DFlash *if anyone uses that combo* | high — full LDS bank-conflict diagnostic + mmq_screen plumbing | only if 27B mq6 + DFlash becomes a real workload |
| 5 | (HFQ3) — separate plan, gated on demand vs HFQ6 | — | — | — | likely higher demand than HFQ6/HFQ8 per AGENTS.md |

**Decision rule:** do priorities 1 and 2 *only if* Priority 0 shows a
real workload using these quants on gfx906. Otherwise defer the
entire plan. The lessons from PR #158's diagnostic-first methodology
+ the closed dot8 PRD's negative result both point toward "don't
build speculative kernel optimizations."

### 5.2 Coherence-gate cost (per glm5 3.4)

Each phase needs ~30 min to ~1 hr for `coherence-gate.sh` +
`coherence-gate-dflash.sh` (where applicable) per kernel batch.
Bake into the per-phase totals.

### 5.3 Per-arch scope

This work is **gfx906-only**. gfx11 / gfx12 (RDNA3 / RDNA4) WMMA
paths are unaffected. `gemv_hfq6g256.gfx1201.hip` and other RDNA-
specific HFQ6 kernels need no changes.

---

## 6. What's not blocked by this analysis

- The HFQ4 production path (PR #158 work) is not affected.
- The existing wave32 HFQ6 / HFQ8 paths remain functional throughout
  (Phase A adds wave64 alongside; doesn't remove wave32).
- gfx11 / gfx12 WMMA paths unchanged.
- mq6 / mq8 production deployments continue working at current
  performance unless / until Priority 0 + Phase A land.

---

## 7. References

- `docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`
  §"Phase 13" (LM-head dp4a port — analogue for batched GEMM)
- `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`
  (the MMQ redesign that priority 4 would mirror; spent 4 days on
  the LDS bank-conflict diagnostic alone)
- `docs/plans/gfx906-dot8-port.md` (closed; on `feat/gfx906-dot8-port`
  branch, not in master). Q1.2 NRMSE result documents the Q4_1
  failure mode that constrains §4 of this doc.
- `kernels/src/gemv_mq8g256.hip` — the MQ8 dp4a-on-int8 reference
  implementation; HFQ8 Phase A item 4 mirrors this.
- PR #158 (`afb84bd` on master) — the HFQ4 reference implementation
- HFQ4 reference kernels:
  - `kernels/src/gemv_hfq4g256_residual_wave64_prefetch.hip` — the
    ILP-prefetch pattern to mirror
  - `kernels/src/fused_gate_up_hfq4g256_wave64_dp4a.hip` — the
    dp4a-on-fused-GEMV pattern
  - `kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh` — the
    MMQ kernel body to adapt for HFQ6 Phase C
- Adversarial reviews (folded into v2):
  - `docs/plans/gfx906-hfq6-hfq8-port-plan-rev-claude.md`
  - `docs/plans/gfx906-hfq6-hfq8-port-plan-rev-gemini.md`
  - `docs/plans/gfx906-hfq6-hfq8-port-plan-rev-glm5.md`
- AGENTS.md §A (MQ3 production status), §5 (perf-bench
  reproducibility requirements), CLAUDE.md (coherence-gate
  requirements per kernel change).
