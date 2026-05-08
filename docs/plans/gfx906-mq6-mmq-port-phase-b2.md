# Phase B.2 — HFQ6 MMQ-streaming port (gfx906)

**Status:** scoping (revised 2026-05-08 after three-way review)
**Author:** Claude Opus 4.7
**Predecessor:** Phase B.1 (BT=8→16 on dp4a fused kernels, +14.5 % cumulative; commits `2bee6e6` + `ff9e210`).
**Reference impl:** `kernels/src/gemm_hfq4g256_residual_mmq_gfx906_*.hip` + `_body.cuh` (PR #158).
**Review status:** v2 — folds findings from claude / gemini / glm5 adversarial reviews.

## Revision history

- **v1 (commit `65be766`)** — initial scope, 5 sessions, set+add chain dispatch.
- **v2 (commit `4819ae4`)** — folded combined adversarial review:
  - 3 correctness landmines added to session-1 spec (x_dm, 0.25f, mmq_screen)
  - dispatch architecture corrected (parallel `_set` calls, not chained `_add`)
  - load strategy committed (row-coalesced, 12 uints/tid)
  - GO/NO-GO threshold tightened (≥10 % over wave64_dp4a, was 30 %)
  - session count 5 → 7
  - new validation gates: DFlash spec-verify, 27B model, per-element max-abs-err
  - pre-session-1 checklist added (10 items)
- **v2.1 (this rev, 2026-05-08)** — pre-S1 checklist executed:
  - cherry-picked `5768fe4` → `c54445b` (capture_mode + DDTree on audit)
  - added 4 dp4a unit tests to `test_hfq6_gemm.rs`; gated WMMA tests on gfx11+
  - mmq_screen threshold default verified: gfx906 = 0.50 (max-abs-err per row)
  - mq4 + mq6 baselines confirmed (598.7 / 190.8 / 44.0 tok/s, 3.14× gap)
  - 27B-mq6 model not on disk — gate deferred or skipped
  - DFlash mq6 has no baseline ever — gate dropped, coherence prefill is proxy
  - `should_use_mmq` threshold confirmed at B≥8 for gfx906
  - Plan §4 S5 + §6 GO/NO-GO updated to reflect dropped DFlash gate.

## 1. Goal

Close the prefill gap between mq4 and mq6 on the 9B model at pp128 by
porting the PR-#158-style MMQ-streaming kernel family from HFQ4 to HFQ6.

**Baseline (audit branch `feat/gfx906-hfq6-hfq8-analysis` HEAD = `65be766`):**

| 9B pp128 | Current | Source |
|---|---:|---|
| mq4 prefill | 598.6 tok/s | rocprof, BT=8 baseline |
| mq6 prefill | **189.9 tok/s** | post-B.1.1 (commit `ff9e210`) |
| Gap | **3.15×** | |

> **Note on baseline reconciliation:** PR #187 (which lands Phase A only)
> ships at ~162 tok/s. The 189.9 number includes B.1.1's BT=16 propagation
> (`ff9e210`), which is on this audit branch but not yet in PR #187. After
> the BT=16 commits land upstream, the public number updates to 189.9.

**Bandwidth-bound floor:** mq6 has 1.47× more weight bytes per output
element than mq4 (200 / 136). Floor = `mq4_pp128 / 1.47 = 407 tok/s`.
This is the BW-floor parity target; below this is genuine inefficiency.

**Targets (re-stated as ratios, not absolute tok/s):**

- **Realistic:** ≥ floor × 1.05 → **425 tok/s** at the audit-branch mq4
  reference. Closes the gap from 3.15× to ~1.41× (= 1.47 × 0.97).
- **Stretch:** ≥ floor × 1.10 → **450 tok/s**. Achievable only if
  unpack overhead is fully overlapped with memory loads.

The earlier "≥ 460 tok/s / 30 % gap" framing in v1 was arithmetically
inconsistent (598.6 × 0.7 = 419, not 460) and anchored on the mq4
prefill at the moment of writing. Use the ratio form so the target
auto-tracks future mq4 improvements.

## 2. What we're porting (mq4 reference)

The mq4 prefill on gfx906 dispatches **only** kernels from the
`gemm_hfq4g256_residual_mmq_gfx906_x{N}` family — fused gate_up /
qkv / qkvza are NOT used as standalone kernels at prefill.

### 2.1 Two distinct add modes, two Rust entry points

> **Correction from v1:** the v1 plan described "set + add chain" across
> projections. That's wrong. The HFQ4 dispatchers issue **parallel
> `_set` calls into N separate output buffers**, not chained `_add` into
> a shared buffer. The `_set` vs `_add` modes are orthogonal to projection
> chaining.

The HFQ4 family ships **3 add-mode variants per size:**

- `_x{N}` — runtime `add` argument (data-dependent path, used when
  `m % 128 != 0` or `batch_size % mmq_x != 0`)
- `_full_set_x{N}` — compile-time `add=0` (overwrite Y)
- `_full_add_x{N}` — compile-time `add=1` (Y += sum, residual fuse)

Two Rust dispatcher entry points wrap them:

| Rust fn | Add mode | Used by |
|---|---|---|
| `gemm_hfq4g256_residual_mmq_gfx906` (`dispatch.rs:7005`) | add=1 (residual fuse) | `gemm_hfq4g256_residual()` for **wo / w_down / lm_head** — single MMQ call per layer site, accumulates onto residual stream |
| `gemm_hfq4g256_mmq_set_gfx906` (`dispatch.rs:7125`) | add=0 (overwrite) | Fused dispatchers (`gemm_qkv_*`, `gemm_qkvza_*`, `gemm_gate_up_*`) — **N parallel calls into N separate Y buffers**, each writes its own projection's output |

The HFQ6 port needs **both entry points**.

### 2.2 Fused dispatch pattern (verified from `dispatch.rs`)

`gemm_qkvza_hfq4g256` at B>1 on gfx906 (`dispatch.rs:3429-3456`):
- `_mmq_set` for `y_qkv` (a_qkv weights)
- `_mmq_set` for `y_z` (a_z weights)
- **Falls through to `gemm_qkvza_hfq4g256_fp16_wave64`** for `beta + alpha` — their M=128 wastes 75 % of MMQ_Y=128 row-tiles, so the dispatcher passes `qkv_m=0, z_m=0` and lets the fused kernel handle the small-M tail.

`gemm_qkv_hfq4g256` at B>1 on gfx906 (`dispatch.rs:3854-3873`):
- 3 `_mmq_set` calls into `y_q`, `y_k`, `y_v`.

`gemm_gate_up_hfq4g256` at B>1 on gfx906 (`dispatch.rs:4238-4252`):
- 2 `_mmq_set` calls into `y_gate`, `y_up`.

All three dispatchers call `ensure_q8_1_mmq_x` ONCE before the fan-out
and pass the same `xq` pointer to all sibling MMQ calls (per the
"quantize once and reuse" comment at `dispatch.rs:1217-1220`). The HFQ6
port must mirror this amortization.

### 2.3 Topology (unchanged from HFQ4)

- Block: `(64, 4, 1)` = 256 threads = 4 wave64s
- Grid: `(M / 128, N / mmq_x, 1)` (1-D rows × 1-D batch tiles)
- mmq_y = 128 (rows per block — fixed)
- mmq_x ∈ {8, 16, 24, 32, 40, 48, 56, 64} (batch tile width — runtime
  selected: greedy `mmq_x = min({8,…,64} | mmq_x ≥ batch_size})` for
  small batch, else 64)
- LDS budget per block: 19 KiB at mmq_x=8, up to 30 KiB at mmq_x=64
  (within the 32 KiB → 2 WGs/CU target on the 64 KiB/CU cap)

### 2.4 Per-output-tile algorithm (Option C "Window Streaming")

```
for kg in 0..(K / 256):                       # one HFQ6 group per outer iter
    for window in 0..2:                       # 2 Q8_1 blocks per group
        load_q8_1_tile_coalesced<mmq_x>(...)  # 128B per row × mmq_x cols
        load_hfq6_tile_streaming<x_stride>(...)  # 96B per row × MMQ_Y rows
        __syncthreads()
        for sub in 0..4:                      # 4 sub-blocks of 32 K-elements
            vec_dot_dp4a_streaming<mmq_x>(...)
        __syncthreads()
write_back_*_templated<mmq_x, need_check>(Y, sum, ..., add)
```

### 2.5 Key design choices (carry-over from HFQ4)

- **LDS-staged X (activations) AND A (weights).** The 8-byte sc/zp
  per HFQ6 group plus 96 B of weight bytes per window per row decode
  into LDS once per window; all 4 wavefronts read from LDS for the
  inner dp4a loop. This is the architectural difference from the dp4a
  fused kernels (which keep A in registers via the 1-wave-per-row
  design).
- **Compile-time `add` modes.** `_full_add_x{N}` and `_full_set_x{N}`
  variants compile out the runtime `if (add)` in writeback —
  hot-loop branch elimination.
- **`x_stride` per-mmq_x.** mmq_x ≥ 64 uses stride 40 (b128 ds_read,
  4-way bank conflict, b128 issue rate dominates). mmq_x < 64 uses
  stride 33 (b32 ds_read, 0-way bank conflict). PMC-validated for
  HFQ4 in `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`;
  may need re-tuning for HFQ6 (see session 2).
- **mmq_screen safety net.** A new `mmq_screen_weight_hfq6()`
  function (separate from the HFQ4 one — see §4 session 1) runs each
  weight through MMQ vs FP16 reference at first use; if NRMSE >
  threshold, the dispatcher falls through to FP16 wave64 instead.
  Catches pathological quant groups (#87).

## 3. Deltas: HFQ4 → HFQ6 quant

| Property | HFQ4 | HFQ6 | Impact on port |
|---|---|---|---|
| Bits per weight | 4 | 6 | Unpack changes — 6 source bytes → 8 unsigned weights, vs 4 bytes → 8 signed nibbles |
| Group size (K) | 256 | 256 | **Same.** Outer-loop count unchanged. |
| Group bytes | 136 | **200** | Per-row bytes scale 1.47×. LDS x_qs same size (decoded ints); only the load path widens. |
| Header bytes | 8 (sc + zp) | 8 (sc + zp) | **Same.** |
| Weight bytes per group | 128 | 192 | Window split: HFQ4 = 64 B/window × 2 windows. HFQ6 = 96 B/window × 2 windows. |
| Load volume per tid per window | 32 B (8 uints) | **48 B (12 uints)** | 1.5× per-tid volume. May increase MemUnitBusy during load phase. |
| Sign convention | nibble - 8 (signed for dp4a) | unsigned q ∈ [0, 63], fits int8 directly | See §3.1 below — **two correctness landmines** lurk here. |
| Decoded int8 packing | 4 nibbles → 1 int (8-bit shift each) | 4 q6 → 1 int (q6 fits in int8 since ≤63) | **Same int_a/int_b layout** — both use 4 int8 packed in int32. |

**Key insight:** the HFQ4 MMQ kernel's `int_a` / `int_b` layout
(packed int8, dp4a-consumable) is identical to what HFQ6 produces.
Only the **unpack step** differs (6 source bytes per lane vs 4). The
`vec_dot_dp4a_streaming`, `load_q8_1_tile_coalesced`, and
`write_back_residual_templated` stages are byte-for-byte reusable.

### 3.1 Two correctness landmines in the math identity

These are the highest-risk findings from the three-way review. Both are
silent-failure-class bugs that the v1 plan's NRMSE < 0.5 % validation
**cannot catch** (systematic biases survive NRMSE checks). Both are
reasoned from the math identity below; both are easy to get wrong via
copy-paste from HFQ4.

#### 3.1.1 LANDMINE 1: `x_dm` precomputation must NOT carry the `+ 8·sc` term

**HFQ4 `body.cuh:87`:**
```c++
x_dm[i] = make_float2(sc, zp + 8.0f * sc);  // ← compensates the (n - 8) shift
```

**HFQ6 must use:**
```c++
x_dm[i] = make_float2(sc, zp);              // ← no shift, zp passes through
```

**Math derivation:**

For HFQ4: weights are stored unsigned `n ∈ [0, 15]`, packed as signed
int8 `(n - 8) ∈ [-8, 7]` for dp4a:
```
true_value(k) = sc · n(k) + zp = sc · ((n(k) - 8) + 8) + zp
              = sc · (n(k) - 8) + (zp + 8·sc)
```
After full sub-block sum: `Σ_k true_value · x = sc·sumi + (zp + 8·sc)·sum_x`
⇒ `zp_eff = zp + 8·sc`.

For HFQ6: weights are stored unsigned `q ∈ [0, 63]`, packed directly as
signed int8 (since 63 < 127, fits without shift):
```
true_value(k) = sc · q(k) + zp
```
After full sub-block sum: `Σ_k true_value · x = sc·sumi + zp·sum_x`
⇒ `zp_eff = zp` (no shift compensation).

**Why this is high-risk:** the natural workflow for the port is "copy
`body.cuh`, replace `load_hfq4_tile_streaming` with HFQ6 unpack". The
`x_dm` line lives **inside** `load_hfq4_tile_streaming` (body.cuh:87),
so a careful port replaces it. But if the developer copies the entire
function as a starting template and only edits the unpack arithmetic,
**`zp + 8·sc` survives the port** and biases every output by `+8·sc·sum_x`
per group. NRMSE on synthetic activations (mean-zero after layernorm)
would stay low because `sum_x ≈ 0` per sub-block; per-element
max-absolute-error against a non-mean-zero activation set IS
discriminating.

#### 3.1.2 LANDMINE 2: the `0.25f` factor must NOT leak from the dp4a kernel

**Existing dp4a kernel (`gemm_hfq6g256_residual_wave64_dp4a.hip:116`):**
```c++
acc[b] += sc * d_x * (float)sumi + zp * sum_x * 0.25f;
//                                            ^^^^^
//   per-lane share factor (4 lanes × 0.25 = 1·sum_x per sub-block)
```

**HFQ6 MMQ body must NOT have the `0.25f`:**
```c++
sum[idx] += scale_w * d_x * (float)sumi + zp_eff * sum_x;
//                                                  ^^^^^^
//             no factor — full sub-block per thread, all 32 elements
```

**Why:**

In the dp4a kernel, each lane handles 8 K-elements (out of 32 in a
sub-block). `sum_x` is the **full sub-block-wide** activation sum (32
elements). Each lane's contribution is `1/4` of `zp · sum_x`; after the
warp reduction sums 4 lanes' contributions, the per-row total is
`zp · sum_x` per sub-block. ✓

In the MMQ body, `vec_dot_dp4a_streaming` (`body.cuh:161`) processes the
**full sub-block per thread** via the inner `vdr=8` loop:
- `sumi = Σ_{32 elements (full sub-block)} q · x_int`
- `sum_x` = same full sub-block sum
- Per-thread contribution: `sc · d_x · sumi + zp_eff · sum_x` (NO 0.25)

**Why this is high-risk:** the v1 plan explicitly references the dp4a
kernel as the source for "shift algebra" but doesn't call out that the
**accumulation formula** differs. Off-by-4 on the zp term gives a
systematic 25 % under-shoot of the bias contribution per row.

**Discriminating unit test** (catches both LANDMINE-1 and LANDMINE-2):

Run with `q ≡ q_const` (all weights equal a constant) and
`x ≡ x_const` (all activations equal a constant). Expected output:
`M · K · (sc · q_const + zp) · x_const`.

- LANDMINE-1 violation: extra `+ 8·sc · K · x_const` per element
- LANDMINE-2 violation: zp term off by 4× (only 25 % of correct value)
- Both: detectable by absolute equality, no floating-point margin needed

## 4. Implementation plan (7 sessions estimated, revised v2)

> **Re-budget:** v1 estimated 5 sessions; consensus from three-way
> review is 6-8. Settled at 7 with explicit per-session deltas.

### Session 1: HFQ6 unpack helper + mmq_x=8 prototype + screen function (~1.25 sessions)

Goal: prove the architecture works on the simplest size before
touching the full sweep, with all three correctness landmines closed.

**Deliverables:**

- `kernels/src/gemm_hfq6g256_residual_mmq_gfx906_body.cuh` — copy
  HFQ4 body, then:
  - **Replace** `load_hfq4_tile_streaming` with `load_hfq6_tile_streaming`
    (see §4.1 below — load strategy is committed, not handwaved).
  - **Verify** `x_dm[i] = make_float2(sc, zp)` (NO `+ 8·sc` — landmine 1).
  - **Verify** the per-thread accumulation in `vec_dot_dp4a_streaming`
    uses `zp_eff * sum_x` with NO `0.25f` (landmine 2).
- `kernels/src/gemm_hfq6g256_residual_mmq_gfx906_x8.hip` — wrapper
  exporting all 3 add-mode variants (`_x8`, `_full_set_x8`,
  `_full_add_x8`) following HFQ4's pattern.
- `dispatch.rs::gemm_hfq6g256_residual_mmq_gfx906()` (add=1, residual)
  AND `dispatch.rs::gemm_hfq6g256_mmq_set_gfx906()` (add=0, overwrite)
  — both Rust entry points, single-mmq_x=8 paths for now (extended in
  S2).
- `dispatch.rs::mmq_screen_weight_hfq6()` — separate function, NOT a
  refactor of the HFQ4 one. Uses `gemm_hfq6g256_residual_fp16` as
  reference (already exists at `dispatch.rs:8210`) and the new MMQ
  kernel as candidate. Factor the shared scaffolding (synthetic
  activation gen, upload, max-err comparison loop) into
  `mmq_screen_compare(reference_fn, candidate_fn, threshold)` to
  avoid 50 lines of duplication.
- `crates/hipfire-runtime/examples/test_hfq6_mmq.rs` — NRMSE +
  per-element max-abs-err parity vs `gemm_hfq6g256_residual_wave64_dp4a`
  AND vs CPU reference. Includes:
  - Aligned shape: M=3584, K=4096, B=8 (mmq_x=8 hits `_full_*`)
  - Non-aligned shape: M=3000, K=4096, B=13 (exercises `need_check=true` and
    data-dependent `_x8` path — this is where MMQ correctness bugs
    historically hide)
  - Constant-weight discriminator: `q ≡ 5, x ≡ 1.0` and check
    `sum == M · K · (sc·5 + zp)` to absolute equality (catches landmine 1+2)

**Validation thresholds:**

- NRMSE < 0.005 vs CPU reference (HFQ6 quantization noise floor)
- **Per-element max-abs-err < 1e-3** vs wave64_dp4a reference (catches
  systematic bias from x_dm or 0.25f leak)
- Constant-weight test: bit-exact equality (Σ exactly `M·K·(sc·q+zp)`)

**Risk-tracked items:**

- **Load strategy commitment** (see §4.1).
- **Register-pressure check.** rocprof's `SALUInsts` and `VALUInsts`
  per kernel; if SALU/(SALU+VALU) jumps significantly above HFQ4's
  ratio, explore `v_alignbit_b32` / `v_lshl_or_b32` as scalar-shift
  optimizations.

### Session 1 GO/NO-GO (revised threshold)

**Go criterion:** prototype must beat current `gemm_hfq6g256_residual_wave64_dp4a`
at B=8 by **≥10 %** per-call wall time. (v1 said 30 %; that was anchored
on cross-quant baseline, wrong reference.)

If <10 % improvement: redesign before scaling out. Possible pivots:
multi-wave LDS-staged dp4a (rejected in B.1.1 but might work with
proper LDS budget), or accept Path 1's cap and ship without B.2.

### Session 2: Size sweep + per-mmq_x stride PMC tuning (~1.5 sessions)

Add the remaining 7 size variants (x16, x24, x32, x40, x48, x56, x64).
Each is a 5-line wrapper × 3 add-modes = 15 LOC per variant ≈ 105 LOC
total wrapper code.

**Stride PMC sweep (NEW in v2 — was unbudgeted in v1):**

The HFQ4 `x_stride_for<mmq_x>() = mmq_x >= 64 ? 40 : 33` cutover was
empirically tuned for HFQ4. HFQ6's heavier unpack changes the
dp4a-issue-rate-vs-bank-conflict tradeoff. Sweep grid:

- mmq_x ∈ {32, 48, 56, 64} (boundary candidates)
- stride ∈ {33, 40} (HFQ4's two choices)
- 5 representative weight matrices (q, k, gate, up, down across one
  layer)
- ~20 measurements, ~2 hours

**LDS budget re-verification.** Same shape as HFQ4 (`MMQ_Y * x_stride * 4
+ 1024 + mmq_x * Y_STRIDE * 4`). Need to re-verify ≤ 32 KiB on the
larger mmq_x with whatever x_stride wins.

**Validation:**

- rocprof shows all 8 variants with reasonable runtime (no >5× outliers)
- `mmq_screen_weight_hfq6` passes for all weight matrices at startup
- mq6 prefill bench shows monotonic improvement across the sweep
- Stride choice committed in `body.cuh::x_stride_for_hfq6<>()` template

### Session 3: Wire up dispatch + retarget gate_up / qkv / qkvza (~1.25 sessions)

> **Architecture corrected from v1:** issue **N parallel `_mmq_set` calls
> into N separate output buffers**, NOT chained set+add into a shared Y.

Branch the existing HFQ6 dispatchers' B>1 paths to route to MMQ when
`should_use_mmq(arch, batch_size) && self.arch == "gfx906"`. Mirrors
HFQ4's pattern at `dispatch.rs:3429-3456` / `3854-3873` / `4238-4252`.

- `gemm_qkvza_hfq6g256` at B>1 dispatches → screen q/z weights, if
  safe: `ensure_q8_1_mmq_x` once → 2 calls of
  `gemm_hfq6g256_mmq_set_gfx906` (q, z) → fall through to
  `gemm_qkvza_hfq6g256_fp16_wave64` for beta+alpha (small-M heuristic;
  pass `qkv_m=0, z_m=0`).
- `gemm_qkv_hfq6g256` at B>1 dispatches → 3 `_mmq_set` calls (q, k, v).
- `gemm_gate_up_hfq6g256` at B>1 dispatches → 2 `_mmq_set` calls
  (gate, up).
- `gemm_hfq6g256_residual` at B>1 on gfx906 routes to
  `gemm_hfq6g256_residual_mmq_gfx906` (add=1, single call).
- `gemm_hfq6g256_batched_lmhead` adds an MMQ branch for the big-K
  W_out matrix (parallels HFQ4 at `dispatch.rs:7748-7749`).

**Quantize-once amortization** (clarified per L4 review finding): each
fused dispatcher MUST call `ensure_q8_1_mmq_x()` ONCE before the fan-out
and pass the same `xq` pointer to all sibling MMQ calls. Do NOT
re-quantize per projection.

**Validation:** coherence-gate clean on 9b.mq6, prefill rocprof shows
the new MMQ kernels accounting for the bulk of GEMM time, kernel
launch overhead measurement (rocprof `Wave_Front_Latency_Sum` — should
match HFQ4's 800-launch pattern at pp128).

### Session 4: Fast paths for full / non-full + lm_head finalize (~1 session)

The `_full_set_x{N}` and `_full_add_x{N}` variants only fire when
`m % 128 == 0 && batch_size % mmq_x == 0`. The `_x{N}` (data-dependent
`add`) variant handles the non-full case (last batch tile when
`batch_size % mmq_x != 0`, etc.).

Verify the dispatcher correctly picks `_full_*` when shapes align and
falls back to `_x{N}` otherwise. Wire batched_lmhead's MMQ branch (if
not done in S3).

**Validation:** all unit shapes (M=3584/3072/4096, K=4096) hit the
`_full_*` fast path; coherence still clean.

### Session 5: Polish + perf + correctness validation (~1.5 sessions)

Multiple validation gates (NEW in v2):

**Perf gates:**

- mq6 9B pp128 ≥ 425 tok/s (target), stretch ≥ 450 tok/s
- ~~27B mq6 pp128~~ — **DEFERRED.** No `qwen3.5-27b.mq6` model on
  disk (per pre-S1 item 8). Either quantize one (~10 min) before S5
  or skip; not blocking.
- ~~DFlash mq6 spec-verify gate~~ — **DROPPED** (per pre-S1 item 9).
  No DFlash mq6 baseline exists to regress against; project has
  never benched DFlash + mq6 (drafter compatibility unverified).
  Launch-overhead risk for small mmq_x will surface in the coherence
  prefill numbers (which already cover mq6 reasoning prompt at
  pp36-class sizes — close enough to DFlash B=8-16 verify shape).

**Correctness gates:**

- coherence-gate 7/7 on 9b.mq6
- mq6 reasoning prompt prefill_tok_s ≥ mq4-class numbers (~330 in the
  coherence harness vs current 175)
- ~~DFlash coherence-gate (3-tier attractor check)~~ — DROPPED for
  same reason as the perf gate above.

**Tuning items:**

- mmq_screen threshold tuning for HFQ6. **gfx906 default per
  `dispatch.rs:595` is `0.50` (per-row max-abs-err, NOT NRMSE)**
  — verified during pre-S1 item 3. Start with the same default for
  HFQ6; tune only if false-positives observed during screen runs.
- HIPFIRE_MMQ_K_FILTER + HIPFIRE_MMQ_LAYER_FILTER + HIPFIRE_MMQ_DIAG_*
  env-var debug knobs (parallel of HFQ4 implementation).
- Per-quant cutover threshold: validate HFQ6's optimal `should_use_mmq`
  cutover matches HFQ4's `batch_size ≥ 8`. If different, parameterize
  `should_use_mmq` over dtype.
- `profile.rs::hfq6g256_weight_bytes` helper (closes plan v3.2.4
  follow-up item 5).

**Outputs:** final mq6 9B + 27B pp128 bench, dev-log writeup, plan
v3.2.6 errata.

### Session 3 GO/NO-GO

If rewiring gate_up / qkv / qkvza to multiple MMQ calls regresses (e.g.
extra kernel-launch overhead outweighs the per-kernel speedup), keep
the fused kernels and only ship MMQ for the residual sites (which
already use one call per projection — wo / w_down / lm_head).

The risk is most acute at small batch sizes (B=8, mmq_x=8) where
amortization is minimum. mq4 doesn't expose this because at AR decode
(B=1) it doesn't enter the MMQ path at all (`should_use_mmq` returns
false for B<8). HFQ6 at the **DFlash spec-verify path** runs at B=8-16,
exactly where MMQ x8/x16 has minimum amortization. The S5 DFlash gate
catches this.

## 4.1 Load strategy commitment (was undefined in v1)

**Decision:** row-coalesced 12-uints-per-tid load, with per-tid local
re-pack into 8 (int_a, int_b) pairs.

**Math:**

- 96 weight bytes/row/window × 128 rows = 12,288 B/window
- = 24 uints/row × 128 rows = 3072 uints/window
- ÷ 256 tids = **12 uints/tid** per window

**Per-tid layout:**

- 12 uints = 48 bytes = 8 (int_a, int_b) pairs (since 8 pairs × 6 bytes = 48)
- Each tid loads 12 uints sequentially (4-byte aligned, coalesced
  across tids), reorganizes the bytes locally into 8 pair-unpacks,
  writes 16 dst ints to LDS.

**Skeleton (from gemini's review, adapted):**

```c++
// Per-window, per-tid: 12 uints loaded coalesced, 8 pairs unpacked.
const int per_tid_uints = 12;
const int per_tid_pairs = 8;

uint32_t buf[per_tid_uints];
#pragma unroll
for (int u = 0; u < per_tid_uints; ++u) {
    const int task_id = tid * per_tid_uints + u;
    const int row_in_window = task_id / 24;  // 24 uints per row per window
    const int uint_in_row = task_id % 24;
    const int row = (row0 + row_in_window < M) ? (row0 + row_in_window) : (M - 1);
    const char* gp = A + ((long long)row * groups_per_row + kg) * 200;
    buf[u] = *(const unsigned int*)(gp + 8 + window * 96 + uint_in_row * 4);
}

// Local re-pack: bytes 0-5 → pair 0, bytes 6-11 → pair 1, ..., bytes 42-47 → pair 7.
// Each pair = 8 q6 weights = 2 dst ints (int_a, int_b).
const uint8_t* bytes = (const uint8_t*)buf;
#pragma unroll
for (int pair = 0; pair < per_tid_pairs; ++pair) {
    const uint8_t b0 = bytes[pair*6 + 0];
    const uint8_t b1 = bytes[pair*6 + 1];
    const uint8_t b2 = bytes[pair*6 + 2];
    const uint8_t b3 = bytes[pair*6 + 3];
    const uint8_t b4 = bytes[pair*6 + 4];
    const uint8_t b5 = bytes[pair*6 + 5];
    // Unpack 8 q6 ∈ [0, 63] from 6 bytes (existing algebra from
    // gemm_hfq6g256_residual_wave64_dp4a.hip:79-86):
    const unsigned int q0 = b0 & 63;
    const unsigned int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
    const unsigned int q2 = (b1 >> 4) | ((b2 & 3)  << 4);
    const unsigned int q3 = b2 >> 2;
    const unsigned int q4 = b3 & 63;
    const unsigned int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
    const unsigned int q6 = (b4 >> 4) | ((b5 & 3)  << 4);
    const unsigned int q7 = b5 >> 2;
    const int int_a = (int)((q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24));
    const int int_b = (int)((q4 & 0xFF) | ((q5 & 0xFF) << 8) | ((q6 & 0xFF) << 16) | ((q7 & 0xFF) << 24));
    // Write to x_qs at the per-row, per-pair offset (matches HFQ4 layout).
    const int row_idx = (tid * per_tid_pairs + pair) / 16;  // which row this pair belongs to
    const int pair_in_row = (tid * per_tid_pairs + pair) % 16;
    x_qs[row_idx * x_stride + 2 * pair_in_row + 0] = int_a;
    x_qs[row_idx * x_stride + 2 * pair_in_row + 1] = int_b;
}
```

This is the pseudocode; session 1 implementation may need tuning of
the per-tid task → row/pair mapping for optimal coalescing on gfx906's
memory subsystem. Verify with rocprof's `MemUnitBusy` and
`L1CacheHitRate` PMC counters.

## 5. Risks and mitigations (v2)

| Risk | Mitigation |
|---|---|
| **LANDMINE-1: x_dm carries `+ 8·sc` term from HFQ4 copy-paste** | §3.1.1 + S1 spec calls out explicitly. S1 validation includes constant-weight discriminator test. |
| **LANDMINE-2: `0.25f` factor leaks from dp4a kernel into MMQ body** | §3.1.2 + S1 spec calls out explicitly. Constant-weight test catches off-by-4 on zp term. |
| **LANDMINE-3: mmq_screen_weight HFQ4-only — no abstraction exists today** | §4 S1 specs `mmq_screen_weight_hfq6()` as separate function with shared `mmq_screen_compare()` helper. |
| HFQ6 unpack in LDS-staging path is materially slower than HFQ4 due to 1.5× source bytes | §4 S1 GO/NO-GO at ≥10 % over wave64_dp4a. Profile S1 prototype before S2 size sweep. If <10 %, redesign (multi-wave LDS-staged dp4a) before scaling. |
| Global memory coalescing degrades for HFQ6's byte-granularity unpack | §4.1 commits to row-coalesced 12-uints/tid pattern. PMC verification (`MemUnitBusy`, `L1CacheHitRate`) in S1. |
| LDS budget overflow at large mmq_x with HFQ6's heavier unpack | Reuse HFQ4's x_qs LDS layout exactly (decoded ints, same 32/row/group). Only the load step widens, not the storage. |
| mmq_screen false-positives on edge-case HFQ6 weights | Reuse HFQ4 mmq_screen infrastructure pattern, separate per-dtype function, validate threshold empirically (likely lower than HFQ4's 0.005). |
| **Capture_mode silent breakage** | The `&& !self.capture_mode` guards are on PR2 (commit `5768fe4`) but NOT yet on this audit branch. Pre-S1 checklist item: cherry-pick `5768fe4` before starting. |
| DDTree spec-decode breakage on HFQ6 lm_head MMQ | DDTree arms in `speculative.rs` (commit `5768fe4`) call the dispatcher fn `gemm_hfq6g256_batched_lmhead`, not a specific kernel — once we add the MMQ branch in the dispatcher, DDTree picks it up automatically. This is by design from `5768fe4`, not lucky abstraction. |
| Per-mmq_x x_stride tuning takes longer than scoped | §4 S2 budgets +0.5 session for stride PMC sweep. Sweep grid pre-defined (mmq_x ∈ {32, 48, 56, 64} × stride ∈ {33, 40}). |
| **Launch overhead at small batch (DFlash spec-verify B=8-16)** | §4 S5 adds explicit DFlash spec-verify regression gate. S3 GO/NO-GO triggers rollback to fused kernels for fused-projection sites if launch overhead dominates. |
| 27B model regresses vs Phase B.1 baseline | §4 S5 adds 27B mq6 pp128 datapoint. |
| Heavier 6-bit unpack pressures registers (more SALU shifts) | §4 S1 PMC observation tracks `SALUInsts` / `VALUInsts` ratio; v_alignbit_b32 / v_lshl_or_b32 explored if SALU jumps. |
| Test surface for the dp4a reference is empty | Pre-S1 checklist: add dp4a unit tests to `test_hfq6_gemm.rs` (~50 lines × 4 kernels). Otherwise S1 validation against dp4a is circular. |

## 6. Decision points (revised v2)

- **Pre-S1 checklist (10 items, see §7) MUST be complete before S1 starts.**
- **S1 GO/NO-GO:** prototype must beat current wave64_dp4a at B=8 by
  **≥10 %** per-call wall time (revised from 30 %). If <10 %: redesign
  before S2 size sweep.
- **S3 GO/NO-GO:** if rewiring fused dispatchers regresses on the
  coherence prefill numbers at small-batch shapes (mq6 reasoning
  prompt at pp36 — closest available proxy for DFlash spec-verify
  B=8-16, since project has no DFlash mq6 baseline), keep fused
  kernels and ship MMQ only for the residual sites. Originally the
  v2 plan named "DFlash mq6" as the gate; pre-S1 item 9 confirmed
  no such baseline exists, so we use the coherence-harness mq6
  reason prompt's prefill_tok_s as the proxy signal.

## 7. Pre-session-1 checklist (v2 — required before any kernel code)

> **Status: complete (2026-05-08).** Items 1-2 + 4 + 10 done with code
> changes / measurements; items 3, 5-9 done as desk checks or
> reclassified. Outcomes recorded inline below.

1. [x] **Land Phase A guards on the audit branch.** ✅ Cherry-picked
   `5768fe4` → audit branch as `c54445b`. All 5 HFQ6 dp4a dispatch
   sites now have `&& !self.capture_mode` (verified at
   `dispatch.rs:7747, 8046, 8211, 8392, 8876, 9317`). DDTree arms
   for HFQ6G256 / MQ6G256 in `speculative.rs` also landed via the
   pick. Build clean.
2. [x] **Add dp4a unit tests to `test_hfq6_gemm.rs`.** ✅ Added Tests
   6-9 covering all 4 HFQ6 wave64_dp4a kernels (residual, gate_up,
   qkv, qkvza). Each test: random weights+activations, CPU
   dequantize + matmul reference, max_err comparison. Also gated
   the WMMA tests (2/4/5/10) with `if gpu.arch.starts_with("gfx11")
   || gpu.arch.starts_with("gfx12")` since they use
   `__builtin_amdgcn_wmma_*` which gfx906 doesn't have.

   **Results on gfx906 (audit branch HEAD):**
   - residual scalar: max_err=0.0002, 0/8192 bad — baseline (weight quant only)
   - gate_up scalar: max_err=0.0002, 0 bad
   - residual dp4a: max_err=0.0562, 30/16384 bad (0.18 %)
   - gate_up dp4a: max_err 0.053-0.054
   - qkv dp4a: max_err 0.057-0.060
   - qkvza dp4a: max_err 0.051-0.058

   The dp4a tests show ~250× higher max-err than scalar because
   dp4a carries Q8_1 activation quantization in addition to HFQ6
   weight quant. Bad-element rate < 0.2 % at relative-err > 10 %
   threshold; absolute err ≈ 0.1 % of typical output magnitude.
   **Functionally correct vs CPU reference.**

3. [x] **Verify the actual `mmq_screen_threshold` default.** ✅
   Verified at `dispatch.rs:595`:
   ```rust
   let mmq_screen_threshold_default: f32 = if arch == "gfx906" { 0.50 } else { 0.10 };
   ```
   **The metric is per-row max-abs-err** (not NRMSE as the v1 plan
   claimed). gfx906 default = 0.50; override via
   `HIPFIRE_MMQ_SCREEN_THRESHOLD`. The mmq_screen comparison
   formula at `dispatch.rs:1314-1330`: per-row, take max of
   `abs(ref_out[i] - mmq_out[i])` across batch and rows; reject
   weight if `worst_err > threshold`. **HFQ6 should start with the
   same 0.50 default**; tune in S5 only if false-positives observed.

4. [x] **Verify mq4 prefill at audit-branch HEAD.** ✅ Measured
   2026-05-08 with 5-run JIT-warm bench:

   | 9B pp128 | Prefill (median) | Decode | Spread |
   |---|---:|---:|---:|
   | mq4 | **598.7 tok/s** | n/a | 0.1 % |
   | mq6 | **190.8 tok/s** | **44.0 tok/s** | 0.5 % prefill |

   Gap = 598.7 / 190.8 = **3.14×**. Bandwidth-bound floor
   = 598.7 / 1.47 = **407.3 tok/s**. Targets unchanged: realistic
   ≥ 425 (= floor × 1.05), stretch ≥ 450 (= floor × 1.10).

   Logs: `/tmp/pre-s1-mq4.log` + `/tmp/pre-s1-mq6.log`.

5. [x] **Spec freeze for `load_hfq6_tile_streaming`.** ✅ §4.1 in
   this plan is the spec. Per-tid task → row/pair mapping verified
   on paper: 12 uints/tid × 256 tids = 3072 uints/window matches
   24 uints/row × 128 rows. 8 pairs/tid × 6 bytes/pair × 256 tids
   = 12,288 bytes/window matches 96 B/row × 128 rows. ✓
6. [x] **Discriminating unit test ready.** ✅ Spec'd in §3.1: the
   constant-weight test (`q ≡ q_const`, `x ≡ x_const`) checks
   absolute equality `Σ == M·K·(sc·q+zp)·x`. To be added to
   `test_hfq6_mmq.rs` in S1 (deliverable, not pre-S1).
7. [x] **Non-aligned shape in test matrix.** ✅ Spec'd in §4 S1 as
   M=3000, K=4096, B=13. To be added to `test_hfq6_mmq.rs` in S1.
8. [⚠️] **27B baseline measured.** PARTIAL — `qwen3.5-27b.mq6` does
   NOT exist on disk. Available models: `qwen3.5-27b.mq3` and
   `qwen3.6-27b.mq6` (different model family). Skipping 27B-mq6
   regression gate for S5 unless the model gets quantized.
   Decision deferred: either quantize a 27B-mq6 (~10 min) before
   S5 starts, or remove the 27B regression gate from S5.
9. [⚠️] **DFlash spec-verify baseline.** RECLASSIFIED — DFlash mq6
   has NEVER been benched in this project (per
   `docs/perf-checkpoints/2026-05-06-mq6-baselines.md:185-194`).
   The `dflash_branch_bench.sh` script uses mq4 only. There is no
   pre-existing baseline to record; the DFlash mq6 spec-verify gate
   in S5 would need to be **introduced** during B.2, not preserved.
   This means S3's "regress vs DFlash mq6" GO/NO-GO is currently
   undefined. Two options: (a) drop the DFlash gate entirely
   (rely on coherence + AR-decode regression checks); (b) wire up
   DFlash mq6 first (separate ~1-session task to verify drafter
   compatibility). Recommend (a) — the launch-overhead risk shows
   up in coherence prefill numbers if it's real.
10. [x] **Confirm `should_use_mmq(gfx906, B) → true at B≥8`.** ✅
    Verified at `dispatch.rs:248`: `let arch_min_batch: usize = if
    arch == "gfx906" { 8 } else { 256 };`. MMQ port inherits this
    threshold.

After this checklist, session 1 can begin. Without it, session 1's
GO/NO-GO criterion is undefined and session 3's rewiring will be
wrong.

## 8. Out of scope (deferred)

- HFQ8 MMQ (plan §3.2.5 — explicitly ruled out).
- MoE-indexed HFQ6 MMQ (plan v3.2.4 follow-up item 4 — ~1
  additional session post-Phase-B).
- LDS bank-conflict tuning beyond the §4 S2 stride sweep (only
  re-tune if PMC flags a regression).
- Refactoring `mmq_screen_weight` into a dtype-parameterized form
  (option β from review) — defer to separate cleanup PR.
- Sub-byte vectorized loads (Strategy B/C in load options) — try
  Strategy A (committed) first; only revisit if PMC flags load-side
  bottleneck.

## 9. Open questions (none of the reviews covered)

These can be addressed during S5 polish; not blocking S1.

1. **Compile-time `add` template parameter cardinality.** HFQ4 ships
   8 size variants × 3 add-modes = 24 kernel exports. HFQ6 needs the
   same. Per-variant overhead ~15 LOC × 24 = 360 LOC of wrapper code.
2. **Daemon-binary kernel-cache invalidation.** When new HFQ6 MMQ
   kernels ship, daemon binaries need to recompile the
   `.hipfire_kernels/gfx906/` cache. Multi-second JIT pause on first
   prefill after upgrade — worth a release-note line.
3. **Kernel name collision risk in `.hipfire_kernels`.** HFQ6 names
   sibling HFQ4 names; cache directory roughly doubles in size. No
   collision but worth a sanity check.

## 10. References

- Reference impl: `kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh`
- HFQ6 6-bit unpack pattern: `kernels/src/gemm_hfq6g256_residual_wave64_dp4a.hip:78-100`
- HFQ4 mmq_screen: `dispatch.rs:1263 (mmq_screen_weight)`
- HFQ4 residual MMQ dispatch (add=1): `dispatch.rs:7005 (gemm_hfq4g256_residual_mmq_gfx906)`
- HFQ4 set MMQ dispatch (add=0): `dispatch.rs:7125 (gemm_hfq4g256_mmq_set_gfx906)`
- HFQ4 fused-projection rewiring patterns:
  - qkvza: `dispatch.rs:3429-3456`
  - qkv: `dispatch.rs:3854-3873`
  - gate_up: `dispatch.rs:4238-4252`
- `should_use_mmq` threshold: `dispatch.rs:224-247` (gfx906 default B≥8)
- `ensure_q8_1_mmq_x` quantize helper: `dispatch.rs:1200-1254`
- Phase A capture_mode + DDTree fix commit: `5768fe4` (PR #187)
- Phase B.1 cumulative writeup: `docs/perf-checkpoints/2026-05-07-phase-a-cumulative-mq6.md` (Phase B.1.1 section)
- PR #158 design doc: `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`
- v3.2.5 errata (Phase B framing): `docs/plans/gfx906-mq6-mq8-port.md` v3.2.5
