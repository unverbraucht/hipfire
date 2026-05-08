# gfx906 MQ6 + MQ8 — kernel-coverage analysis

**Status:** Draft v3 (2026-05-06, scope reframe to MQ6/MQ8).
Branch: `feat/gfx906-hfq6-hfq8-analysis` (branch name carries v1
history; not renamed).
**Hardware:** AMD Instinct MI50 (gfx906, Vega 20)
**Predecessor:** PR #158 (gfx906 HFQ4 dp4a + AR-decode optimizations,
merged as `afb84bd`).
**Reviews integrated:** `gfx906-mq6-mq8-port-plan-rev-{claude,gemini,glm5}.md`
(co-located in this directory). v1 had 5 blocking errors caught by
adversarial review; v2 corrected the factual claims; v3 reframes
the scope from raw HFQ6/HFQ8 (not deployed) to MQ6/MQ8 (deployed).

### v3 scope reframe (2026-05-06)

v1 and v2 used "HFQ6 / HFQ8" as the framing because that's the
kernel-family naming. **The actually-deployed quant formats are
MQ6 and MQ8** (FWHT-rotated variants). The on-disk model registry
ships `qwen3.5-9b.mq6` (and zero `qwen3.5-9b.hf6`); the quantize
tool emits `mq8` but has no `hf8` format at all. Rotation is
essentially free at runtime — one small per-layer
`mq8_rotate_quantize_x` / `fused_rmsnorm_rotate_mq` kernel call,
already in the tree — and gives real quality benefit. **MQ8 is
the correct int8 target for production gfx906 work; HFQ8 work is
deprioritized indefinitely.**

The per-quant kernel-coverage analysis below is unchanged: at the
GEMV/GEMM layer, MQ6 uses the HFQ6-family kernels (and MQ8 uses the
HFQ8-family kernels) with one extra activation-rotate kernel layered
in front. The kernel surfaces "HFQ6 has wave32 GEMV" and "MQ6 has
wave32 GEMV" describe the same kernels.

### v3.1 errata (2026-05-06): `gemv_mq8g256.hip` was not actually buildable on gfx906

§3.2 of v3 framed MQ8 as Priority 1 partly because "the dp4a-on-int8
inner loop is already shipped in `gemv_mq8g256.hip`; Phase A items are
mechanical mirrors with the same proven inner loop." **That was wrong
about the *gfx906* path.** The kernel as-shipped used
`__builtin_amdgcn_sudot4` for the dp4a call, which lowers to
`v_dot4_i32_iu8` (mixed-sign int8 dp4a). That instruction needs the
`dot8-insts` target feature — RDNA3+ only. On gfx906 (Vega 20, MI50)
the kernel **failed to compile** with `error: '__builtin_amdgcn_sudot4'
needs target feature dot8-insts`. The kernel had been validated on
gfx1100 / gfx1201 only; there was no shipped gfx906 mq8 path.

Discovered 2026-05-06 during the Priority 0 baseline run. Fix in
commit `ee0fac6`: substitute `sudot4(true, w, true, x, acc, false)` →
`sdot4(w, x, acc, false)` (signed×signed dp4a, gfx906+, dot2-insts).
Math is identical — both operands are signed int8 (Q8_1 activations
+ symmetric MQ8 weights `[-127, 127]`); the sudot4 mixed-mode form
was gratuitous. Cross-arch portability preserved per LLVM's per-arch
syntax docs (sdot4 is supported on gfx906/908/9/10/11/12).

**Implications for the priority list (§5.1):**

- **MQ8 Phase A is no longer "low risk because the inner loop ships."**
  The fixed B=1 kernel was validated only as far as a single-process
  bench on Qwen 9B mq8: 45.4 tok/s, p50 21.46 ms/tok, tight
  per-token determinism (1.4% spread). No coherence-gate run yet
  (NFS-bound on this box). Treat the inner loop as freshly-validated,
  not battle-tested.
- **The "MQ8 first" reasoning weakens.** The argument was "no MMQ
  port required and the dp4a inner loop already ships" — only the
  first half survives. MQ6 Phase A's competitive case (wave32 → wave64
  mechanical port pattern is the same one PR #158 used for HFQ4 and
  is genuinely shipped) is now relatively stronger.
- **Audit-the-other-mq-kernels work item added.** §6 priority list
  should include a sweep for other latent gfx906-only build failures
  in MQ/HFQ kernels using RDNA3+-only intrinsics. Cheap (~½ session,
  pure compile-test) and forces the "is this kernel actually shipped
  on the target arch?" question to be answered before any Phase A
  estimate is treated as load-bearing.

This errata does not invalidate the v3 scope reframe (MQ6/MQ8 are
still the deployed targets). It does mean every "shipped on gfx906"
claim in the body of this doc deserves a build-verification step
before being relied on for prioritization.

### v3.2 errata (2026-05-06): MQ8 runtime dispatch is not wired for per-layer use

After v3.1 fixed the kernel build, an end-to-end bench on
`qwen3.5-9b.mq8` produced 45.4 tok/s with deterministic per-token
timing — but the inference is **invalid**. See
`docs/perf-checkpoints/2026-05-06-mq8-runtime-dispatch-audit.md` for
the full discovery sequence.

**Root cause:** the per-layer prefill batched dispatch in
`crates/hipfire-arch-qwen35/src/qwen35.rs` excludes `MQ8G256` from
all 14 `is_mq` matchers (lines 3946, 4118, 4147, 4194, 4243, 4508,
4538, 4576, 4651, …). MQ8 weights silently fall through to
`gemm_qkvza_hfq4g256` etc., which read at HFQ4-format byte stride
(136 B/group) when MQ8 is 258 B/group. Prefill produces corrupted
DeltaNet state and KV cache; gen consumes that corrupted state.

**Why it stayed undetected:** MQ8 was originally shipped (commit
`246501a`, 2026-04-08) "targeting dp4a on gfx1100" and only ever
wired into the lm_head tied-embedding path (`bf0ba43`, 2026-04-13:
explicit comment *"Not a current path"*). No production model has
ever shipped MQ8 per-layer weights. `coherence-gate.sh` has no mq8
entry. The gfx906 build failure (fixed in `ee0fac6`) masked the
deeper issue.

**Implications for plan §3.2 / §5:**

- **§3.2 Phase A item 4 ("batched GEMM") is a correctness
  prerequisite, not an optimization.** The `gemm_*_mq8g256_*`
  kernels don't exist AND the runtime dispatch sites that would
  call them are absent (14 sites in the arch crate).
- **The "MQ8 first" priority ranking is no longer defensible.**
  The "smaller scope" argument survives in nominal terms but the
  gap is wider than counted: implementing MQ8 batched means
  writing 4–7 new kernels (qkvza / qkv / gate_up / residual /
  MoE-indexed × wave64) AND wiring 14 dispatch sites. MQ6 Phase A
  is comparatively better-defined: the wave32 path through bare
  `gemm_qkvza_hfq6g256` works end-to-end today.
- **§5.1 priority list is reordered in v3.2:** MQ6 Phase A
  becomes priority 1; MQ8 Phase A becomes priority 4 (deferred
  until a production model ships raw MQ8 per-layer weights, OR a
  measured advantage over MQ6 motivates the full wiring work).
- **Audit-method gap:** §5.5 build-tested kernel sources but did
  not test runtime dispatch wiring. §5.4 (Priority 0.5 audit) is
  expanded to include grepping arch crates for `is_mq` /
  dtype-matchers and confirming every quant format the loader
  produces is handled at every per-layer call site.

**MQ8 scope after v3.2:** restricted to "lm_head-tier optimization,
not primary weight format." The `weight_gemv` MQ8G256 dispatch in
`crates/hipfire-runtime/src/llama.rs:601, 646, 680, 733` works
correctly and serves the lm_head tied-embedding path used by
production mq4-format models. That path benefits from `ee0fac6` on
gfx906; no further MQ8 work is in scope without a deployed
per-layer MQ8 model.

### v3.2.2 errata (2026-05-06): dp4a leads, wave64 is scaffolding; Phase A is decode-only

Two corrections after the wave64 HFQ6 residual port shipped (`466f1a6`)
and the bench result landed (`docs/perf-checkpoints/2026-05-06-wave64-hfq6-residual-experiment.md`):

**Correction 1 — Lever attribution recalibrated against PR #158 data.**

The original v3 framing implied wave64 ports were the dominant
Phase A lever (1.5-2× lift was projected). Re-reading
`docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`
and the original wave64 port commit (`166451d`, MI300X) shows the
attribution was wrong:

| HFQ4 lever (PR #158 data) | 9B mq4 decode lift |
|---|---:|
| Pre-investigation (wave64 already shipped via `166451d`/`a293e15`) | 50.7 tok/s |
| + ILP-prefetch on `gemv_residual` (`3ef127d`) | 54.4 (+7.3 %) |
| + dp4a on `fused_gate_up` (`5a45260`) | 58.5 (+7.5 %) |
| + dp4a on `fused_qkv` + `fused_qkvza` (HEAD) | 58.9 (+0.7 %) |
| **Total** | **+16.2 %** |

Wave64 alone for `gemv_residual` measured **+3.2 %** on MI300X
(within noise per `166451d`). The +4.8 % figure that v3 anchored on
was the *prefetch* lever applied on top of an already-wave64 kernel,
not the wave64 port itself. Today's HFQ6 wave64 residual experiment
on gfx906 confirms: 9B mq6 decode +2.9–3.3 % (matches HFQ4
sibling).

**The dominant lever for HFQ6/MQ6 on gfx906 is dp4a on the fused
GEMVs**, not wave64 alone. Wave64 is a foundation — the dp4a kernels
ship as wave64+dp4a hybrids (mirror `fused_gate_up_hfq4g256_wave64_dp4a.hip`).
Phase A reordered accordingly:

| Phase | Old (v3.2) | New (v3.2.2) |
|---|---|---|
| A.1 | Wave64 GEMV ports (5 kernels) | Wave64 residual GEMV (foundation) — single shipped kernel |
| **A.2** | dp4a port (Phase B, "optional, PMC-gated") | **dp4a-on-fused-GEMV (3 kernels: gate_up, qkv, qkvza) — primary lever, expected +7-8 % decode** |
| A.3 | MoE-indexed (5 → 10 kernels) | Same; ports the dp4a path through the MoE family |
| A.4 | Wave64 batched GEMM (LM-head) | Same; lower priority |

§3.1.2's "Phase B" framing is reclassified to **Phase A primary
lever** in the v3.2.2 reordering. Plan §3.1.2 keeps the PMC-entry-gate
recommendation (verify VALUBusy < 50% on the production hot kernel
before committing) — that's still the right risk-mitigation for the
~1-session port.

**Correction 2 — Phase A is decode-only; prefill needs Phase C.**

The implicit framing of v3 was that Phase A improves both prefill
and decode. The wave64 HFQ6 residual experiment falsifies this: 9B
mq6 prefill Δ was **0.0 %** (within 0.4 % spread), decode Δ was
+2.9–3.3 %. The mechanism is structural:

- **Decode B=1** dispatches through `weight_gemv` →
  `gemv_*_with_rotate` (or for HFQ6, `gemv_hfq6g256_residual` etc.).
  These are the kernels Phase A touches.
- **Prefill B>1** dispatches through `gemm_qkvza_hfq6g256` etc. (the
  batched fused-GEMM family). These are *different kernels*, not
  touched by Phase A's wave64/dp4a-fused work.

The MQ-specific kernels in the tree (`gemv_mq*_with_rotate`,
`mq*_rotate_quantize_x`) are GEMV-shape (decode); the rotation
pre-pass (`fused_rmsnorm_rotate_mq_batched`, `fused_silu_mul_mq_rotate`)
serves both prefill and decode but is one launch per layer, not per
weight matrix. **MQ-quant-specific kernels are primarily a decode
optimization.** Prefill-heavy workloads (long-context code/doc Q&A,
document understanding) on mq6 need the batched-GEMM MMQ port —
that's plan §3.1.3 Phase C, which is a separate ~5-session effort
covering the `gemm_*_residual_mmq_gfx906_x{N}.hip` family for HFQ6.

The §5.1 priority list now reflects this with explicit
prefill-vs-decode columns (see §5.1 below).

### v3.2.3 errata (2026-05-07): Phase A decode shipped — measured +41-50 % vs +15-18 % calibrated target

Phase A.1a/b/c shipped 2026-05-06 → 2026-05-07 across commits `466f1a6`,
`692d792`, `ba246c4`. Cumulative decode lift on gfx906 mq6:

| Stage | 9B pp32 | 9B pp128 | 27B pp32 | 27B pp128 |
|---|---:|---:|---:|---:|
| wave32 (`850848a`) | 31.1 | 30.3 | 10.2 | 10.1 |
| + A.1a wave64 | 32.0 | 31.3 | 10.6 | 10.5 |
| + A.1b prefetch | 32.3 | 31.7 | 10.9 | 10.7 |
| **+ A.1c dp4a-fused** | **44.0** | **42.8** | **15.3** | **15.0** |
| **Cumulative Δ** | **+41.5 %** | **+41.3 %** | **+50.0 %** | **+48.5 %** |

**~2.5-3× better than the v3.2.2 calibrated +15-18 % target.** Full
analysis in `docs/perf-checkpoints/2026-05-07-phase-a-cumulative-mq6.md`.

**Why the calibration underestimated:** v3.2.2 anchored on PR #158's
+16.2 % HFQ4 attribution, but PR #158's measurements ran on a
baseline that **already had wave64 in place** (`166451d` predates
the +16.2 % bench). HFQ4's `5a45260` measured +7.5 % from gate_up
dp4a only because the wave64-vs-wave32 win was already baked in.
HFQ6 had **zero** gfx906 GEMV optimization pre-Phase-A, so today's
A.1c captured three layered wins simultaneously: wave32→wave64
lane utilization, scalar-call→fused-call dispatch overhead, and
dp4a vs scalar FP ALU throughput.

**27B sees more lift than 9B (+50 % vs +41 %)** — projection family
is a larger fraction of decode time at scale.

**mq6 decode is now bandwidth-bound at ~31 % of HBM2 peak**, matching
the post-PR-158 mq4 baseline. Phase A is genuinely complete on the
decode side; further GEMV-level levers won't deliver meaningful lift
without bandwidth-reduction approaches (smaller quants, KV
optimization, speculative decode).

### v3.2.3 errata (2026-05-07): prefill is now the dominant remaining gap

Phase A delivered ~0 % on prefill (as v3.2.2 predicted: prefill goes
through batched GEMM, not GEMV). But the gap relative to mq4 is
striking now that decode is closed:

| Workload | mq4 9B pp128 | mq6 9B pp128 | Gap |
|---|---:|---:|---:|
| Decode | 59 tok/s | 42.8 tok/s | 1.4× (matches BW-bound 1.47× ratio) |
| **Prefill** | **594 tok/s** | **46.8 tok/s** | **12.7×** |

mq6 prefill is **8.6× slower than its bandwidth-bound floor** would
predict. The gap is kernel-architecture: mq4 prefill takes the
PR #158 dp4a-MMQ path (`gemm_hfq4g256_residual_mmq_gfx906_x{N}` family,
+5×); mq6 prefill falls through to wave32 scalar `gemm_qkvza_hfq6g256`
with no gfx906 optimization ever. **This is the natural Phase B target
for v3.2.3.**

**[CORRECTED in v3.2.5]** This errata's numbers and architectural
diagnosis are both stale. The 46.8 tok/s figure was measured before
Phase A.3 (`94643cb`) wired the dp4a fused GEMMs into the prefill
path; the "falls through to wave32 scalar" claim was true at v3.2.3
write-time but has not been true since A.3. Re-baseline and revised
Phase B framing in **v3.2.5 errata** below.

### v3.2.4 errata (2026-05-07): Phase A review follow-ups (non-blocking, deferred)

Three-way review (self / glm5 / gemini) of `feat/gfx906-mq6-phase-a-dp4a`
caught three blocking gaps and three non-blocking polish items. The
blockers shipped in commit `5768fe4` on PR #187 (capture-mode guards on
5 HFQ6 dp4a sites, DDTree HFQ6/MQ6 dispatch arms, stale comment on
`gemv_dp4a_enabled`). The non-blockers are tracked here as Phase A
follow-ups.

**Why deferred:** all three are observability / defensive hardening,
not correctness. None affect tok/s, coherence, or model output. They
are batched into a single dispatch-coverage cleanup pass to be run
before Phase B (prefill MMQ), where adding new fused dp4a paths makes
the audit script + profiler updates pay for themselves.

**How to apply:** when a new dp4a fused GEMV/GEMM ships, add its byte
counts to `profile.rs`, its timer to `dispatch.rs`, and its assert at
the Rust entrypoint — at the same time as the dispatch-arm wire-up.
Don't ship a kernel without observability.

#### Phase A follow-up items (deferred to pre-Phase-B cleanup)

4. **Multi-GPU + MoE LA fused dp4a coverage** — gemini noted no MoE
   dp4a path for HFQ6/MQ6, only single-GPU AR fused. Confirmed by
   inspection: §3.1 Phase A does not enumerate MoE-indexed kernels
   for the new dp4a path; the MoE fused gate_up/down on HFQ6 still
   takes the wave32 FP fallback. Estimated effort: ~1 session
   (mechanical port from HFQ4 MoE patterns once Phase B lands).

5. **profile.rs + dispatch timers for HFQ6 dp4a path** — gemini 2.2 +
   2.3. Add `hfq6g256_weight_bytes` (200 B/group × N groups) helper
   to `crates/rdna-compute/src/profile.rs`. Wire `begin_timer` /
   `end_timer` on the 7 new HFQ6 dispatchers (qkv / qkvza / gate_up
   / residual / batched_lmhead × {fn body, dp4a branch}). Without
   this, the new kernels are invisible to the bandwidth profiler.
   Estimated effort: ~½ session.

6. **Defensive assert on gemv_dp4a_enabled at Rust entry** — glm5
   finding 6/13. Add `assert!(gemv_dp4a_enabled(&self.arch))` at the
   top of every `*_dp4a` Rust function. Today the dispatcher gates
   the call, but a future caller could forget — the assert turns a
   silent wave32-mismatch into a loud panic. Mirrors the wave64
   guard pattern. Estimated effort: ~10 lines, ~5 min.

13. **`scripts/audit-dispatch-coverage.sh`** — claimed in PR #187
    body but not added. Should sweep every dispatch site and check
    that {wave64, dp4a, capture-mode-guard} match the canonical HFQ4
    pattern. Either add the script in this cleanup pass or remove
    the claim from the PR body. Estimated effort: ~½ session for the
    script + integration into pre-commit hook chain.

**Sequencing:** items 5 + 6 + 13 fold into the Phase B prep pass
(audit before adding new dispatch sites). Item 4 follows Phase B once
the prefill MMQ kernel is in place — MoE fused gate_up reuses the same
Q8_1 quantize amortization as the AR path.

### v3.2.5 errata (2026-05-07): Phase B scoping — gap is 3.6×, not 12.7×

Re-baselined mq4 vs mq6 prefill on current HEAD (post-A.3). The
v3.2.3 errata's 12.7× gap quote was pre-Phase-A.3; A.3 already closed
most of it.

| 9B pp128 | mq4 | mq6 | Gap |
|---|---:|---:|---:|
| Prefill (median, JIT-warm) | 598.6 tok/s | 164.9 tok/s | **3.6×** |

**rocprof per-kernel breakdown (mq6 pp128, gfx906):**

| Kernel | % time | Avg ns/call |
|---|---:|---:|
| `gemm_gate_up_hfq6g256_wave64_dp4a` | **45.5 %** | 10.99 ms |
| `gemm_hfq6g256_residual_wave64_dp4a` | 28.5 % | 3.45 ms |
| `gemm_qkvza_hfq6g256_wave64_dp4a` | 17.0 % | 5.49 ms |
| `gemm_qkv_hfq6g256_wave64_dp4a` | 4.7 % | 4.58 ms |
| **Top 4 (Phase A.3 dp4a)** | **95.7 %** | |

**Architectural finding:** mq4 prefill on gfx906 dispatches **no
gate_up / qkv / qkvza kernels at all**. Every prefill matmul goes
through the `gemm_hfq4g256_residual_mmq_gfx906_x{N}` family — 8 size
variants × {set, add} = 16 kernels sharing a body header. Each of
the 9 projections per layer (q / k / v / z / α / β / gate / up /
down) becomes one residual-shaped MMQ call (`set` for first
projection of a fused group, `add` for subsequent ones accumulating
into Y). This is the PR #158 win — small-tile dp4a-MMQ streaming,
not big fused kernels. mq6 today still uses big fused kernels at
prefill and pays for the LDS-tile / occupancy mismatch.

**Two paths for Phase B:**

- **Path 1 (~2-3 sessions): optimize the 4 dp4a fused GEMMs in
  place.** LDS-tile redesign starting on `gemm_gate_up_*` (45.5 %
  alone), prefetch tuning, x_stride sweep. Stays within the
  Phase A.3 dispatch architecture. Expected lift: 1.5-2 ×.
- **Path 2 (~5 sessions): port the MMQ-streaming family to HFQ6.**
  Build `gemm_hfq6g256_residual_mmq_gfx906_x{N}` family + retarget
  dispatchers. This is what mq4 uses for its 60 % win. Expected
  lift: 3-4 × (parity with mq4).

**Sequencing decision:** start with Path 1 kernel #1
(`gemm_gate_up_hfq6g256_wave64_dp4a`). Cheapest first improvement,
validates whether the dp4a-fused path has LDS / prefetch headroom.
If gate_up yields only +5-10 % after a session of LDS-tuning,
that's evidence the dp4a kernels are near their ceiling and we
should pivot to Path 2.

**Phase B.0 (instrumentation) demoted to non-blocker:** rocprof +
`--stats` provides kernel-name-level breakdown without touching
dispatch.rs. Adding `begin_timer` to HFQ6 dispatchers is still
useful for the production daemon profile dump
(`HIPFIRE_PROFILE=1`) but doesn't gate Phase B.1. Plan v3.2.4
item 5 stays a non-blocking follow-up.

**How to apply:** for any future "what's the bottleneck?" question
in this codebase, **rocprof on a representative bench is the first
move, not adding `begin_timer`.** rocprof catches everything
(including kernels we forgot to instrument); `begin_timer` only
catches what we remembered. Use the latter for in-process
correlation against application-level metrics, not for kernel-level
bottleneck identification.

### v3.2.6 errata (2026-05-08): Phase B shipped — Path 2 (MMQ port) won 2.94× at 9B, 3.52× at 27B

Phase B done in 5 sessions (vs 5-7 budgeted). Both paths from v3.2.5
implemented; results contradict v3.2.5's prediction direction.

**Path 1 (BT tuning + multi-wave): capped at +14.5 %.**

| Phase | Commit | mq6 9B pp128 | Cumulative Δ |
|---|---|---:|---:|
| Pre-Phase-B (BT=8 baseline) | — | 165.8 | — |
| B.1 gate_up only (BT=8→16) | `2bee6e6` | 178.0 | +7 % |
| B.1.1 all 4 dp4a (BT=8→16) | `ff9e210` | 190.8 | +15 % |

Multi-wave / 4-rows-per-block experiments (B.1.1 second half) all
regressed at gfx906 due to occupancy loss (256-thread blocks → 10
blocks/SIMD vs 21 with 64-thread). v3.2.5's lower-bound prediction
"+1.5–2 ×" overshot — Path 1's actual ceiling is +1.15 ×.

**Path 2 (MMQ-streaming port): exceeded v3.2.5's upper-bound prediction.**

| Model | Pre-B.2 (kill-switch) | B.2 (MMQ on) | Speedup | v3.2.5 prediction |
|---|---:|---:|---:|---|
| 9B mq6 pp128 | 190.8 tok/s | **561.2** | **2.94×** | 3-4× ✓ within range |
| 27B mq6 pp128 | 54.8 tok/s | **192.8** | **3.52×** | (not predicted; bigger M → bigger win) |
| Decode (both) | unchanged | unchanged | 1.00 × | n/a (MMQ never fires at B=1) |

**Bandwidth-bound floor: exceeded by 38 %.** v3.2.5 calculated 407
tok/s (= mq4_pp128 / 1.47 weight-byte ratio) as the parity target;
B.2 hit **561 tok/s = 38 % over floor**. The prediction
underestimated amortization wins:
- Q8_1 quantize shared across sibling projections (one quantize
  per fused-projection group, not per-call)
- LDS-staged A reused across batch tiles (mmq_x=64 → 4 row-tile
  reuses per batch tile)
- Compile-time `_full_set` / `_full_add` variants eliminating
  branch overhead in the writeback hot loop

**Final mq6/mq4 prefill ratio: 0.939 — within 6 % of mq4 parity.**
v3.2.5 framed this as "stretch"; we hit it.

**Path 2 effort breakdown (5 sessions actual):**

| Session | Outcome |
|---|---|
| Pre-S1 checklist | 10 items closed; 2 deferred (27B model + DFlash baseline never benched) |
| S1 — body.cuh + x8 + x64 + dispatchers + screen | 6/6 correctness tests pass; LANDMINE discriminator catches both math-identity bugs |
| S2 — size sweep + per-batch routing | mmq_x ∈ {8..64} all correct; perf wins at B=16 + B≥40 |
| S3 — dispatcher rewiring (qkvza/qkv/gate_up/residual) | end-to-end **+194 % wall** on 9B pp128 |
| S4 — lm_head MMQ + fast paths | small marginal change (lm_head is small fraction of pp128) |
| S5 — debug knobs + final validation | kill-switch verified, mq4 regression check (-0.2 %), 27B bench |

**Effort attribution surprises:**

1. **S1's "≥10 % at B=8" GO/NO-GO was the wrong reference workload.**
   At B=8, MMQ x8 is actually 16 % SLOWER than wave64_dp4a (block
   size mismatched to small grid; 112 waves vs 1792). Strict
   adherence to the threshold would have aborted before discovering
   the 3× win at production B=128. **Lesson: GO/NO-GO thresholds
   must anchor on the actual production workload, not a microbench
   convenient size.** Updated: B.1.1 BT=16 ceiling diagnosis at B=128
   should be the reference for any future MMQ-class experiment on
   gfx906.

2. **The plan's three correctness landmines (`x_dm` shift, `0.25f`
   factor, dispatch architecture) all surfaced during S1 spec work.**
   The discriminator unit test (q ≡ 5, x ≡ 1.0) caught nothing
   because the plan flagged them upfront and the implementation got
   them right on first try. Pre-spec'ing landmines worked.

3. **lm_head MMQ (S4) added almost nothing (+0.04 %).** The lm_head
   is a small fraction of pp128 GEMM time (M=vocab=248k is huge but
   it fires once per prefill, not 64×/layer like the residual sites).
   In hindsight S4 wasn't worth a session of its own; could have
   been folded into S3.

4. **27B regression validation (deferred from pre-S1 item 8 because
   no Qwen 3.5 27B mq6 model existed)** turned out to show a *bigger*
   speedup than 9B (3.52× vs 2.94×), not a regression. The win
   scales WITH model size because larger M means more `MMQ_Y=128`
   row tiles fully utilized. **Use Qwen 3.6-27B for future 27B-class
   benches** — same architecture family, on disk, no quantize step
   needed.

**Items deferred to future errata (status as of v3.2.6 write-time):**

- Per-mmq_x stride PMC tuning (S2.4 secondary): non-monotonic perf at
  B=24/32 not yet diagnosed; B=16 and B≥40 work fine but the
  intermediate batches fall through to wave64_dp4a. Probably worth
  ~½ session of investigation. [**Closed in v3.2.7** — b128 cliff at
  mmq_x>=32, B=32 now wins 1.15× and B=40-56 improved 16-20 %.]
- MoE-indexed HFQ6 MMQ (plan v3.2.4 item 4): no MoE mq6 model on
  disk, defer until one ships. [**Still open.**]
- `audit-dispatch-coverage.sh` (plan v3.2.4 item 13): **closed** — script
  shipped in repo at `scripts/audit-dispatch-coverage.sh`, runs the
  per-quant DType matcher coverage audit (the silent-corruption gate
  from plan §5.4 part 2). Surfaces 14 pre-existing MQ-family coverage
  gaps in qwen35.rs (MQ8 missing from 14 matchers, MQ2 missing from
  several) which are out-of-scope for Phase B.2 but worth fixing in
  a separate cleanup PR.
- Phase A defensive `assert!(gemv_dp4a_enabled(arch))` on dp4a Rust
  fns (plan v3.2.4 item 6): still pending. [**Closed in v3.2.7**
  — `debug_assert!` added at top of all 11 dp4a Rust fns.]
- DFlash spec-verify mq6 baseline: project has no DFlash + mq6
  workflow today; B.2 used the coherence-harness mq6 reasoning prompt
  as a proxy and confirmed no decode regression. [**Established in
  v3.2.7** — qwen3.6-27b.mq6 + qwen36-27b-dflash-mq4 baseline at
  τ=5.18, decode 4.74 → 15.05 tok/s (3.18×) after capture-mode gate
  lift.]

**Items closed by Phase B:**

- profile.rs HFQ6 byte counts (plan v3.2.4 item 5): **closed** —
  added `hfq6g256_weight_bytes` / `gemv_hfq6g256_bytes` /
  `gemm_hfq6g256_bytes` in S1.
- Phase A `&& !self.capture_mode` guards on audit branch
  (cherry-picked from PR2): **closed** in pre-S1 item 1
  (commit `c54445b`).
- DDTree HFQ6/MQ6 dispatch arms: **closed** (same cherry-pick).

**Calibration takeaway for future plans:**

The HFQ4 reference attribution was a *lower bound*, again — same
direction as v3.2.3's takeaway from Phase A decode (+41-50 % vs
+15-18 % predicted). For new quant ports on gfx906, treat
HFQ4-derived numbers as **estimates of the last incremental lever's
effect**, with realistic upside from amortization wins that compound
when:
- the quantize step is shared (Q8_1 once per fused-projection group)
- LDS-staged data is reused across batch tiles
- the kernel architecture matches mq4's (Window Streaming with
  `_full_set` / `_full_add` compile-time variants)

**Coverage table update (§1.1 row 2 — HFQ6):** all four columns now
✓. The dp4a/MMQ batched cell that was ✗ in v3.2.3 → now ✓ via Phase
B.2 commits `8755a35` + `1acef95` + `bcce686` + `9705856`.

**References:**
- B.2 plan: `docs/plans/gfx906-mq6-mmq-port-phase-b2.md` v2.3
- Cumulative dev-log: `docs/perf-checkpoints/2026-05-07-phase-a-cumulative-mq6.md`
- Path 1 ceiling: `2bee6e6` + `ff9e210`
- Path 2 (S1+S2+S3+S4+S5+27B): `8755a35`, `1acef95`, `bcce686`, `9705856`

### v3.2.7 errata (2026-05-08): Phase B.2 polish — DFlash unlock + cleanup

Three follow-up items shipped post-v3.2.6 close-out:

**1. b128 cliff lowered to mmq_x≥32** (commit `3ac7a3d`).

S2's perf sweep had B=32 at 0.96× (regression vs wave64_dp4a). PMC
diagnosis: at mmq_x=32, the b32 LDS path issues 8 ds_read_b32 per
inner ALU iter, choking the LDS pipeline (MemUnitBusy collapsed to
13.8 %, kernel idle not stalled). HFQ4's b128 cliff at `mmq_x >= 64`
was wrong for HFQ6's heavier unpack — lowered to `mmq_x >= 32` in
both `body.cuh::x_stride_for<>()` and the `vec_dot_dp4a_streaming`
`if constexpr`. Stride must follow (b128 reads need stride=40 for
16-byte alignment).

| Batch | mmq_x | Before | After | Speedup before | After |
|---|---|---:|---:|---:|---:|
| 32 | 32 | 381 µs | 316 µs | 0.96× ❌ | **1.15× ✓** |
| 40 | 40 | 415 | 348 | 1.13× | **1.35×** |
| 48 | 48 | 466 | 391 | 1.26× | **1.51×** |
| 56 | 56 | 542 | 433 | 1.21× | **1.51×** |
| 64+ | 64 | (already b128) | unchanged | | |

`hfq6_mmq_winning_size` updated: B≥32 routes (was B≥40). End-to-end
9B pp128 unchanged at 561 tok/s — pp128 always picks mmq_x=64 which
was already on b128. Forward-looking for spec-decode and other
small-batch shapes.

**2. capture_mode gate lifted for HFQ6 MMQ** (commit `fa8785b`) —
unlocked DFlash mq6.

Initial DFlash mq6 bench (post-Phase-B baseline) showed only 4.74
tok/s decode — investigation revealed 90 % of DFlash time was in
`gemm_*_hfq6g256_fp16` fallback kernels, not MMQ. Root cause: the
4 HFQ6 MMQ branches had `&& !self.capture_mode` guards (leftover
from Phase A's wave64_dp4a hipMemset-during-capture bug). MMQ is
actually capture-safe after the warmup pass populates
`ensure_q8_1_mmq_x`'s scratch buffer + JIT cache.

| State                       | DFlash mq6 27B decode | τ      | accept |
|---|---:|---:|---:|
| Pre-fix (MMQ capture-gated) | 4.74 tok/s            | 5.182  | 0.345  |
| Post-fix (MMQ allowed)      | **15.05 tok/s**        | 5.182  | 0.345  |
| Speedup                     | **3.18×**              | (same) | (same) |

τ and accept_rate unchanged → bit-identical output, bit-identical
acceptance pattern. Pure speedup, no quality drift.

Added capture-aware routing helper `hfq6_mmq_route(capture_mode,
batch_size)`:
- `capture_mode=true`: route MMQ at any B≥8 (vs fp16 fallback, MMQ
  always wins; the wins-vs-dp4a heuristic is irrelevant under
  capture — dp4a is gated off by its own memset_async-during-capture
  bug).
- `capture_mode=false`: use original `hfq6_mmq_winning_size` (B=16
  or B≥32 post-b128-cliff fix) — required to win specifically vs
  dp4a, which IS available outside capture.

This separation prevents the B=8 microbench regression vs dp4a from
being pessimized by capture-mode-only DFlash workloads.

**3. v3.2.4 follow-up cleanup** (commit `8528923`).

- `audit-dispatch-coverage.sh` (item 13): already shipped — surfaces
  14 pre-existing MQ-family matcher coverage gaps in qwen35.rs as a
  separate cleanup target.
- Defensive `debug_assert!(gemv_dp4a_enabled(&self.arch))` on all 11
  dp4a Rust fns (item 6). Catches future caller mistakes loud in
  debug; release pays nothing.
- 17 pre-existing `bind_thread` audit violations cleared. 14 fns get
  `// bind_thread: skip — delegated via ensure_q8_1_mmq_x` markers
  (they call into `ensure_q8_1_mmq_x` first which bind_threads); 3
  fns get real `self.bind_thread()?;` (moe_topk + the new HFQ6 MMQ
  dispatchers I introduced in B.2). `verify-bind-thread.sh` now
  reports OK on 305/305 pub fns.

Bonus inside the cleanup commit: `gemm_hfq6g256_residual_mmq_gfx906`'s
internal size-routing was stuck at the pre-v3.2.7 staircase (B≥40
only). Updated to include B=32 per the b128-cliff fix. Added a
`debug_assert!(!self.capture_mode)` on the wave64_dp4a fall-through
path to defensively catch capture-mode misuse.

**Final state — what's left for HFQ6 on gfx906:**

| Item | Status |
|---|---|
| Phase A: decode (Wave64 + ILP-prefetch + dp4a fused) | ✓ shipped |
| Phase B.1: BT propagation (BT=8→16) | ✓ shipped |
| Phase B.2: MMQ-streaming port | ✓ shipped |
| b128 cliff at mmq_x>=32 | ✓ shipped (v3.2.7) |
| capture_mode lift for DFlash | ✓ shipped (v3.2.7) |
| profile.rs HFQ6 byte counts | ✓ shipped (v3.2.4 item 5) |
| audit-dispatch-coverage.sh | ✓ shipped (v3.2.4 item 13) |
| Defensive asserts on dp4a fns | ✓ shipped (v3.2.4 item 6) |
| bind_thread audit clean | ✓ shipped (v3.2.7) |
| MoE-indexed HFQ6 MMQ | ⚠️ deferred (no MoE mq6 model exists) |
| 27B Qwen3.5 (vs 3.6) regression | ⚠️ deferred (used 3.6 as proxy) |
| B=24 b32-path marginal optimization | ⚠️ low-value micro-optimization |

**Final mq6 9B pp128 prefill: 562 tok/s.** vs mq4 9B pp128 = 599
tok/s (same git ref). mq6/mq4 ratio = **0.939 — within 6 % of mq4
parity.** Bandwidth-bound floor exceeded by 38 %. Phase B done.

**Final 27B mq6 prefill: 192 tok/s.** Phase B unlock = 3.52× over
the wave64_dp4a baseline (54.8 → 192). Bigger speedup than 9B
because larger M means more MMQ_Y=128 row tiles utilized.

**Final DFlash mq6 27B decode: 15.0 tok/s.** v3.2.7 capture-mode
fix = 3.18× over the pre-fix baseline (4.74 → 15.05).

**References:**
- v3.2.7 commits: `3ac7a3d`, `fa8785b`, `8528923`
- Dev-log: `docs/perf-checkpoints/2026-05-07-phase-a-cumulative-mq6.md`

### Calibrated decode-lift expectations (gfx906, post-v3.2.2)

| Lever | Expected 9B mq6 decode Δ | Measured (post-v3.2.3) |
|---|---:|---:|
| Wave64 residual GEMV | +3 % (calibrated) | **+2.9 % (measured ✓)** |
| Wave64 + ILP-prefetch on residual | +5-7 % (calibrated) | **+0.9-1.3 % (measured: smaller than HFQ4 sibling, likely because HFQ6 unpack already amortized some load latency)** |
| Wave64 + dp4a on fused GEMVs trio | +7 % (calibrated, HFQ4 reference) | **+35-40 % (measured: 5× the calibrated value — HFQ6 had no prior dp4a optimization)** |
| **Phase A complete (gfx906 mq6 decode)** | **+15-18 % cumulative** (v3.2.2) | **+41-50 % measured (v3.2.3)** |

**v3.2.3 takeaway: the HFQ4 reference attribution was a lower bound,
not an upper bound, when applied to a quant that hadn't been through
the same optimization journey.** Future plan calibrations for new
quant ports should treat HFQ4-derived numbers as estimates of the
*last incremental* lever's effect, not the *full* lever-stack effect.

This document is **analysis-only**. It maps what's missing for MQ6
and MQ8 to reach the same kernel-coverage level we have for HFQ4
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
| **HFQ6** | 6 | 200 | ✓ (wave32) | ✓ + prefetch (Phase A.1b) | ✓ + dp4a (Phase A.1c) | ✓ (Phase A.1a) | ✓ MMQ + dp4a (Phase B.2) | (n/a) | (n/a) |
| **HFQ8** | 8 | 264 | ✓ (wave32) | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| HFQ3 | 3 | 104 | ✓ | ✓ | ✗ | ✗ | ✗ | (n/a) | (n/a) |
| HFQ2 | 2 | 72 | ✓ | ✗ | ✗ | ✗ | ✗ | (n/a) | (n/a) |

| Quant | batched GEMM (B>1: gate_up/qkv/qkvza/residual) | wave64 batched GEMM | dp4a batched | MoE-indexed |
|---|:---:|:---:|:---:|:---:|
| **HFQ4** | ✓ (multiple variants) | ✓ | ✓ MMQ | ✓ wave64 |
| **HFQ6** | ✓ (base / fp16 / dot2 / wmma / wmma_gfx12 / wave64+dp4a / **MMQ-streaming**) | ✓ (Phase A) | ✓ MMQ (Phase B.2) | ✗ (deferred — no MoE mq6 model) |
| **HFQ8** | ✗ | ✗ | ✗ | ✗ |

**Bottom line (updated post-Phase-B + polish, v3.2.7):**

- **HFQ6 has near-parity coverage on gfx906** post-Phase-B + v3.2.7
  polish. 9B mq6 prefill at 562 tok/s = 0.94× of mq4's 599; 27B at
  192 tok/s = 3.52× over the wave64_dp4a baseline. Phase A (decode)
  + Phase B.1 (BT tuning) + Phase B.2 (MMQ-streaming port) closed
  the original 3.15× gap to 1.07×. Remaining gap is the
  bandwidth-byte-overhead floor (HFQ6 has 1.47× more weight bytes
  per output than HFQ4) — fundamentally limited by memory
  bandwidth, not kernel design.
- **DFlash mq6 spec-decode unlocked at 27B** (v3.2.7 capture-mode
  fix). 27B mq6 + qwen36-27b-dflash-mq4 draft = 15.0 tok/s decode,
  τ=5.18, accept=0.345. 3.18× over the pre-fix baseline.
- **HFQ8 runs end-to-end at B=1 today on gfx906** via
  `gemv_hfq8g256` + `attention_hfq8_kv` + `kv_cache_write_hfq8`.
  The gap is throughput at B>1 (no batched GEMM at all) and the
  wave64 / dp4a optimizations available for HFQ4. Phase A approach
  could port to HFQ8 mechanically if a workload demand emerges.
- **MoE-indexed kernel coverage is HFQ4-only.** Five MoE kernel
  files exist for HFQ4 (down + gate_up, indexed + batched
  variants). Zero exist for HFQ6 or HFQ8 — A3B-class models with
  mq6 weights on gfx906 fall through to wave32 FP fallbacks. Phase
  B.2 explicitly out-of-scope; estimated ~1 session post-B.2 once
  an MoE mq6 model ships.

### 1.2 Lever availability at a glance

| Lever | HFQ4 | HFQ6 | HFQ8 | Why |
|---|:---:|:---:|:---:|---|
| wave64 topology (~3 % decode, foundation only) | ✓ shipped | ✓ shipped (Phase A.1a, `466f1a6`) | applicable | mechanical port; no quant dependence; **single-lever lift modest per v3.2.2 errata** |
| ILP-prefetch in residual GEMV | ✓ shipped | ✓ shipped (Phase A.1b) | applicable | mechanical port; per-thread byte count differs (HFQ6=6, HFQ8=8) but pattern transfers |
| dp4a (int8×int8 via `__builtin_amdgcn_sdot4`) | ✓ shipped | ✓ shipped (Phase A.1c+A.2+A.3) | **shipped as MQ8**, see §3.2 | gfx906 has the instruction; works for any int8-dequantizable weight |
| MMQ-streaming (Window Streaming dp4a) | ✓ shipped (PR #158) | ✓ shipped (Phase B.2 + v3.2.7 b128 cliff) | applicable | small-tile dp4a-MMQ; biggest single-lever win for prefill (Phase B.2 = 2.94×–3.52×) |
| Capture-mode-safe MMQ for DFlash | ✓ via HFQ4 path | ✓ shipped (v3.2.7 `fa8785b`) | applicable | DFlash spec-decode unlock; 3.18× decode at 27B mq6 |
| dot8 (`v_dot8_i32_i4`, int4×int4) | ✓ HFQ4 native | **NO — would require lossy 6→4 repack** | **NO — would require lossy 8→4 repack** | hardware is int4×int4 only, no mixed-precision; see §2.4 |
| MFMA / WMMA (CDNA2+ / RDNA3+) | n/a on gfx906 | n/a on gfx906 | n/a on gfx906 | hardware not available |

The dot8 lever **does not apply** to HFQ6 or HFQ8 — see §2.4 for the
full reasoning.

### 1.3 Estimated effort to feature parity (revised v3)

Coverage scope is **MQ6** and **MQ8** — the deployed FWHT-rotated
variants. The kernel-surface analysis below is identical to "HFQ6
GEMV/GEMM" and "HFQ8 GEMV/GEMM" because the FWHT rotation is one
separate per-layer kernel, not part of the GEMV/GEMM inner loop.

| Surface | MQ6 | MQ8 | Notes |
|---|---:|---:|---|
| wave64 GEMV (AR decode B=1) | ✓ shipped (Phase A.1a) | ~¼ session | MQ8 dword-aligned and trivially ports; MQ6 needs split-load handling |
| wave64 residual GEMV + ILP-prefetch | ✓ shipped (Phase A.1b) | ~½ session | mechanical mirror of HFQ4 work |
| Single-token fused GEMVs (gate_up / qkv / qkvza) | ✓ shipped (Phase A.1c) | ~1 session | new GEMV-level surface for both quants |
| Wave64 batched GEMM (LM-head + per-layer residual, dp4a) | ✓ shipped (Phase A.2-A.4) | ~½ session | MQ6 done; MQ8 covered by direct register-tile dp4a IF wiring exists (see v3.2 errata). |
| MoE-indexed kernels (5 files per quant) | ⚠️ deferred (no MoE mq6 model on disk) | ~½ session | A3B / MoE workload coverage |
| **AR-only complete coverage** | **✓ shipped (cumulative ~3 sessions for Phase A)** | **~2.5 sessions** | MQ8 lighter only counts kernels, not the 14 missing dispatch sites — see v3.2 errata for the expanded MQ8 estimate (~5–6 sessions). |
| dp4a port for fused GEMVs (MQ6 only — MQ8 is dp4a from day one) | ✓ shipped (Phase A.1c) | n/a | PMC-gated before commit |
| MMQ-equivalent dp4a path (DFlash verify) | **✓ shipped (Phase B.2 + v3.2.7 — 5 sessions actual, matched v2.3 budget)** | **not needed** (Phase A item 4 covers it) | MQ6 done; MQ8 batched is structurally simpler |

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

### 3.2 MQ8 — extend dp4a-on-int8 from B=1 reference to wave64 / batched / MoE

**Errata note:** the v3 framing called the B=1 kernel "shipped" — it
was, but only on RDNA3+. Until commit `ee0fac6` (2026-05-06) the
kernel failed to compile on gfx906 because it used the RDNA3-only
`sudot4` builtin. See the v3.1 errata in the header section. The
discussion below describes the fixed kernel.

**Reference kernel ships today:** `kernels/src/gemv_mq8g256.hip` is
the int8-weight × Q8_1-activation dp4a GEMV at B=1. Inner loop:

```c
// MagnumQuant MQ8 GEMV: FWHT-rotated symmetric INT8 with dp4a.
// Inner loop uses v_dot4_i32_iu8 for 4x VALU throughput vs FP32.
//
// Weight format per group (258 bytes for 256 elements):
//   [0:2]   f16 scale
//   [2:258] int8[256] quantized FWHT-rotated weights
int dot = __builtin_amdgcn_sudot4(true, wp0, true, xp0, 0,   false);
dot     = __builtin_amdgcn_sudot4(true, wp1, true, xp1, dot, false);
```

The `(true, w, true, x, ...)` flags say both operands are signed int8;
this delivers the 4× ALU throughput PR #158 exploited for HFQ4. **The
gap on gfx906 is everything around the inner loop**: no wave64 variant,
no residual+ILP-prefetch variant, no fused single-token GEMVs, no
batched GEMM, no MoE-indexed kernels.

#### 3.2.1 On-disk format and runtime status

**MQ8 weights:** symmetric signed `q ∈ [-127, 127]`, fp16 scale only,
256 weights per 258-byte group. **FWHT-rotated at quantize time**;
matching activation rotation happens once per layer via the existing
`mq8_rotate_quantize_x` / `fused_rmsnorm_rotate_mq` kernels (already in
the tree at `crates/hipfire-runtime/src/dispatch.rs:2495`). The
rotation is **one small per-layer kernel**, not a per-GEMV cost; the
GEMV inner loop is identical to a hypothetical un-rotated int8-weight
GEMV.

Quantize-tool support: `--format mq8` in `crates/hipfire-quantize/src/main.rs`
(function `quantize_mq8g256` at line 540). No `hf8` format exists in
the tool; raw HFQ8 work is deprioritized indefinitely (see header v3
reframe).

**MQ8 runs end-to-end on gfx906 at B=1 today** via `gemv_mq8g256` plus
the activation-rotate kernel called once per layer.

#### 3.2.2 Coverage gap on gfx906

| Path | MQ8 today | Gap |
|---|---|---|
| Plain GEMV (B=1) | ✓ dp4a (`gemv_mq8g256.hip`) | needs wave64 variant for occupancy parity with HFQ4 |
| Residual GEMV (B=1) | none | needs wave64 + ILP-prefetch |
| Single-token fused GEMV (B=1: gate_up / qkv / qkvza) | none | needs wave64 + dp4a (mirror of HFQ4 fused family) |
| **Batched GEMM (B>1)** | **none** | needs wave64 + dp4a (mirror of `gemm_hfq4g256_wave64_dp4a`) |
| MoE-indexed | none | needs 5 MQ8 kernels (down + gate_up, indexed + batched) |

#### 3.2.3 Lever map for MQ8

| Lever | Status | Notes |
|---|---|---|
| wave64 GEMV (1.5–2× over wave32) | not shipped for MQ8 | mechanical port; int8 weights are 8-byte/thread, dword-aligned, strictly easier than HFQ6 |
| ILP-prefetch in residual | not applicable yet | needs residual variant first; then mechanical |
| dp4a on int8 weights × Q8_1 activations | ✓ shipped at B=1 | extend to wave64 / fused / batched / MoE |
| dp4a-batched (Q8_1 × int8 weights at B>1) | not shipped | direct extension of the B=1 kernel: same inner loop, batched accumulators, no LDS-streaming MMQ required |
| dot8 (`v_dot8_i32_i4`) | NOT applicable (would require lossy 8→4 repack) | see §2.4 |
| LM-head dp4a port | not shipped | mechanical mirror of `gemm_hfq4g256_wave64_dp4a` |

#### 3.2.4 Phase A: wave64 + batched GEMM + MoE (~2.5 sessions)

Five small ports, all dp4a-on-int8 (no FP variant needed since the
B=1 dp4a kernel is the reference):

1. `gemv_mq8g256_wave64.hip` — wave64 mirror of `gemv_mq8g256`. Trivial
   because 8-byte alignment plays nicely with dword loads. ~¼ session.
2. `gemv_mq8g256_residual_wave64.hip` + ILP-prefetch variant. New
   residual surface (doesn't exist today). ~½ session.
3. `fused_gate_up_mq8g256_wave64_dp4a.hip`, `fused_qkv_*`,
   `fused_qkvza_*` — three single-token fused GEMVs at B=1. Mirror
   of the HFQ4 fused-dp4a family. ~1 session.
4. `gemm_mq8g256_wave64_dp4a.hip` — batched GEMM (B>1) for prefill /
   DFlash verify / LM-head. The activations are already int8 and small
   enough to fit in registers per-batch; **no LDS-streaming MMQ
   required** (this is the structural difference from HFQ6 Phase C).
   ~½ session.
5. MoE-indexed: 5 kernels mirroring the HFQ4 family. ~½ session.

**Activation rotation:** the dispatch layer already routes through
`mq8_rotate_quantize_x` / `fused_rmsnorm_rotate_mq` to produce rotated
fp16 input, then through `quantize_q8_1_mmq_ds4` to produce Q8_1
activations. The new wave64 / batched / MoE kernels consume the same
Q8_1 scratch — no new wiring at the runtime layer.

#### 3.2.5 No MQ8 MMQ-streaming port

Q8_1 × int8 weights at B>1 is structurally simpler than HFQ4-MMQ:
weights are already int8, no nibble unpack, no per-mmq_x LDS-stride
sweep. Phase A item 4 covers the batched case directly with batched
dp4a accumulators. **No separate "Phase C" for MQ8.** This is the
structural payoff of doing MQ8 instead of HFQ6 work first.

#### 3.2.6 MQ8 + DFlash

DFlash verify-pass on Qwen 9B mq8 currently dispatches to a B=1 kernel
called N times for the verify batch — functional but slow. After Phase
A item 4, the dp4a-batched path covers it directly. Expected lift: TBD
by Priority 0; direction is positive since the verify pass is
batched-GEMM-dominated.

**Production-relevance check:** MQ8 is the deployed int8 format. The
quantize tool emits it; mq8 models are produced from local hf16/bf16
sources via `hipfire-quantize --format mq8`. Workload-demand is
conditional on whether mq8 deployment becomes meaningful for gfx906
users vs the existing mq4/mq6 paths.

### 3.3 MQ3 — out of scope but flagged

Per AGENTS.md §A: MQ3 is production on gfx11/gfx12, and on gfx906
"MQ3 weights still load and run via per-token GEMV fallback —
correct, just slower." MQ3 has *more* documented production demand
than mq6/mq8 on gfx906.

**The priority list (§5) considers MQ3 alongside MQ6/MQ8 in the
demand check.** A separate MQ3-specific plan would mirror this
document but is not in scope here.

---

## 4. Coherence-validation note (carried from dot8 work)

The closed `gfx906-dot8-port.md` PRD (on the `feat/gfx906-dot8-port`
branch — not in master) established that **int4 activations are not
viable for transformer inference on these models**. Q4_1 NRMSE 18×
Q8_1; geometric floor at ~9-12% worst-block even with asymmetric
quant + smaller groups. That conclusion is load-bearing for MQ6 / MQ8
work too:

- The activation format for any MQ6 / MQ8 dp4a variant must be
  **Q8_1** (the existing format), not Q4_1. (Weight quant choice is
  independent.)
- Any future MQ6 MMQ port (§3.1.3 Phase C) must inherit PR #158's
  `mmq_screen` + `mmq_screen_threshold` mechanism for outlier-row
  rejection.

**The dot8 lever explicitly does not apply to MQ6 or MQ8** — see §2.4.

---

## 5. Recommended priority order (revised v2)

### 5.0 Priority 0: baseline measurement (~½ session, prerequisite)

**Before any kernel work**, run the canonical AR decode + DFlash
benches on existing mq4 / mq6 / mq3 paths on gfx906 with 5-run
deterministic medians per AGENTS.md prompt-md5 / binary-md5
requirements:

- Qwen 9B mq4 AR decode tok/s (sanity baseline against PR #158
  numbers)
- Qwen 9B mq6 AR decode tok/s (target for §3.1 wave64 work)
- Qwen 9B mq3 AR decode tok/s (per AGENTS.md §A: production on
  gfx11/12, runs via fallback on gfx906)
- DFlash 27B mq6 humaneval-0 tok/s (27B mq6 quantized 2026-05-06
  via `hipfire-quantize --format mq6` from Qwen3.6-27B bf16)

**Note (v3.2):** mq8 is excluded from Priority 0. The per-layer
runtime dispatch isn't wired (see v3.2 errata header). A bench
would produce GPU-time numbers but the inference is invalid;
re-add mq8 only if the per-layer wiring is restored.

**Quantize prerequisites** (before bench): the canonical mq6 9B
and 27B targets are produced locally via:

```
hipfire-quantize --format mq6 <hf16-source>  # ~1-2 min for 9B,
                                             # ~5-8 min for 27B
```

(The 9B mq6 already ships in the registry; 27B mq6 is the new
artifact, produced from Qwen3.6-27B bf16 since architectural
config is identical to Qwen3.5-27B.)

Record absolute tok/s + the comparison matrix. Write up at
`docs/perf-checkpoints/2026-05-06-mq6-baselines.md`. **All lift
estimates below are placeholders pending Priority 0.**

### 5.1 Priority list (revised v3.2.3 — Phase A decode complete, prefill is the new headline)

HFQ8 dropped per v3. MQ8 demoted per v3.2. MQ6 sub-phases reordered
per v3.2.2 (dp4a-fused leads). v3.2.3 marks Phase A decode work as
shipped with measured +41-50 % cumulative (vs +15-18 % calibrated
target — see v3.2.3 errata for why) and reframes the priority list
around the now-dominant prefill gap.

| Priority | Phase | Cost | Decode Δ | Prefill Δ | Risk | Status |
|---:|---|---:|---:|---:|---|---|
| 1a | MQ6 wave64 residual GEMV | done (`466f1a6`) | **+2.9 % (9B), +3.9 % (27B) ✓ measured** | 0 % | low | **shipped 2026-05-06** |
| 1b | MQ6 ILP-prefetch on residual | done (`692d792`) | **+0.9 % (9B), +2.8 % (27B) ✓ measured** | 0 % | low | **shipped 2026-05-06** |
| 1c | MQ6 dp4a-on-fused-GEMVs trio | done (`ba246c4`) | **+36.2 % (9B), +40.4 % (27B) ✓ measured** | 0 % | low (gated on `gemv_dp4a_enabled`) | **shipped 2026-05-07** |
| **1a+1b+1c cumulative** | (Phase A decode complete) | done | **+41.5 % (9B), +50.0 % (27B) ✓ measured** | 0 % | — | **shipped — see `2026-05-07-phase-a-cumulative-mq6.md`** |
| **2** | **MQ6 dp4a-on-batched-residual** (`gemm_hfq6g256_residual_wave64_dp4a.hip`, prefill wo + w_down at B>1) | ~1 session | 0 % | est +20-30 % (subset of prefill) | low — mirrors A.1c structure at GEMM shape | **next: Phase A.2** |
| 3 | MQ6 wave64 batched non-residual GEMM (`gemm_qkvza_hfq6g256_wave64.hip` etc.) | ~1 session | 0 % | est +5-10 % | low | natural completion of batched-fused family |
| 4 | MQ6 LM-head wave64 batched (`gemm_hfq6g256_wave64.hip`) | ~½ session | small (lm_head is ~8 % of decode) | small | low | low priority |
| 5 | MQ6 MoE-indexed wave64+dp4a (10 kernels, A3B / MoE workload) | ~1 session | A3B-mq6 dependent | A3B-mq6 dependent | low | only if MoE+MQ6 has demand |
| 6 | MQ6 Phase C (MMQ batched, dp4a streaming) | **5 sessions** | 0 % | up to +5× (mirror of PR #158 HFQ4 result) | high — full LDS bank-conflict diagnostic + mmq_screen plumbing | the heavy lever if prefill-heavy mq6 production demand emerges |
| 7 | MQ8 Phase A (per-layer wiring + 4-7 new batched-GEMM kernels + 14 dispatch sites) | **~5-6 sessions** (per v3.2) | unmeasured | unmeasured | high | **deferred until a production model ships raw MQ8 per-layer weights** |
| 8 | (MQ3) — separate plan, gated on demand | — | — | — | — | likely higher demand than mq6/mq8 per AGENTS.md §A |

**Phase A decode is complete.** The remaining surface is prefill +
LM-head + MoE. Prefill is now the dominant gap (mq6 13× behind mq4).
Phase A.2 (dp4a-batched-residual, ~1 session) is the cheap intermediate
prefill lever; Phase C (MMQ-batched, ~5 sessions) is the heavy lever.

**Decision rule:** do priority 1 *only if* Priority 0 shows a real
workload using mq6 on gfx906. Otherwise defer. The lessons from PR
#158's diagnostic-first methodology + the closed dot8 PRD's negative
result both point toward "don't build speculative kernel
optimizations."

**Why MQ6 leads in v3.2.2:**
- Wave32 path (`gemm_qkvza_hfq6g256`, `gemv_hfq6g256`,
  `fused_rmsnorm_mq_rotate`) works end-to-end on gfx906 today —
  audited in §5.5 plus exercised by Priority 0 baseline (commit
  `850848a`).
- The dp4a-on-fused-GEMVs port is the dominant lever, expected
  +7-8 % decode per HFQ4 reference. Wave64 ports underneath provide
  the foundation (each ~+3 % alone, +15-18 % cumulative when
  combined with dp4a).
- MQ6 ships in 9 distinct production model artifacts in
  `cli/registry.json` (Qwen3.5/3.6 base + Carnice + Qwopus
  finetunes, 0.8B/4B/9B/27B sizes; `--both` shorthand produces
  mq4+mq6 simultaneously). Per `docs/MODELS.md`, mq6 is the
  curated quality tier above mq4.
- The plan body §3.1 documents Phase A subitems (1a–1e) and Phase C
  (MMQ batched). MQ6 priority-1 ranking is grounded in measured
  bench data (today's wave64 +3 %), HFQ4 attribution (PR #158
  +16.2 % cumulative), and deployment reality (registry breadth).

**Why MQ8 is now priority 4 (effectively deferred indefinitely):**

- v3.2 errata: per-layer runtime dispatch is not wired (14 sites in
  the arch crate exclude MQ8G256). Phase A item 4 isn't an
  optimization — it's a correctness prerequisite. Without it, MQ8
  inference for any non-lm_head workload is invalid.
- Kernel surface to write: `gemm_qkvza_mq8g256_wave64_dp4a.hip`,
  `gemm_qkv_mq8g256_wave64_dp4a.hip`,
  `gemm_gate_up_mq8g256_wave64_dp4a.hip`,
  `gemv_mq8g256_residual_wave64.hip`, plus MoE-indexed variants.
  Each needs coherence-gate + correctness validation.
- Runtime wiring: 14 dispatch sites in `qwen35.rs` need MQ8G256
  added to their matchers, plus rotation-aware activation prep
  (the rotated path is more involved than HFQ4's plain rmsnorm).
- No deployed per-layer MQ8 model exists. The lm_head path (which
  IS wired) is served by the existing B=1 GEMV and benefits from
  `ee0fac6` already.

### 5.2 Coherence-gate cost (per glm5 3.4)

Each phase needs ~30 min to ~1 hr for `coherence-gate.sh` +
`coherence-gate-dflash.sh` (where applicable) per kernel batch.
Bake into the per-phase totals.

### 5.3 Per-arch scope

This work is **gfx906-only**. gfx11 / gfx12 (RDNA3 / RDNA4) WMMA
paths are unaffected. `gemv_hfq6g256.gfx1201.hip` and other RDNA-
specific HFQ6 kernels need no changes.

### 5.4 Build + runtime-dispatch audit (~1 session, expanded by v3.2 errata)

The v3.1 audit (§5.5) caught the `sudot4` build failure but missed
the more critical runtime-dispatch gap (v3.2 errata): every
`is_mq` matcher in the arch crate excluded `MQ8G256`, so MQ8
weights were silently corrupted via stride-mismatched HFQ4 reads.
Build-test alone isn't sufficient.

**Two-part audit, mandatory before any MQ-related Phase A work:**

**Part 1 — kernel build verification (already done in §5.5):**

1. `grep -rn '__builtin_amdgcn_\(sudot\|wmma\|s_wait_event\|v_dot.*bf16\|v_dot.*f16\)' kernels/src/`
2. For each match, check the kernel's dispatch path: is it ever
   instantiated on gfx906 today? If yes, build-test it via the
   runtime's JIT path.
3. Build-test every kernel listed in `kernels::*_SRC` constants on
   gfx906 via `hipcc --genco --offload-arch=gfx906`.

**Part 2 — runtime-dispatch verification (new in v3.2):**

For each `DType::*G256` variant the loader can produce
(`crates/hipfire-runtime/src/hfq.rs:417`), verify wiring at every
per-layer call site:

1. `grep -rn 'is_mq = matches!\|is_6bit = matches!\|is_mq3 = matches!\|is_mq8 = matches!' crates/hipfire-arch-*/src/`
2. For each matcher, confirm every loader-producible quant format
   appears in at least one branch. Missing format → silent
   fall-through → corrupted inference.
3. For each per-layer GEMM/GEMV call, confirm the matcher arms
   route to a dtype-correct kernel (e.g. `gemm_qkvza_hfq6g256`
   handles HFQ6 stride; an MQ6/MQ4/MQ8 weight in the `else` arm
   would mis-read).
4. Cross-check `coherence-gate.sh` test matrix at
   `scripts/coherence-gate.sh:84-103` covers each loader-producible
   format end-to-end. Missing entries → no automated detection of
   future dispatch gaps.

**Why both parts before any Phase A:** every "shipped on gfx906"
claim in §3.1 and §3.2 is load-bearing for the priority list. The
v3.2 audit raised MQ8 effort estimate from ~2.5 sessions to ~5-6
sessions because the runtime-dispatch gap was uncounted. Treat
this as Priority 0.5 — between the baseline measurement and any
kernel work — and run **both parts** for any quant format being
considered for production deployment.

### 5.5 Audit results (2026-05-06)

Build-tested every MQ / HFQ6 / HFQ8 kernel that the gfx906 dispatch
routing requests, plus a sample of WMMA / dot2 kernels for
gate-correctness verification. All build via
`hipcc --genco --offload-arch=gfx906 -O3` with the same flags the
runtime uses.

**14 dispatched-on-gfx906 kernels: all build clean.**

| Kernel | Result | Notes |
|---|---|---|
| `gemv_mq8g256.hip` | ✓ post-`ee0fac6` | fixed by v3.1 errata |
| `gemv_mq6g256.hip` | ✓ | shipped wave32 path |
| `gemv_mq4g256.hip` | ✓ | shipped (PR #158) |
| `gemv_hfq6g256.hip` | ✓ | shipped wave32 |
| `gemv_hfq6g256_residual.hip` | ✓ | shipped wave32 |
| `gemv_hfq8g256.hip` | ✓ | shipped wave32 |
| `gemm_hfq6g256_residual.hip` | ✓ | shipped FP32 batched |
| `gemm_hfq6g256_residual_fp16.hip` | ✓ | shipped FP16 batched |
| `gemm_qkvza_hfq6g256.hip` | ✓ | shipped fused batched |
| `gemm_qkv_hfq6g256.hip` | ✓ | shipped fused batched |
| `gemm_gate_up_hfq6g256.hip` | ✓ | shipped fused batched |
| `kv_cache_write_int8.hip` | ✓ | KV cache write path |
| `fused_rmsnorm_mq_rotate.hip` | ✓ | activation-rotate prepass |
| `fused_silu_mul_mq_rotate.hip` | ✓ | activation-rotate prepass |

**WMMA kernels: build FAIL on gfx906, gates are required and correct.**

| Kernel | gfx906 build | Gate |
|---|---|---|
| `gemm_hfq4g256_residual_wmma.hip` | ✗ (`gfx11-insts,wavefrontsize32`) | `has_wmma_f16` (gfx11+) ✓ |
| `gemm_qkvza_hfq4g256_wmma.hip` | ✗ (same) | `has_wmma_f16` ✓ |
| `gemm_gate_up_hfq6g256_wmma.hip` | ✗ (same) | `has_wmma_f16` ✓ |

**dot2 kernels: build OK on gfx906, but dispatch is gated to RDNA2+.**

| Kernel | gfx906 build | Dispatch gate |
|---|---|---|
| `gemm_gate_up_hfq4g256_dot2.hip` | ✓ | `has_dot2_f32_f16()` allowlist excludes gfx906 |
| `gemm_gate_up_hfq6g256_dot2.hip` | ✓ | same |

The `dot2` allowlist at `crates/rdna-compute/src/dispatch.rs:123-130`
explicitly omits gfx906 despite gfx906 hardware supporting
`v_dot2_f32_f16` (it carries the `dot2-insts` feature in LLVM). This
may be a missed FP16-GEMM optimization opportunity for gfx906 —
unmeasured. Treat as a **deferred Phase B candidate** for the MQ6 /
HFQ6 batched GEMM surface, not a blocker. Whether it beats the
current wave32 FP32 path on gfx906 is a PMC-gated experiment.

**Conclusion: no further audit-driven plan changes.** The `sudot4`
bug in `gemv_mq8g256.hip` was a one-off, caused by the kernel being
authored on RDNA3+ hardware without a gfx906 build-test in the loop.
The audit confirms every other "shipped on gfx906" claim in §3.1 /
§3.2 is build-verified. Phase A estimates can be treated as
load-bearing again.

### 5.6 dot2 (`v_dot2_f32_f16`) gfx906 — RULED OUT BY MEASUREMENT (2026-05-06)

The `has_dot2_f32_f16()` allowlist at `dispatch.rs:123-130` excludes
gfx906 even though gfx906 hardware carries the `dot2-insts` feature.
v3.2.1 flagged adding gfx906 to the allowlist as a deferred Phase B
candidate. Tested 2026-05-06; **measured negative on prefill,
unaffected on decode by design.**

| Workload | Δ vs baseline (5-run median) |
|---|---|
| 9B mq6 prefill pp=32 | **-2.4%** |
| 9B mq6 prefill pp=128 | **-2.4%** |
| 27B mq6 prefill pp=32 | **-2.2%** |
| 27B mq6 prefill pp=128 | 0% (3.0% spread) |
| 9B / 27B decode pp={32,128} | 0% (decode never enters the dot2 path) |

**Root cause:** the dot2 dispatch path requires an FP32→FP16
conversion of the entire X buffer before each kernel invocation
(`ensure_fp16_x` at `dispatch.rs:1150+`). Under graph capture
(`HIPFIRE_GRAPH=1`, the bench-cold default), the conversion fires
on every replay because the X data changes per chunk even though the
buffer pointer is stable (see comment at `dispatch.rs:1167`). Per
mq6 prefill layer, three of these conversions land (qkvza + qkv +
gate_up). On gfx906 the dot2 kernel's bandwidth saving (X reads
16 B/iter vs 32 B/iter) does not compensate for the ~180 MB extra
HBM traffic per prefill pass at 27B-scale.

The wave32 scalar baseline reads X directly as FP32 with no
conversion; the dot2 path is net-negative under graph capture.

**Impact on §5.1 priority list:** none. dot2 isn't on the priority
list. The §5.6 was originally documented as a deferred candidate;
this experiment closes the loop.

**See:** `docs/perf-checkpoints/2026-05-06-dot2-gfx906-experiment.md`
for full numbers, kernel-source comparison (VGPR=32 / SGPR=50
identical across scalar/fp16/dot2 — so the regression isn't
register-pressure driven), and the cross-arch note explaining why
dot2 wins on RDNA2/RDNA3 archs but loses on gfx906 (different
graph-capture FP16 plumbing on those archs).

### 5.7 Runtime-dispatch sweep results (v3.2.1, 2026-05-06)

Per §5.4 part 2, swept all 28 `gpu_dtype` matchers in
`crates/hipfire-arch-qwen35/src/qwen35.rs` plus 4 in
`crates/hipfire-arch-qwen35/src/speculative.rs` for coverage of the
12 loader-producible quant types.

**Per-quant matcher coverage:**

| Quant | qwen35.rs `is_mq` | qwen35.rs `is_6bit` | speculative.rs lm_head batched | Production |
|---|---|---|---|---|
| HFQ4G256 | n/a (not rotated) | n/a | ✓ | ✓ ships |
| HFQ4G128 | n/a | n/a | n/a | ✓ ships |
| HFQ6G256 | n/a | ✓ all sites | ✗ (perf miss only — falls through to unbatched) | rare |
| HFQ3G256/G128 | n/a | n/a | n/a | rare |
| MQ4G256 | ✓ all 28 | n/a | ✓ | ✓ ships (default) |
| MQ6G256 | ✓ all 28 | ✓ all 10 | ✗ (perf miss) | ✓ ships |
| **MQ3G256** | **partial 26/28** — missing from MoE-batched LA (line 4651) and MoE-batched FA (line 4802) | n/a | ✓ | ✓ ships gfx11/12 dense; A3B+MQ3 not deployed |
| MQ8G256 | ✗ 0/28 (per §5.4 / dev-log) | n/a | ✗ | only as lm_head |
| **MQ2G256** | **✗ 0/28** | n/a | ✗ | unknown — quantize-tool supports it |
| Q8_0 | n/a | n/a | ✓ | ✓ ships |
| F32 | n/a | n/a | n/a | rare |

**Three latent silent-correctness gaps found beyond MQ8:**

1. **MQ3 dropped from MoE-batched matchers** (lines 4651, 4802 in
   `qwen35.rs`). Pattern: copy-paste from dense LA/FA bodies (which
   include MQ3) into the duplicated MoE bodies (which dropped MQ3).
   Trigger: any MoE model with MQ3 weights — e.g. a hypothetical
   Qwen3.6-35B-A3B mq3 quant. Failure mode: rotation pre-pass
   skipped → activation handed to GEMV without FWHT → wrong
   arithmetic. **No deployed model triggers this today** (only
   `qwen3.6-35b-a3b.mq4` ships).

2. **MQ2G256 has zero matcher coverage** (28/28). Same class as
   MQ8: loader produces it, quantize-tool supports `--format mq2`,
   but every per-layer prefill-batched dispatch site silently
   falls through to HFQ4-stride read. Failure mode: stride-mismatch
   corruption (72 vs 136 B/group). **No deployed `*.mq2` model.**

3. **MoE `use_kernarg_fused` predicate gap** at
   `qwen35.rs:1931`. Currently:
   ```
   let use_kernarg_fused = k == 8 && routed_gate_up_mq4 && x_rot_local.is_some();
   ```
   The check gates on `routed_gate_up_mq4` but not `routed_mq4`
   (which checks `down`). Mixed-precision MoE (gate_up=MQ4 but
   down=MQ6/MQ3) would silently corrupt the down kernel. Trivial
   one-token fix:
   ```
   let use_kernarg_fused = k == 8 && routed_gate_up_mq4 && routed_mq4 && x_rot_local.is_some();
   ```
   **Mixed-precision MoE not deployed today** — `--format <X>`
   produces uniform expert quants — but recommended to land
   alongside any future MoE-mq6 work.

### 5.8 Phase A scope freshness check (v3.2.1, 2026-05-06)

Reconciled §3.1.1's enumerated 5 wave64 ports against the current
kernel tree. All 5 are confirmed missing. **One additional kernel
should be added to Phase A scope:**

- `gemm_hfq6g256_residual_wave64.hip` — wave64 sibling of the
  shipped `gemm_hfq6g256_residual.hip`. The existing wave32
  variant fires at 6 call sites (`qwen35.rs:4130, 4210, 4520,
  4590` for dense LA/FA wo+w_down prefill batched; `llama.rs:1625,
  1680` for the same projection family).

Phase A item 4 in §3.1.1 is named "LM-head batched GEMM" and
implies non-residual `gemm_hfq6g256_wave64.hip` only. The
per-layer batched residual is a separate kernel shape and adds
~½ session to the estimate — **revised Phase A total: ~3.5
sessions** (was ~3 sessions in §1.3 / §3.1.1).

---

## 6. What's not blocked by this analysis

- The HFQ4 / MQ4 production path (PR #158 work) is not affected.
- The existing wave32 MQ6 / B=1 dp4a MQ8 paths remain functional
  throughout (Phase A adds wave64 alongside; doesn't remove wave32).
  *(MQ8 B=1 path verified 2026-05-06 via single-process bench; see
  v3.1 errata.)*
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
- Adversarial reviews (folded into v2; renamed in v3 alongside this doc):
  - `docs/plans/gfx906-mq6-mq8-port-plan-rev-claude.md`
  - `docs/plans/gfx906-mq6-mq8-port-plan-rev-gemini.md`
  - `docs/plans/gfx906-mq6-mq8-port-plan-rev-glm5.md`
- AGENTS.md §A (MQ3 production status), §5 (perf-bench
  reproducibility requirements), CLAUDE.md (coherence-gate
  requirements per kernel change).
