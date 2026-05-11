# DFlash decode overhead — three-lever plan

**Status:** Proposed (2026-05-08).
**Source issue:** [#172](https://github.com/Kaden-Schutt/hipfire/issues/172) — `gfx906 DFlash decode: ~17% dispatch overhead in steady-state — three-lever proposal`.
**Author:** Claude Opus 4.7 (1M ctx) post-review.
**Owner:** TBD.

This plan turns issue #172's three numbered levers into an executable
work order. The picture has shifted since #172 was filed (PR #158
landed, runtime crate was split, commit `9aabdcf` removed env::var
overhead from the gfx906 MMQ residual path), so each lever's scope is
re-stated against the post-`9aabdcf` `master` rather than the original
issue body.

The headline: **all three of #172's numbered levers are still open on
master**, but lever 1's premise has migrated from gfx906 to gfx11
because of PR #158's MMQ kernel redesign. Levers 2 and 3 are
arch-agnostic and unchanged.

---

## 1. Background — what changed since #172 was filed

| Event | Commit/PR | Effect on #172's three levers |
|---|---|---|
| MMQ redesign | `afb84bd` (PR #158, 2026-05-06) | gfx906 MMQ residual moved to `_full_add_*` accumulate-into-Y kernels — **no pre-zero memset needed**. Lever 1's gfx906 framing is structurally fixed. |
| Runtime crate split | `b19251ee` / `081634da` (2026-05-05) | `crates/engine` → `crates/hipfire-runtime`; `qwen35` extracted to `crates/hipfire-arch-qwen35`. Speculative decode lives at `crates/hipfire-arch-qwen35/src/speculative.rs`. Line numbers from #172 (`speculative.rs:2158, :2708`) held; file just moved crates. |
| MMQ env::var removal | `9aabdcf` (PR #177-equivalent, 2026-05-06) | Removed 5+ per-call `env::var` lookups + atomic from `gemm_hfq4g256_residual_mmq_gfx906`. Per-call cost dropped from 5–25 µs to a single bool load. **NOT one of #172's three levers** — separate hot-path fix. The maintainer's #172 comment conflated the two; this plan does not. |

**Net state on `master` today:**
- Lever 1 (gfx906 MMQ residual pre-zero): **moot** on gfx906; **still
  present on gfx11/gfx12 WMMA paths** (three sites, hit per spec-decode
  cycle).
- Lever 2 (sync `argmax_buf` D2H): **unchanged**, both sites still
  synchronous.
- Lever 3 (GPU-side accept/reject): **unchanged**, no `commit_count`
  kernel exists.

---

## 2. Lever inventory (with current code citations)

### Lever A — Hoist per-cycle WMMA pre-zero memsets (gfx11/gfx12)

`crates/rdna-compute/src/dispatch.rs`:

| Site | Function | Path | Per-cycle hits |
|---|---|---|---|
| 8184 | `gemm_mw16_residual_wmma_via_dequant` | gfx11 residual WMMA via per-call FP16 dequant | 1× per residual call |
| 8255 | `gemm_hfq4g256_batched_lmhead` (WMMA arm) | gfx11 lm_head WMMA, MQ4 weights | 2× per cycle (verify + draft) |
| 8294 | `gemm_hfq3g256_batched_lmhead` (WMMA arm) | gfx11/gfx12 lm_head WMMA, MQ3 weights | 2× per cycle (verify + draft) when MQ3 draft |

All three follow the same pattern:

```rust
// dispatch.rs:8181-8186 (representative)
self.fp16_x_source_ptr = std::ptr::null_mut();
match self.active_stream.as_ref() {
    Some(stream) => self.hip.memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
    None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
}
```

The Y buffer is the caller's persistent scratch — `verify_scratch.logits_batch`
and `draft_scratch.logits_batch` for lm_head, the residual chain's Y for
the residual WMMA. **Neither the buffer pointer nor its length changes
across cycles** in steady-state DFlash decode (B is fixed by the
draft config; m and batch_size are loop-invariant).

**Hoist target:** lift the memset out of the dispatch site and into
scratch initialization (one-shot) plus an explicit per-cycle reset
*only when shape changes* (rollback to a smaller B, KV refill, etc.).
The kernel's `y += acc` semantics demand zero-on-first-use, not
zero-on-every-call.

#### Constraints (observed during code read)

1. **`fp16_x_source_ptr = null` stomp.** Adjacent to the memset; needed
   because `ensure_fp16_x` caches FP32→FP16 conversion keyed on source
   pointer, and DFlash callers reuse the same pointer with new data
   (comment at dispatch.rs:8233). This stomp must stay per-call. **It
   is not part of the memset cost** — only the buffer-fill is.
2. **Non-WMMA fallback path** at 8295+ (per-batch GEMV loop) does not
   pre-zero. Anything we do must preserve the WMMA-only zero behavior
   without affecting the GEMV fallback.
3. **Coherence:** `gemm_*_residual_wmma` does `y += acc`. If the buffer
   is hoisted-zeroed once and the kernel runs N times without a
   reset, the N-th call accumulates N applications of `acc` instead
   of one. **This is the load-bearing invariant** — every dispatch must
   either run on a freshly-zeroed Y or use a `_replace`-style kernel
   variant.

#### Two implementation shapes (pick one)

- **A1 — Caller-managed zero, dispatch trusts it.** Move the memset to
  `VerifyScratch::new` / `DraftScratch::new` and to the rollback path.
  Dispatch becomes precondition-trusting. Simpler diff, but the
  invariant is enforced by hoping callers behave.
- **A2 — Add a `_replace` kernel variant.** Compile a sibling kernel
  that does `y = acc` instead of `y += acc`. First-call-of-cycle uses
  `_replace`, subsequent calls (none today, but future fusion paths)
  use `_add`. No memset needed at all. Larger diff (new kernel
  variants), but eliminates the invariant rather than relocating it.

**Recommendation:** A1 first. The current code has only one caller per
buffer per cycle, so the invariant collapses to "callers zero their
scratch once at allocation." A2 is the cleaner long-term shape and
worth a follow-up issue if the WMMA dispatch surface grows.

#### Estimated lift

#172's profile pointed to ~1.5 ms/cycle on the residual `fillBufferAligned`
gaps (gfx906 framing). On gfx11 with three sites at ~50–500 µs each, the
upper bound is similar — 1–2 ms/cycle. At 27B-3.5 LRU DFlash 199 tok/s
≈ 5 ms/token, that's a 20–40% per-cycle ceiling. Realistic floor:
3–5% decode tok/s.

### Lever B — Async D2H of `argmax_buf` (arch-agnostic)

`crates/hipfire-arch-qwen35/src/speculative.rs`:

| Site | Caller | Buffer | Bytes | Sync? |
|---|---|---|---|---|
| 2151–2159 | verify path, greedy batched argmax | `verify_scratch.argmax` (B × i32) | 4 × B (≤64 at B=16) | Yes — `gpu.hip.memcpy_dtoh` |
| 2701–2710 | draft path, greedy batched argmax | same | 4 × (B-1) | Yes — `gpu.hip.memcpy_dtoh` |

```rust
// speculative.rs:2151-2160 (verify-side, abridged)
let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b)?;
let mut host_idx = vec![0i32; b];
{
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, b * 4)
    };
    gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;   // ← sync, blocks the chain
}
for &idx in &host_idx { argmax_per_pos.push(idx as u32); }
```

Sync D2H here pins both ends of the cycle: the next launch chain
(repeat-penalty / accept loop / next draft) can't start until the
copy returns to the host. The D2H itself is small (≤64 bytes), so
the actual cost is **not the bandwidth** — it's the
`hipDeviceSynchronize`-flavored stall.

#### Plan

1. Replace `gpu.hip.memcpy_dtoh` with `memcpy_dtoh_async` on `active_stream`
   (already in scope per the comment at speculative.rs:2425).
2. Defer `stream_synchronize` until the host actually consumes
   `host_idx`. In the current shape that's the line immediately after
   the copy block (`for &idx in &host_idx`), so the overlap window is
   small *unless* the CPU bookkeeping (repeat-penalty, n-gram block,
   accept comparison) is reordered between the async D2H launch and
   the sync.
3. Reorderable CPU work that can hide the copy:
   - Build the `prev_committed` slice for repeat-penalty / n-gram block
     (already on hand from caller; doesn't depend on argmax).
   - Compute draft-side `cactus_delta` constants if applicable.
   - Pre-build the next launch chain's kernarg blob via `launch_maybe_blob`
     if graph-capture is active.

**Realistic lift:** 1–2% in isolation, 2–3% if combined with reordering.
Lever B's value depends on what work is reorderable; without
reordering it caps at the cost of one HIP stream sync (~10–50 µs/cycle).

#### Constraints

- Both sites (verify and draft) are inside the `if !host_path_active`
  branch of greedy decode. Temperature-sampled paths (`use_temp_sampling`)
  go through a different branch with full-logits D2H — out of scope for
  this lever.
- `argmax_buf` is reused across cycles. The async copy must complete
  before the next cycle's `argmax_f32_batched` writes to the same buffer.
  Trivially satisfied by the same-stream ordering, but explicit
  invariant to assert.
- The verify-side B and draft-side (B-1) hosts are different lengths;
  do not share a single `host_idx` allocation.

### Lever C — GPU-side accept/reject decision (arch-agnostic)

The current shape: after lever B's D2H, the host runs:

```rust
// (sketch, not direct quote — runs at speculative.rs:~2160-2200)
for i in 0..b {
    if argmax_per_pos[i] != drafted[i] { /* reject from i */ break; }
    /* accept */
}
```

This is `b` integer compares + a scan; trivially GPU-friendly.

#### Plan

1. Add a kernel `dflash_commit_count` taking
   `(argmax_per_pos: *const i32, drafted: *const u32, B: i32, out_count: *mut i32)`.
   Single workgroup, B threads, reduction (or warp ballot at B≤32).
   Writes the longest accepted prefix length.
2. Replace the B×4 D2H from lever B with a 4-byte D2H of `commit_count`.
3. Repeat-penalty / n-gram-block decisions still need the actual
   accepted token IDs, but those can come from `drafted[..commit_count]`
   on the host (already in scope) — no need to D2H argmax for the
   accept-or-reject decision itself.

#### Constraints

- Repeat-penalty / n-gram-block apply *before* the argmax in the draft
  path (they edit logits in-place). On the draft side, lever C's kernel
  needs to land in a place that doesn't reorder these. Concretely: only
  the **verify-side** accept/reject is straightforwardly GPU-foldable;
  the draft-side path does host-side logit shaping that depends on
  full-row download.
- Diff is larger than levers A/B because it adds a new kernel + dispatch
  binding + two call-site refactors.

#### Estimated lift

#172 estimated 1–3%. With a 4-byte D2H replacing a B×4 D2H, the
bandwidth saving is negligible; the win comes from getting rid of the
synchronous stall pattern entirely on the verify side, *if* lever B
hasn't already eliminated it. So **lever C is partially redundant with
lever B** — picking it up after B is done provides only the residual
async-stream-sync cost.

**Recommendation:** treat lever C as **stretch / prep work** for any
future graph-capture of `spec_step_dflash`, not as a primary
performance lever. If lever B hits its 2–3% target, lever C's
incremental value is low (~1%); if lever B is blocked by reordering
constraints, lever C becomes more attractive.

---

## 3. Phase plan

**Execution order:** gfx906 (MI50) first, gfx1100 (7900 XTX) on
maintainer-triggered handoff. Each phase below tags
**🔬 gfx906** (executable on MI50) or **🔁 gfx1100 handoff** (requires
the maintainer to run on the 7900 XTX box).

### Phase 0 — Diagnostic re-baseline

Before touching code, confirm the workload still looks like #172's
trace did. The MMQ redesign + env::var removal both landed *after* the
trace, so the steady-state percentages may have shifted.

1. **🔬 gfx906 (MI50) baseline.** *Start here.*
   - Same prompt as #172 (`dflash_spec_demo` Qwen 3.5 27B mq4 +
     drafter, humaneval-0, `--max 16 --no-chatml`, post-JIT).
   - Capture `rocprofv3 --kernel-trace` filtered to last 1.5 s of
     decode steady-state (drop the load-and-prefill warmup).
   - Confirm gfx906 MMQ residual gaps have collapsed post-PR-#158 —
     the trace should **no longer** show `fillBufferAligned` between
     `gemm_hfq4g256_residual_mmq_gfx906_full_*` kernels.
   - Compute: GPU-busy %, median/p90/p99 inter-kernel gap, top 5
     longest gaps with their kernel-pair context. Compare to #172's
     pre-#158 trace (line `Decode dispatch overhead = ~17%`).
   - The remaining steady-state overhead is then attributable to
     lever B (and, if measurable, lever C). Record the new percentage.
   - Run `benchmarks/scripts/bench_dflash_27b_gfx906.sh` (commit
     `dd6e410`) for the end-to-end tok/s baseline. Record prompt md5
     + binary md5 + model md5 per CLAUDE.md.
2. **🔁 gfx1100 handoff — Phase 0 baseline.**
   - When gfx906 Phase 0 is complete and code work is about to begin,
     ping the maintainer to run the same trace on 7900 XTX with the
     canonical bench config: Qwen 3.6 27B mq3 LRU, `--no-chatml`,
     `--kv-mode asym3`, max=120, PEP-8 strict prompt,
     `prompt_normalize=true`. Expected baseline: **199 tok/s τ=10.36 ±2 %**.
   - Ask the maintainer to capture the same `rocprofv3 --kernel-trace`
     summary and confirm whether dispatch.rs:8184/8255/8294 actually
     show as top-5 gaps. If yes, Phase 2 is justified; if no, Phase 2
     scope-redirects.
3. **Commit the baseline trace summaries** to
   `docs/perf-checkpoints/2026-05-XX-dflash-decode-rebaseline-gfx906.md`
   first; gfx1100 file lands when the handoff returns.

**Exit criteria:** numeric "before" baseline for gfx906, gfx1100
handoff scheduled. **No code changes in Phase 0.**

### Phase 1 — Lever B (async argmax D2H), both sites

Smallest blast radius, arch-agnostic. Validates the active_stream
infrastructure is working as advertised on both arches.

1. Replace `gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?` at
   `speculative.rs:2158` and `:2708` with `memcpy_dtoh_async` on
   `gpu.active_stream` (asserted Some by the existing init at
   `:2425+`).
2. Audit whether the immediately-following host loop can be deferred
   past any of the host-side work currently between it and the next
   GPU launch. If yes, reorder; if no, just call `stream_synchronize`
   immediately before the loop (no overlap, but matches current
   semantics).
3. **🔬 gfx906 validation** — run locally:
   - `scripts/coherence-gate-dflash.sh` on Qwen 3.5 27B mq4 + drafter.
   - `benchmarks/scripts/bench_dflash_27b_gfx906.sh` for end-to-end
     decode tok/s. Compare to Phase 0 gfx906 baseline.
4. **🔁 gfx1100 handoff** — at this point, ping the maintainer:
   - Run `scripts/coherence-gate-dflash.sh` on the canonical 27B mq3 LRU
     config.
   - Run the gfx1100 bench (canonical config from Phase 0) and report
     decode tok/s + τ vs the 199 tok/s τ=10.36 baseline.
   - Drift budget ±5 %; real win shows up as ≥+2 % decode tok/s.

**Exit criteria:** ≥+1 % decode tok/s on gfx906; coherence gate green;
gfx1100 handoff returned with non-regressing numbers. Sub-1 % delta on
gfx906 is acceptable as "no regression"; plan moves to Phase 2 anyway
because levers A and B+C are independent.

### Phase 2 — Lever A (memset hoist), gfx11/gfx12 only

This is the lever-1 retarget. Pure gfx906 work would be a no-op since
PR #158 already removed the per-cycle MMQ residual pre-zeros.

**Note:** Phase 2 is **gfx11/gfx12-only** by construction. On gfx906
this phase is a regression check, not a perf win — the code change
won't traverse on the MI50. Code can be written and unit-tested on
gfx906; the perf gate is gfx1100-only.

1. **Choose A1 vs A2 implementation shape.** Recommendation: A1
   (caller-managed zero) for the first PR. Document A2 as a follow-up
   if the WMMA dispatch surface grows.
2. **Phase 2a — `gemm_hfq4g256_batched_lmhead` (dispatch.rs:8255).**
   - Add a one-shot zero of `verify_scratch.logits_batch` and
     `draft_scratch.logits_batch` in their respective `*Scratch::new`
     constructors (locate via `git grep -n "logits_batch:" crates/hipfire-arch-qwen35/`).
   - Move the per-call memset out of the dispatch site, leaving the
     `fp16_x_source_ptr = null` stomp in place (it's an unrelated
     correctness fix per the comment at 8233+).
   - Add a debug_assert documenting the precondition: "Y must be zero
     on first call per allocation lifetime."
3. **Phase 2b — `gemm_hfq3g256_batched_lmhead` (dispatch.rs:8294).**
   - Mirror change for the MQ3 draft path. Same scratch buffers, same
     precondition.
4. **Phase 2c — `gemm_mw16_residual_wmma_via_dequant` (dispatch.rs:8184).**
   - Lower-priority — only used when `HIPFIRE_MW16=1` is opted in
     (see dispatch.rs:7631). This path isn't on the default DFlash
     decode hot path. Address only if Phase 0's gfx1100 trace shows it
     as a top-5 gap; otherwise skip and file as a follow-up.
5. **Validation:**
   - **🔬 gfx906 (regression check only):** run
     `bench_dflash_27b_gfx906.sh` and `coherence-gate-dflash.sh`
     locally; expect ±2 % from Phase 1 gfx906 numbers (no win, no loss
     — Phase 2's code path doesn't traverse on MI50).
   - **🔁 gfx1100 handoff (perf gate):** ping the maintainer to run
     the canonical 27B mq3 LRU bench. ≥+1 % decode tok/s required to
     declare Phase 2 a win.
   - Additionally: dump `verify_scratch.logits_batch` contents at
     end-of-cycle for 3 cycles, confirm bytes are byte-identical to
     pre-change (the `y += acc` invariant means any over-accumulation
     would show as garbage in the next cycle's argmax). This dump is
     arch-agnostic and runnable on gfx906.

**Exit criteria:** ≥+1 % decode tok/s on gfx1100 (handoff numbers);
coherence gate green on both arches; gfx906 within ±2 % drift.

### Phase 3 — Lever C (GPU accept/reject) — STRETCH

Only execute if Phases 1+2 underdeliver vs the combined estimate
(2–4% gfx1100, 1–3% gfx906). If Phase 1 alone hits 2 %+ on gfx906,
lever C is not worth its diff size.

If executed:

1. New kernel `dflash_commit_count` in
   `crates/rdna-compute/src/kernels/` (location: alongside
   `argmax_batched.hip`).
2. Wire into `crates/rdna-compute/src/dispatch.rs` via
   `pub fn dflash_commit_count(...)`.
3. Refactor verify-side accept loop at `speculative.rs:~2160-2200` to
   call the kernel and consume a 4-byte D2H of `commit_count`.
4. Leave draft-side host-logit-shaping path untouched (constraint
   from §2 lever C).
5. **Validation:**
   - **🔬 gfx906** — run locally as in Phase 1.
   - **🔁 gfx1100 handoff** — ping maintainer for canonical bench.

**Exit criteria:** +1 % marginal lift over Phase 1+2 combined on at
least one arch, OR clear evidence the diff sets up a future
graph-capture win.

---

## 4. Risk register

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Phase 2 invariant violation (caller skips zero, kernel over-accumulates) | High | Low | `debug_assert!` on first call per allocation lifetime; coherence gate catches structural attractors. |
| Phase 1 reordering breaks repeat-penalty / n-gram-block ordering | Medium | Low | Lever B's reorder window is small; if reorder is non-trivial, fall back to immediate stream_synchronize (no overlap, no semantic change). |
| Phase 0 trace shows the top-5 gaps have moved entirely (e.g. KV-cache slot management dominates post-#158) | Medium | Medium | Phase 0 is explicitly diagnostic-first; if levers A/B/C don't match the trace, scope-redirect before writing code. |
| gfx906 hardware unavailable for validation | Low | Medium | Levers B and C are arch-agnostic by code path; gfx1100 validation covers the correctness surface. gfx906 perf gate becomes "rerun when hardware available." |
| Phase 2 conflicts with future work that adds new WMMA dispatch sites | Low | Low | A2 (kernel `_replace` variant) is the long-term cleaner shape; track as follow-up issue if dispatch surface grows. |
| Within-session perf noise hides real wins/losses | Medium | High | Use `scripts/probe_commits.sh master HEAD` per CLAUDE.md perf-methodology rule, not in-session A/B. Multi-run with stddev. |

---

## 5. Out of scope

- **Full graph-capture of `spec_step_dflash`.** #172's own §"What's NOT
  a lever" rules this out: at 83 % steady-state busy, even perfect
  graph capture caps at 17 % lift, and the blast radius (KV pointer
  manipulation, dynamic B, `pld_spine` / `cactus_delta` / `repeat_penalty`
  branches) is large for modest payoff.
- **AR decode (batch=1).** Already covered by `HIPFIRE_GRAPH=1` graph
  capture; not the workload #172 is about.
- **Prefill.** Single-launch-per-layer-stack, not dispatch-overhead
  bound.
- **Full-logits non-greedy (temperature sampling) path** at
  `speculative.rs:~2141, :2174, :2719`. Different sync characteristics
  (B × vocab × 4 = ~15 MB D2H per cycle), separate analysis if/when
  temp sampling becomes the production hot path.
- **Per-DN-state memsets on draft rollback** (`speculative.rs:294-311`,
  ~48 memsets/cycle on 27B). Already async-stream-gated; only fires on
  rollback events, not steady-state. File as a separate issue if
  rollback frequency turns out to dominate post-this-plan.
- **lm_head pre-zero memsets at dispatch.rs:4475/4545** referenced in
  the comment at `speculative.rs:2425+`. These are *also* gated on
  `active_stream` being set; the gate is already in place.
- **Lever 1 on gfx906.** Structurally fixed by PR #158's
  `_full_add_*` kernel redesign; no code change needed.

---

## 6. Validation matrix

Per the user request: separate gates per arch.

| Phase | gfx1100 (7900 XTX) | gfx906 (MI50) | Notes |
|---|---|---|---|
| 0 | rocprofv3 baseline + bench numbers | rocprofv3 baseline + bench numbers | Required before any code change. |
| 1 | bench gate ≥+1 % decode tok/s OR ±2 % no-regression | same | Lever B is arch-agnostic. |
| 2 | bench gate ≥+1 % decode tok/s | bench gate ±2 % (regression check only) | Lever A is gfx11/gfx12-only by construction. |
| 3 (stretch) | bench gate ≥+1 % marginal | bench gate ≥+1 % marginal | Only if Phase 1+2 underdelivers. |

**Canonical bench configs (per CLAUDE.md):**

- **gfx1100:** Qwen 3.6 27B mq3 LRU, `--no-chatml`, `--kv-mode asym3`,
  max=120, PEP-8 strict prompt with `prompt_normalize=true`. Expected
  baseline: **199 tok/s τ=10.36 ±2 %**. Drift >5 % from this is a
  regression — `git bisect` against this number, not session-recalled
  peaks.
- **gfx906:** Qwen 3.5 27B mq4, `dflash_spec_demo` via
  `benchmarks/scripts/bench_dflash_27b_gfx906.sh` (commit `dd6e410`).
  Pre-#158 baseline: ~17 % steady-state overhead. Post-#158 baseline:
  TBD in Phase 0 — likely lower since MMQ residual was a major source.

**Coherence gates (mandatory):**

- `scripts/coherence-gate.sh` (general) — kernels, dispatch, fusion,
  rotation, rmsnorm, forward pass changes.
- `scripts/coherence-gate-dflash.sh` (spec-decode token-attractor
  guard) — DFlash, DDTree, slow-path-kill changes. **All three levers
  in this plan trigger this gate.**

**Three-tier dflash thresholds (per CLAUDE.md):**

- Tier 1 (first 128 tokens): unique_token_ratio ≥ 0.15, max_single_token_freq ≤ 0.50.
- Tier 2 (last 128 tokens): unique_token_ratio ≥ 0.30, max_single_token_freq ≤ 0.50.
- Tier 3 (full output): 3-gram density ≤ 50 % in final half; full-output
  unique-token-ratio ≥ 0.10.

Tight stddev on a spec-decode bench is **suspicious, not reassuring**
(per the 2026-04-26 CASK m-fold incident). Real acceptance noise is
wider; if Phase 1 shows σ < 1 % on tok/s across runs, that is grounds
for a correctness re-check, not confidence.

---

## 7. Estimated total lift

Bounded reasoning (not a promise):

- Phase 1 (lever B): 1–3 % decode tok/s, both arches.
- Phase 2 (lever A, gfx11 only): 3–5 % on gfx1100, 0 % on gfx906.
- Phase 3 (lever C, stretch): 0–1 % marginal.

**Combined upper bound:** ~7 % on gfx1100, ~3 % on gfx906.

#172's original estimate was 17 % steady-state overhead → bounded
upper bound. With PR #158 having already collapsed the gfx906 MMQ
residual gaps, the gfx906 share of the original 17 % is mostly
realized; the remaining gfx906 budget is ≤5 %.

The plan is worth executing if the realistic floor (3 % gfx1100,
2 % gfx906) holds. If Phase 0's re-baseline shows post-#158 gfx906
overhead has already collapsed below 5 %, **drop the gfx906 portion
entirely** and ship gfx11-only.

---

## 8. References

- Issue #172 (`gfx906 DFlash decode: ~17% dispatch overhead in steady-state — three-lever proposal`)
- PR #158 (`feat(gfx906): MMQ kernel redesign + AR decode optimizations`) — commit `afb84bd`
- Commit `9aabdcf` (`fix(gfx906): remove per-call diag overhead from MMQ residual hot path`)
- `crates/rdna-compute/src/dispatch.rs` — dispatch sites for lever A
- `crates/hipfire-arch-qwen35/src/speculative.rs` — sites for levers B and C
- `benchmarks/scripts/bench_dflash_27b_gfx906.sh` — gfx906 perf gate
- CLAUDE.md §"Perf benchmarking" — bench methodology
- CLAUDE.md §"Coherence Gate" + §"DFlash Coherence Gate" — gates
- CLAUDE.md §"Prompt-structure τ sensitivity" — prompt md5 requirement

---

## 9. Open questions

1. **Phase 0 first or commit to plan first?** This document commits to
   the plan. A maintainer who wants to defer until Phase 0 numbers are
   in can treat sections 3–7 as conditional on Phase 0 confirming the
   trace shape.
2. **Is gfx12 (RDNA4) in scope for lever A?** dispatch.rs:8294 fires on
   gfx12 too (the `has_wmma_f16_gfx12(&self.arch)` branch). Plan
   currently treats gfx12 as same-as-gfx11. If gfx12 hardware is in the
   validation pool (gfx1201), include it in Phase 2 gates.
3. **Owner for the work.** Not set in this draft.
