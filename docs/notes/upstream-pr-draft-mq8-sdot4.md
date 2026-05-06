# Upstream PR draft: gemv_mq8g256 sudot4 → sdot4 (gfx906 buildability)

Status: **draft, not yet pushed.** Single-commit PR for `ee0fac6`.

## Push commands

```
# from the audit branch, cherry-pick the single fix commit onto a fresh
# branch off upstream/master so the PR diff is one commit, no
# audit/research churn.
git fetch upstream
git checkout -b fix/gfx906-mq8-sdot4 upstream/master
git cherry-pick ee0fac6
git push -u origin fix/gfx906-mq8-sdot4
gh pr create --repo Kaden-Schutt/hipfire \
    --title "fix(kernels): port gemv_mq8g256 from sudot4 to sdot4 — fixes gfx906 build" \
    --body "$(cat docs/notes/upstream-pr-draft-mq8-sdot4.md | sed -n '/^## PR body$/,$p' | tail -n +2)"
```

## PR body

---

## Summary

`kernels/src/gemv_mq8g256.hip` (and its co-located `mq8_rotate_quantize_x` entry point) used `__builtin_amdgcn_sudot4` for the dp4a inner loop. That builtin lowers to `v_dot4_i32_iu8` (mixed signed/unsigned int8), which requires the LLVM `dot8-insts` target feature — **RDNA3+ only**. On gfx906 (Vega 20, MI50/MI60) the kernel fails to compile:

```
error: '__builtin_amdgcn_sudot4' needs target feature dot8-insts
```

This blocks any inference path that JIT-compiles `gemv_mq8g256` on gfx906 — most concretely, mq4-format models with `embed_tokens` quantized as MQ8 (the lm_head tied-embedding path).

## Fix

4 builtin call sites in `kernels/src/gemv_mq8g256.hip`, plus a doc-comment correction:

```diff
- int dot = __builtin_amdgcn_sudot4(true, wp0, true, xp0, 0, false);
- dot     = __builtin_amdgcn_sudot4(true, wp1, true, xp1, dot, false);
+ int dot = __builtin_amdgcn_sdot4(wp0, xp0, 0, false);
+ dot     = __builtin_amdgcn_sdot4(wp1, xp1, dot, false);
```

(Same pattern at the other two call sites; comment line updated from `v_dot4_i32_iu8` to `v_dot4_i32_i8`.)

## Why the substitution is safe

Both operands at this call site are signed int8:
- **Activations** are Q8_1 signed int8 (per the existing `block_q8_1_mmq` layout).
- **Weights** are symmetric MQ8 in `[-127, 127]` (per `quantize_mq8g256` in `crates/hipfire-quantize/src/main.rs`).

So the sudot4 mixed-mode form is gratuitous — sdot4 (which lowers to `v_dot4_i32_i8`, **gfx906+, dot2-insts**) does the same arithmetic.

Per LLVM's [AMDGPUUsage](https://llvm.org/docs/AMDGPUUsage.html) reference, the i1 flags in `sudot4` are signedness selectors (`true` = signed). So `sudot4(true, w, true, x, acc, false)` is mathematically identical to `sdot4(w, x, acc, false)` on hardware that supports both — there's no math difference, only a hardware-feature-availability difference.

`clamp=false` in both paths; both wrap on i32 overflow. Saturation is irrelevant here because a 256-element int8 dp4a sum maxes at ~2M, ten bits below i32 overflow.

## Cross-arch portability

`sdot4` is supported on **gfx906, gfx908, gfx9, gfx10, gfx11, gfx12** per LLVM's per-arch syntax docs ([gfx906](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html)). RDNA3 (gfx1100) and RDNA4 (gfx1201) retain `v_dot4_i32_i8` alongside the newer `v_dot4_i32_iu8` variant.

Build-tested on three target archs after the fix:

| Arch | `hipcc --genco --offload-arch=<arch>` |
|---|---|
| gfx906 | ✓ (was: `error: needs feature dot8-insts`) |
| gfx1100 | ✓ |
| gfx1201 | ✓ |

## Verified on hardware

`qwen3.5-9b.mq8` (produced via `hipfire-quantize --format mq8` from `Qwen/Qwen3.5-9B`) on MI50 / gfx906:

- Kernel compiles cleanly, no errors / no warnings.
- Bench (`bench_qwen35_mq4 ... --prefill 32 --gen 50`) runs to completion.
- Per-token decode: min 21.12 / p50 21.46 / max 21.71 ms (1.4% spread over 50 tokens — consistent with deterministic GPU work).

**Important caveat:** the bench produced GPU work but the inference is invalid because `crates/hipfire-arch-qwen35/src/qwen35.rs` excludes `MQ8G256` from all 14 `is_mq` matchers — MQ8 weights silently fall through to HFQ4-stride read in the prefill batched path. This is a separate issue (reported in audit work; orthogonal to this fix) and only matters if you try to use MQ8 as a per-layer weight format. **This PR doesn't fix or affect that gap; it only makes the kernel buildable so the lm_head tied-embedding path works.**

## What this fixes in production

- mq4-format Qwen 3.5+ models on gfx906: `embed_tokens` is quantized as MQ8 for quality (per `bf0ba43`'s loader wiring + `crates/hipfire-quantize/src/main.rs` defaults). The lm_head tied-embedding GEMV at the end of every forward pass dispatches `gemv_mq8g256_with_rotate` via `weight_gemv` in `crates/hipfire-runtime/src/llama.rs`. Pre-fix, this path fails at JIT.
- Future per-layer MQ8 use: this PR is a prerequisite for any work that tries to use MQ8 weights end-to-end on gfx906 (e.g. mq8 models, mixed-precision MoE with int8 experts).

## Test plan

- [x] `hipcc --genco --offload-arch=gfx906 -O3 -I/opt/rocm/include kernels/src/gemv_mq8g256.hip` (compiles clean post-fix; failed pre-fix)
- [x] `hipcc --genco --offload-arch=gfx1100 ...` (compiles clean — kernel still works on archs that had `dot8-insts`)
- [x] `hipcc --genco --offload-arch=gfx1201 ...` (compiles clean)
- [x] End-to-end JIT-compile via runtime on gfx906 (MI50): `qwen3.5-9b.mq8` loads, prefill runs, decode runs, deterministic per-token timing
- [ ] Maintainer-side: coherence-gate run on a model that exercises the lm_head tied-embedding MQ8 path (any mq4-format Qwen3.5+ model). The path's coverage by the existing `scripts/coherence-gate.sh` matrix is unverified — recommend adding a row that checks lm_head dispatch on gfx906 specifically.

## What's NOT in this PR

- No changes to `crates/hipfire-arch-qwen35/src/qwen35.rs` per-layer dispatch matchers. Those gaps are documented separately and not in scope here.
- No new MQ8 batched kernels. The `gemv_mq8g256.hip` was already present in the tree at gfx1100/1201; this PR just makes it work on gfx906 too.
- No wave64 / dp4a-MMQ / occupancy work. This is a buildability fix, not a perf fix.

## Refs

- LLVM AMDGPUUsage doc on sdot4/sudot4: https://llvm.org/docs/AMDGPUUsage.html
- Earlier upstream commits introducing MQ8: `246501a` (MagnumQuant MQ8 + dp4a, targeting gfx1100), `bf0ba43` (load MQ4/MQ8 tied lm_head embeddings).
- Closes a gfx906-only build break that has been latent since `246501a` (Apr 8) — nobody hit it before because no one had end-to-end-tested mq8 inference on gfx906 specifically until 2026-05-06.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
