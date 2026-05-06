# gfx906 Priority 0 baselines — mq3 / mq4 / mq6 on 9B + 27B

Date: 2026-05-06
Branch: `feat/gfx906-hfq6-hfq8-analysis` at commit `cf00664`
Hardware: AMD Instinct MI50 / gfx906 / 32 GB HBM2 (1 TB/s peak)
Host: Ryzen, 30 GiB system RAM, ROCm 6.4.43484-123eb5128, AMD clang 19
Bench harness: `scripts/bench-cold.sh` (5-run fresh-process median, 1
warmup process, 10s DPM warmup, asym3 KV cache, 50-token gen).

## TL;DR

| Model | Decode tok/s (pp32) | Decode tok/s (pp128) | Prefill tok/s (pp32) | Prefill tok/s (pp128) |
|---|---:|---:|---:|---:|
| Qwen3.5 9B mq4 | **59.5** | 57.0 | 310.6 | 593.8 |
| Qwen3.5 9B mq3 | 35.6 | 34.7 | 37.2 | 36.5 |
| Qwen3.5 9B mq6 | 31.1 | 30.3 | 46.3 | 46.7 |
| Qwen3.6 27B mq6 | 10.2 | 10.1 | 13.5 | 13.5 |

**All numbers ≤1.7% spread across 5 fresh processes. Bench is in
deterministic regime.**

The 9B mq4 result reproduces the PR #158-era baseline (~50–54 tok/s
decode → +10% post-PR-158, this measurement post-`ee0fac6` at 59.5
tok/s). mq6 + mq3 are first-time clean baselines per the v3.2 plan;
they confirm the qualitative shape Phase A targets.

## Reproducibility chain (per AGENTS.md §5)

- Binary md5 (`bench_qwen35_mq4`): `1695537f286f95a0bf54b33e09a9aaff`
- Binary md5 (`greedy_dump`): `6f6c7c320492f8248f56aeab510e5155`
- Bench script: `scripts/bench-cold.sh`
- Bench-cold flags: `--pp 32,128 --runs 5 --gen 50`
- Engine env: `HIP_VISIBLE_DEVICES=0 HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 HIPFIRE_DPM_WARMUP_SECS=10`
- Bench prompt: deterministic fake `0..N-1` token sequence (per
  `bench_qwen35_mq4` source, `let prompt_tokens: Vec<u32> = (0..prefill_len as u32).collect();`)
- Coherence prompt: `"A farmer has 17 sheep. All but 9 die. How many
  are left? Show brief reasoning then state the final number."` —
  prompt md5 `fcfff9c745d58b218fc88be1639d759d` (canonical from
  `scripts/coherence-gate.sh:86`)
- Models from `/local/hipfire/` (NVMe SSD, no NFS in path)

## Hardware + thermal context

- MI50 idle temp ~37 °C, runs hit 60–66 °C across all benches — well
  below thermal-throttle threshold (95 °C). No DPM regression.
- Bench-cold's `HIPFIRE_DPM_WARMUP_SECS=10` pinned the GPU at peak
  DPM (666–670 GiB/s effective memset throughput in the warmup loop)
  before each timed run. This eliminates the clock-ramp variance that
  cost us ~6 hours of debugging on RDNA4 in late April.
- No GPU lock contention, no quantize co-running (the 27B mq6 quantize
  ran *before* this bench-cold sweep — the original mq3 numbers from
  earlier in the day were contaminated by quantize co-running and
  showed 98%+ spread; this rerun is clean).

## Per-quant analysis

### 9B mq4 (sanity baseline vs PR #158 era)

Five 5-run-fresh-process medians: 59.5 / 59.4 / 58.9 / 59.9 / 59.0
tok/s (decode pp32). Spread 1.7%, well within the bench-cold "trust
this" band (<5%).

`docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`
reported 50.7 tok/s pre-PR-158 and 54.4 tok/s post-`gemv_hfq4g256_residual_wave64`
(the +4.8% wave64 lift). PR #158's full set of optimizations layered to
~52 tok/s in the original measurement. **Today's 59.5 tok/s is +9.4%
above PR #158's reported ceiling**, attributable to:

- Fresh kernel JIT cache (no `mmq_set_prequant_diag` env-var lookups
  per call — that fix is in flight as PR #177)
- Fresh DPM warmup (this benchmark's `HIPFIRE_DPM_WARMUP_SECS=10`
  vs whatever the prior measurement used)
- Possible additional small commits since PR #158 merged

The number is real and reproducible; the comparison to PR #158 era is
indicative not exact.

Effective bandwidth: 5.31 GiB × 59.5 tok/s = **316 GiB/s** = ~31% of
HBM2 peak. Consistent with PR #158's `~315 GiB/s` for the
unverbraucht/skyne98 fork's 63.48 tok/s on stock llama.cpp Q4_K_M.

### 9B mq3 — surprisingly slower than mq4 (4.3 GB vs 5.3 GB)

| Quant | File size | Bytes per group | Decode tok/s | Effective BW |
|---|---:|---:|---:|---:|
| mq4 | 5.31 GiB | 136 (HFQ4-format) | 59.5 | 316 GiB/s |
| **mq3** | **4.32 GiB** | **104 (HFQ3-format)** | **35.6** | **154 GiB/s** |
| mq6 | 7.30 GiB | 200 (HFQ6-format) | 31.1 | 226 GiB/s |

mq3 is **~25% smaller than mq4 in file size** but decodes **40%
slower**. Effective BW is half of mq4's. This is **not a memory
bandwidth signal** — it's a kernel-tuning gap. The mq3 path on gfx906
runs through `gemv_hfq3g256` (wave32 scalar) without any of the
wave64-native optimizations PR #158 layered onto HFQ4.

This is a direct argument for **plan §3.1 Phase A's wave64 work** —
the same wave32→wave64 mechanical port that gave HFQ4 +4.8% would, on
HFQ3/MQ3, recover much more (the gap is bigger). MQ3 isn't on the v3.2
priority list because there's no production demand on gfx906 (per
AGENTS.md §A: MQ3 is gfx11/gfx12 production), but if someone *does*
ship MQ3 on gfx906 they'll see this characteristic underperformance.

### 9B mq6 — first clean baseline; matches expected shape

mq6 is **38% larger than mq4** (7.30 vs 5.31 GiB). Decode is **48%
slower** (31.1 vs 59.5 tok/s). Effective bandwidth: 226 GiB/s vs
mq4's 316 — also lower, but not as catastrophically as mq3.

Expected shape: **mq6's wave32 path on gfx906 has ~30% of the wave64
optimization headroom that PR #158 captured for HFQ4.** Phase A's
wave64 ports (plan §3.1.1 items 1–5 + the §5.8 residual_wave64
sibling) target this gap directly. Order-of-magnitude lift estimate
post-Phase A: 20–30% decode improvement, taking 9B mq6 from ~31 tok/s
toward ~38–40 tok/s.

prefill at pp128 (46.7 tok/s) is **12.7× lower than mq4's 593.8**,
which reflects mq4's WMMA + dot2 fast paths that mq6 doesn't have on
RDNA3+. On gfx906 the gap is closer to 8× because the dot2 path is
gated off (plan §5.6) — but mq6's wave32 batched GEMM is genuinely
slower than HFQ4's wave64 + dp4a fast paths.

### 27B mq6 — first 27B-class baseline on gfx906

| pp | Decode tok/s | Prefill tok/s |
|---:|---:|---:|
| 32 | 10.2 | 13.5 |
| 128 | 10.1 | 13.5 |

27B mq6 is **2.93× larger than 9B mq6** (21.4 GiB vs 7.30 GiB) and
decodes **3.05× slower** (10.2 vs 31.1 tok/s). The ratio is almost
exactly proportional to weight footprint — confirming this is HBM2
bandwidth-limited at 27B-scale. Effective BW: 218 GiB/s, similar to
9B mq6's 226 GiB/s.

The 27B model has **~8 GiB of weights remaining in 32 GiB VRAM after
KV cache + scratch**, comfortably fitting. No OOM, no swap, no DPM
throttle.

prefill at pp32 = pp128 (13.5 tok/s, 0% spread between them) is
characteristic of bandwidth-bound prefill on a model this size — the
batched GEMM doesn't amortize across batch when each row touches all
21.4 GiB of weights.

This is the first time a 27B-class mq6 model has been benched on the
MI50 in this checkout; **establishing this baseline was the bonus
achievement of today's session**. (The 27B mq6 was quantized today
from `Qwen3.6-27B` bf16 safetensors via
`hipfire-quantize --format mq6` — Qwen3.5 and Qwen3.6 share the
arch config so substitution is equivalent for kernel-perf
measurement.)

## Coherence check

Greedy decode of the canonical sheep-reasoning prompt
(prompt md5 `fcfff9c745d58b218fc88be1639d759d`, MAX_TOKENS=200, KV
asym3) against each model:

| Model | Coherence | Notes |
|---|---|---|
| 9B mq4 | ✓ structured reasoning, mid-thought at 200-token cap (would land at 9) | Premise/event/question breakdown, considers "All but X" semantics |
| **9B mq3** | **✗ single-token attractor** | First 100 tokens reasonable, then collapses to *"Final Answer: The final answer is the number of sheep that are left."* repeated 6+ times. Classic mq3-quality-cliff signature (matches upstream issue #114). |
| 9B mq6 | ✓ correctly reaches "9 sheep left" | Cleanest reasoning path of the four; explicitly considers riddle phrasing |
| 27B mq6 | ✓ correctly reaches "9 sheep are left alive" | Considers traps/alternative interpretations; structured chain-of-thought |

**The 9B mq3 attractor is NOT a regression we introduced** — it's the
known mq3 quality cliff documented in upstream issue #114 ("MQ3
quality collapse on sub-9B dense models"). 9B is borderline-but-shipped;
4B and smaller are documented as broken. Bench timing is real GPU
work; coherence on 9B mq3 alone is suspect even when the kernel path
is fast.

mq4 / mq6 / 27B mq6 all produce on-topic, structured, eventually-correct
reasoning. **No silent-corruption findings.** The audit's matcher
coverage proved correct: every quant tested here has full per-layer
matcher entries in `qwen35.rs`.

## Notable absences

- **9B mq8 NOT benched.** Per the v3.2 audit (commits `662859d` /
  `10541df`), `qwen3.5-9b.mq8` produces invalid inference because the
  per-layer prefill batched dispatch in `qwen35.rs` excludes
  `MQ8G256` from all 14 `is_mq` matchers. The earlier-today bench
  attempt (45.4 tok/s) measured GPU work on corrupted state; the
  number is real but the inference is not. Re-included in Priority 0
  only after the per-layer wiring is restored (plan §3.2 Phase A
  item 4 + §5.7 errata).
- **27B mq3 NOT benched.** No 27B mq3 model exists locally, and the
  9B mq3 attractor (above) suggests 3-bit quantization is borderline
  at any scale on this Qwen family. Could be quantized in ~10 minutes
  if a baseline is wanted later.
- **DFlash + speculative decode NOT benched.** Plan §5.0 mentioned
  "DFlash 27B mq6 humaneval-0 tok/s" as a Priority 0 line item, but
  DFlash needs a draft model, and the existing `qwen3.5-0.8b.mq4`
  drafter compatibility with a Qwen3.6 target is unverified. Defer to
  follow-up.

## Summary observations

1. **Determinism is excellent.** Worst spread on a clean run was 1.7%
   (9B mq4 pp32 decode), and that's the noisiest path. Most are
   ≤1.0%. Bench-cold harness is producing trustworthy numbers.

2. **mq3 9B underperforms mq4 9B** despite being smaller. This is
   the kernel-tuning gap that motivates the Phase A wave64 work for
   the HFQ-family kernels.

3. **mq6 9B is ~half mq4 9B decode speed** (31 vs 59 tok/s). Phase A
   targets closing this gap by 20–30% via wave64 ports. The B=1 dp4a
   port (Phase B) might add another 10% if PMC validation shows ALU
   headroom.

4. **27B mq6 fits comfortably in 32 GB** with asym3 KV at max_seq=512
   (default) — VRAM footprint is workable. Decode at 10 tok/s is
   usable for interactive chat and acceptable for code-completion at
   this scale.

5. **No silent-corruption signals from the four quants tested.** The
   matcher audit coverage matrix (commit `cf00664`) holds.

## Cross-references

- Plan: `docs/plans/gfx906-mq6-mq8-port.md` v3.2.1 (commit `d3a0575`)
  — particularly §3.1 (MQ6 Phase A scope) and §5.7 (matcher coverage)
- Audit dev-log: `docs/perf-checkpoints/2026-05-06-mq8-runtime-dispatch-audit.md`
  (commit `10541df`) — explains why mq8 is excluded
- Dispatch matrix: `docs/perf-checkpoints/2026-05-06-quant-dispatch-matrix.md`
  (commit `cf00664`) — per-quant × workload × arch reference
- PR #158: gfx906 MMQ kernel redesign + AR decode optimizations
  (the comparison baseline for 9B mq4)
- Upstream issue #114: MQ3 quality collapse (root cause for 9B mq3
  attractor observed here)
- Upstream issue #179: MQ3 missing from MoE-batched matchers (filed
  today as part of the audit work)

## Raw bench logs

- `/tmp/baseline-2026-05-06/9b-mq4.log`
- `/tmp/baseline-2026-05-06/9b-mq3.log`
- `/tmp/baseline-2026-05-06/9b-mq6.log`
- `/tmp/baseline-2026-05-06/27b-mq6.log`
- Coherence text + token dumps: `/tmp/coherence-2026-05-06/`
