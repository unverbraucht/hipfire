# Issue #172 — gfx1151 reproduction baseline

Date: 2026-05-06
Hardware: AMD Ryzen AI Max+ 395 — Strix Halo APU, gfx1151 iGPU (Radeon 8060S),
          shared LPDDR5X system memory, Zen 5 CPU.
Workload: DFlash 27B Qwen3.5 + matched draft, humaneval-0 prompt, `--max 80`,
          `--ctx 2048`, `--no-chatml`, mean_B=16, accept_rate ~0.68.

## Headline

**The 17 % steady-state dispatch overhead reported in issue #172 does not
reproduce on gfx1151.** Excluding a one-time graph-capture event, gfx1151
runs at **96.3 % GPU busy** in steady-state with median inter-kernel gap
of 2.7 µs and p99 12 µs.

| Metric | gfx906 (issue #172) | gfx1151 (this run, no verify_graph) |
|---|---:|---:|
| Steady-state GPU busy | 83.4 % | **96.3 %** |
| Median inter-kernel gap | 10.2 µs | **2.7 µs** |
| p99 gap | 1450 µs | **12 µs** |
| max gap (excl. graph capture) | 12000 µs | 5760 µs (cycle boundary) |
| Dispatch overhead | ~17 % | **<4 %** |

Probable root cause: gfx906 was hosted on an older / slower CPU; the iGPU
on Strix Halo is tightly coupled to a Zen 5 CPU and shares the memory
controller, so kernel-launch path latency is dominated by CPU clock and
memory hierarchy, not PCIe round-trip + driver IRQ semantics.

## Trace details

### With verify_graph (default)

Run: `dflash_spec_demo --max 80`, `HIPFIRE_VERIFY_GRAPH` unset.

```
steady-state: 8655 kernels, wall 1177 ms, busy 573 ms (48.7%)
gaps: median 2.7us, p90 7.4us, p99 12.1us, max 563252us
```

The `max 563 ms` outlier sits between `gemm_hfq4g256_residual_wmma_k2`
(verify lm-head) and `argmax_f32_batched`. It happens once at cycle 2,
which matches the demo's `[verify-graph] captured for B=16 with 1188
blobs` log. **It's the graph-capture serialization stall, not a
recurring dispatch overhead.** Excluding it, busy% rises to 93.4 %.

### Without verify_graph (HIPFIRE_VERIFY_GRAPH=0)

Same workload, graph capture disabled.

```
steady-state: 10518 kernels, wall 1494 ms, busy 1438 ms (96.3%)
gaps: median 2.7us, p90 7.4us, p99 12.0us, max 5760us
```

Top remaining gaps:
- 5.8 ms — copyBuffer → embedding_q8 (cycle boundary, host bookkeeping)
- 4.2 ms — copyBuffer → copyBuffer
- 0.56 ms — copyBuffer → mq_rotate_x
- 0.47 ms — copyBuffer → mq_rotate_x
- 0.31 ms — copyBuffer → mq_rotate_x

The 5.8 ms gap matches issue #172's "6-12 ms once per spec-cycle"
boundary gap (sync D2H of argmax + accept/reject + kernarg rebuild +
embedding lookup setup). At ~1.3 cycle boundaries per 1.5 s window,
total cycle-boundary cost is ~6 ms of 1494 ms = **0.4 % of decode**.
Not 17 %.

## Levers from issue #172

| Lever | Issue's claim (gfx906) | Estimated on gfx1151 |
|---|---|---|
| 1. Hoist per-cycle residual-GEMM output memsets | 3-5 % | <0.5 % (median gap is already 2.7 µs; the lm-head memset isn't blocking the chain) |
| 2. Async D2H of argmax_buf + deferred sync | 1-2 % | ~0 % (B*4 byte D2H is irrelevant in 96 % busy regime) |
| 3. GPU-side accept/reject | 1-3 % | <0.5 % (collapses the 5.8 ms cycle-boundary gap somewhat, but it's a one-cycle-cost item) |

Combined upside on gfx1151: **<1 %** vs **5-10 % on gfx906**.

Lever 1 is also harder than the issue suggests: the residual-WMMA kernel
does `y += acc`, so the memset can't simply be hoisted to scratch
allocation — y would carry stale values across cycles. A correct fix
needs a non-residual `y = acc` WMMA variant or a deferred memset
overlapped with CPU bookkeeping. Not worth the work for <0.5 % on this
hardware.

## Reproduction

```sh
# Build
cargo build --release -p hipfire-runtime --example dflash_spec_demo --features deltanet

# Trace
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
HIPFIRE_VERIFY_GRAPH=0 \
rocprofv3 --kernel-trace -d ./trace -o decode --output-format csv -- \
    target/release/examples/dflash_spec_demo \
        --target $HOME/.hipfire/models/qwen3.5-27b.mq4 \
        --draft  $HOME/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
        --prompt "$(cat benchmarks/prompts/humaneval_0_has_close_elements.txt)" \
        --max 80 --ctx 2048 --no-chatml
```

Reproducibility metadata:
- commit: (fix/issue-172-dflash-dispatch-overhead, base 262e5f6)
- binary md5: 7ca34fe13f13d55242516559ff81e8b1
- prompt md5: 5333a1f70d884060807676347c0edb93 (humaneval_0)

## Recommendation

Close issue #172 as **not actionable on gfx1151**. The pattern is
gfx906-specific. Reasons documented above; the root cause likely lives
in the host-CPU + dedicated-GPU dispatch path and does not propagate
to APU-style integrated topologies.

If gfx906 nodes (MI50-class boxes) remain in the supported matrix and
issue #172 is still considered worth pursuing there, the fix needs to
be arch-gated to gfx906 and validated on gfx906 hardware, not blanket
applied. The CPU-speed component of the root cause also means a
gfx906 node with a faster CPU might see less of the overhead than the
diagnostic node showed.
