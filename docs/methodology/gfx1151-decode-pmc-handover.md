# Handover: profile dots.ocr decode attention with PMC on gfx1151

**Why this exists:** the decode-attention bandwidth-bound hypothesis can't be
confirmed with hardware counters on the gfx1100 (7900 XTX) box — rocprofv3's
memory/cache/instruction counters all read back a flat zero there (details
below). gfx1151 (Strix Halo, RDNA 3.5) on the 96 GB UMA box may have a working
PMC path. If it does, a one-afternoon profiling pass answers the open question.

- **Branch:** `feat/dots-ocr-phase-3-daemon` @ `d8bfe006`
- **Box of record for the question:** maintainer's gfx1151 (see
  `reference_gfx1151_for_large_calibration.md` in memory)
- **Tool on the gfx1100 where this was attempted:** ROCm 7.2.3, rocprofv3 1.1.0

## The question we want answered

Decode is ~70% attention. The analytical model says decode attention runs at
~185 GB/s on the 7900 XTX — roughly ¼ of the card's ~700 GB/s peak — and it is
*not* occupancy-bound (VGPR=32, 480 workgroups for the split-K kernel). That
points at **memory traffic**, not compute or occupancy, as the ceiling.

The recent change (commit `d8bfe006`) adds a **GQA-aware flash decode kernel**
(`attention_flash_gqa`) that loads each K/V head once and reuses it across the
6 query heads that share it (GQA ratio 12:2 = 6:1). The microbench shows it's
~+44% faster than the per-head split-K kernel. **The hypothesis we cannot yet
prove on gfx1100:** the speedup comes from a ~6× reduction in KV fetch traffic.

Three numbers would confirm or kill it:

| Counter | Question it answers |
|---|---|
| `FETCH_SIZE` (bytes from HBM) per kernel | Does `attention_flash_gqa_partial` actually fetch ~6× fewer bytes than `attention_flash_partial`? |
| `GL2C_HIT` / `GL2C_MISS` ratio | Is the KV cache streaming from HBM (low hit%) or partly resident in L2 (high hit%)? Tells us whether the GQA reuse is winning in L2 or in HBM. |
| `SQ_INSTS_VALU` (or VALU busy) | Sanity floor: confirm the kernel is not secretly compute-bound. |

## Exact repro

The microbench is **pre-existing** —
`crates/rdna-compute/examples/bench_decode_attention.rs`. It runs the real
decode-attention shape (single-token query against a long F32 KV cache,
`n_heads=12, n_kv_heads=2, head_dim=128, max_seq=12000`) and dispatches each
attention variant so they can be profiled side by side. No GPU-lock wrapper
needed — it's a plain binary, not a self-locking hipfire tool.

```bash
cargo build --release --example bench_decode_attention
# plain timing run (no profiler):
./target/release/examples/bench_decode_attention --seq 5100 --iters 100
```

Kernels of interest in the dispatch stream (names are stable):

| Kernel | grid | wg | VGPR | role |
|---|---|---|---|---|
| `attention_flash_partial` | 61440 | 128 | 32 | split-K decode, **per query head** (480 wg) |
| `attention_flash_reduce`  | 1536  | 128 | 16 | split-K combine |
| `attention_flash_gqa_partial` | 10240 | 128 | 16 | **GQA**: per kv-head, reused across group (80 wg = 480/6) |
| `attention_f32`           | 3072  | 256 | 24 | naive baseline (grid = n_heads) |
| `attention_q8_0_kv`       | 3072  | 256 | 32 | Q8 KV-cache variant (rejected lever; profile only for contrast) |

The **80 vs 480 workgroup count** for gqa vs non-gqa is the structural fingerprint
of the reuse, and it *did* read back correctly even on gfx1100 (dispatch
metadata works; only the HW counter values are dead). FETCH_SIZE per kernel is
the missing piece.

## How to run PMC (and the gfx1100 gotchas to skip past)

1. **One counter per invocation.** rocprofv3's `--pmc` does **not** replay —
   `--help` says *"job will fail if entire set of counters cannot be collected
   in a single pass."* On gfx1100 a multi-counter set didn't error cleanly, it
   **hung** (had to `timeout`/`pkill`). So sweep one counter per run and merge
   on `Dispatch_Id` / `Kernel_Name`. The bench is deterministic (fixed LCG
   seeds, identical dispatch sequence every run), so cross-run merge is exact.

2. **Keep `--iters 1`.** PMC serializes dispatches; `--iters 100` is unnecessary
   for counts and makes the run drag. One iteration gives one clean dispatch per
   kernel (plus one warmup dispatch — discard the first of each pair).

3. **Always wrap in `timeout`** so a hang can't wedge the session.

Single-counter sweep (this is the workaround that *did* run on gfx1100, just
returned zeros for the memory counters):

```bash
mkdir -p /tmp/pmc-sweep
for C in FETCH_SIZE WRITE_SIZE GL2C_HIT GL2C_MISS SQ_INSTS_VALU SQ_WAVES; do
  timeout 60 rocprofv3 --pmc "$C" --output-format csv \
    -d /tmp/pmc-sweep -o "$C" -- \
    ./target/release/examples/bench_decode_attention --seq 5100 --iters 1
done
```

Each run writes `<C>_counter_collection.csv`. Column 9 is `Kernel_Name`,
column 17 is `Counter_Value`. Merge per kernel:

```bash
for f in /tmp/pmc-sweep/*_counter_collection.csv; do
  c=$(basename "$f" _counter_collection.csv)
  echo "== $c =="
  awk -F, 'NR>1 && $9 ~ /attention/ {print "  "$9": "$17}' "$f"
done
```

## What the gfx1100 box returned (so you don't re-derive it)

rocprofv3 1.1.0 / ROCm 7.2.3 on gfx1100: the **infrastructure runs** (no more
signal-6 crash that earlier ROCm had), per-dispatch **metadata is solid**
(grid, workgroup, VGPR/SGPR/LDS, timestamps), but the **HW counter readback is
mostly dead**:

| Counter | gfx1100 result |
|---|---|
| `SQ_WAVES` | ✅ nonzero (96000) |
| `SQ_BUSY_CYCLES` | ✅ nonzero (~31M) |
| `SQ_INSTS_VALU`, `SQ_WAVE_CYCLES`, `SQ_INST_CYCLES_VMEM`, `SQ_WAIT_INST_ANY` | ❌ flat 0 |
| `FETCH_SIZE`, `WRITE_SIZE`, `GL2C_HIT`, `GL2C_MISS`, `MemUnitBusy`, `L2CacheHit` | ❌ flat 0 |
| `GRBM_COUNT`, `GRBM_GUI_ACTIVE`, `GPUBusy` | ❌ flat 0 |
| `SQ_INSTS_VMEM`, `TA_FLAT_LOAD_WAVEFRONTS`, `TCP_*`, `TCC_*`, `MemUnitStalled`, `VALUUtilization`, `VALUBusy` | ❌ rejected as unsupported |

Only `SQ_WAVES` and `SQ_BUSY_CYCLES` came back — neither answers the bandwidth
question. This is consistent with the RDNA3 SQ/TCC hardware perf-counter path
not being fully wired in ROCm's dispatch-mode PMC.

**First thing to check on gfx1151:** run the single-counter sweep above and look
at whether `FETCH_SIZE` / `GL2C_HIT` / `GL2C_MISS` read **nonzero**. If they do,
gfx1151's PMC works where gfx1100's doesn't, and the full answer is one merge
away. If they're zero there too, RDNA3.5 has the same gap and we fall back to the
analytical bandwidth model + kernel-trace timing as the tools of record (see
`docs/methodology/perf-benchmarking.md`).

## Fallback if PMC is dead on gfx1151 too

`scripts/rocprof-wrap.sh` (kernel-trace mode, no `--pmc`) **does** work on
gfx1100 and gives reliable per-kernel wall-clock. Combined with the known byte
volumes (KV cache = `seq_len × n_kv_heads × head_dim × 4 B` for F32), that yields
an *effective* GB/s per kernel without HW counters — which is how the 185 GB/s
figure was derived in the first place. The GQA traffic-reduction claim can be
checked indirectly: if `attention_flash_gqa_partial` wall-clock scales with
`n_kv_heads × head_dim` bytes (not `n_heads × head_dim`), the reuse is real.
