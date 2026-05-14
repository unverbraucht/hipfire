# GPTQ quantize panic — `TensorSpill` path collision under concurrent runs

## TL;DR

`hipfire-quantize`'s `TensorSpill` scratch file is hard-coded to
`<output_dir>/.hipfire_quant_spill.tmp` (`main.rs:2281`). The path
includes neither the PID nor any suffix derived from the output file
name. Any second `hipfire-quantize` process whose output lives in the
same directory (which is almost always `~/.hipfire/models/` in
practice) will:

1. `File::create(&path)` on the existing spill file → **truncate to
   zero bytes**, invalidating the first process's spill data.
2. When the second process finishes, its `TensorSpill::drop` calls
   `std::fs::remove_file(&self.path)` → **unlink the spill path
   entirely**.

The first process is still writing through its own `BufWriter<File>` to
the now-unlinked inode (Linux keeps the inode alive while the FD is
open, so the writes don't fail). When the first process eventually
reaches `write_hfq` (line 4617) and tries `File::open(&spill.path)?`
to read back the spilled tensor data, the path no longer exists →
`Err(Os { code: 2, kind: NotFound })` → `.unwrap()` at
`main.rs:4617:83` panics. **Hours of GPTQ Hessian work are lost at the
final write step.**

Observed on commit `fb2b9278` (master branch as of 2026-05-13 11:26).
The panic site `main.rs:4617:83` corresponds to the `.unwrap()` after
`write_hfq(...)`; the line numbering shifted by a few in my newer
commits, but the bug is the same.

## Reproducer (observed 2026-05-13 → 2026-05-14)

Two quantize processes ran concurrently with overlapping windows, both
writing into `/home/kread/.hipfire/models/`:

| Process | Start | End | Command | Result |
|---|---|---|---|---|
| **GPTQ** (PID 256323) | 2026-05-13 11:26:48 | **2026-05-14 01:44:29 (panic)** | `--awq --gptq <hessian> --gptq-damp 0.01 --gptq-max-damp 1.0 --format mq4 --imatrix ... --output qwen3.5-9b.mq4-awq-gptq` | rc=101, 14 MB partial output |
| **My MQ6-lm_head chain** (`/tmp/quantize_awq_mq6lm_q8conv_now.sh`) | 2026-05-13 21:45:41 | 2026-05-13 21:53:06 | `HIPFIRE_QUANTIZE_CONV_Q8=1 HIPFIRE_QUANTIZE_LM_HEAD_MQ6=1 --awq --imatrix ... --output qwen3.5-9b.mq4-awq-q8conv-mq6lmhead` | rc=0, valid 5.57 GB output |

Both processes share `spill_dir = output_path.parent() = /home/kread/.hipfire/models`
and therefore share `.hipfire_quant_spill.tmp` (`main.rs:2281`).

Timeline of the spill file lifecycle:

```
11:26  GPTQ A: TensorSpill::new() → File::create(spill_path) → opens FD#A,
              truncates to 0 bytes, ready
11:26→21:45  GPTQ A: layer-by-layer Hessian pass, calls spill.spill(...)
              many times via its BufWriter (FD#A). File grows.

21:45  My quantize B: TensorSpill::new() → File::create(spill_path) →
              TRUNCATES the existing inode to 0 bytes (or creates a new
              inode at the same path; details depend on filesystem and
              prior unlink state — both lose GPTQ's data either way).
              GPTQ A's FD#A still references its original inode, so
              GPTQ A's writes continue to "succeed" but to an orphaned
              inode (after my Drop, below).
21:45→21:53  My quantize B writes its own quant data to the spill file.
              GPTQ A simultaneously continues writing through FD#A
              (now pointing at a different inode than the path).

21:53  My quantize B: TensorSpill::drop → remove_file(spill_path).
              Path is unlinked. My quantize B exits clean (its work
              already completed before this drop runs).

21:53→01:44  GPTQ A: continues. Its BufWriter still has FD#A; writes
              still "succeed" at FD layer. No errors visible.

01:44  GPTQ A: reaches write_hfq(...), which does
              File::open(&spill.path)? at main.rs:2396 to read back the
              spilled tensor data. Path was unlinked at 21:53 → NotFound
              → panic at main.rs:4617:83 (the .unwrap()).
```

The proximate signal — `Os { code: 2, kind: NotFound, message: "No such
file or directory" }` returned from `File::open(&spill.path)` —
matches this story exactly.

## Source pointers

`crates/hipfire-quantize/src/main.rs:2273-2313` — `TensorSpill`:

```rust
struct TensorSpill {
    file: std::io::BufWriter<File>,
    path: PathBuf,
    offset: u64,
}

impl TensorSpill {
    fn new(dir: &Path) -> std::io::Result<Self> {
        let path = dir.join(".hipfire_quant_spill.tmp");  // ← hardcoded, shared
        let file = std::io::BufWriter::with_capacity(
            4 * 1024 * 1024,
            File::create(&path)?,  // ← truncates existing
        );
        Ok(Self { file, path, offset: 0 })
    }
    // ...
    fn cleanup(self) { drop(self); }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);  // ← unlinks path on drop
    }
}
```

`crates/hipfire-quantize/src/main.rs:2393-2418` — `write_hfq` read-back:

```rust
if let Some(spill) = spill {
    let _ = spill.flush();
    let mut spill_reader = std::io::BufReader::new(
        File::open(&spill.path)?            // ← line 2396 — fails with NotFound
                                            //   if a concurrent quantize
                                            //   already removed the path
    );
    // ... copy spilled tensors into the output file ...
}
```

`crates/hipfire-quantize/src/main.rs:3817-3818` — where `spill` is
constructed (output dir = parent of `--output`):

```rust
let spill_dir = output_path.parent().unwrap_or(Path::new("."));
let mut spill = TensorSpill::new(spill_dir).ok();
```

## Fixes (preferred → fallback)

### Option 1 (Preferred) — make the spill path unique per run

Smallest, most targeted change. Anything that prevents two concurrent
processes from picking the same path works:

```rust
fn new(dir: &Path) -> std::io::Result<Self> {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let path = dir.join(format!(".hipfire_quant_spill.{pid}.{nanos}.tmp"));
    // ...
}
```

PID alone is enough to prevent the concurrent-run collision (the PID
namespace is per-process unique on Linux). The nanosecond suffix only
protects against the unlikely edge case of PID reuse across two
quantize runs sharing a single inode within the same epoch tick.

Plus: error on `File::create` if the path exists, instead of silently
truncating. Defensive belt-and-suspenders:

```rust
use std::fs::OpenOptions;
let file = OpenOptions::new()
    .read(true).write(true).create_new(true)
    .open(&path)?;  // create_new fails if path exists
```

### Option 2 — use the `tempfile` crate's `NamedTempFile`

Already in the workspace dep graph (used by tests). Picks a unique
filename in the system temp dir or in a caller-chosen dir, with the
right semantics by construction:

```rust
let tmp = tempfile::NamedTempFile::new_in(dir)?;
let (file, path) = tmp.into_parts();
// path is auto-deleted by TempPath drop; preserve via path.keep() if needed
```

Larger diff but eliminates the entire class of "two quantize runs
share a scratch path" bug.

### Option 3 — co-locate spill with the output file

Change `let path = dir.join(format!(".hipfire_quant_spill.{output_filename}.tmp"))`
where `output_filename = output_path.file_name()`. Cheap, no PID
needed, but two runs targeting the same output filename (e.g.,
resuming) would still collide. The user told me to delete the old
output before re-quantizing tonight, so they hit this case in practice.

Recommendation: **Option 1 (PID+nanos suffix + `create_new`)**. Minimal
diff, no new deps, defends against both the concurrent-run case and
the same-output-name re-quantize case.

## Defense-in-depth: don't lose work at the final write

Even with the spill-path fix, the panic happens AFTER the entire GPTQ
Hessian pass completed (the log shows
`Quantized params: 8953528320 (100.0%)` immediately before the panic).
14 hours of computation evaporated on a write-step bug. Two
mitigations:

1. **Spill the in-memory quantized tensor data to a sidecar file
   periodically**, not just at the very end. Then the final
   `write_hfq` is a pure copy from `<output>.partial.hfq` to
   `<output>.hfq`, and a panic at any earlier step leaves a
   resumable artifact instead of a 14 MB header.
2. **Replace the `.unwrap()` at `main.rs:4617:83` with a `?`**
   propagating the error to `main`, plus a Drop-tolerant cleanup
   for `spill`. Doesn't save the work, but exits with a clean
   error message + non-zero exit code instead of `thread 'main'
   panicked`. Easier to file/CI on.

Both are worth doing. The first is a multi-day change (touches the
quant-loop output protocol); the second is a 1-line correctness fix.

## Coverage gate

`scripts/coherence-gate.sh` doesn't currently exercise GPTQ at all
(GPTQ requires a precomputed Hessian sidecar, which the gate doesn't
ship). Adding `qwen3.5-0.8b.mq4-awq-gptq` to the gate matrix would
not have caught the spill-path collision (it's a concurrency bug, not
a code-path bug), but a unit test for concurrent `TensorSpill::new`
would:

```rust
#[test]
fn tensor_spill_paths_are_per_process_unique() {
    let dir = tempfile::tempdir().unwrap();
    let s1 = TensorSpill::new(dir.path()).unwrap();
    let s2 = TensorSpill::new(dir.path()).unwrap();
    assert_ne!(s1.path, s2.path, "spill paths must be unique across overlapping spills");
}
```

That single test would have caught this regression before merge.

## Recovery

GPTQ on 9B is ~14 hours wall. Hessian sidecar
(`benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.hessian.bin`)
already exists, so re-running GPTQ once the spill bug is fixed costs
only the Hessian-quantize pass, not the Hessian collection.

The Hessian sidecar took its own multi-hour pass to produce — keep it
intact.

## Related

- The MQ4G256 + AWQ + GPTQ output path was the only artifact for the
  Stage B (GPTQ on MQ4) row of `docs/plans/qwen35-mq4-quality-gap.md`
  §5 Phase A revised. Re-quantize after fix to unblock Stage B
  measurement.
- The chain script `/tmp/quantize_awq_mq6lm_q8conv_now.sh` was the
  culprit second-quantize, but the root cause is the spill-path
  hardcoding, not "two concurrent quantizes is invalid usage." Users
  *will* run multiple quantizes targeting the same models dir; the
  quantizer must tolerate it.

(2026-05-14 ~06:30 CEST)
