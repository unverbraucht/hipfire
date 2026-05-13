# Engine surgery — Phase 2 + 3a planning doc (2026-05-13)

Sequel to `00-summary.md`. Phase 1c+1d closed with the verdict
"engine drift is structural, not closable by per-weight precision". This
doc plans the next decision point: **whether kernel-level rewrite of
hipfire's compute graph is tractable**, and if so which kernels to start
with.

## Why this exists

Phase 1 audited LA layer 0 only. The data:

- Per-stage drift at stages 0-12 is small (≤ 0.012 rL2)
- Stage 13 (recurrence output) jumps to ~0.045 rL2 and STAYS THERE
  regardless of upstream input precision (F16 conv cut stages 11/12 by
  50%, stage 13 unchanged)
- Model-output KLD regresses for every F16-precision intervention

This rules out **per-position input drift** as the dominant contributor
to recurrence-output drift, and rules out **per-weight quant noise** as
the lever. What it does *not* tell us:

1. How much of the 0.045 stage-13 rL2 is intrinsic to hipfire's
   `gated_delta_net_f32` kernel implementation vs. is driven by some
   subtle property of the inputs we haven't isolated.
2. How the **FA layers** (LayerType::FullAttention) contribute. Closed
   investigation showed FA-7 at 0.157 rL2, FA-23 at 0.189 — these are
   *higher* per-layer than LA-4's 0.140. We have audited zero FA stages.
3. Where, within the FA pipeline, drift is introduced vs amplified.

Phase 2 (FA pipeline localization) and Phase 3a (recurrence
matched-input probe) answer 2 and 1 respectively. Together they
identify the 1-2 kernels most-worth surgical rewrite.

## Reference target

llama.cpp natively supports Qwen3.5 DeltaNet and scores **0.002 nats KLD
vs HF transformers BF16** (cross_engine_check.py). That's the achievable
floor — within ~3× of float-precision noise. llama.cpp's C++ ggml graph
is the de facto reference implementation in the open ecosystem; "byte-
match llama.cpp's compute order" is a *defined* engineering scope, not
open-ended exploration.

Whatever surgery we do should be benchmarked against llama.cpp's per-
stage rL2 (probe: dump intermediate state from llama.cpp scoring path,
compare to hipfire's same stages).

## Phase 2 — FA pipeline localization (1-2 days)

### What to instrument

The FA branch of `forward_scratch_layers` in
`crates/hipfire-arch-qwen35/src/qwen35.rs` has this kernel sequence
(per-token, decode mode):

| stage | hipfire kernel | output buffer | n_elems |
|---:|:---|:---|---:|
| 0 | (layer entry, pre-norm residual) | `s.x` | dim |
| 1 | `fused_rmsnorm_mq_rotate` / plain rmsnorm | `s.tmp` | dim |
| 2 | `weight_gemv_prerotated(wq)` | `s.q` | n_heads × head_dim |
| 3 | `weight_gemv_prerotated(wk)` | `s.k` | n_kv_heads × head_dim |
| 4 | `weight_gemv_prerotated(wv)` | `s.v` | n_kv_heads × head_dim |
| 5 | `q_norm` per-head RMSNorm on Q | `s.q` | n_heads × head_dim |
| 6 | `k_norm` per-head RMSNorm on K | `s.k` | n_kv_heads × head_dim |
| 7 | `rope_partial_halfsplit_f32` on Q,K | `s.q`, `s.k` | (both) |
| 8 | KV-cache write (Q8/asym3/fp32) | cache buffer | — |
| 9 | attention compute (FA / softmax) | `s.attn_out` | n_heads × head_dim |
| 10 | `weight_gemv_prerotated(wo)` | accum tmp | dim |
| 11 | residual add | `s.x` | dim |

12 stages × 3 target layers (FA-3, FA-7, FA-23) × 2048 positions = 73k
records per dump file. Per-file ~250 MB. Total disk ~1.5 GB for both
engines.

### HF oracle (Phase 2)

Extend `scripts/dump_hf_la_stages.py` to a new
`scripts/dump_hf_fa_stages.py` that monkey-patches
`Qwen3_5Attention.forward` to capture matching intermediates. The
PyTorch reference does:

```python
hidden = self.input_layernorm(hidden)  # stage 1
q = self.q_proj(hidden)                # stage 2
k = self.k_proj(hidden)                # stage 3
v = self.v_proj(hidden)                # stage 4
q = q.view(...); k = k.view(...); v = v.view(...)
q = self.q_norm(q)                     # stage 5
k = self.k_norm(k)                     # stage 6
q, k = apply_rotary_pos_emb(q, k, ...) # stage 7
# kv cache update (stage 8)
attn = scaled_dot_product_attention(...) # stage 9
attn = self.o_proj(attn)               # stage 10
hidden = residual + attn               # stage 11
```

DecoderLayer pre/post hooks capture stage 0 / 11. Monkey-patch captures
stages 1-10.

### Hipfire instrumentation

Add a parallel `dump_fa_stage` helper to `qwen35.rs` (alongside the
existing `dump_la_stage`) and an env-gate `HIPFIRE_DUMP_FA_STAGES`.
Mirror the layer-target selection via `HIPFIRE_DUMP_FA_LAYER`. 11
call sites in the FA branch, gated identically to the LA path.

### Phase 2 deliverable

A per-stage rel_L2 table for FA-3 / FA-7 / FA-23, sorted by stage and
layer. Specifically answer:

- Which stage in the FA path introduces the bulk of per-layer drift?
- Is the dominant kernel the same across FA-3 / FA-7 / FA-23 (one
  kernel issue compounding) or does it shift (multiple distributed
  issues)?
- How does FA-3 stage drift compare to LA-0 stage drift at matched
  stages (e.g., post-rmsnorm)? Are the layer-types' per-stage kernels
  contributing differently?

If a single FA stage dominates (>50% of per-layer rL2 jump): that
kernel is the surgery target. If drift is spread across 4+ stages:
the bottleneck is graph-level fusion, not a single kernel.

## Phase 3a — Recurrence matched-input probe (1 day)

### Hypothesis

Stage 13 (recurrence output) shows 0.045 rL2 at LA-0. We've established:

1. fp32 vs fp64 internal state in `gated_delta_net_f32` is byte-identical
   (closed investigation probe c.3)
2. Per-position input precision interventions don't reduce stage 13
   drift (this audit Phase 1c-1d)

If we feed `gated_delta_net_f32` with **HF's exact tensors** (q/k/v/α/β
at stage 11/12/10/6/7 boundaries) and compare its output to HF's
`chunk_gated_delta_rule` output, the residual is the recurrence-kernel
implementation drift in isolation.

### Methodology

1. Run the existing Phase 1 LA-stages dump on hipfire and HF — done
   already, files at `/data/cache/hipfire/audit-2026-05-13/`.
2. Read HF's stage 11 (q post-l2norm+scale), stage 12 (k post-l2norm),
   stage 10 (v), stage 6 (gated alpha), stage 7 (gated beta).
3. Write a new diagnostic example `recurrence_matched_input` in
   `crates/hipfire-runtime/examples/`:
   - Loads HF's per-position tensors from the dump file
   - For each position 0..2048:
     - Uploads HF's q/k/v/α/β to GPU
     - Calls `gpu.gated_delta_net_f32(...)` with those inputs
     - Downloads `s.dn_attn_out` (stage 13 buffer)
   - Writes to a new `HIPFIRE_DUMP_LA_STAGES`-format file, stage_id 13
     only, layer_idx = (target layer)
4. Compare against HF's stage 13 captured in the Phase 1 HF dump file.

Result: stage 13 rL2 with bit-exact inputs. If ≥ 0.01: recurrence
kernel itself diverges from HF's `chunk_gated_delta_rule` and is the
surgery target. If ≤ 0.001: recurrence is faithful; the residual at
stage 13 in our Phase 1c data was actually coming from upstream
through some path we haven't accounted for (most likely the GQA repeat-
interleave or the state initialization).

### Phase 3a deliverable

A single rL2 number: "recurrence kernel intrinsic drift with bit-exact
inputs". Binary outcome: recurrence kernel needs surgery or not.

## Decision criteria (after Phase 2 + 3a)

| outcome | surgery target | est. effort | go/no-go |
|---|---|---|---|
| FA single-kernel dominant + recurrence clean | rewrite that FA kernel | 1-2 weeks | GO |
| FA distributed (>4 stages) + recurrence clean | graph-level rewrite of FA fusion | 3-4 weeks | reconsider |
| FA distributed + recurrence diverges | both | 4-6 weeks | NO-GO — calibration only |
| FA clean + recurrence diverges | rewrite recurrence kernel | 1-2 weeks | GO |
| Both clean | no localized bug; floor is genuinely structural | n/a | NO-GO — calibration only |

The "GO" outcomes have a defined kernel to rewrite. The "NO-GO" outcomes
mean engine surgery is exploration, not engineering — switch entirely
to calibration (AWQ Stage A on 0.8B, then Stage B/C).

## Phase 4 — Kernel rewrite (if GO)

Scope: byte-match the dominant kernel to llama.cpp's implementation.
Llama.cpp source for Qwen3.5 is in
`ggml-cuda/argsort.cu`, `ggml-cuda/rope.cu`, `ggml-cuda/ssm-scan.cu`
(equivalent paths), and the dispatcher in `ggml-cuda.cu`. The C++
implementations are short enough to read end-to-end and port to HIP.

Phase 4 milestones:

1. **Port the dominant kernel** (1 week): write `<kernel>_v2.hip`
   alongside the existing one. Env-gate `HIPFIRE_KERNEL_V2_<name>=1`
   so we can A/B test without touching the production path. Verify
   per-stage rL2 against HF: target 0.001 at that stage.

2. **Model-output validation** (0.5 day): re-run eval_hipfire with
   the env-gate on. Target: KLD drop ≥ 0.02 nats (i.e., 0.08 → 0.06).
   If drop < 0.01: the kernel was identified correctly but the fix
   isn't moving the floor — investigate why, may need to rewrite
   multiple kernels in tandem.

3. **AWQ compose-check** (0.5 day): re-run the AWQ-quantized model on
   the v2 kernel path. Confirm AWQ above-floor delta still positive
   on the new floor. If AWQ's benefit shrinks, the systematic bias the
   kernel had has been replaced with a less-correctable form;
   re-validate the AWQ scale calibration.

4. **Production graduation gate** (0.5 day):
   - Coherence-gate passes (./scripts/coherence-gate.sh)
   - DFlash coherence-gate passes for spec-decode paths
   - Within-arch perf benchmark shows no regression > 5%
   - 20-chunk full-eval KLD confirms the 5-chunk improvement

If all pass: flip the v2 kernel to default, remove the env-gate.

## Sequencing with gfx1100 AWQ/GPTQ work

The gfx1100 calibration work runs entirely in the quantizer + AWQ
sidecar path. The engine surgery here runs in `kernels/src/*.hip` and
`crates/rdna-compute/src/dispatch.rs`. No direct conflict.

Interaction at the benchmark layer:

- Today: AWQ on 9B measured at −32.6% above-floor KLD. That's relative
  to today's engine floor.
- If engine surgery succeeds: floor drops, AWQ's *absolute* KLD also
  drops, but the *above-floor delta* may shrink (engine bias was
  partially helping AWQ's apparent benefit).
- Phase 4's AWQ compose-check (4.3) catches this and re-validates.

Recommended sequencing:

1. **Now**: Phase 2 audit (1-2 days, parallel with AWQ-on-0.8B work)
2. **Day 3**: Phase 3a probe (1 day)
3. **Day 4**: Decision review — if GO, draft Phase 4 plan; if NO-GO,
   close the engine-surgery branch and ship calibration as the answer
4. **Day 5+**: if GO, Phase 4 kernel rewrite (1-2 weeks)
   In parallel: AWQ on 0.8B continues on the current engine
5. **Post-rewrite**: re-validate AWQ on the v2 kernel path (0.5 day),
   re-bench 0.8B + 9B cohorts (1 day)

Total wall to decision: **~4 days**. Total wall to surgery shipped (if
GO): **~3 weeks** including validation gates.

## Files

To be created during Phase 2/3a:

- `crates/hipfire-arch-qwen35/src/qwen35.rs`: `dump_fa_stage` helper +
  11 call sites + `HIPFIRE_DUMP_FA_STAGES` / `HIPFIRE_DUMP_FA_LAYER`
  env-gates
- `scripts/dump_hf_fa_stages.py`: HF oracle for FA pipeline
- `scripts/compare_fa_stages.py` (or extend `compare_la_stages.py`)
- `crates/hipfire-runtime/examples/recurrence_matched_input.rs`:
  Phase 3a probe
- `docs/investigations/2026-05-13-engine-drift-residual-audit/02-phase2-results.md`
- `docs/investigations/2026-05-13-engine-drift-residual-audit/03-phase3a-results.md`
- `docs/investigations/2026-05-13-engine-drift-residual-audit/04-decision.md`

## Exit conditions for the engine-surgery branch

- After Phase 2+3a, if no single kernel exceeds 50% of per-layer drift
  contribution: close branch, switch to calibration-only path
- After Phase 4 milestone 2, if model-output KLD drops < 0.01 nats:
  the kernel rewrite was correctly targeted but the floor isn't kernel-
  localizable — close, switch to calibration
- After Phase 4 milestone 4, if any gate fails: revert v2 kernel, close

The audit infrastructure stays in-tree regardless of outcome.
