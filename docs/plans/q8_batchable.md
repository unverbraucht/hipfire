# Adding Q8 to `is_batchable_la` — scope, findings, and tiered plan

**Status:** Notes / scoping doc (deferred work)
**Last updated:** 2026-05-12
**Owner:** hipfire quant eval
**Companion docs:** `docs/plans/issue-113-quant-quality-eval.md` (PRD), `docs/plans/qwen35-mq4-quality-gap.md` (§Stage 0 floor measurement), `docs/plans/awq_hipfire.md`

---

## 1. Why this matters

The 9B engine-drift floor (`q8f16` candidate vs BF16 ref) measured at **KLD = 0.5735** on gfx1100 (qwen35-mq4-quality-gap.md §Stage 0). That measurement was made in eval_hipfire's **auto-fallback per-token path** because `DType::Q8_0` (what `q8f16` loads as, per `hfq.rs:495-497`) is not in `is_batchable_la`'s always_ok set.

All current 4-bit candidates (MQ4, MQ4-AWQ, HFP4G32, MFP4G32 when batchable) run through eval_hipfire's **batched-prefill path**. The PRD §5.3 documents a measured **~7% kernel-path bias** between the two modes (gfx1100 9B MQ4 full slice: prefill 0.817 vs per-token 0.876, Pearson 0.949, CIs non-overlapping, sign-test p < 1e-100).

So when we say "MQ4 absolute KLD − Q8 floor = quantization-attributable component" we are mixing scoring modes. The intended `KLD(MQ4_prefill) − KLD(Q8_per-token)` subtraction is contaminated by the same kernel-path bias the PRD §5.3 explicitly warns against cross-comparing.

**The clean fix is to score Q8 in prefill mode too**, which requires Q8_0 in `is_batchable_la`. This doc lays out what that takes.

---

## 2. What was tried 2026-05-12 — Tier 1 (one-line + smoke)

### 2.1 The change

```rust
// crates/hipfire-runtime/src/llama.rs::is_batchable_la
// crates/hipfire-arch-qwen35/src/qwen35.rs::is_batchable_la
let always_ok = matches!(dt,
    DType::MQ4G256 | DType::HFQ4G256
    | DType::MQ6G256 | DType::HFQ6G256
    | DType::Q8_0      // <-- added
);
```

Plus a corresponding update to the `is_batchable_la_unsupported_dtypes` test and an `is_batchable_la_q8_0_now_always_ok` smoke test.

### 2.2 The result

Smoke run on `/data/cache/hipfire/qwen3.5-9b.q8f16` (`--scoring-mode prefill --kv-mode asym3 --max-chunks 1`):

```
KV cache: asym3 (K rotated-3b 100B + V Q8 272B = 372 B/head, ...)
  chunk    1/1  scored     1023/    1023  (100.0%, 78 tok/s)
eval_hipfire: scored 1023 tokens in 13.2s (78 tok/s)
eval_hipfire: slice-mean KLD = 0.000000  mean NLL = NaN  PPL = NaN
```

Three observations:

1. **The build succeeded** (after also stubbing the pre-existing AWQ-batched build break at `llama.rs:697` — see §3.1).
2. **Batched dispatch did fire** — 78 tok/s vs per-token's 11 tok/s on the same model.
3. **The numbers are garbage.** KLD = 0.0, NLL = NaN. This is exactly the "silent prefill corruption" the `qwen35.rs:3863-3870` comment warns about: HFQ4-stride-layout kernels reading Q8_0-stride data.

The change was reverted; the AWQ-batched stub kept (it's a real fix; see §3.1).

### 2.3 Why it failed

`forward_prefill_chunk` in `crates/hipfire-arch-qwen35/src/qwen35.rs` dispatches per-DType for the LA layer body via hard-coded match arms calling MQ4/HFQ4-family kernels. Concrete call sites (line numbers from this branch):

| line | call | format-specific? |
|---:|---|---|
| 2021 | `gpu.fused_qkvza_hfq4g256(...)` | yes — HFQ4 stride |
| 2122 / 2153 / 4072 / 4100 / 4637 | `gpu.fused_silu_mul_rotate_mq_batched(...)` | yes — MQ FWHT path |
| 4352 / 4391 | `gpu.gemm_qkvza_hfq6g256(...)` / `_hfq4g256(...)` | yes — HFQ stride |
| 4538 / 4558 / 4649 / 4669 | `gpu.gemm_hfq6g256_residual(...)` / `_hfq4g256_residual(...)` | yes |
| 4584 / 4616 | `gpu.gemm_gate_up_hfq6g256(...)` / `_hfq4g256(...)` | yes |
| 4713 / 4750 | `gpu.gemm_qkv_hfq6g256(...)` / `_hfq4g256(...)` | yes |

When `is_batchable_la(Q8_0) == true`, forward_prefill_chunk enters the batched path with Q8_0 weights but calls (e.g.) `gemm_hfq4g256_residual` which reads weight bytes assuming HFQ4's 104-byte-per-group layout. Q8_0 uses 34-byte blocks. The kernel runs but reads garbage; the result is zero or NaN logits.

---

## 3. Side findings during the exploration

### 3.1 Pre-existing build break in current HEAD

`llama.rs:697` calls `gpu.fused_rmsnorm_rotate_mq_awq_batched` which is referenced but never defined. Phase 2a (commit `e51a3cd9`) added only the non-batched `fused_rmsnorm_rotate_mq_awq`; Phase 2b (commit `a4265ce4`) added the call site for the batched variant but the kernel wrapper was never landed.

Fixed in this session as a temporary stub at `dispatch.rs` immediately after `fused_rmsnorm_rotate_mq_batched`:

```rust
pub fn fused_rmsnorm_rotate_mq_awq_batched(
    &mut self,
    x: &GpuTensor,
    weight: &GpuTensor,
    awq_scale: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    batch_size: usize,
) -> HipResult<()> {
    for i in 0..batch_size {
        let x_row = x.sub_offset(i * k, k);
        let xr_row = x_rot.sub_offset(i * k, k);
        self.fused_rmsnorm_rotate_mq_awq(&x_row, weight, awq_scale, &xr_row, k, eps)?;
    }
    Ok(())
}
```

Correctness-equivalent to the batched non-AWQ kernel (same math, just N launches instead of 1). Slower than a true batched kernel would be. Replace with a proper grid.x = batch_size kernel launch when AWQ-prefill becomes a perf-target.

### 3.2 Available Q8 kernels — inventory (`kernels/src/`)

What exists today:

```
attention_flash_q8_0_{reduce,tile}.hip
attention_q8_0_kv{,_batched,_timed}.hip
attention_hfq8_kv.hip, attention_q8kv.hip
embedding_q8{,_batched}.hip
gated_delta_net_q8{,.gfx1200,_tree}.hip
gemm_q8_0_batched.hip                ← single batched GEMM, MAX_BATCH=16
gemv_q8_0{,_wide}.hip                ← per-token GEMV
gemv_q8hfq{,_wide}.hip
gemv_hfq8g256.hip, gemv_mq8g256.hip
kv_cache_write_q8{,_0,_0_batched,_hfq8}.hip
kv_fold_q8.hip, pflash_score_q8_kv.hip
triattn_score_q8.hip
```

What **does not exist** for Q8 (compare to MQ4/HFQ4 inventory in `kernels/src/`):

- `fused_qkv_q8_0` (no — MQ4 has 3 wave-variant fused QKV kernels)
- `fused_qkvza_q8_0` (no — Q4 has 3 wave-variants)
- `fused_gate_up_q8_0` (no — Q4 has 3)
- `gemm_qkv_q8_0` / `gemm_qkv_q8_0_wmma` (no)
- `gemm_qkvza_q8_0` (no)
- `gemm_gate_up_q8_0` (no — Q4 has 6 variants spanning dot2/fp16/wave64/wmma/wmma_ldsx)
- `gemm_q8_0_residual` (no — Q4 has 16+ variants including gfx906 MMQ x{8,16,24,32,40,48,56,64} stack)
- `fused_silu_mul_rotate_q8` (no — MQ family kernels are FWHT-specific; Q8 doesn't use FWHT, so the analogue would be a non-rotate fused silu_mul)

The attention + KV-write side is in good shape (Q8 is the canonical live-decode path); the LA-prefill side has nothing.

---

## 4. The work — tiered

### Tier 2 — fall-back wrappers (~1–2 days; correctness, not perf)

Goal: make Q8_0 enter `is_batchable_la::always_ok` and produce correct logits, even if slower than MQ4/HFQ4.

Approach: for each LA-batched dispatch site, add a Q8_0 arm that calls existing per-row GEMV kernels in a loop, or chunks through `gemm_q8_0_batched` at MAX_BATCH=16. No new HIP kernels written.

Concrete edits to `crates/hipfire-arch-qwen35/src/qwen35.rs` (representative — full list in §2.3):

```rust
// Pattern at each site (e.g. ~line 4584 gate_up dispatch):
match wsh.gpu_dtype {
    DType::HFQ4G256 => gpu.gemm_gate_up_hfq4g256(...)?,
    DType::HFQ6G256 => gpu.gemm_gate_up_hfq6g256(...)?,
    DType::Q8_0     => gemm_gate_up_q8_0_via_chunks(gpu, ..., MAX_Q8_BATCH=16)?,  // NEW
    _ => panic!("unsupported dtype for batched gate_up: {:?}", wsh.gpu_dtype),
}
```

Where each `*_via_chunks` helper iterates the input batch in chunks of 16 and calls `gemm_q8_0_batched` per chunk. For QKV/gate_up the kernel call shape is "[N rows] × [K cols]" — split N into ceil(N/16) sub-calls.

For `fused_silu_mul_rotate_mq_batched` (MQ family, lines 2122/4072/etc.), Q8_0 weights don't use FWHT — so the Q8_0 arm should NOT call this kernel at all. Instead, plain `silu_mul_f32_batched` + skip rotation. This may mean the LA preamble for Q8 layers is structurally different from MQ layers — careful audit needed.

**Effort breakdown:**

| step | wall | risk |
|---|---|---|
| Audit each of 6+ dispatch sites; document required Q8 alternative | ~3 h | low — code-reading only |
| Implement `*_via_chunks` helpers (or direct call to `gemm_q8_0_batched`) | ~4–6 h | low; underlying kernel exists |
| Handle the FWHT-elimination for Q8 LA preamble | ~2–4 h | medium — possible new code path |
| Smoke + 50-chunk validation against the existing per-token Q8 floor | ~1 h compute + 1 h analysis | low |

**Expected throughput after Tier 2:** the unfused chunked-at-16 Q8 path should run at ~2–5× per-token decode speed (vs MQ4's ~20× speedup over per-token). Plenty for the floor diagnostic; not for ship benches.

**Validation pass criteria:**
1. `KLD(q8_prefill_tier2 vs BF16) ≈ KLD(q8_per-token vs BF16)` within the ~7% kernel-path mode-bias range (so within 0.04 nats of the 0.5735 per-token floor). If it lands at ~0.55 ± 0.04, that confirms the path is correct and roughly closes the methodology gap.
2. Same NLL/PPL within similar tolerance.

### Tier 3 — production parity (~2–3 weeks)

Goal: Q8 prefill at MQ4-class throughput, ship-ready.

Write fused/wmma Q8 batched kernels matching the MQ4/HFQ4 suite:

| family | kernels needed | reference |
|---|---|---|
| Fused LA preamble | `fused_qkv_q8_0`, `fused_qkv_q8_0_wave64`, `fused_qkv_q8_0_wave64_dp4a` (3) | mirror `fused_qkv_hfq4g256*` shape |
| Fused LA + gate | `fused_qkvza_q8_0` × 3 wave-variants | mirror `fused_qkvza_hfq4g256*` |
| FFN gate+up | `fused_gate_up_q8_0` × 3 wave-variants | mirror `fused_gate_up_hfq4g256*` |
| Batched plain GEMM | `gemm_qkv_q8_0`, `gemm_qkv_q8_0_wmma`, gfx12 sibling | mirror `gemm_qkv_hfq4g256*` |
| Batched residual GEMM | `gemm_q8_0_residual`, `gemm_q8_0_residual_wmma`, `_wave64`, etc. (~6–8 variants) | mirror `gemm_hfq4g256_residual*` family |
| `gemm_gate_up_q8_0` family | × 6 variants (dot2/fp16/wave64/wmma/...) | mirror `gemm_gate_up_hfq4g256*` |
| Non-FWHT SwiGLU | `silu_mul_f32_batched` (likely already exists; verify) | check `kernels/src/` |

≈ **10–14 new HIP kernels**, each with arch-specific siblings, each with a wave32+wave64 path. The MQ4 suite was originally ~3 weeks for a kernel-experienced contributor.

Tier 3 only makes sense when Q8 prefill becomes a **regular production target** — neither the current eval matrix nor the live-decode path needs it.

---

## 5. Decision matrix

| use case | enough? |
|---|---|
| One-off Q8 floor diagnostic to validate the ~0.57 number | **Per-token mode is enough** (current state — what 0.5735 was measured in). The kv-ablation experiment + Step 2 pos-0 logit check (both in flight as of 2026-05-12) triangulate the floor's components without Q8 prefill. |
| Future cohorts that report `Δ = KLD(MQ4_X) − KLD(Q8)` and want the Q8 in the SAME scoring mode | **Tier 2 sufficient.** Slower but correct; numbers comparable to MQ4 prefill rows. |
| Production ship target for Q8 prefill (e.g., long-context serving where prefill latency matters) | **Tier 3 needed.** Separate project. |

**Current recommendation (2026-05-12):** stay on per-token for Q8 floor measurements. Note the ~7% kernel-path-bias caveat in any Q8-vs-MQ4 delta calculation, and accept ~7% as an irreducible methodological uncertainty until Tier 2 lands. Don't write Tier 3 unless Q8 prefill becomes a ship target.

---

## 6. Open questions

- **Q8_0 group layout vs HFQ4G256.** HFQ4G256 packs 256 weights per group with a fixed scale-zero header layout (104 B/group). Q8_0 packs 32 weights per block with 2 B scale (34 B/block). For Tier 2 fallback wrappers chunked at 16 rows, the kernel's expectation of "block stride" needs verifying — `gemm_q8_0_batched.hip` should already be Q8-stride-correct, but the LA call site needs to pass the right `k * elem_size_per_block` arithmetic.
- **Does `fused_silu_mul_rotate_mq_batched` need a Q8 sibling?** Q8 weights don't use FWHT, so this kernel is bypassed entirely. The LA preamble for Q8 layers would compose `silu + mul` separately. But the residual chain in `forward_prefill_chunk` may assume an `x_rot` output buffer from the SwiGLU step — need to thread through.
- **MoE path.** Most of the eval-relevant rows are dense Qwen3.5-9B; MoE Qwen3.5-A3B's MoE path has its own dispatch using `gemv_hfq4g256_moe_*` kernels. Adding Q8 there is a separate effort. For the floor diagnostic, dense is enough.
- **Does Q8_0 prefill batching have any DN-state interaction?** DN state is the recurrent linear-attention state. The Q8 kernels for DN (`gated_delta_net_q8.hip`) already work in the per-token live-decode path. forward_prefill_batch's DN path calls `gated_delta_net_q8_batch_seq` which is sequential-per-token by design. For Q8, this remains unchanged — Q8 only affects the GEMM-side of the LA layer, not the DN recurrence.

---

## 7. Files affected (Tier 2)

- `crates/hipfire-runtime/src/llama.rs::is_batchable_la` — add `Q8_0` to `always_ok`
- `crates/hipfire-arch-qwen35/src/qwen35.rs::is_batchable_la` — same
- `crates/hipfire-arch-qwen35/src/qwen35.rs::forward_prefill_chunk` — Q8_0 arms at 6+ dispatch sites
- Unit tests in `llama.rs` (lines 4097–4140) — move Q8_0 to the always_ok test
- `crates/rdna-compute/src/dispatch.rs` — add `gemm_gate_up_q8_0_via_chunks`, `gemm_qkv_q8_0_via_chunks`, `gemm_q8_0_residual_via_chunks` helper wrappers (or call `gemm_q8_0_batched` directly from the dispatch sites — author's choice)

## 8. References

- `docs/plans/issue-113-quant-quality-eval.md` §5.3 — measured prefill-vs-per-token kernel-path bias
- `docs/plans/qwen35-mq4-quality-gap.md` §Stage 0 — 0.5735 KLD floor result
- `docs/plans/awq_hipfire.md` — companion plan for the AWQ work that introduced the in-flight build break (§3.1 above)
- `crates/hipfire-runtime/src/hfq.rs:495-497` — Q8F16 → DType::Q8_0 binding
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3863-3870` — the "silent prefill corruption" warning comment that this exploration validated
- `kernels/src/gemm_q8_0_batched.hip` — the one batched Q8 GEMM that exists today (MAX_BATCH=16)
