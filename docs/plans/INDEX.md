# `docs/plans/` — index of active investigations and design plans

**Purpose:** a single map of the in-flight investigation threads in hipfire. Each thread groups one or more plan / investigation / audit docs around a coherent technical question, points at the master doc for that thread, and identifies what links to what.

When a new doc lands in `docs/plans/` or `docs/investigations/`, add it here under the relevant thread (or open a new thread).

When a thread completes and ships, move its docs to `docs/investigations/<date>-<topic>/` per the existing convention, then drop the thread from this index with a one-line "shipped in PR #X" note kept for searchability.

**Last refreshed:** 2026-05-11.

---

## Thread 1 — Quantization format roadmap (HFP4 / MFP4 commit) — under review

**Master doc:** `qwen35-mq4-quality-gap.md`
**Rebuttal + KLD update:** `hfp4-fivetide-rebuttal-perspective.md` (2026-05-11 + 2026-05-12)
**Status:** Strategic claim under review, pending multi-metric resolution. Fivetide's PPL analysis (2026-05-11) showed MFP4 losing on PPL; their KLD measurement (2026-05-12 update) showed FWHT helps E2M1 on KLD — opposite signal. Neither metric alone is sufficient; downstream task metrics are the next yardsticks.

The strategic format decision for the next several years. MQ4 → HFP4 (PR #224) → MFP4 (PR #225) closed the per-weight format-quality levers; the remaining gap to Unsloth Dynamic 2.0 is activation-aware calibration (imatrix). **However:** fivetide initially measured +25–94% PPL regression of MFP4G32 vs MQ4G256 on Qwen3.5 dense, and then their own KLD follow-up reversed the picture (FWHT helps E2M1 on KLD). The format roadmap is still useful for organizing engineering work; the strategic conclusion ("commit to HFP4") is suspended pending downstream-task evidence. Phase A (imatrix calibration) is still the right engineering — just on top of a baseline-format question that's empirically open.

**Phase A entry point:** `qwen35-mq4-quality-gap.md` §5 (lines 393+). The 2026-05-12 framing update adds two prerequisite steps (Step 0 — bench expansion to emit MSE + KLD + PPL + HumanEval per format variant; Step 0.5 — reproduce fivetide's PPL + KLD numbers in-tree as the multi-baseline reference table) before any quantizer-side change lands. Each L4/L5 step then runs against three baseline formats (MQ4G256, MFP4G32, HFP4G32-unrotated) rather than assuming MFP4 as the calibration baseline. Net: 5–7 weeks (vs original 4–6) for evidence-based format decision rather than projection-based.

| Doc | Role |
|---|---|
| `qwen35-mq4-quality-gap.md` | **Master roadmap** (carries a 2026-05-11 header annotation flagging the rebuttal). Per-lever taxonomy (L1–L5), what HFP4 closed, what's missing, format-extension plan, future-proofing for Gemma / Qwen2.5-VL, gfx906 acceleration analysis, Phase A/B/B'/C/D sequencing. |
| `hfp4-fivetide-rebuttal-perspective.md` | **Empirical rebuttal.** Fivetide's PPL data, the kurtosis / Lloyd-Max codebook analysis showing INT4-uniform beats E2M1 at g=32 on post-FWHT weights, methodology gaps in fivetide's analysis, honest meta-lesson on per-weight-MSE vs model-quality, concrete next-actions for measurement-driven resolution. |
| `gfx906-moe-kernel-gaps.md` | Phase B' prerequisite (dropped in priority while format question is under review). 3-way-cross-validated audit of 8 gaps in gfx906 MoE kernel coverage vs the dense path. |
| `qwen35-gguf-moe-bridge.md` | GGUF → hipfire MoE pipeline gaps. Unmerged patches for multi-shard loader + arch_id mapping + 3D expert split. |

**Adjacent context:** `mq-sub4bit-prd.md`, `mq-sub4bit-roadmap.prd`, `mq-sub4bit-research-queue.md`, `mq3-rounding-out-precompute-leverage.prd`, `mq3-lloyd-wmma-prefill.md`, `mq-lloyd-batched-prefill-followup.md`, `PR-115-*.md` — earlier sub-4-bit research that informed the HFP4 design but predates it.

---

## Thread 2 — MoE coherence: the `<think>` spiral on Qwen3.6-35B-A3B

**Master doc:** `qwen35-moe-coherence-investigation.md`
**Status:** Investigation complete; PR #228 shipped with workaround flag; engine-side runtime fix pending (Phase 11 plan).

What started as a "validate PR #228 rmsnorm fix" smoke test re-exposed a `<think>` infinite-loop spiral on A3B reasoning. Eleven investigation phases later, root cause is identified as **A3B precision-fragile architecture, not a quantization-format issue.** Quantization-side levers are exhausted; remaining work is runtime-side (sampler intervention, vLLM FP16-router contract, period-N block-attractor detection).

| Doc | Role |
|---|---|
| `qwen35-moe-coherence-investigation.md` | **Master timeline.** Phases 1–11, H1–H4 hypothesis tracker, engine-pass plan revision. Phase 11 falsifies the quant-quality-gap hypothesis via MFP4 smoke test. |
| `qwen35-moe-rmsnorm-fix.md` | PR #228 audit rev3 — the signed-off version of the MoE final-norm GemmaRMSNorm convention fix. |
| `qwen35-moe-precision-vllm-comparison.md` | vLLM cross-reference. Router + `shared_expert_gate` are FP16 in vLLM; `topk_weights` is FP32. Hipfire's Q8 router (`ee1be8a`) is the inverse of vLLM's contract — informs the engine-side runtime fix. |
| `../investigations/2026-05-08-qwen35-rmsnorm-audit/rev-glm5.md` | glm-5's review that caught my first wrong-direction audit and pushed me to the correct diagnosis. |
| `../investigations/2026-05-08-qwen35-rmsnorm-audit/concession.md` | My point-by-point concession to glm-5 with verifiable steps against vLLM / llama.cpp / HF references. |

**Cross-links:**
- The Phase 11 falsification of "format quality fixes spiral" is the load-bearing motivation for moving from "fix MQ4" to "ship HFP4 + ship runtime fix separately" — see Thread 1 §6 closing.
- The vLLM router precision contract (FP16) is a candidate fix-Phase-11-residual lever and links into Thread 1's L5d per-tensor bit allocation work (HFP8E4M3G32 router as one possible implementation).

---

## Thread 3 — gfx906 / Vega 20 perf (MI50 dev box)

**Master doc:** none — this is a multi-front thread spanning prefill, decode, MoE, and family completion.
**Status:** Live. The dev box is gfx906; every kernel-perf change must regress through here.

Hipfire's gfx906 dense path reaches 95% of stock llama.cpp at HFQ4 pp512 (PR #158). MoE has known gaps. DFlash decode has a ~17% dispatch overhead. Three sub-threads:

| Doc | Sub-thread |
|---|---|
| `gfx906-mmq-prd.md` | **Dense prefill** — the PRD that motivated PR #158's MMQ kernel redesign. Largely shipped. |
| `gfx906-moe-kernel-gaps.md` | **MoE prefill + decode** — 8 audited gaps. See Thread 1 Phase B' for the planned closing PR. |
| `dflash-decode-overhead-three-lever.md` | **DFlash decode dispatch overhead** — three independent levers from issue #172. None shipped yet; lever 1's premise migrated from gfx906 to gfx11 post-#158. |

**Adjacent context:** `path_d.md` — DDTree pipelining; relevant on gfx906 because the dispatch-overhead issue compounds with per-step DDTree state mutations on this arch.

---

## Thread 4 — Multi-token prediction (MTP) for Qwen3.5/3.6 dense

**Master doc:** `qwen-mtp-integration.md`
**Status:** Research-only deliverable; no code written. Blocked on PR #228 (now landed) and gated on user prioritization vs other work.

Qwen3.5/3.6 ship native MTP heads — small auxiliary modules that, given the last-layer hidden state + just-sampled token, predict the *next* token without rerunning the full model. vLLM and llama.cpp both consume these as speculative-decoding proposers. The doc scopes the integration: head parsing, runtime wiring into existing spec-decode infrastructure, phased implementation.

| Doc | Role |
|---|---|
| `qwen-mtp-integration.md` | **Master research deliverable** — scope, architecture, risks, phases. Hard prerequisite (PR #228 MoE final-norm fix) now satisfied. |

**Cross-links:** consumes hidden states whose magnitude was corrected by Thread 2's PR #228. MTP integration is a future Thread 2 follow-up, but logically independent of the A3B spiral (MTP targets dense models; A3B is MoE).

---

## Thread 5 — Speculative decoding (DFlash / DDTree / PFlash) infrastructure

**Master doc:** none — speculative decoding has several independently-tracked variants.
**Status:** Live, multi-track.

Hipfire's speculative-decoding family includes:
- **DFlash** — drafter-flush + spec-decode. Mature; ships in master.
- **DDTree** — drafter draft-tree. Mature; ships.
- **PFlash** — speculative-prefill (faster prefill via partial-FA + verify). Earlier-stage.

| Doc | Sub-thread |
|---|---|
| `dflash-decode-overhead-three-lever.md` | DFlash dispatch overhead (Thread 3 overlap). |
| `ddtree-path-c-main-path-first-from-lucebox.prd` | DDTree path C — main-path-first lazy FA re-verify. |
| `pflash-speculative-prefill.prd` | PFlash design. |
| `hetero-pflash-dflash.prd` | Combined PFlash + DFlash hetero pipeline. |
| `path_d.md` | DDTree pipelining (issue #38). |

**Adjacent context:** MTP integration (Thread 4) consumes the same spec-decode hooks; logically these threads compose.

---

## Thread 6 — Multi-GPU pipeline-parallel (PP)

**Master doc:** `multi-gpu-pp.md`
**Status:** Stages 1–9 implemented on `feat/multi-gpu-pp` (external contributor alpineQ).
**Target:** v0.2.0 release.

Adjacent context: `docs/multi-gpu.md` (user-facing reference). Logically independent of the other threads; doesn't intersect Thread 1's format work or Thread 2's coherence investigation.

---

## Thread 7 — CDNA / MI300X port (calibration optimization)

**Master doc:** `cdna-calibration-optimization.prd`
**Status:** Speculative; depends on whether MI300X hardware becomes available for development.

Cross-references the format roadmap (Thread 1) — CDNA1 (gfx906) acceleration is analyzed in `qwen35-mq4-quality-gap.md` §1.4; CDNA2/3 (gfx908/gfx90a/MI300X) would extend to MFMA-INT8 if hardware appears.

---

## Thread 8 — Tokenizer hot-path correctness & hardening

**Master doc:** none — work happens directly in PRs (#201, #202, #203, #226, #229, #230) without a standalone plan doc.
**Status:** Live. BPE O(N²) → O(N log N) shipped (#201); merge_rank cache shipped (#226); SentencePiece allocator-free scan in flight (#229); interned merge symbols + loud OOV at construction in flight (#230).

The tokenizer has been under sustained attack since #201 found the BPE O(N²) bug that caused minute-long prefill stalls on long prompts. Three follow-up PRs split the residual work along orthogonal axes:

| PR / commit | Subject | Status | Closes |
|---|---|---|---|
| #201 (upstream-merged, in tree) | BPE encoder O(N²) → O(N log N) | shipped | — |
| #226 (merged commit `743e23d`) | BPE `merge_rank` cache on Tokenizer (37× TTFT on long prompts) | shipped | half of #202 |
| #229 (commit `48d985f`, open as PR) | SentencePiece allocator-free scan + `merge_rank` → `merge_pair_rank` rename + sp_tests module | in flight | remainder of #202 |
| #230 (commit `8b3b32c`, open as PR) | Interned merge symbols (`Vec<(String,String)>` → `Vec<u32>` rank-ordered ids) + loud OOV (`MissingByteSymbol`/`MissingMergeOperand`/`MissingMergeResult` errors at constructor) | in flight | BPE half of #203 |
| (commit `350131f`) | Combined review-round addresses (F1/F3/F4/F7/GLM-tok-2/G-tup) | shipped | review follow-ups |

**Open correctness gap explicitly deferred:** the **SentencePiece half of #203** — `encode_sentencepiece`'s single-char fallback silently *dropping* missing chars — is **not addressed** by #230. PR #230's body recommends leaving #203 open until both halves land, or splitting it into a separate SP-specific issue. Likely shape: `encode_strict` returning `Result<Vec<u32>, EncodeError>`.

**Memory note for future agents:** `project_upstream_pr201_bpe.md` in user-memory ("fork is 41 commits behind upstream on BPE fix") is **stale**. The BPE O(N²) fix landed via #226 (merge_rank cache) on this branch; long-prompt prefill should no longer stall in tokenizer.

---

## Thread 9 — Attention numerics correctness & perf

**Master doc:** none — distributed across PRs and inline comments.
**Status:** Live. PR #222 (open) addresses two issues in `attention_dflash_f32`. The 1-ULP softmax-renorm attractor in `qwen35.rs:1996-2003` is comment-only; period-N block-attractor detection from Thread 2 Phase 11 has no design doc.

| PR / commit | Subject | Status | Impact |
|---|---|---|---|
| #222 (open as PR, NOT yet in our tree) | `attention_dflash_f32`: tiled online-softmax + full-workgroup V-accumulation in Phase C | **pending merge** | (1) **Correctness fix at L ≥ 16128** — pre-fix kernel allocates `(L + block_size) * 4` bytes of LDS, blows the 64 KB gfx1100 per-WG limit at L≥16128; crashes hetero PFlash+DFlash at the canonical 16K bench. (2) **+41% kernel time** (130 ms → 77 ms) from full-workgroup V-accumulation in Phase C when nthreads > head_dim (128 lanes idle in the old loop). |

**Why this matters for the rest of the index:**
- PR #222 is a hard prerequisite for the hetero PFlash+DFlash work in Thread 5 (`hetero-pflash-dflash.prd`). Without it, hetero canonical 16K bench crashes.
- The attractor work in Thread 2 Phase 11 (period-N block-attractor detection) is a *different* attention-numerics issue at the per-token logit level; PR #222 is a kernel-side LDS/correctness fix at the kv-length level. The two are not redundant.
- The 1-ULP softmax-renorm attractor (`qwen35.rs:1996-2003`) sits in a third category — quantization-induced softmax precision drift. Currently mitigated by a comment-only sentinel. Should grow a design doc when imatrix calibration (Thread 1 Phase A L5c) lands, because activation-weighted LS may modify the precision profile that the existing mitigation assumes.

---

## Topics still not tracked (candidates for future threads)

After absorbing Threads 8 and 9 above, the remaining gaps:

| Candidate topic | Why it might matter | Where would it live |
|---|---|---|
| **KV-cache quantization audit** | hipfire ships asym3, asym4, q8, fp16, fp32 KV variants. No equivalent "what's missing / next levers" doc exists; it's all in commit messages. | `docs/plans/kv-cache-quant-*.md` |
| **Sampler / decoding strategy** | All sampling code is currently kernel-side or in `crates/hipfire-runtime/src/sampler.rs`; no design doc tracks min_p, repetition penalty, mirostat status. Thread 2 Phase 11's "sampler intervention" lever is also undocumented. | `docs/plans/sampling-strategy.md` (if needed) |
| **Cross-vendor compute deprecation rationale** | CLAUDE.md says Vulkan/RADV is out of scope (issue #44 closed 2026-04-25). No standalone doc explains the rationale for future contributors. | `docs/plans/cross-vendor-rationale.md` |
| **Build / packaging / distribution** | No PRD covers the long-term packaging story (cargo dist, deb packages, Docker, etc.). May not be needed in the design tree at all. | `docs/distribution.md` |
| **Tracing / observability** | hipfire daemon emits JSONL events. No doc tracks the schema, no metrics-format integration plan exists. | `docs/plans/observability.md` (if needed) |

KV-cache audit is the highest-value remaining gap — it directly affects perf *and* coherence (Q8 drift on DeltaNet has its own mitigation in `qwen35.rs` per CLAUDE.md "DeltaNet state: FP32 (MoE model — Q8 drift mitigation)").

---

## How to add a new thread

1. If the new doc is the *master* for its thread, give it a self-explanatory filename in `docs/plans/`.
2. If it's a supporting doc (audit, review, sub-PRD), use a descriptive filename and link from the master.
3. Add a row to this index under the relevant thread, or open a new thread.
4. When the thread ships: move the docs to `docs/investigations/<date>-<topic>/`, leave a one-line shipped-in-PR note here.

## How to find what you need

- "Why is A3B spiraling?" → Thread 2 (`qwen35-moe-coherence-investigation.md`)
- "Why does hipfire use HFP4?" → Thread 1 (`qwen35-mq4-quality-gap.md`)
- "What's missing on gfx906?" → Thread 3 (start with `gfx906-moe-kernel-gaps.md` for MoE; PR #158 changelog for dense)
- "How do we add Gemma?" → Thread 1 §4.3 (`qwen35-mq4-quality-gap.md`)
- "Should we calibrate with imatrix?" → Thread 1 Phase A (`qwen35-mq4-quality-gap.md` §5)
- "When does the format change?" → It doesn't. Thread 1 commits to HFP4 for the next several years; all extension is inside reserved bits.
- "Why is long-prompt prefill slow?" → Should no longer be — Thread 8 (#226 shipped the BPE `merge_rank` cache, 37× TTFT). If still slow, check PR #229/#230 status.
- "Why does encoder silently drop chars?" → Thread 8. BPE half closed by #230 (loud OOV); SP half still open in #203.
- "What does PR #222 do?" → Thread 9. Fixes `attention_dflash_f32` LDS overflow at L ≥ 16128 (crash → correct) + 41% kernel time win. Prerequisite for hetero PFlash+DFlash (Thread 5).
