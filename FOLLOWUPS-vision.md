# Vision pipeline — pre-existing follow-ups

Surfaced during the May 2026 vision review (see `vision_rev_claude.md`,
`vision_rev_gemini.md`, `vision_rev_glm5.md`) but **out of scope** for the
`fix/vision-parity` branch (commits `b47ba99a`, `8c415626` and follow-ups).
These are either pre-existing concerns inherited from `d6db6ae8` (the OpenAI
vision wire-format PR) or deeper architectural items that need their own
design pass.

Each item is sized roughly. File one ticket per group, or one umbrella issue
referencing this doc.

---

## A. Daemon / generate_vl hygiene (from `d6db6ae8`)

### A1. `max_think_tokens` think-pair tracker uses UTF-8 string search

**Location:** `crates/hipfire-runtime/examples/daemon.rs:4739-4747`
**Source:** GLM-5 review item A3.

The think-budget tracker decodes all `streamed_tokens` to a UTF-8 string via
`tokenizer.decode_bytes(&streamed_tokens)` every iteration and does
`rfind("💭")` / `rfind("_FACTORY_END_")` on the raw string. This is:

1. **O(N²)** — re-decoding all streamed tokens each step (300×300 ≈ 90K
   decodes for a 300-token think block).
2. **Fragile** — relies on the specific UTF-8 byte sequence of the emoji
   thinking markers. If a future Qwen variant tokenizes them differently,
   `rfind` silently breaks and thinking-mode enforcement stops working.

**Fix:** track think depth via token IDs (`think_pair` is already available).
The text-only `generate()` path may have the same pattern; audit both.

**Size:** ~half-day.

### A2. `generate_vl` is 381 lines, mixes streaming + think-enforcement + ngram + attractor + decode

**Location:** `crates/hipfire-runtime/examples/daemon.rs:4444-4824`
**Source:** GLM-5 review item D5.

Extracting the decode loop into a helper would improve readability and make
future changes (e.g. multi-image, video) less invasive.

**Size:** ~1 day refactor + careful testing.

### A3. Early returns in `generate_vl` could leak GPU resources if anyone adds pre-return allocations

**Location:** `crates/hipfire-runtime/examples/daemon.rs:4484, 4495, 4506, 4539, 4590`
**Source:** GLM-5 review item C2.

At these `write_error` + `return` points, no GPU tensors have been allocated
within the function — safe today. But if anyone refactors to pre-allocate
anything before these points, the early returns become leaks.

**Fix:** add a comment documenting the "no GPU allocation before this point"
invariant, OR refactor to a single exit point with `?`-style error
propagation.

**Size:** ~30 min (comment) or ~1 day (refactor).

### A4. Daemon's `assistant_prefix` is hardcoded

**Location:** `crates/hipfire-runtime/examples/daemon.rs:990` (noted as follow-up in the file).
**Source:** GLM-5 review item D3.

The VL token IDs (`<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`) are
correctly resolved from the tokenizer at runtime. `assistant_prefix` follows
the same pattern but is hardcoded — apply the runtime-lookup fix to that
string too.

**Size:** ~1 hour.

---

## B. CLI image handling (from `d6db6ae8`)

### B1. `https://` image URLs are silently dropped

**Location:** `cli/index.ts:1530`
**Source:** GLM-5 review item C1; `docs/plans/completions_vision.md §Postponed`.

`extractContent` only processes `data:` URLs. `https://` URLs are silently
ignored — the request proceeds as text-only with no indication to the client.

**Fix:** return `rejectImage("https image URLs not supported — use data: URIs")`
instead of silently dropping. Defer the fetch implementation itself.

**Size:** ~30 min.

### B2. Multi-turn VL rejection is a correctness landmine

**Location:** `cli/index.ts:1700-1703`
**Source:** GLM-5 review item B4.

The CLI rejects images in non-last user turns. The daemon resets per-request
(`cli/index.ts:1344`), so there's no multi-turn state to worry about today.
But: if anyone removes the per-request reset (e.g., to add text-only
multi-turn caching), the image-in-last-turn-only constraint silently breaks.

**Fix:** add a comment at the daemon's reset call documenting the assumption
the CLI relies on. Better: add a server-side enforcement.

**Size:** ~30 min (comment) or ~half-day (server enforcement).

### B3. Large image base64 strings spike memory

**Location:** `cli/index.ts:906-913`
**Source:** GLM-5 review item C4.

For a 30 MB raw photo, the base64 string is ~40 MB, then serialized into JSON
body. Hits the daemon's `MAX_BASE64_ENCODED_LEN` of 40 MB. Memory spike is
worth noting; not currently a bug.

**Fix:** consider streaming/chunked upload for VL eventually.

**Size:** not urgent.

---

## C. Vision pipeline architecture

### C1. 189 GPU alloc/free pairs per image in `vision_forward`

**Location:** `crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs::vision_forward` per-layer loop.
**Source:** GLM-5 review item B2; my review item 16 (partial).

7 GPU tensors (`tmp`, `qkv`, `attn_out`, `proj`, `tmp2`, `fc1`, `fc2`) × 27
vision layers = 189 alloc/free round-trips per image. The pool reduces real
cost, but it's still bookkeeping overhead and means the vision path is
allocation-dominated rather than compute-dominated for the framing case.

**Fix:** pre-allocate scratch buffers at weight-load time (sized to the
largest possible image at `max_pixels`) and reuse across layers. Also
naturally eliminates the implicit single-stream-invariant dependency (see
the comment added in this branch at the layer-loop head).

**Size:** ~1 day. Touches `VisionWeights` storage shape.

### C2. `pos_embed` upload per image — bottleneck for batched/video

**Location:** `crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs::vision_forward`
**Source:** GLM-5 review item B5; Gemini G8; my review item 16.

Today: CPU pos_embed table is ~10 MB. Every `vision_forward` interpolates a
fresh `(n, h)` slice and uploads it. For single-image requests this is fine.
For batched VL or video processing the upload becomes a real overhead.

**Fix:** when batching arrives, either (a) pre-allocate a max-cap-sized GPU
buffer and only upload the interpolated slice (current behavior — fine), or
(b) move bilinear sampling onto the GPU and keep the full table resident.

**Size:** deferred until batching/video is on the roadmap.

### C3. `vit_attention_opt` is now dead code on the vision path

**Location:** `crates/rdna-compute/src/dispatch.rs::vit_attention_opt`
**Source:** my review item 17.

`vision_forward` calls `vit_attention_f32` (the simple variant), not
`vit_attention_opt`. The optimized variant exists but doesn't apply 2D
rotary — so it can't be plugged in without first adding rotary to it.

**Fix:** either delete `vit_attention_opt` or extend it to take cos/sin and
apply rotary inline (would also fuse the kernel launch — a small win).

**Size:** ~half-day either way.

### C4. No VL output coherence gate

**Location:** `scripts/`
**Source:** GLM-5 review item C6; my review item 7.

Text decode has `coherence-gate.sh` + `coherence-gate-dflash.sh`. VL has
nothing equivalent — the `benchmarks/vision/run_bench.sh` comparison is
manual and not part of CI. Given that the vision pipeline already shipped
**one silent quality regression** (the R/B/G swap + patch ordering bugs), a
VL coherence gate would have caught the symptom much earlier.

**Fix:** add `coherence-gate-vl.sh` that runs the daemon on a fixed prompt +
image fixture and asserts:
- non-zero token output (hard fail on panic/zero).
- unique-token-ratio on first/last 128 tokens (catches attractors).
- substring presence of expected keywords on a known reference image
  (e.g., "Shiba" / "Doge" on `doge.jpeg`, "Louvre" on `scene_1.jpg`).

**Size:** ~1 day to wire up, mostly mirroring the existing
`coherence-gate-dflash.sh` pattern.

### C5. Pure-color channel-order tests are decorative

**Location:** `crates/hipfire-arch-qwen35-vl/tests/channel_order.rs`
**Source:** my review item 10.

Three of the four tests use pure colors `(255,0,0)`, `(0,255,0)`, `(0,0,255)`.
These cannot detect a transpose or a non-swap permutation — only a channel
swap. Only `mixed_pixel_keeps_rgb_order` is a real channel-order test.

**Fix:** delete the three pure-color tests OR repurpose them to assert
specific output indices (e.g., that channel-0 byte at pixel (0,0) doesn't
equal the channel-1 byte for a `(10, 200, 50)` image).

**Size:** ~30 min. Low priority — they're not actively harmful, just not
load-bearing.

### C6. Resize-filter precision: still 0.002 → 9.2e-5 residual

**Location:** `crates/hipfire-arch-qwen35-vl/src/image.rs::preprocess_dynamic_image`
**Source:** my review item 5 (now resolved in this branch; remaining residual is a precision item).

The branch swapped `FilterType::Triangle` → `FilterType::CatmullRom`,
dropping rel-L1 vs HF BICUBIC from 0.002 to ~9.2e-5. That's "very close —
likely just precision" per `diff_dumps.py`. Further closure would require
matching PIL's exact BICUBIC kernel weights, which the `image` crate
doesn't expose — would need a custom resize. Not worth doing unless
benchmark regression points at it.

**Size:** not urgent.

---

## D. Cross-cutting infrastructure

### D1. ~~Three pre-existing `bind_thread` misses in `dispatch.rs::pflash_score_*`~~ — RESOLVED

**Resolution:** Fixed in the vision-parity branch as a tiny prep commit
(see `fix/324-vision-parity`). The 3 wrappers delegate to
`pflash_score_fwht_kv_impl` which already calls `bind_thread()`, so the
correct fix was the documented `// bind_thread: skip — delegates to …`
marker, not a redundant binding call. `./scripts/verify-bind-thread.sh`
now reports OK and the pre-commit hook no longer requires `--no-verify`
on dispatch.rs touches.

### D2. `smart_resize` integer-overflow defense

**Location:** `crates/hipfire-arch-qwen35-vl/src/image.rs::smart_resize`
**Source:** GLM-5 review item B3.

`(height * width) as f64` can overflow `usize` on 32-bit targets for very
large images. The `from_bytes` path has a dimension-bomb guard; the path
variant relies on `image::open()` doing its own check. Defense-in-depth
rather than an active bug.

**Fix:** add a saturating multiplication or assert on `height.checked_mul(width)`.

**Size:** ~30 min.

### D3. `--no-verify` policy lives only in personal memory

**Location:** project memory `feedback_speed_gate_local_offset.md` (not in-repo)
**Source:** my review item 20.

The reasoning "vision-only change doesn't affect text-decode tok/s, speed-gate
is N/A on this local box where the maintainer's baseline is +10%" lives in a
personal memory file. Future contributors won't have that context.

**Fix:** codify the `--no-verify` policy in `CLAUDE.md` or
`CONTRIBUTING.md`, OR make the speed-gate itself smarter (skip when no
hot-path file is staged).

**Size:** ~1 hour (doc), or ~half-day (smarter gate).

---

## How to use this list

The branch is otherwise green: kernels work, bench is 12/12, rel-L1 vs HF is
~9e-5. The items above are quality, hygiene, and follow-up work that
**doesn't block** merging this branch — but should be tracked so they don't
get lost.

Suggested triage:

- **Open as separate tickets immediately:** D1 (5-min fix that unblocks future
  dispatch work), C4 (VL coherence gate — direct response to the regression
  class that motivated this branch).
- **Roll into a "vision pipeline cleanup" umbrella ticket:** A1–A4, B1–B2,
  C1, C3, C5.
- **Defer until use case appears:** B3, C2, D2 (large-image / batched-VL
  optimizations).
- **Codify:** D3 (project documentation).
