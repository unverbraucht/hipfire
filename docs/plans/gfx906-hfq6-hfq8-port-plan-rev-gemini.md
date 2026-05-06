# Adversarial Review: gfx906 HFQ6 + HFQ8 Port Plan

**Reviewer:** Gemini CLI
**Date:** 2026-05-06
**Subject:** Analysis of `docs/plans/gfx906-hfq6-hfq8-port.md`

## 1. Executive Summary of Risks

The plan to port HFQ6 and HFQ8 to gfx906 (MI50) accurately maps the coverage gaps but contains several "hidden" implementation hurdles that will likely blow out the estimated session counts. While the hardware analysis is technically sound (including the presence of `v_dot8_i32_i4`), the transition from 4-bit (dword-aligned) to 6-bit (non-aligned) weights introduces a non-linear increase in complexity.

Key findings:
- **The "Unpack Penalty":** HFQ6's 6-bit-to-8-bit unpack is ~2.5x more ALU-intensive than HFQ4's nibble unpack. On the ALU-constrained gfx906, this risks erasing the gains from dp4a.
- **Alignment and Prefetch Complexity:** HFQ6 uses 6 bytes per thread (8 weights). This breaks dword alignment for weight loads, making the "mostly-mechanical" prefetch port (Phase A) a significant VGPR-heavy redesign involving split-load handling.
- **LDS Bank Conflicts:** A 6-byte stride in the MMQ path (Phase C) is a "worst-case" scenario for LDS bank conflicts, potentially requiring padding that reduces effective LDS capacity.
- **Accuracy Concerns:** The reconstruction term `(zp + 32 * sc) * sum_x` for HFQ6 amplifies quantization noise in the `sum_x` term significantly more than the 4-bit variant.
- **Wavefront Underutilization:** The existing HFQ6 kernels are wave32-encoded, which run at 50% throughput on gfx906 hardware. Phase A correctly targets this, but the lift might be capped by dispatch overhead.

---

## 2. Technical Deep Dive

### 2.1 The dp4a "Lift" on HFQ6/HFQ8
The plan correctly identifies `v_dot8_i32_i4` as an available instruction on gfx906, but correctly notes it would require a lossy repack for HFQ8. 
1. **Instruction Set:** Verified that `v_dot8_i32_i4` is present on gfx906 (Vega 20/MI50).
2. **HFQ6 ALU Bottleneck:** Unpacking 4 weights from 3 bytes into `int8` lanes for `v_dot4` requires a complex sequence of shifts and masks. On gfx906, which lacks the high ALU-to-memory ratio of newer CDNA architectures, this "Unpack Penalty" could make the dp4a path **slower** than the FP path.
3. **HFQ8 Strategy:** The plan dismisses dp4a for HFQ8, but ignores the **activation bandwidth** win. Switching to Q8_1 activations would reduce X-traffic by 4x. Even if the math is FP32, the memory win might justify a "pseudo-MMQ" path for HFQ8.

### 2.2 Alignment and VGPR Pressure (Phase A)
The plan describes Phase A as "mostly-mechanical." 
- **The 6-Byte Problem:** HFQ6 weights are 6 bytes per thread. Standard vectorized loads (`dwordx2`, `dwordx4`) will always be misaligned for some threads in the warp. 
- **Prefetch Complexity:** Implementing a software pipeline (like `gemv_hfq4g256_residual_wave64_prefetch`) with 6-byte loads requires staging more VGPRs to handle the split-dword logic. This could drop occupancy from 8 waves to 4-5 waves, negating the prefetch benefit.

### 2.3 LDS Bank Conflicts in MMQ (Phase C)
The HFQ6 MMQ port is the highest risk.
- A 6-byte per-thread stride in LDS is a guaranteed source of 2-way bank conflicts (32 banks, 4-byte width).
- **Mitigation Cost:** To avoid conflicts, weights must be padded to 8 bytes in LDS, increasing LDS footprint by 33%. This might limit the `mmq_x` batch size or increase the number of syncs required, hitting the "sync-stalling" regime of gfx906.

### 2.4 Accuracy: The Reconstruction Term
For HFQ6 (range 0-63), the midpoint shift is 32.
- `acc += sc * dx * sumi + (zp + 32 * sc) * sum_x`
- Errors in the `sum_x` term (accumulated quantized activations) are amplified by `32 * sc`.
- In long-context or high-dynamic-range blocks, this could lead to divergence that was not present in the 4-bit (shift-8) implementation. This requires rigorous NRMSE validation against a FP16 reference before release.

### 2.5 Leveraging gfx906 Specialized ISA
The plan should explicitly target gfx906-specific instructions to mitigate the dequantization overhead:
1. **`v_perm_b32` for HFQ6 Unpacking:** This is the "secret weapon" for 6-bit unpacking. It allows for arbitrary byte-shuffling and can be used to align the 6-byte blocks into dword-friendly structures before bit-extraction, potentially reducing the ALU cost of Phase B.
2. **`v_dot8_i32_i4` for HFQ4/HFQ8:** While HFQ8 is already 8-bit, `v_dot8` (int4x8) could be utilized if weights are repacked. However, for asymmetric HFQ8, the standard `v_dot4_i32_i8` remains the most robust choice for Q8_1 activations.
3. **`v_add_lshl_u32` and `v_and_or_b32`:** These GCN3/GFX9+ specific "triple-operand" instructions should be used to fuse the mask+shift logic of the 6-bit unpacker, keeping the ALU pipeline full without increasing VGPR pressure.

---

## 3. Review of "Sessions" and Priority

| Phase | Estimated | Reviewer Assessment |
|---|---|---|
| HFQ6 Phase A | 2 sessions | **3.5 sessions** (Non-aligned prefetch is hard) |
| HFQ8 Phase A | 1.5 sessions | **1 session** (Aligned, should be trivial) |
| HFQ6 Phase B | 1 session | **2 sessions** (ALU-overhead tuning) |
| HFQ6 Phase C | 2-3 sessions | **5 sessions** (LDS padding + sync tuning) |

**Recommendation Changes:**
1. **Prioritize HFQ8 Phase A:** This is the lowest-hanging fruit. Because HFQ8 is 8-bit aligned, the wave64 port will be significantly easier and provide a "clean" baseline for wave64 gains.
2. **PMC Gate for Phase B:** Do not implement the HFQ6 dp4a path unless the Phase A wave64 FP kernel shows >40% "ALU-limited" profile in rocprof. If it's purely memory-bound, the dp4a unpack overhead will be "free," but if it's already compute-stalled, dp4a will regress.
3. **Restructure MMQ Phase C:** Consider a "DWord-Padded" LDS layout for HFQ6 weights during the MMQ staging to avoid bank conflicts at the cost of capacity.

---

## 4. Final Verdict

The plan is technically sophisticated but underestimates the **"Alignment Friction"** of HFQ6. The 6-byte granularity of HFQ6 is fundamentally less "friendly" to GCN/CDNA hardware than the 4-byte/8-byte granularity of HFQ4/HFQ8. 

**Revised Guidance:** Start with HFQ8 to prove the wave64 lift, then tackle HFQ6 Phase A with a focus on VGPR-efficient split-load handling. Treat HFQ6 Phase C as a research project rather than a direct port.
