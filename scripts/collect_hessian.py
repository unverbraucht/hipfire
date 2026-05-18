#!/usr/bin/env python3
"""collect_hessian.py — calibration-time Hessian collector for Stage B GPTQ.

Phase A Stage B Phase 1.1 per `docs/plans/gptq.md` v2.

`crates/hipfire-runtime/examples/imatrix_collect.rs` is a llama.cpp subprocess
wrapper (no native forward pass), so we use HF transformers + ROCm PyTorch here
for the calibration forward pass. The output Hessian binary is consumed by
`crates/hipfire-quantize/src/gptq.rs` (Phase 2) at quantize time.

Loads a BF16 model, runs a forward pass on calibration tokens, registers
`nn.Linear` forward hooks that accumulate per-tensor outer products
`H_t = (1/N) * sum_t x_t · x_t^T`, writes to a sidecar binary file.

Calibration subset matches the GPTQ paper's scale: 128 sequences × 2048
tokens = 262144 calibration tokens. The Hessian is a smoothed expectation
that converges well at this scale; tokenizer disagreement with hipfire's
tokenizer is averaged out (per `docs/plans/gptq_plan_rev_synthesis.md`
Topic 6).

Tensor coverage: all `nn.Linear` modules whose name matches the GPTQ
whitelist (mirrors the quantizer's `awq_eligible` plus the non-AWQ MQ4
projections we want to GPTQ-quantize: o_proj/out_proj/down_proj).

Output file format (HFHS — Hipfire Hessian Sidecar, v1):

  Header (24 bytes):
    [0..4)    magic     u8[4]  = b"HFHS"
    [4..8)    version   u32_le = 1
    [8..16)   n_tensors u64_le
    [16..24)  reserved  u64_le = 0

  Per-tensor record (variable length):
    [0..4)              name_len    u32_le
    [4..4+name_len)     name        utf8 bytes
    [4+name_len..+4)    expert_idx  u32_le   (default 0; reserved for MoE)
    [+4..+4)            K           u32_le
    [+4..+4)            dtype_flag  u32_le   (1=F32, 2=F64)
    [+4..+(K*K*sz))     payload     K*K * sizeof(dtype) bytes (row-major,
                                                                native endian)

Hessian values are float32 by default. The quantizer promotes to FP64 for
Cholesky.

Usage:
    .venv/bin/python3 scripts/collect_hessian.py \\
        --model    <hf-model-id-or-path> \\
        --output   <path-to-out.hessian.bin> \\
        [--n-sequences 128] \\
        [--ctx-len    2048] \\
        [--corpus     <hf-dataset-id-or-text-file>] \\
        [--device     cuda] \\
        [--dtype      bfloat16]

Examples:
    # Qwen3.5-9B, 128 sequences × 2048 ctx from wikitext-2 train
    .venv/bin/python3 scripts/collect_hessian.py \\
        --model /data/cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202.../ \\
        --output ~/.hipfire/refs/qwen3.5-9b-bf16.hessian.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Tensor name patterns that should get GPTQ Hessians. Matches hipfire's
# AWQ whitelist plus the non-AWQ MQ4 projections (o_proj/out_proj/down_proj).
# Pattern check is done on the FULL module name (e.g.
# "model.layers.0.self_attn.q_proj"); we match against the last 2-3 components.
GPTQ_TARGET_SUFFIXES = (
    # Attention input projections
    "q_proj", "k_proj", "v_proj", "qkv_proj",
    # Attention output (no AWQ but yes GPTQ)
    "o_proj",
    # MLP
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
    # Linear-attention (Gated DeltaNet)
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
    "out_proj",
    # MoE router
    "gate",
    # Final logits projection — added 2026-05-18 for gptq_lm_head_awq.md
    # Phase 1. lm_head (HF dense + multimodal naming) and `output` (GGUF
    # twin). Only matters when --lm-head-format mq4-awq is selected at
    # quantize time; the Hessian is collected unconditionally because the
    # cost is small (one extra K×K FP64 matrix, K=hidden) and lets us
    # decide at pack time whether to use it.
    "lm_head", "output",
    # Vision encoder (Qwen3.5/3.6 VL) — added 2026-05-18 for
    # gptq_lm_head_awq.md Phase 3 quant-side prep. Names match the
    # visual tower's attention QKV / attention output / MLP first &
    # second linears, plus the merger MLPs. Catches modules whose last
    # path segment is one of these literals — e.g. `model.visual.blocks.<N>.
    # attn.qkv`, `model.visual.blocks.<N>.mlp.linear_fc1`,
    # `model.visual.merger.linear_fc2`. AWQ eligibility for vision is
    # NOT yet enabled (the runtime AWQ-aware vision kernels are the
    # gating dependency — see plan §3.3). Adding Hessian coverage now
    # lets a single Stage B pass also support future vision AWQ + GPTQ
    # work without a second 5-10h re-collection.
    "qkv",         # visual attn fused QKV  (last segment of .attn.qkv)
    "proj",        # visual attn output projection  (last segment of .attn.proj)
    "linear_fc1",  # visual MLP fc1 (blocks AND merger)
    "linear_fc2",  # visual MLP fc2 (blocks AND merger)
)


def is_gptq_target(name: str) -> bool:
    """Returns True if this module name should have a Hessian collected."""
    last = name.rsplit(".", 1)[-1]
    if last in GPTQ_TARGET_SUFFIXES:
        return True
    # `mlp.gate.weight` (MoE router) — matched by trailing ".gate" — but we
    # need to disambiguate from `gate_proj`. The check above on the literal
    # last segment handles both.
    return False


def _extract_layer_idx(mod_name: str) -> int | None:
    """Parse the layer index from a transformer module name.

    Recognizes the conventional `.layers.<N>.` infix in both HF
    safetensors naming (`model.language_model.layers.0.linear_attn.q_proj`)
    and the MTP head's `mtp.layers.0.self_attn.q_proj`. Layers in MTP
    are treated as a separate index space; we return `None` to lump
    them into the "always-included" bucket (they're a fixed small set
    and their tensors fit in RAM trivially).

    Returns `None` for modules without a recognizable layer infix
    (lm_head, embed_tokens, top-level norms, vision encoder).
    """
    parts = mod_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _safetensors_keys(model_path: str) -> set[str]:
    """Returns the set of all tensor keys stored in the safetensors files,
    stripped of `.weight` suffix to match against module names.

    Used to translate HF's flattened in-memory module names (e.g.
    `model.layers.0.q_proj` for Qwen3.5 ForCausalLM) to the canonical
    safetensors / .hfq naming (`model.language_model.layers.0.q_proj`).
    """
    import json
    p = Path(model_path)
    keys: set[str] = set()
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        with idx.open() as f:
            data = json.load(f)
        keys = set(data["weight_map"].keys())
    else:
        # Single-file safetensors path
        from safetensors import safe_open  # type: ignore[import-not-found]
        for st_file in p.glob("*.safetensors"):
            with safe_open(st_file, framework="pt") as f:
                keys.update(f.keys())
    # Strip `.weight` so the keys align with the module-name basis (modules
    # without `.weight` such as norms/biases are not GPTQ targets anyway).
    return {k.removesuffix(".weight") for k in keys if k.endswith(".weight")}


def _translate_to_stored_name(mod_name: str, stored_keys: set[str]) -> str | None:
    """Find the canonical safetensors-stored name for a hooked module.

    Strategy: longest-suffix match. If `model.layers.0.q_proj` is the
    in-memory name, look for any stored key ending in
    `.layers.0.q_proj` — and pick the longest match (typically
    `model.language_model.layers.0.q_proj` for Qwen3.5 ConditionalGen).

    Returns the stored name (without `.weight`) on success, or None if no
    suffix-matching key exists in the safetensors (which would mean the
    module is in-memory only and not stored — unusual; skip with a warning).
    """
    # Strip the leading "model." (common prefix in both naming conventions).
    # We match on the trailing portion after the FIRST "." — typically
    # ".layers.N.<...>" — to be robust against the language_model. insertion.
    if mod_name in stored_keys:
        return mod_name
    # Build a candidate suffix by stripping a single leading component
    # ("model.") — most common HF wrapper pattern.
    trail = mod_name.split(".", 1)[1] if "." in mod_name else mod_name
    # Find any stored key whose suffix matches.
    matches = [k for k in stored_keys if k.endswith(trail)]
    if not matches:
        return None
    # If multiple, pick the longest (most fully-qualified — e.g. prefers
    # `model.language_model.layers.0.q_proj` over a hypothetical bare
    # `layers.0.q_proj`).
    matches.sort(key=len, reverse=True)
    return matches[0]


class HessianAccumulator:
    """One per nn.Linear we hook. Accumulates H = sum_t x_t · x_t^T (FP32, K×K).

    Two backing-store modes:
      - **in-RAM** (default): `H = np.zeros((K, K), dtype=np.float32)`. Fast,
        no I/O cost per `update`. Suitable for ≤9B-class models where the
        sum of all per-tensor Hessians fits comfortably in system RAM
        (9B's 248 entries total ~33 GB, fits in 32 GB tight; 4B fits easily).
      - **memmap** (use `accumulator_dir`): each Hessian is a
        `np.memmap` file at `<accumulator_dir>/{escaped_name}.f32.bin`.
        Updates go through the OS page cache; resident DRAM stays bounded.
        Required for 27B-class models where the full accumulator set is
        126 GB (FP32) — 4× system RAM. Local SSD typical write throughput
        (~500 MB/s on the md0 RAID) makes the per-token overhead ~1 ms.
        NFS-backed memmap is workable but throttles by ~10× (~10 ms/token).

    Promoted to FP64 only at the quantizer for Cholesky (per plan v2 §2).
    """

    def __init__(self, name: str, K: int, accumulator_dir: Path | None = None) -> None:
        self.name = name
        self.K = K
        self.n_tokens = 0
        self.accumulator_dir = accumulator_dir
        if accumulator_dir is None:
            # In-RAM accumulator (current behavior for ≤9B).
            self.H = np.zeros((K, K), dtype=np.float32)
            self._mmap_path: Path | None = None
        else:
            # Memmap-backed accumulator. File-per-tensor; OS handles paging.
            # Escape name → filename: replace dots/slashes for filesystem safety.
            safe_name = name.replace("/", "_").replace(".", "_")
            self._mmap_path = accumulator_dir / f"{safe_name}.f32.bin"
            self._mmap_path.parent.mkdir(parents=True, exist_ok=True)
            # mode='w+' zeros the file at open. Total bytes = K * K * 4.
            self.H = np.memmap(
                self._mmap_path,
                dtype=np.float32,
                mode="w+",
                shape=(K, K),
            )

    def update(self, x_2d: np.ndarray) -> None:
        """x_2d: shape [num_tokens, K] in FP32 on CPU.

        We use BLAS gemm via numpy: H += x.T @ x. Single-precision; the
        rounding error per accumulation is ~1 ULP per entry per token —
        across 262k tokens, accumulated error stays well below the
        damping λ that GPTQ adds anyway.

        For memmap-backed `self.H`, the `+=` is a read-modify-write of
        K² FP32 values per update. OS page cache buffers; explicit
        `H.flush()` happens once at `finalize` time (not per update).
        """
        assert x_2d.ndim == 2 and x_2d.shape[1] == self.K, \
            f"{self.name}: shape mismatch {x_2d.shape} vs K={self.K}"
        # numpy's gemm path: x.T @ x is the standard outer-product
        # accumulation. For x shape [N, K], x.T @ x gives [K, K].
        self.H += x_2d.T @ x_2d
        self.n_tokens += x_2d.shape[0]

    def finalize(self) -> np.ndarray:
        """Returns H / n_tokens as an FP32 K×K array (the actual
        expectation E[x x^T]).

        For memmap-backed accumulators, this materializes a fresh in-RAM
        K×K FP32 copy of the data normalized by `n_tokens`. The caller
        is expected to immediately write it to the HFHS sidecar and drop
        the reference, so peak RAM is one Hessian at a time (≤1.2 GB
        for K=17408 — fits comfortably).
        """
        if self.n_tokens == 0:
            return np.asarray(self.H, dtype=np.float32)
        if isinstance(self.H, np.memmap):
            self.H.flush()
        return (np.asarray(self.H, dtype=np.float32) / self.n_tokens).astype(np.float32)

    def close(self) -> None:
        """Release the memmap and (optionally) delete the backing file.
        Safe to call multiple times; no-op for in-RAM accumulators.
        Called after `finalize()` to free OS resources during the
        sidecar-write loop."""
        if isinstance(self.H, np.memmap):
            self.H._mmap.close()  # type: ignore[attr-defined]
        self.H = None  # type: ignore[assignment]
        # Caller may unlink self._mmap_path after the sidecar is written;
        # we don't auto-delete to ease debugging a partial run.


def build_hook(acc: HessianAccumulator):
    """Returns a forward-pre hook that captures the input activations.

    Hooks are registered on `nn.Linear`; the input is `x` (the matmul
    input, shape [..., K]). We flatten leading dims and move to CPU FP32
    before accumulating.
    """

    def hook(module, inputs):  # noqa: ARG001 — module is unused but required by API
        x = inputs[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))
        # Cast to FP32 on the GPU then transfer — avoids a host-side
        # cast and minimizes PCIe bytes (BF16→FP32 is 2x more bytes,
        # but the .cpu() transfer is the bottleneck either way).
        x_cpu = x.to(dtype=torch.float32).detach().cpu().numpy()
        acc.update(x_cpu)

    return hook


def merge_hfhs_files(out_path: Path, partial_paths: list[Path]) -> None:
    """Concatenate per-pass partial HFHS files into a single sidecar.

    The HFHS v1 format is a header (n_tensors u64) followed by sequential
    per-tensor records — so a "merge" is: sum the n_tensors fields,
    write the header, then copy each partial's record bytes verbatim.
    All partials must be HFHS v1 with the same dtype convention.

    Used by the multi-pass collector (--n-passes N): each pass writes
    its layers' Hessians to a partial, then this function combines them.
    Partial files are unlinked after merge.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_tensors = 0
    record_blocks: list[tuple[Path, int, int]] = []  # (path, offset, length)
    for pp in partial_paths:
        with pp.open("rb") as f:
            header = f.read(24)
            if header[:4] != b"HFHS":
                raise IOError(f"{pp} is not an HFHS file (magic={header[:4]!r})")
            version = struct.unpack("<I", header[4:8])[0]
            if version != 1:
                raise IOError(f"{pp} HFHS version {version} != 1")
            n_t = struct.unpack("<Q", header[8:16])[0]
            total_tensors += n_t
            # Record block = everything after the 24-byte header
            file_size = pp.stat().st_size
            record_blocks.append((pp, 24, file_size - 24))

    with out_path.open("wb") as f:
        # Write merged header
        f.write(b"HFHS")
        f.write(struct.pack("<I", 1))                  # version
        f.write(struct.pack("<Q", total_tensors))      # n_tensors (summed)
        f.write(struct.pack("<Q", 0))                  # reserved
        # Stream each partial's records, then unlink it. This keeps peak
        # disk usage at ~one-partial-size above the growing output, vs
        # holding all partials until the very end (which would peak at
        # 2x total final size).
        for pp, off, length in record_blocks:
            with pp.open("rb") as g:
                g.seek(off)
                # Stream in 64 MB chunks so we don't fault the whole partial
                # into RAM during the copy.
                CHUNK = 64 * 1024 * 1024
                remaining = length
                while remaining > 0:
                    buf = g.read(min(CHUNK, remaining))
                    if not buf:
                        break
                    f.write(buf)
                    remaining -= len(buf)
            try:
                pp.unlink()
            except OSError as e:
                print(f"  WARN: failed to unlink {pp}: {e}", file=sys.stderr)


def collect_one_pass(
    model,
    seqs: list[torch.Tensor],
    stored_names: set[str],
    pass_idx: int,
    n_passes: int,
    accumulator_dir: Path | None,
    device: str,
) -> dict[str, HessianAccumulator]:
    """Register hooks for layers in this pass, run forward, return accs.

    `layer_idx % n_passes == pass_idx` selects which layers participate.
    Modules without a recognizable `.layers.<N>.` infix (lm_head etc.)
    are not GPTQ-eligible by `is_gptq_target` anyway, so we never see
    them here. MTP modules use the `mtp.layers.<N>.` namespace; their
    layer_idx is in a separate index space from the main model, but
    falls under the same modulo policy (so MTP layer 0 lands in pass 0
    when n_passes ≥ 1 — fine since MTP is small).
    """
    accs: dict[str, HessianAccumulator] = {}
    handles = []
    skipped_no_layer_idx = 0
    skipped_other_pass = 0

    for mod_name, module in model.named_modules():
        if not (isinstance(module, torch.nn.Linear) and is_gptq_target(mod_name)):
            continue
        layer_idx = _extract_layer_idx(mod_name)
        if n_passes > 1:
            if layer_idx is None:
                # No layer infix → assign to pass 0 by convention.
                if pass_idx != 0:
                    skipped_no_layer_idx += 1
                    continue
            else:
                if layer_idx % n_passes != pass_idx:
                    skipped_other_pass += 1
                    continue

        K = module.in_features
        stored = _translate_to_stored_name(mod_name, stored_names)
        if stored is None:
            print(f"  WARN: no safetensors key matches {mod_name!r} — skipping",
                  file=sys.stderr)
            continue
        accs[stored] = HessianAccumulator(stored, K, accumulator_dir=accumulator_dir)
        handles.append(module.register_forward_pre_hook(build_hook(accs[stored])))

    if n_passes > 1:
        print(f"      pass {pass_idx+1}/{n_passes}: {len(accs)} hooks "
              f"({skipped_other_pass} skipped → other passes, "
              f"{skipped_no_layer_idx} skipped → no layer infix)")
    else:
        print(f"      {len(accs)} hooks registered "
              f"({len(set(acc.K for acc in accs.values()))} distinct K dims)")
    if accs:
        print(f"      K range: {min(acc.K for acc in accs.values())}.."
              f"{max(acc.K for acc in accs.values())}")

    t0 = time.time()
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            seq = seq.unsqueeze(0).to(device)
            # `logits_to_keep=1` (renamed from `num_logits_to_keep` in
            # transformers ≥4.50) makes lm_head only project the LAST
            # token, not all `ctx_len` positions. For 27B with
            # vocab=248320 + ctx=2048, full logits = ~1 GB BF16 — and
            # that allocation lands on whichever device lm_head sits
            # on (typically cuda:0 under device_map="auto"), which
            # may already be near its --max-gpu-mem cap. The Hessian
            # collector doesn't use logits at all; only the hooks on
            # the GPTQ-target Linear layers (which fire before lm_head)
            # contribute to the accumulators.
            try:
                model(seq, logits_to_keep=1)
            except TypeError:
                # Older transformers: use num_logits_to_keep
                try:
                    model(seq, num_logits_to_keep=1)
                except TypeError:
                    # Even older: just run full forward (works for <27B)
                    model(seq)
            if (i + 1) % 8 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(seqs) - i - 1) / rate
                print(f"      seq {i+1}/{len(seqs)} "
                      f"({elapsed:.1f}s elapsed, {rate:.2f} seq/s, ETA {eta:.0f}s)")
    print(f"      forward pass complete: {time.time() - t0:.1f}s")

    for h in handles:
        h.remove()
    return accs


def write_hessian_file(out_path: Path, accs: dict[str, HessianAccumulator]) -> None:
    """Serialize all accumulated Hessians to HFHS v1 binary format.

    For memmap-backed accumulators, each tensor is finalized → written →
    closed → backing file unlinked in turn, so peak in-flight RAM is
    O(K_max² · 4 bytes) = ~1.2 GB for K=17408. Without this streaming
    discipline, 27B's 126 GB total Hessian set would never fit in RAM
    even temporarily.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Count valid tensors first so we can write the correct n_tensors
    # header without scanning twice through the accumulator dict.
    valid_count = sum(1 for acc in accs.values() if acc.n_tokens > 0)

    with out_path.open("wb") as f:
        # Header
        f.write(b"HFHS")
        f.write(struct.pack("<I", 1))                  # version
        f.write(struct.pack("<Q", valid_count))        # n_tensors
        f.write(struct.pack("<Q", 0))                  # reserved

        for name, acc in accs.items():
            if acc.n_tokens == 0:
                print(f"  WARN: {name} accumulated 0 tokens — skipping", file=sys.stderr)
                acc.close()
                continue
            H_final = acc.finalize()                      # FP32 K×K, in-RAM copy
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))   # name_len
            f.write(name_bytes)                           # name
            f.write(struct.pack("<I", 0))                 # expert_idx (default 0)
            f.write(struct.pack("<I", acc.K))             # K
            f.write(struct.pack("<I", 1))                 # dtype_flag (1 = F32)
            f.write(H_final.tobytes(order="C"))           # K*K*4 bytes
            del H_final
            # Free the memmap (if any) and unlink the backing file —
            # we already serialized the data into the .hessian.bin.
            backing_path = acc._mmap_path
            acc.close()
            if backing_path is not None:
                try:
                    backing_path.unlink()
                except OSError as e:
                    print(f"  WARN: failed to unlink {backing_path}: {e}", file=sys.stderr)


def load_calibration_text(corpus: str, n_sequences: int, ctx_len: int,
                          tokenizer) -> list[torch.Tensor]:
    """Returns a list of n_sequences token tensors, each of length ctx_len.

    `corpus` is either a HF datasets identifier (e.g. "wikitext") or a
    path to a plain-text file. For HF datasets we use the "train" split
    by default (avoids data leakage against wikitext-2-test which is the
    eval slice).
    """
    if Path(corpus).is_file():
        text = Path(corpus).read_text()
        all_tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    else:
        # HF datasets path. Default: wikitext-2-raw-v1 train split.
        if "/" in corpus or corpus == "wikitext":
            ds_name = corpus if "/" in corpus else "wikitext"
            cfg = "wikitext-2-raw-v1" if ds_name == "wikitext" else None
            ds = load_dataset(ds_name, cfg, split="train", trust_remote_code=False)
        else:
            ds = load_dataset(corpus, split="train", trust_remote_code=False)
        # Concatenate text rows; the dataset's "text" field is conventional.
        text_field = "text" if "text" in ds.column_names else ds.column_names[0]
        text = "\n\n".join(row[text_field] for row in ds if row[text_field].strip())
        all_tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    n_total_tokens = all_tokens.shape[0]
    needed_tokens = n_sequences * ctx_len
    if n_total_tokens < needed_tokens:
        raise SystemExit(
            f"corpus has only {n_total_tokens} tokens, need {needed_tokens} "
            f"({n_sequences} seqs × {ctx_len} ctx)"
        )

    seqs = [all_tokens[i * ctx_len: (i + 1) * ctx_len] for i in range(n_sequences)]
    return seqs


def _gen_fwht_signs(seed: int, n: int = 256) -> np.ndarray:
    """Replicate hipfire's `gen_fwht_signs` (main.rs:537-543) — a tiny LCG
    that produces ±1 signs from a u32 seed. Must match bit-for-bit so the
    Python noise-injection round-trip is identical to runtime's FWHT."""
    state = np.uint32(seed)
    out = np.empty(n, dtype=np.float32)
    mul = np.uint32(1103515245)
    inc = np.uint32(12345)
    mask = np.uint32(0x7fffffff)
    for i in range(n):
        state = (state * mul + inc) & mask
        out[i] = 1.0 if ((state >> 16) & 1) == 1 else -1.0
    return out


def _fwht_butterflies_inplace(x: np.ndarray) -> None:
    """Walsh-Hadamard transform butterflies on a length-256 array (no
    scaling, no sign-flipping). Matches the inner loop of
    `cpu_fwht_256` in main.rs:518-531."""
    assert x.shape == (256,)
    stride = 1
    while stride < 256:
        i = 0
        while i < 256:
            for j in range(stride):
                a = x[i + j]
                b = x[i + j + stride]
                x[i + j] = a + b
                x[i + j + stride] = a - b
            i += stride * 2
        stride <<= 1


def _fwht_256_inplace(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> None:
    """Forward FWHT-256 with hipfire's sign/scale convention.
    F(x) = D_s2 · (H · (D_s1 · x)) / 16.
    Matches `cpu_fwht_256` in main.rs:515-534 bit-for-bit."""
    assert x.shape == (256,)
    x *= signs1
    _fwht_butterflies_inplace(x)
    x *= 0.0625 * signs2  # 1/16 = 1/sqrt(256)


def _fwht_256_inplace_inverse(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> None:
    """Inverse of `_fwht_256_inplace`.
    F^-1(y) = D_s1 · (H · (D_s2 · y)) / 16.
    Matches `load_any_as_f32`'s inverse-FWHT path in qwen35.rs (the
    runtime's MQ-load-time inverse rotation that recovers BF16-like
    values from MQ4-stored bytes)."""
    assert x.shape == (256,)
    x *= signs2
    _fwht_butterflies_inplace(x)
    x *= 0.0625 * signs1


def _quantize_mq4_block_roundtrip(block: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Apply hipfire's MQ4G256 quant-then-dequant round-trip to a single
    256-element block. Mirrors `quantize_mq4g256` (main.rs:549-587) for
    one block: FWHT → per-block min/max → quant to 0..15 grid → dequant.
    Returns the noisy (FP32) values that the runtime would observe.
    """
    assert block.shape == (256,)
    rotated = block.copy()
    _fwht_256_inplace(rotated, signs1, signs2)
    min_val = float(rotated.min())
    max_val = float(rotated.max())
    rng = max_val - min_val
    if rng <= 0.0:
        return block.copy()
    scale = rng / 15.0
    inv_scale = 1.0 / scale
    # Quantize → 4-bit grid index
    q_idx = np.clip(np.floor((rotated - min_val) * inv_scale + 0.5), 0.0, 15.0)
    dequant_rot = q_idx * scale + min_val
    # Inverse FWHT recovers the BF16-like values in original basis with
    # the structured 4-bit-quant noise baked in. Mirrors runtime's
    # load_any_as_f32 path for qt=13 (MQ4G256).
    out = dequant_rot.copy()
    _fwht_256_inplace_inverse(out, signs1, signs2)
    return out


def _inject_conv1d_mq4_noise_in_place(model) -> None:
    """For every `linear_attn.conv1d` weight in `model`, replace it with
    the FWHT → MQ4 round-trip in-place. The resulting forward pass
    produces activations matching what hipfire's runtime sees when conv1d
    is stored as MQ4G256 (i.e., the default for `--format mq4` without
    `HIPFIRE_QUANTIZE_CONV_Q8=1`).

    Used by the calibration-mismatch experiment: collect Hessians with
    MQ4-noisy conv1d activations so GPTQ's OBS corrections account for
    the noise it'll face at inference. Hypothesis (2026-05-14): if
    confirmed, fixes the GPTQ-regression-without-Q8-conv1d issue.
    """
    signs1 = _gen_fwht_signs(42, 256)
    signs2 = _gen_fwht_signs(1042, 256)
    count = 0
    for mod_name, module in model.named_modules():
        if not mod_name.endswith("linear_attn.conv1d"):
            continue
        w = module.weight.detach().to(torch.float32).cpu().numpy()
        # Conv1d weight: [out_ch, in_ch=1, kernel=4] in PyTorch. Hipfire
        # treats it as a flat array; we mirror that.
        flat = w.reshape(-1).astype(np.float32, copy=True)
        n = flat.shape[0]
        assert n % 256 == 0, f"{mod_name}: numel {n} not divisible by 256"
        for b in range(n // 256):
            block = flat[b * 256:(b + 1) * 256]
            flat[b * 256:(b + 1) * 256] = _quantize_mq4_block_roundtrip(block, signs1, signs2)
        # Restore shape + dtype + device
        noisy = torch.from_numpy(flat.reshape(w.shape)).to(
            module.weight.dtype).to(module.weight.device)
        with torch.no_grad():
            module.weight.copy_(noisy)
        max_err = float(np.abs(flat.reshape(w.shape) - w).max())
        mean_abs = float(np.abs(flat.reshape(w.shape) - w).mean())
        print(f"  {mod_name}: shape={tuple(w.shape)} max_err={max_err:.4e} "
              f"mean_abs_err={mean_abs:.4e}")
        count += 1
    print(f"      injected MQ4 noise into {count} conv1d weights")


def main():
    ap = argparse.ArgumentParser(description="Collect per-tensor Hessians for Stage B GPTQ.")
    ap.add_argument("--model", required=True,
                    help="HF model identifier or local path to BF16 model dir.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output Hessian sidecar (.hessian.bin).")
    ap.add_argument("--n-sequences", type=int, default=128,
                    help="Calibration sequences (default 128, matches GPTQ paper).")
    ap.add_argument("--ctx-len", type=int, default=2048,
                    help="Tokens per calibration sequence (default 2048).")
    ap.add_argument("--corpus", default="wikitext",
                    help="HF dataset ID or path to plain-text file "
                         "(default 'wikitext' = wikitext-2-raw-v1 train).")
    ap.add_argument("--device", default="cuda",
                    help="Device for the forward pass (default 'cuda').")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Model dtype (default bfloat16).")
    ap.add_argument("--max-gpu-mem", type=str, default=None,
                    help="Per-GPU memory cap for HF's device_map='auto' (e.g. "
                         "'14GiB'). Required for 27B-class models on the "
                         "dual-16-GB-GPU box: without it, 'auto' tries to pack "
                         "the full 54 GB model into 32 GB total VRAM and OOMs "
                         "during _init_weights' FP32 cast. Leave unset for "
                         "≤9B models that fit comfortably on the GPUs.")
    ap.add_argument("--max-cpu-mem", type=str, default=None,
                    help="CPU memory cap for HF's device_map='auto' (e.g. "
                         "'60GiB'). Defaults to 60GiB when --max-gpu-mem is "
                         "set. Includes CPU RAM + disk-backed offload "
                         "(safetensors mmap), so can exceed system RAM.")
    ap.add_argument("--accumulator-dir", type=Path, default=None,
                    help="If set, each per-tensor Hessian is backed by a "
                         "memmap file under this directory rather than held "
                         "in system RAM. Required for ≥27B models where the "
                         "total Hessian footprint (~126 GB FP32 for "
                         "Qwen3.6-27B) exceeds system RAM. Local SSD path "
                         "(e.g. /tmp/hipfire-hessian-acc/) is fastest; "
                         "NFS works but throttles update throughput ~10x. "
                         "Files are unlinked after the HFHS sidecar is "
                         "written. For ≤9B, leave unset and use in-RAM "
                         "accumulators (no I/O overhead).")
    ap.add_argument("--n-passes", type=int, default=1,
                    help="Number of forward passes over the corpus. Each "
                         "pass collects Hessians for a disjoint subset of "
                         "transformer layers (layer_idx %% n_passes == "
                         "pass_idx). Set N=4 for 27B-class models where "
                         "the full-set in-RAM accumulator (~126 GB) "
                         "exceeds system RAM — per-pass peak is ~30 GB. "
                         "Forward-pass wall time scales linearly with N "
                         "(N=4 for 27B → ~4h vs ~1h at N=1). Each pass "
                         "writes its Hessians to a partial .hessian.bin.partN "
                         "file; a final merge step concatenates them into "
                         "--output. Default 1 (no chunking, same behavior "
                         "as before for ≤9B models).")
    ap.add_argument("--inject-conv1d-mq4-noise", action="store_true",
                    help="Round-trip every linear_attn.conv1d weight through "
                         "FWHT-256 + MQ4-quantize-then-dequantize BEFORE the "
                         "forward pass. Tests the calibration/inference "
                         "distribution-mismatch hypothesis for GPTQ on "
                         "Qwen3.5 DeltaNet (2026-05-14): default Hessian "
                         "collection uses BF16 conv1d, but inference with "
                         "--format mq4 (no --HIPFIRE_QUANTIZE_CONV_Q8) "
                         "gets MQ4 conv1d → activation distribution shifts → "
                         "GPTQ's OBS corrections amplify rather than dampen "
                         "the noise. Injecting matching noise into the "
                         "Hessian-collection forward pass should let GPTQ "
                         "compensate correctly. See docs/plans/gptq_bug.md "
                         "and master doc §5 calibration-mismatch entry "
                         "(filed 2026-05-14).")
    args = ap.parse_args()

    print(f"=== Hessian collector — Stage B Phase 1.1 ===")
    print(f"  model:        {args.model}")
    print(f"  output:       {args.output}")
    print(f"  corpus:       {args.corpus}")
    print(f"  sequences:    {args.n_sequences} × {args.ctx_len} ctx "
          f"= {args.n_sequences * args.ctx_len} calibration tokens")
    print(f"  device:       {args.device}")
    print(f"  dtype:        {args.dtype}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"\n[1/4] Loading tokenizer + model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    # device_map="auto" → HF/accelerate uses meta tensors, skipping the
    # random-init step that would otherwise allocate the full model in
    # GPU memory before safetensors weights replace them.
    #
    # For 27B-class models on this dual-16-GB-GPU box, "auto" without
    # an explicit max_memory cap tries to pack the full model onto
    # GPUs (54 GB BF16 → spills past 32 GB total VRAM) and OOMs during
    # _init_weights' FP32 cast (`init.normal_(module.weight.float())`).
    # The --max-gpu-mem flag caps per-GPU usage so the remainder lands
    # on CPU/disk. Setting to e.g. "14GiB" leaves ~2 GB headroom per
    # card for activations + KV cache during the forward pass.
    if args.device == "cuda":
        if args.max_gpu_mem is not None:
            # Build a per-device max_memory dict. accelerate accepts
            # GPU indices as either int or str; we use int for clarity.
            n_gpus = torch.cuda.device_count()
            max_memory = {i: args.max_gpu_mem for i in range(n_gpus)}
            max_memory["cpu"] = args.max_cpu_mem or "60GiB"
            print(f"      max_memory: {max_memory}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model = model.to(args.device)
    model.eval()
    print(f"      loaded in {time.time() - t0:.1f}s")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"      VRAM in use: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f}/"
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if args.inject_conv1d_mq4_noise:
        print(f"\n[1.5/4] Injecting MQ4 noise into linear_attn.conv1d weights "
              f"(seeds 42/1042, group=256)...")
        _inject_conv1d_mq4_noise_in_place(model)

    if args.n_passes < 1:
        raise SystemExit("--n-passes must be >= 1")

    # HF's AutoModelForCausalLM flattens multimodal submodules; we resolve
    # in-memory module names back to safetensors-stored names so the Rust
    # quantizer can look Hessians up by the same key as the .hfq tensors.
    stored_names = _safetensors_keys(args.model)

    print(f"\n[3/4] Loading calibration corpus...")
    t0 = time.time()
    seqs = load_calibration_text(args.corpus, args.n_sequences, args.ctx_len, tokenizer)
    print(f"      loaded {len(seqs)} calibration sequences in {time.time() - t0:.1f}s")

    if args.n_passes == 1:
        # Single-pass (default, original behavior for ≤9B).
        print(f"\n[2/4] Registering hooks + running forward pass (1 pass)...")
        accs = collect_one_pass(
            model, seqs, stored_names,
            pass_idx=0, n_passes=1,
            accumulator_dir=args.accumulator_dir,
            device=args.device,
        )
        if not accs:
            raise SystemExit("no GPTQ-target Linear modules matched — nothing to collect")
        sample_acc = next(iter(accs.values()))
        print(f"      total tokens accumulated per tensor (sample): {sample_acc.n_tokens}")
        print(f"\n[4/4] Writing Hessian sidecar to {args.output}...")
        t0 = time.time()
        write_hessian_file(args.output, accs)
        print(f"      wrote {args.output.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.1f}s")
    else:
        # Multi-pass: N forward passes, each over a disjoint layer set.
        # Per-pass peak RAM is bounded by N; total wall is N × single-pass.
        print(f"\n[2-3/4] Multi-pass collection ({args.n_passes} passes — "
              f"required for 27B-class models to bound per-pass accumulator RAM)")
        partial_paths: list[Path] = []
        for pass_idx in range(args.n_passes):
            print(f"\n  --- Pass {pass_idx+1}/{args.n_passes} ---")
            accs = collect_one_pass(
                model, seqs, stored_names,
                pass_idx=pass_idx, n_passes=args.n_passes,
                accumulator_dir=args.accumulator_dir,
                device=args.device,
            )
            if not accs:
                print(f"      WARN: pass {pass_idx} yielded 0 hooks — skipping")
                continue
            partial = args.output.with_suffix(args.output.suffix + f".part{pass_idx}")
            print(f"      writing pass {pass_idx} → {partial} ({len(accs)} tensors)")
            t0 = time.time()
            write_hessian_file(partial, accs)
            print(f"      wrote {partial.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.1f}s")
            partial_paths.append(partial)
            # Free RAM held by the accumulators of this pass before the next.
            del accs

        print(f"\n[4/4] Merging {len(partial_paths)} partial HFHS files → {args.output}")
        t0 = time.time()
        merge_hfhs_files(args.output, partial_paths)
        print(f"      wrote {args.output.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.1f}s")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
