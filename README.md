# hipfire

LLM inference engine for AMD RDNA GPUs. Written in Rust. Faster than llama.cpp at generation on every model tested.

## What it does

Takes a quantized language model and runs it on your AMD GPU. Generates text at **59.9 tok/s for Qwen3-8B** and **262 tok/s for Qwen3-0.6B** on an RX 5700 XT ($200 GPU from 2019).

Includes **TurboQuant KV cache** (2/3/4-bit, FWHT + norm correction) for 14.2x KV compression with coherent output, and **Qwen3.5 DeltaNet** support (gated linear attention at 190 tok/s).

No Python runtime. No ROCm link-time dependency. Loads `libamdhip64.so` via `dlopen` at runtime.

## v0.1.0 Highlights

- Qwen3.5 DeltaNet inference working (0.8B/2B/4B/9B)
- TurboQuant KV cache: turbo2 (2-bit, 14.2x), turbo3 (3-bit, 9.8x), turbo4 (4-bit, 7.5x)
- Register-only FWHT via `__shfl_xor` (zero shared memory, zero barriers)
- HFQ2 weight quantization kernel (19 VGPRs, 66.8 tok/s for 8B)
- Layer-adaptive KV: first/last layers at FP32, middle layers at turbo
- GPU-side max-probability kernel for confidence checking
- Early-exit forward pass infrastructure

## Performance

Measured on AMD RX 5700 XT (gfx1010, RDNA1, 8GB GDDR6, 448 GB/s peak).

### Generation (decode)

| Model | hipfire | llama.cpp | Ratio |
|-------|---------|-----------|-------|
| **Qwen3-8B** | **59.9 tok/s** | 44.3 tok/s | **1.35x** |
| **Qwen3-8B long (1000+ tok)** | **52.7 tok/s** | 42.8 tok/s | **1.23x** |
| **Qwen3-0.6B** | **262 tok/s** | 193.6 tok/s | **1.35x** |
| **Qwen3.5-0.8B (DeltaNet)** | **190 tok/s** | N/A | -- |

### TurboQuant KV Cache

| KV Config | Qwen3-8B tok/s | KV Compression | Quality |
|-----------|---------------|----------------|---------|
| Q8 (default) | 59.9 | 3.88x | good |
| turbo2 (2-bit) | 55.1 | **14.2x** | good |
| turbo4 (4-bit) | 54.5 | 7.5x | good |
| turbo3 (3-bit) | 52.0 | 9.85x | good |

All turbo configs produce coherent output. Norm-corrected quantization eliminates systematic drift.

### Prefill (prompt processing)

| Model | hipfire | llama.cpp | Ratio |
|-------|---------|-----------|-------|
| Qwen3-8B | 108 tok/s | 189.2 tok/s | 0.57x |
| Qwen3-0.6B | 1053 tok/s | 1534 tok/s | 0.69x |

hipfire wins all generation benchmarks. llama.cpp wins prefill via rocBLAS GEMM.

## Quick Start

```bash
# Build
cd hipfire
cargo build --release

# Quantize a model from HuggingFace safetensors
cargo run --release -p hipfire-quantize -- \
  --input path/to/Qwen3-8B/ \
  --output models/qwen3-8b.hfq \
  --format hfq4

# Run inference
cargo run --release --example infer_hfq -- models/qwen3-8b.hfq "Hello"

# With TurboQuant KV cache (2-bit, 14.2x compression)
cargo run --release --example infer_hfq -- models/qwen3-8b.hfq --turbo2 "Hello"

# Qwen3.5 DeltaNet models
cargo run --release --features deltanet --example infer_qwen35 -- models/qwen35-0.8b.hfq "Hello"
```

### Requirements

- AMD GPU with ROCm (tested on RDNA1 gfx1010, should work on RDNA2+)
- `hipcc` in PATH (from ROCm installation)
- Rust 1.75+

## How it works

### Weight quantization: HFQ4

Weights are stored in HFQ4 (HipFire Quantized 4-bit) format -- designed for maximum GPU occupancy on RDNA.

Each block of 256 weights: `[f32 scale][f32 zero][128 packed nibble bytes]` = 136 bytes. The GEMV kernel uses **18 VGPRs** -- half what llama.cpp's Q4_K uses (39 VGPRs). Lower register pressure = more concurrent wavefronts = better memory latency hiding = higher effective bandwidth (282 GB/s vs ~210 GB/s).

Also supports HFQ2 (2-bit, 19 VGPRs, 66.8 tok/s -- quality requires incoherence processing).

### TurboQuant KV cache

Norm-corrected quantization for the KV cache using Fast Walsh-Hadamard Transform:

1. **Normalize** each KV vector to unit L2 norm
2. **FWHT rotate** via register-only `__shfl_xor` (zero shared memory barriers)
3. **Lloyd-Max quantize** to optimal centroids (2/3/4-bit)
4. **Norm correction**: store `original_norm / reconstruction_norm` per head

This guarantees exact L2 norm preservation and decorrelated quantization error. The FWHT butterfly runs entirely in 8 VGPRs.

### Qwen3.5 DeltaNet

Gated Delta Network linear attention with recurrent S-state. The 128x128 state matrix fits exactly in RDNA1's 64KB LDS. Supports Q8 and FP32 state quantization.

### Runtime kernel compilation

HIP kernels are embedded as C++ string constants in the Rust source. On first use, each kernel is compiled to `.hsaco` via `hipcc --genco` and cached to `/tmp/hipfire_kernels/`. Source hashing ensures stale caches are recompiled.

## Architecture

```
hipfire/
├── crates/
│   ├── hip-bridge/          # Safe Rust FFI to libamdhip64.so via dlopen
│   ├── rdna-compute/        # HIP kernel compilation, dispatch, GPU tensor ops
│   ├── engine/              # Model loading, forward pass, tokenizer
│   └── hipfire-quantize/    # Quantizer: HuggingFace safetensors -> .hfq
├── bench/                   # Benchmark data and scripts
├── scripts/                 # Benchmark automation
└── docs/
    ├── BENCHMARKS.md         # Full benchmark tables
    ├── TURBO3_DESIGN.md      # TurboQuant KV design document
    ├── RESEARCH_RDNA1.md     # RDNA1 optimization research synthesis
    ├── FLASH_DECODE_PLAN.md  # Flash Decoding design (planned)
    ├── MATH_SYNTHESIS.md     # Mathematical optimization analysis
    └── PERF_COMPARISON.md    # hipfire vs llama.cpp detailed comparison
```

## Supported Models

| Model | Architecture | Weight format | VRAM | Generation tok/s |
|-------|-------------|-------------|------|-----------------|
| Qwen3-8B | Transformer + GQA | HFQ4-G256 | ~4.4 GB | 59.9 |
| Qwen3-0.6B | Transformer + GQA | HFQ4-G128 | ~0.5 GB | 262 |
| Qwen3.5-0.8B | DeltaNet + Attention | HFQ4 | ~0.3 GB | 190 |
| TinyLlama-1.1B | Transformer | HFQ4-G256 | ~0.7 GB | 222 |

Quantize any LLaMA/Qwen3/Qwen3.5 model from HuggingFace with `hipfire-quantize --format hfq4`.

## Roadmap

- [ ] Vision model support (Qwen-VL, LLaVA)
- [ ] E8 lattice 2-bit weight quantization (QuIP#-style)
- [ ] Flash Decoding for long context (4-5x attention at 2K+)
- [ ] HTTP server mode (OpenAI-compatible API)
- [ ] Embedded tokenizer from HFQ metadata (remove GGUF fallback)

## License

MIT
