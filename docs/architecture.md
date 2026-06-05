# hipfire — System Architecture Diagram

**v0.2.0 | 2026-05-25**

## L1: Crate Topology (Build-Time)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLI / User-Facing Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ hipfire  │  │ hipfire  │  │ hipfire  │  │ hipfire  │  │  CLI TUI  │    │
│  │ serve    │  │  run     │  │  pull    │  │  config  │  │  (TypeScript│    │
│  │  (daemon)│  │ (attach) │  │  (hf hub)│  │  (kv)    │  │   cli/)    │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│     ┌──────────────────────────────────────────────────────────────┐      │
│     │                  daemon.rs (hipfire-runtime)                 │      │
│     │  - HTTP server (port 11435, OpenAI-compatible REST API)      │      │
│     │  - model hot-swap, prompt framing, chat_template (Jinja)     │      │
│     │  - multi-turn session state, EOS detection, tool-call parse  │      │
│     └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐
│  hipfire-runtime    │  │     rdna-compute     │  │   Arch Crates (×4)      │
│  (inference engine) │  │ (kernel dispatch)    │  │                          │
│                     │  │                      │  │  ┌──────────────────┐   │
│  ┌───────────────┐  │  │  ┌────────────────┐  │  │  │ hipfire-arch-   │   │
│  │ llama.rs      │  │  │  │ dispatch.rs    │  │  │  │   qwen35        │   │
│  │ tokenizer.rs  │  │  │  │ kernels.rs     │  │  │  └──────────────────┘   │
│  │ sampler.rs    │  │  │  │ profile.rs     │  │  ┌──────────────────┐   │
│  │ hfq.rs        │  │  │  │ (JIT + cache)  │  │  │ hipfire-arch-   │   │
│  │ gguf.rs       │  │  │  └────────────────┘  │  │   qwen35-vl     │   │
│  │ bf16_loader.rs│  │                         │  │ hipfire-arch-llama│   │
│  │ prompt_frame.rs│  │                         │  │ hipfire-arch-qwen2│   │
│  │ eos_filter.rs  │  │                         │  └──────────────────┘   │
│  │ tool_call.rs   │  │                         │  ┌──────────────────┐   │
│  │ multi_gpu.rs   │  │                         │  │ hipfire-arch-toy│   │
│  │ weight_pager.rs│  │                         │  │ (reference impl) │   │
│  └───────────────┘  │                         │  └──────────────────┘   │
│                     │  ┌────────────────────┐  │                         │
│  [cfg(deltanet)]    │  │ GPU bridge layer   │  │                         │
│                     │  │                    │  │                         │
│  ┌───────────────┐  │  │ ┌──────────────┐  │  │                         │
│  │ dflash.rs     │  │  │ │ hip-bridge   │  │  │                         │
│  │ ddtree.rs     │  │  │ │ (dlopen HIP) │  │  │                         │
│  │ triattn.rs    │  │  │ │              │  │  │                         │
│  │ cask.rs       │  │  │ │ hsa-bridge   │  │  │                         │
│  │ cpu_router.rs │  │  │ │ (dlopen HSA) │  │  │                         │
│  └───────────────┘  │  │ └──────────────┘  │  │                         │
│                     │  └────────────────────┘  │                         │
└─────────────────────┘                         └──────────────────────────┘
        │                                                    │
        └───────────────┬────────────────────────────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │  Kernel Family (476)   │
            │  kernels/src/*.hip     │
            └────────────────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │  ROCm Stack             │
            │  libamdhip64.so  │      │
            │  libhsa-runtime64.so  │  │
            └────────────────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │  AMD GPU                │
            │  RDNA3: gfx1100/1102   │  │
            │  CDNA3: gfx942          │  │
            │  CDNA2: gfx90a          │  │
            │  VEGA2: gfx906          │  │
            │  RDNA4: gfx1201         │  │
            └────────────────────────┘
```

## L2: Runtime Data Flow (Inference Cycle)

```
                    ┌──────────────────────────────────────────────────────┐
                    │                    Prompt Ingest                    │
                    │  CLI/HTTP → tokenizer → prompt_frame → BPE encode   │
                    └──────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼────────────────────────────────────┐
                    │                 │                                    │
                    ▼                 ▼                                    ▼
        ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐
        │    Prefill Pass    │  │   Decode Loop      │  │   Speculative Dec   │
        │  (all tokens at   │  │   (one token at    │  │   (draft + verify)  │
        │   once, batched)  │  │   a time)          │  │                     │
        └────────────────────┘  └────────────────────┘  └─────────────────────┘
                    │                 │                                    │
                    ▼                 ▼                                    ▼
┌──────────────────────────┐ ┌────────────────────────────────────────────────────┐
│ Layer-by-layer forward:  │ │  Decode (per-token, per-layer):                   │
│                           │ │                                                   │
│  1. embedding             │ │  1. rmsnorm (input)                              │
│  2. rmsnorm               │ │  2. fused_qkv  (Q+K+V projection)                │
│  3. fused_qkv             │ │  3. rope_partial (positional encoding)           │
│  4. rope_partial          │ │  4. attention_flash (or dflash / triattn)        │
│  5. attention_flash       │ │  5. fused_gate_up (FFN: gate + up projection)   │
│  6. fused_gate_up         │ │  6. silu_mul (gate activation)                   │
│  7. silu_mul              │ │  8. gemv_residual  (FFN down + residual add)     │
│  8. gemv_residual         │ │                                                   │
│  └→ Δnet layers:         │ │  MoE (a3b) path:                                 │
│     conv1d_silu_split     │ │  1. rmsnorm                                      │
│     alpha_gate            │ │  2. moe_softmax_topk                             │
│     gated_delta_net       │ │  3. moe_scatter_permute                          │
│     repeat_interleave     │ │  4. moe_gate_up_unscatter                        │
│     conv1d_silu           │ │  5. moe_down_combine                             │
│     deinterleave          │ │  6. moe_down_combine (residual)                  │
│                           │ │                                                   │
│  (repeat for all layers)  │ │  LM Head:                                        │
│                           │ │  embedding (transposed, as output weight)        │
│  LM Head:                 │ │  (or gemm for batched vocab)                     │
│  embedding (transposed)   │ │                                                   │
│  softmax / argmax         │ │  Sampler:                                        │
│                           │ │  topk_logits → sample_top_p → tokenizer.decode  │
└───────────────────────────┘ └────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────────┐
                    │                  Output Path                         │
                    │  token → eos_filter → chat_template → HTTP/TUI      │
                    └──────────────────────────────────────────────────────┘
```

## L3: Quantization Formats (Weight → Kernel Mapping)

```
                          ┌──────────────┐
                          │  Model File   │
                          │  .hfq / .gguf │
                          └──────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌────────────┐  ┌─────────┐  ┌─────────┐
            │  HFQ4-G256  │  │  HFQ6   │  │  MQ4    │
            │  (4-bit)    │  │  (6-bit)│  │  (4-bit) │
            │  136 B/grp  │  │  200 B  │  │  136 B  │
            └──────┬──────┘  └────┬────┘  └────┬────┘
                   │              │             │
           ┌───────┼──────┐ ┌────┼────┐  ┌────┼────┐
           │              │     │     │  │            │
           ▼              ▼     ▼     ▼  ▼            ▼
     ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
     │ GEMV     │  │ Fused    │  │ GEMM     │  │ WMMA     │
     │ (decode) │  │ (decode) │  │ (prefill)│  │ (prefill)│
     │          │  │          │  │          │  │          │
     │ wave64   │  │ wave64   │  │ wave64  │  │ wave32   │
     │ [fp16]   │  │ [dp4a]   │  │ [fp16]  │  │ [fp16]   │
     └──────────┘  └──────────┘  └──────────┘  └──────────┘

  Arch-specific routing (same weight format → different kernels per GPU):

  ┌──────────┬──────────────────┬──────────────────┬──────────────────┐
  │ Format   │ gfx11 (RDNA3)   │ gfx942 (CDNA3)  │ gfx906 (VEGA2)  │
  ├──────────┼──────────────────┼──────────────────┼──────────────────┤
  │ HFQ4-G256│ fused_*_wave64   │ fused_*_v2      │ fused_*_wave64   │
  │ (decode) │ gemv_w64_pref    │ gemv_w64_dp4a   │ gemv_w64_pref    │
  │          │                  │                  │  + dp4a variants │
  ├──────────┼──────────────────┼──────────────────┼──────────────────┤
  │ HFQ4-G256│ gemm_wmma (K4)   │ gemm_wmma       │ gemm_mmq_gfx906  │
  │ (prefill)│                  │                  │  + mmq_x{8..64}  │
  ├──────────┼──────────────────┼──────────────────┼──────────────────┤
  │ HFQ6-G256│ fused_*_mq4lloyd │ fused_*_wave64  │ fused_*_dp4a     │
  │ (decode) │                  │                  │                  │
  ├──────────┼──────────────────┼──────────────────┼──────────────────┤
  │ MQ3      │ fused_*_mq3lloyd │ (not ported)     │ (not ported)     │
  │ (decode) │                  │                  │                  │
  └──────────┴──────────────────┴──────────────────┴──────────────────┘
```

## L4: Kernel Family (by Compute Category)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KERNEL FAMILY MAP (~476 kernels)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ EMBEDDING ──────────────────────────────────────────────────────────┐   │
│  │  embedding_hfq4g128.hip              embedding_hfq4g256.hip          │   │
│  │  embedding_hfq4g256_batched.hip      embedding_q4k.hip               │   │
│  │  embedding_q8.hip                    embedding_q8_batched.hip        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ FUSED PROJECTION (2-way: gate+up) ──────────────────────────────────┐   │
│  │  fused_gate_up_hfq4g256.hip           fused_gate_up_hfq4g256_wave64.hip│   │
│  │  fused_gate_up_hfq4g256_wave64_dp4a.hip                              │   │
│  │  fused_gate_up_hfq6g256_wave64_dp4a.hip                              │   │
│  │  fused_gate_up_hfq4g256_v2.gfx942.hip                                │   │
│  │  fused_gate_up_mq3g256_lloyd.hip    fused_gate_up_mq3g256_lloyd.gfx11│   │
│  │  fused_gate_up_mq4g256_lloyd.hip    fused_gate_up_mq4g256_lloyd.gfx11│   │
│  │  fused_gate_up_q4k.hip                                                       │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ FUSED PROJECTION (3-way: q+k+v) ────────────────────────────────────┐   │
│  │  fused_qkv_hfq4g256.hip             fused_qkv_hfq4g256_wave64.hip    │   │
│  │  fused_qkv_hfq4g256_wave64_dp4a.hip                                 │   │
│  │  fused_qkv_hfq6g256_wave64_dp4a.hip                                 │   │
│  │  fused_qkv_hfq4g256_v2.gfx942.hip                                   │   │
│  │  fused_qkv_mq3g256_lloyd.hip      fused_qkv_mq3g256_lloyd.gfx11     │   │
│  │  fused_qkv_mq4g256_lloyd.hip      fused_qkv_mq4g256_lloyd.gfx11     │   │
│  │  fused_qkv_q4k.hip                                                       │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ FUSED PROJECTION (4-way: q+k+v+z+β) ────────────────────────────────┐   │
│  │  fused_qkvza_hfq4g256.hip           fused_qkvza_hfq4g256_wave64.hip  │   │
│  │  fused_qkvza_hfq4g256_wave64_dp4a.hip                               │   │
│  │  fused_qkvza_hfq6g256_wave64_dp4a.hip                               │   │
│  │  fused_qkvza_hfq4g256_v2.gfx942.hip                                 │   │
│  │  fused_qkvza_mq3g256_lloyd.hip   fused_qkvza_mq3g256_lloyd.gfx11    │   │
│  │  fused_qkvza_mq4g256_lloyd.hip   fused_qkvza_mq4g256_lloyd.gfx11    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ GEMV / GEMM (FFN down + residual) ──────────────────────────────────┐   │
│  │  gemv_hfq4g256_residual.hip              gemv_hfq4g256_residual_wave64.hip│   │
│  │  gemv_hfq4g256_residual_wave64_prefetch.hip                          │   │
│  │  gemv_hfq6g256_residual_wave64.hip                                   │   │
│  │  gemm_hfq4g256_wave64.hip                  gemm_hfq4g256_wave64_dp4a.hip  │   │
│  │  gemm_hfq4g256_residual_wave64_dp4a.hip                             │   │
│  │  gemm_hfq4g256_residual_mmq.hip                                      │   │
│  │  gemm_hfq4g256_residual_mmq_gfx906_body.cuh                          │   │
│  │  gemm_hfq4g256_residual_mmq_gfx906_x{8..64}.hip                      │   │
│  │  gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh                           │   │
│  │  gemm_qkv_hfq4g256_mmq_gfx906_body.cuh                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ ATTENTION ──────────────────────────────────────────────────────────┐   │
│  │  attention.hip              attention_flash.hip          attention_dflash.hip│   │
│  │  attention_causal_batched.hip                                                  │   │
│  │  attention_flash_asym2_tile{_batched}.hip                                     │   │
│  │  attention_flash_asym3_tile{_batched}.hip                                     │   │
│  │  attention_flash_asym4_tile{_batched}.hip                                     │   │
│  │  attention_flash_asym_reduce_batched.hip                                     │   │
│  │  attention_flash_fwht{2,3,4}_tile{_batched}.hip                              │   │
│  │  attention_flash_q8_0_{tile,reduce}.hip                                     │   │
│  │  attention_hfq4_kv.hip    attention_hfq8_kv.hip                              │   │
│  │  attention_q4kv.hip       attention_q8kv.hip                                 │   │
│  │  attention_q8_0_kv{_batched}.hip    attention_q8_0_kv_timed.hip             │   │
│  │  attention_int8_kv.hip    attention_int8c_kv.hip                             │   │
│  │  attention_int8c_f16_kv.hip                                                  │   │
│  │  vit_attention.hip      vit_attention_opt.hip                                │   │
│  │  triattn_accumulate.hip   triattn_score_{asym2,asym3,asym4}.hip             │   │
│  │  triattn_score_q8.hip  pflash_score_q8_kv.hip                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ NORM / ACTIVATION / POSITIONAL ──────────────────────────────────────┐   │
│  │  rmsnorm.hip        rmsnorm_reduce.gfx942.hip                           │   │
│  │  layernorm.hip     l2_norm.hip                                          │   │
│  │  rope.hip         rope_batched.hip                                      │   │
│  │  rope_partial_interleaved{_batched}.hip                                 │   │
│  │  rope_partial_halfsplit{_batched}.hip                                   │   │
│  │  silu.hip         silu_mul.hip   softplus.hip   sigmoid.hip   sigmoid_mul.hip│   │
│  │  softmax.hip      max_prob.hip                                          │   │
│  │  conv1d_decode.hip conv1d_silu.hip  conv1d_silu_split.hip               │   │
│  │  conv1d_silu_split_tree.hip  apply_rope_2d_vision.hip                   │   │
│  │  gated_delta_net.hip (in arch crate)                                   │   │
│  │  alpha_gate.hip                                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ SAMPLING / SCORING ──────────────────────────────────────────────────┐   │
│  │  argmax.hip       argmax_batched.hip                                    │   │
│  │  topk_logits.hip  topk_logsumexp_batched.hip                            │   │
│  │  sample_top_p.hip  softmax_prob_gather_batched.hip                     │   │
│  │  cross_entropy_loss.hip                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ MoE (a3b) ───────────────────────────────────────────────────────────┐   │
│  │  moe_softmax_topk_k8{_batched}.hip                                     │   │
│  │  moe_topk_renorm_k8{_batched}.hip                                      │   │
│  │  moe_scatter_histogram_k8.hip  moe_scatter_offsets_k8.hip              │   │
│  │  moe_scatter_permute_k8.hip    moe_scatter_fused_k8.hip                │   │
│  │  moe_gate_up_unscatter_k8.hip                                           │   │
│  │  moe_down_combine_k8_batched.hip  moe_down_combine_grouped_k8.hip      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ ROTATION / PREPROCESS ───────────────────────────────────────────────┐   │
│  │  rotate_x_mq_awq.hip   rotate_with_rms.gfx942.hip                      │   │
│  │  mq_rotate_x_dual.gfx12.hip  dequant_hfq4g256_to_f16.hip               │   │
│  │  deinterleave{_batched}.hip  transpose.hip                             │   │
│  │  repeat_interleave_qk{_batched}.hip                                    │   │
│  │  fused_qk_l2_norm_scale{_interleave_f32_batched}.hip                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ UTILITIES ───────────────────────────────────────────────────────────┐   │
│  │  add.hip  add_inplace.hip  mul.hip  scale_f32.hip  bias_add.hip        │   │
│  │  scaled_add_inplace.hip                                             │   │
│  │  pack_f32_to_fp8.gfx12.hip                                          │   │
│  │  bench_q8_fp16wmma.hip (dev only)                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## L5: Inference Loop (Detailed)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE CYCLE (per-token decode)                       │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌──────┐     ┌───────┐     ┌──────────────────────────────┐
  │CPU:  │────▶│GPU:   │────▶│ GPU: Forward Pass (per layer) │
  │token │     │embed  │     │                                 │
  │→GPU  │     │token→ │     │  ┌──────────────────────────┐  │
  │      │     │hidden │     │  │ rmsnorm(hidden)           │  │
  └──────┘     │(1D)  │     │  ├──────────────────────────┤  │
       │       └───────┘     │  │ fused_qkv(hidden→Q,K,V) │  │
       │                     │  ├──────────────────────────┤  │
       │                     │  │ rope_partial(Q,K)        │  │
       │                     │  ├──────────────────────────┤  │
       │                     │  │ attention_flash(Q,K,V,    │  │
       │                     │  │   kv_cache) → attn_out   │  │
       │                     │  ├──────────────────────────┤  │
       │                     │  │ fused_gate_up(hidden) →   │  │
       │                     │  │   gate⊗up (2 projections) │  │
       │                     │  ├──────────────────────────┤  │
       │                     │  │ silu_mul(gate, up)       │  │
       │                     │  ├──────────────────────────┤  │
       │                     │  │ gemv_residual(out, W_down│  │
       │                     │  │   , hidden) → hidden'    │  │
       │                     │  └──────────────────────────┘  │
       │                     │           │                     │
       │                     │           │ (repeat N layers)   │
       │                     │           ▼                     │
       │                     │  ┌──────────────────────────┐  │
       │                     │  │ embedding(logits, W^T)   │  │
       │                     │  │ → logit[TOKEN_VOCAB]     │  │
       │                     │  └──────────┬───────────────┘  │
       │                     │             │                   │
       │                     │             ▼                   │
       │                     │  ┌──────────────────────────┐  │
       │                     │  │ argmax / sample_top_p    │  │
       │                     │  │ → next_token_id (CPU)    │  │
       │                     │  └──────────┬───────────────┘  │
       │                     │             │                   │
       │                     │             ▼                   │
       │                     │  ┌──────────────────────────┐  │
       │                     │  │ KV cache write (asym{2,3,│  │
       │                     │  │  4} flash attention path) │  │
       │                     │  └──────────────────────────┘  │
       │                     │                                 │
       └─────────────────────┴─────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────┐
  │                    DFLASH SPECULATIVE DECODE PATH                           │
  │                                                                             │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │                    Draft Model (5-layer Qwen3, MQ4, 0.55-0.92 GB)    │  │
  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │  │
  │  │  │ Layer 1-4│  │ Layer 1-4│  │ Layer 1-4│  │  Layer 5 │            │  │
  │  │  │ full attn│  │ full attn│  │ full attn│  │ full attn│            │  │
  │  │  │ +cross-at│  │ +cross-at│  │ +cross-at│  │ +cross-at│            │  │
  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │  │
  │  │       │           │            │            │                       │  │
  │  │       └───────────┴────────────┴────────────┘                       │  │
  │  │                          │                                          │  │
  │  │                   seed_pred(token)                                  │  │
  │  │                   → candidate tokens {t1, t2, ..., tN}              │  │
  │  └────────────────────────────┬───────────────────────────────────────┘  │
  │                               │                                          │  │
  │                               ▼                                          │  │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │                    Target Model (full, 5-27 GB)                      │  │
  │  │                                                                     │  │
  │  │  ┌────────────────────────────────────────────────────────────┐     │  │
  │  │  │  Parallel Verify: run all N candidates through target     │     │  │
  │  │  │  attention_flash (batched) + argmax_batched               │     │  │
  │  │  │                                                            │     │  │
  │  │  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                  │     │  │
  │  │  │  │ cand1│  │ cand2│  │ cand3│  │ candN│                   │     │  │
  │  │  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                  │     │  │
  │  │  │     ▼         ▼         ▼         ▼                        │     │  │
  │  │  │  target_attn ×N  →  argmax_batched                          │     │  │
  │  │  │     │                                                      │     │  │
  │  │  │  match? → accept (advance N tokens) or reject (take 1)     │     │  │
  │  │  │                                                            │     │  │
  │  │  │  [optional] DDTree: tree-shaped verify                      │     │  │
  │  │  │  ┌───────────────────────────────────────────┐             │     │  │
  │  │  │  │  ┌──────┐                                │             │     │  │
  │  │  │  │  │t1,t2,│ → verify batch 1  ┌──────────┐│             │     │  │
  │  │  │  │  │  t3 │           │        │ continue ││             │     │  │
  │  │  │  │  │     │ → verify batch 2  └──────────┘│             │     │  │
  │  │  │  │  └─────┘           │                   │             │     │  │
  │  │  │  └────────────────────┴───────────────────┘             │     │  │
  │  │  └─────────────────────────────────────────────────────────┘     │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────┐
  │                    KV CACHE ARCHITECTURE                                   │
  │                                                                             │
  │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
  │  │  Full (fp16)    │  │  asym2 (4-bit K,  │  │  asym3 (3-bit K,  │          │
  │  │  K=fp16, V=fp16 │  │  4-bit V)         │  │  3-bit V)        │          │
  │  └─────────────────┘  └──────────────────┘  └──────────────────┘          │
  │       │                       │                       │                    │
  │       │ (default)            │ (flash attention       │ (flash attention   │
  │       │                      │  with FWHT decode)     │  with FWHT decode) │
  │       ▼                       ▼                       ▼                    │
  │  attention.hip     attention_flash_asym2/3/4_tile  attention_flash_fwht   │
  │  attention_flash   + triattn variants             {2,3,4}_tile           │
  │  attention_hfq4/8_kv                                                              │
  └────────────────────────────────────────────────────────────────────────────┘
```

## L6: Build System & Kernel JIT

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        BUILD / COMPILE FLOW                                │
└────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────────────────────────────────┐
  │ .hip src │────▶│ hipcc (ROCm HIP Compiler)                │
  │  kernels │     │  -O3 -march=gfxXXX                       │
  │  /src/   │     │  --gpu-architecture=gfxXXX               │
  │          │     │  -DNDEBUG                                 │
  └──────────┘     └─────────────┬────────────────────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ .hsa / .o     │
                         │ (compiled      │
                         │  kernels)      │
                         │                │
                         │  cache_dir:    │
                         │  ~/.hipfire/   │
                         │  cache/        │
                         │  {arch}-{md5}/ │
                         └───────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌────────────┐ ┌────────────┐ ┌────────────┐
            │ hipModule  │ │ hipKernel  │ │ hipGraph   │
            │ .load()    │ │ .launch()  │ │ (capture)  │
            │            │ │ (per-call) │ │ (graph    │
            │            │ │            │ │  replay)   │
            └────────────┘ └────────────┘ └────────────┘

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  KERNEL CACHE KEY:  arch_id + md5(source + compile_flags + define_list) │
  │  HIT → load cached .hsa  │  MISS → compile + cache                      │
  │                                                                   │
  │  Define injection for mmq_x:                                        │
  │    wrapper.hip: #include "body.cuh"                                 │
  │    + runtime:   #define MMQ_X_VAL 24                                │
  │    → unique .hsa per mmq_x (x8, x16, x24, ..., x64)                 │
  │                                                                   │
  │  Graph capture mode (HIPFIRE_GRAPH=1):                              │
  │    1st cycle:  capture forward pass into hipGraphExec               │
  │    2nd+ cycle: replay graph (eliminates host→dispatch sync per kernel)│
  └─────────────────────────────────────────────────────────────────────────┘
```

## L7: Model Loading (Multi-Source)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        MODEL LOADING PATHS                                 │
└────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐
  │ .hfq        │    │ .gguf       │    │ .safetensors  │    │ .bin      │
  │ (hipfire)   │    │ (GGUF v3)   │    │ (HuggingFace) │    │ (raw)     │
  └──────┬──────┘    └──────┬──────┘    └──────┬───────┘    └─────┬─────┘
         │                  │                  │                   │
         ▼                  ▼                  ▼                   ▼
  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐
  │ hfq.rs       │  │ gguf.rs      │  │ safetensors_   │  │ bf16_loader │
  │ (metadata +  │  │ (tensor      │  │ source.rs      │  │ .rs         │
  │  weight idx) │  │  parsing)    │  │ (numpy format) │  │             │
  └──────┬───────┘  └──────┬───────┘  └──────┬─────────┘  └────┬────────┘
         │                  │                  │                  │
         └──────────────────┴──────────────────┴──────────────────┘
                                │
                                ▼
                   ┌────────────────────────┐
                   │  load_any_as_f32()     │
                   │  (12 quant formats)    │
                   │                        │
                   │  HFQ4-G256  (4-bit)    │
                   │  HFQ6-G256  (6-bit)    │
                   │  MQ4 / MQ8    (4/8-bit)│
                   │  MQ3 (Lloyd) (3-bit)   │
                   │  Q4_K / Q8_0 (llama.cpp)│
                   │  FP16 / BF16           │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │  GpuTensor (GPU VRAM)  │
                   │  + GpuTensor alias     │
                   │    (PARO rotation      │
                   │     metadata sharing)  │
                   └────────────────────────┘
```

## L8: Observability & Testing

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    QUALITY ASSURANCE LAYER                                 │
└────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │ coherence-gate   │   │ hipfire-detect   │   │ hipfire-atlas    │
  │ -dflash.sh       │   │ (behavioral      │   │ (kernel atlas)   │
  │                   │   │  detectors)      │   │                  │
  │ - 27b-dflash-prose│   │  - token attractors│  │ - ISA Fit View  │
  │ - 27b-dflash-code │   │  - special-token  │   │ - Phase-aware   │
  │ - 27b-ddtree-*   │   │    leaks          │   │ - Measurement   │
  │ - MQ3/MQ6 checks │   │  - n-gram density │   │   schema        │
  │                   │   │  - tool-call shape│  │ - JSONL writer  │
  │ Gates:            │   └──────────────────┘   └─────────────────┘
  │  - zero tokens    │
  │  - max_token_freq │
  │  - unique tokens  │
  │  - no panic       │
  └───────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │  PROFILING STACK:                                                    │
  │                                                                       │
  │  rocprof → PMC counters (VALUBusy, MemUnitStall, L2 Hit, etc.)      │
  │         → kernel-level dispatch profiling (profile.rs)                │
  │         → hipfire-atlas → JSONL → Kernel Atlas visualization         │
  │                                                                       │
  │  HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1 → isolate Q8_1 quant cost          │
  │  HIPFIRE_FP16_LAYER_MIN/MAX → per-layer KLD attribution sweep        │
  │  HIPFIRE_PROMPT_TOKEN_HEAT=1 → BPE merge-rank heat map               │
  │  HIPFIRE_HOST_TIMING → per-cycle host-side timing                     │
  └──────────────────────────────────────────────────────────────────────┘
```

## L9: External Interfaces

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SURFACES                                   │
└────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │  CLI (TypeScript / Bun)                                              │
  │                                                                       │
  │  hipfire serve  → long-lived HTTP daemon (port 11435)                │
  │  hipfire run    → attach to running daemon, stdin/stdout chat        │
  │  hipfire pull   → download from HuggingFace → ~/.hipfire/models/    │
  │  hipfire config → key-value config in ~/.hipfire/config.json         │
  │                                                                       │
  │  chat.ts         → full chat session (OpenAI API compat)             │
  │  chat_pure.ts    → direct daemon interaction (no OpenAI wrapping)    │
  │  index.ts        → CLI command router (264KB, 7000+ lines)           │
  │  registry.json   → model catalog for `hipfire pull`                  │
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │  OpenAI-Compatible REST API (daemon)                                 │
  │                                                                       │
  │  POST /v1/chat/completions  → chat with system/user/assistant turns  │
  │  POST /v1/completions       → raw completion                        │
  │  GET  /v1/models            → list loaded models                    │
  │                                                                       │
  │  Supports: stream=true (SSE), tool_call, function_call,             │
  │  system_prompt, max_tokens, temperature, top_p, logprobs            │
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Tool Calling Pipeline                                               │
  │                                                                       │
  │  1. Tool spec → JSON → Jinja chat_template ({{ tool | tojson }})   │
  │  2. Model generates tool-call token sequence                         │
  │  3. tool_call.rs parses raw tokens → structured call                 │
  │  4. parseToolCalls (TypeScript) handles spec/flat/XML malformation   │
  │  5. Return OpenAI-compatible tool_call object                        │
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Multi-GPU (hipfire-runtime/src/multi_gpu.rs)                        │
  │                                                                       │
  │  - Weight paging: model weights distributed across multiple GPUs    │
  │  - Per-layer GPU assignment with overlap-aware scheduling           │
  │  - Peer-to-peer memory copy between GPUs for layer boundaries       │
  │  - GPU 0 runs the sampler + argmax; others run forward pass         │
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Redline (crates/redline)                                            │
  │                                                                       │
  │  Bare libdrm / direct-KMD dispatch — bypass HIP entirely.            │
  │  Experimental path to zero-hip overhead on supported AMDs.           │
  └──────────────────────────────────────────────────────────────────────┘
```

## L10: Cross-Cutting Concepts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-CUTTING ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ARCH TRAIT (hipfire-runtime/src/arch.rs)                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ trait Architecture {                                                  │  │
│  │   fn config_from_hfq(&self, meta: &HfqMetadata) -> &ArchConfig;      │  │
│  │   fn load_weights(&mut self, gpu: &mut Gpu, meta: &HfqMetadata);     │  │
│  │   fn forward_prefill_batch(&mut self, gpu, x, y, batch);             │  │
│  │   fn forward_decode(&mut self, gpu, x, y);                          │  │
│  │   fn hidden_state_size(&self) -> usize;                              │  │
│  │   fn kv_layer_indices(&self) -> &[usize];                            │  │
│  │   fn arch_id(&self) -> u16;                                           │  │
│  │ }                                                                     │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Dispatched from runtime to arch crate:                                     │
│    Qwen35 → hipfire-arch-qwen35 (DeltaNet + FullAttention hybrid)         │
│    Qwen35-VL → hipfire-arch-qwen35-vl (vision tower + hybrid)             │
│    Llama-family → hipfire-arch-llama (plain MHA, GQA, RoPE)               │
│    Qwen2 → hipfire-arch-qwen2 (attention_bias=true, GQA)                   │
│                                                                             │
│  DISPATCH LAYER (rdna-compute/src/dispatch.rs)                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  12K+ lines. Every compute path in hipfire funnels through here.     │  │
│  │                                                                       │  │
│  │  Input:  (arch, quant_format, kernel_type, M, K, N)                  │  │
│  │  Process: arch gate → quant gate → kernel variant selection          │  │
│  │  Output: hipLaunchKernel / hipGraphExec / hipExtLaunchKernel      │  │
│  │                                                                       │  │
│  │  Arch gates: gemv_dp4a_enabled(), has_mmq_dp4a_or_wmma()            │  │
│  │  Quant gates: should_use_mmq(), hfq3_dp4a_enabled()                 │  │
│  │  Variant sel: mmq_x{8..64}, _full_add vs _bounds, wave64 vs wave32  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  SPECULATIVE DECODE (deltanet feature gate)                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  DFlash (dflash.rs): native Rust draft forward, 5-layer Qwen3        │  │
│  │    - No persistent KV cache (recomputed from target_hidden every step)│  │
│  │    - GdnTape: pre-conv1d capture for correct tape-replay rollback    │  │
│  │    - DeltaNetTape: O(1) rollback on aborted trajectories              │  │
│  │    - SpecPair: tokenizer round-trip verification                      │  │
│  │                                                                       │  │
│  │  DDTree (ddtree.rs): tree-shaped verify with batched parallel path   │  │
│  │    - b12-k2 (12 nodes, 2-way fan) / b22 variants                     │  │
│  │    - Stale-context overlap (Path D) — roadmap                        │  │
│  │                                                                       │  │
│  │  TriAttn (triattn.rs): KV eviction policy for extended context       │  │
│  │    - cask.rs: KV cache eviction trigger + hysteresis (cask_beta)     │  │
│  │    - 3-attention variants (asym2, asym3, asym4)                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
