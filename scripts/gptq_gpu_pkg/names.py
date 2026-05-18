"""Name translations between safetensors / HFHS Hessian / imatrix GGUF.

Mirrors `main.rs::safetensors_to_ggml_name` and the HFHS key convention
documented at `docs/plans/gptq-hessian-format.md` §3.1.

Three naming conventions in play:

1. **Safetensors weight names** — e.g. `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`.
   These are what the input model file uses and what the manifest's
   `weights.safetensors` re-emits verbatim.

2. **HFHS Hessian keys** — same as (1) but with the trailing `.weight`
   stripped (per format spec §3.1, also Rust at `main.rs:4411`).

3. **Imatrix GGUF keys** — ggml-style flat names: `blk.0.attn_qkv.weight`.
   Translation is implemented by `safetensors_to_ggml_name` in Rust;
   ported below to `to_ggml_name`.

Tensors that don't have a ggml mapping (norms, A_log, conv1d, dt_bias,
small 1D) return None — caller falls back to non-imatrix (no AWQ).
"""

from __future__ import annotations


def to_hfhs_key(safetensors_name: str) -> str:
    """Strip the trailing `.weight` to produce the HFHS sidecar key.

    HFHS spec §3.1: keys are stored without the `.weight` suffix so
    that one Hessian can be referenced by either the FP16-side weight
    (`*.weight`) or some hypothetical future bias-only tensor — the
    activation H is the same either way.
    """
    return safetensors_name.removesuffix(".weight")


def to_ggml_name(safetensors_name: str) -> str | None:
    """HF safetensors name → ggml-style `blk.{N}.{slot}.weight`.

    Mirrors `main.rs::safetensors_to_ggml_name` exactly. Returns None
    for tensor names that don't have a mapping (norms, A_log, conv1d,
    dt_bias, top-level params). Caller (`imatrix_weights_for`)
    treats None as "no AWQ for this tensor".
    """
    # Drop the "language_model." or "model." prefix.
    if safetensors_name.startswith("model.language_model."):
        normalized = safetensors_name[len("model.language_model."):]
    elif safetensors_name.startswith("model."):
        normalized = safetensors_name[len("model."):]
    else:
        normalized = safetensors_name

    # Top-level passthroughs.
    if normalized == "embed_tokens.weight":
        return "token_embd.weight"
    if normalized == "lm_head.weight":
        return "output.weight"
    if normalized == "norm.weight":
        return "output_norm.weight"

    # Per-layer: layers.{N}.{slot}.weight
    if not normalized.startswith("layers."):
        return None
    rest = normalized[len("layers."):]
    if "." not in rest:
        return None
    layer_idx, slot_full = rest.split(".", 1)
    if not slot_full.endswith(".weight"):
        return None
    slot = slot_full[:-len(".weight")]

    translation = {
        # MLP
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        # FullAttention
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        # LinearAttention (Gated DeltaNet)
        "linear_attn.in_proj_qkv": "attn_qkv",
        "linear_attn.in_proj_z": "attn_gate",
        "linear_attn.in_proj_a": "ssm_alpha",
        "linear_attn.in_proj_b": "ssm_beta",
        "linear_attn.out_proj": "ssm_out",
    }.get(slot)
    if translation is None:
        return None
    return f"blk.{layer_idx}.{translation}.weight"


def awq_eligible(safetensors_name: str, *, f1_only: bool = False) -> bool:
    """Mirror of `main.rs::awq_eligible`.

    F1 set: input-side projections (q_proj, k_proj, v_proj, qkv_proj,
    wqkv, gate_proj, up_proj, w_gate, w_up, gate_up_proj, in_proj_*,
    mlp.gate, router).

    F2 expansion (default ON, per master-doc 2026-05-14): adds
    output-side projections (o_proj, wo, out_proj, down_proj, w_down).

    Pass `f1_only=True` to exclude the F2 additions — for A/B
    comparison parity with `HIPFIRE_AWQ_F1_ONLY=1` Rust runs.
    """
    f1_match = any(safetensors_name.endswith(s) for s in (
        # Full-attention input projections (HF + fused variants).
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "qkv_proj.weight", "wqkv.weight",
        # MLP input projections.
        "gate_proj.weight", "up_proj.weight",
        "w_gate.weight", "w_up.weight",
        # MoE fused expert gate+up.
        "gate_up_proj.weight",
        # MoE router (HF naming) — also `router.weight` for non-HF arches.
        "mlp.gate.weight", "router.weight",
        # Final logits projection — added 2026-05-18 (gptq_lm_head_awq.md).
        # Semantically input-side: lm_head's "activation" is the final
        # hidden state, post-RMSNorm. Only used by quantize when the
        # untied-embedding guard passes AND --lm-head-format mq4-awq is
        # set; otherwise lm_head stays at Q8 (default).
        "lm_head.weight", "output.weight",
    )) or ".in_proj_" in safetensors_name  # linear-attn input substrings
    if f1_only:
        return f1_match
    f2_match = any(safetensors_name.endswith(s) for s in (
        # Output-side projections (F2 added 2026-05-14).
        "o_proj.weight", "wo.weight",
        "out_proj.weight",
        "down_proj.weight", "w_down.weight",
    ))
    return f1_match or f2_match
