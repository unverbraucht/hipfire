#!/usr/bin/env bash
# 4B MQ3 sweep — compare RTN / AWQ-only / GPTQ-only / AWQ+GPTQ at 3-bit
# to figure out whether uniform-MQ3 collapse can be averted by activation-
# aware quant (master-doc §5 reports uniform MQ3 RTN collapse on every
# locally-tested model size).
#
# Output: 4 .hfq files under ~/.hipfire/quantized/mq3-sweep/, each ~2 GB.
# Validation: copy all 4 to gfx1100, run coherence-gate on each, see which
# (if any) produce coherent output.
#
# Usage:
#   bash scripts/mq3_sweep_4b.sh
#
# Wall: ~25 min/variant × 4 = ~1h40m total. Each variant runs Stage C
# (Python+CUDA → manifest) then Stage D (Rust → .hfq).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

. ~/.cargo/env  # cargo on PATH for Stage D

MODEL=/data/models/qwen/Qwen3.5-4B
HESSIAN=/data/hipfire-refs/qwen3.5-4b-bf16.hessian.bin
IMATRIX="$REPO_ROOT/benchmarks/quality-baselines/refs/qwen3.5-4b-bf16.imatrix.gguf"

OUT_BASE="$HOME/.hipfire/gptq-precomputed"
HFQ_BASE="$HOME/.hipfire/quantized/mq3-sweep"
LOG_BASE="$HOME/.hipfire/mq3-sweep-logs"

mkdir -p "$OUT_BASE" "$HFQ_BASE" "$LOG_BASE"

# Ensure the Rust binary is built (--precomputed-gptq-path with n_bits=3 dispatch).
cargo build --release -p hipfire-quantize > "$LOG_BASE/00-cargo-build.log" 2>&1

run_variant() {
    local NAME=$1
    local USE_HESSIAN=$2   # "yes" / "no"
    local USE_IMATRIX=$3   # "yes" / "no"

    echo "=================================================================="
    echo "[$(date '+%H:%M:%S')] Variant: $NAME"
    echo "  hessian=$USE_HESSIAN imatrix=$USE_IMATRIX"
    echo "=================================================================="

    local MANIFEST_OUT="$OUT_BASE/qwen3.5-4b-mq3-$NAME"
    local HFQ_OUT="$HFQ_BASE/qwen3.5-4b.mq3-$NAME.hfq"
    local STAGE_C_LOG="$LOG_BASE/$NAME-stage-c.log"
    local STAGE_D_LOG="$LOG_BASE/$NAME-stage-d.log"

    rm -rf "$MANIFEST_OUT" "$HFQ_OUT"

    # Build python command with conditional flags.
    local PY_ARGS=(
        --input "$MODEL"
        --alpha 0.55
        --bits 3
        --output "$MANIFEST_OUT"
        --devices cuda:0 cuda:1
        --verbose
    )
    if [[ "$USE_HESSIAN" == "yes" ]]; then
        PY_ARGS+=(--hessian "$HESSIAN")
    fi
    if [[ "$USE_IMATRIX" == "yes" ]]; then
        PY_ARGS+=(--imatrix "$IMATRIX")
    fi

    echo "[$(date '+%H:%M:%S')] Stage C — Python+CUDA → manifest"
    ./.venv-cuda/bin/python scripts/gptq_cuda.py "${PY_ARGS[@]}" \
        > "$STAGE_C_LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] Stage C done. Last 3 log lines:"
    tail -3 "$STAGE_C_LOG"

    echo "[$(date '+%H:%M:%S')] Stage D — Rust → .hfq"
    ./target/release/hipfire-quantize \
        --input "$MODEL" \
        --output "$HFQ_OUT" \
        --format mq3 \
        --precomputed-gptq-path "$MANIFEST_OUT" \
        > "$STAGE_D_LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] Stage D done. Output: $(ls -lh "$HFQ_OUT" | awk '{print $5}')"
    echo

    # Clean up the manifest to save disk — keep only the .hfq + logs.
    rm -rf "$MANIFEST_OUT"
}

# Order: cheapest → most expensive, so we get partial results fast.
# RTN is the cheapest (no Hessian load, no GPTQ inner loop).
run_variant rtn        no  no
run_variant awq-only   no  yes
run_variant gptq-only  yes no
run_variant awq-gptq   yes yes

echo "=================================================================="
echo "All 4 variants complete."
echo "Output: $HFQ_BASE/"
ls -lh "$HFQ_BASE/"
echo
echo "md5 sums:"
md5sum "$HFQ_BASE"/*.hfq
echo
echo "Logs: $LOG_BASE/"
echo "Next: copy all 4 .hfq files to gfx1100, run coherence-gate on each."
echo "  RTN expected to collapse (master-doc §5)."
echo "  AWQ-only / GPTQ-only / AWQ+GPTQ: the open question."
