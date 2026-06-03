#!/bin/bash
# Profile hipfire decode path using rocprofv2
# Runs the `run` example for a short decode, capturing kernel timing
set -e

HIPFIRE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$HOME/.hipfire/models/qwen3.6-27b.mq4"
BIN="$HIPFIRE_DIR/target/release/examples/run"
OUTDIR="$HIPFIRE_DIR/.codeinsight+research/rocmprof-decode"
mkdir -p "$OUTDIR"

echo "=== hipfire decode profiling ==="
echo "Output dir: $OUTDIR"

# Run with kernel trace to see all kernel launches and durations
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Small prompt, short decode (5 tokens) to capture typical decode kernels
PROMPT_FILE="$HIPFIRE_DIR/benchmarks/prompts/humaneval_3_below_zero.txt"
if [ ! -f "$PROMPT_FILE" ]; then
    PROMPT="Write a function to add two numbers."
else
    PROMPT=$(cat "$PROMPT_FILE" | head -c 200)
fi

echo "Running hipfire decode with rocprofv2 kernel trace..."
rocprofv2 --kernel-trace -d "$OUTDIR/kt" \
    "$BIN" "$MODEL" \
    --prompt "$PROMPT" \
    --max 5 \
    --no-chatml \
    2>&1 | tail -5

echo ""
echo "=== Kernel trace output ==="
ls -la "$OUTDIR/kt/" 2>/dev/null || echo "No kernel trace output found"