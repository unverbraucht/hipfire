#!/usr/bin/env bash
# 9B GPTQ overnight pipeline — CUDA path.
#
# Replaces the ~14h Rust-CPU GPTQ run with the Python+CUDA pipeline at
# `scripts/gptq_cuda.py` (per `docs/plans/gptq_cuda.md`). Target wall:
# 1-3h on the dual RTX 5070 Ti box.
#
# Stage 1 (Python+CUDA): build the precomputed-gptq manifest.
# Stage 2 (Rust):        consume manifest, emit .hfq.
#
# Usage:
#   bash scripts/gptq_9b_overnight.sh \\
#       [--alpha 0.55] \\
#       [--source-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha>/] \\
#       [--hessian /data/hipfire-refs/qwen3.5-9b-bf16.hessian.bin] \\
#       [--imatrix benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.imatrix.gguf] \\
#       [--manifest-out ~/.hipfire/gptq-precomputed/qwen3.5-9b-mq4-awq-gptq-q8conv-f2/] \\
#       [--hfq-out ~/.hipfire/quantized/qwen3.5-9b.mq4-awq-gptq-q8conv-f2-cuda.hfq] \\
#       [--devices "cuda:0 cuda:1"]
#
# Validation steps after this finishes (per plan §5.2 + §9):
#   - copy the .hfq to the gfx1100 box for runtime eval
#   - `eval_hipfire n=512 q8-KV` on master vs this .hfq
#   - paired-t on per-chunk NLL vs F2 α=0.55 anchor (KLD 0.1830 / NLL 2.1730 / PPL 8.79)

set -euo pipefail

ALPHA=0.55
SOURCE_MODEL=""
HESSIAN="/data/hipfire-refs/qwen3.5-9b-bf16.hessian.bin"
IMATRIX="$(git rev-parse --show-toplevel)/benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.imatrix.gguf"
MANIFEST_OUT="$HOME/.hipfire/gptq-precomputed/qwen3.5-9b-mq4-awq-gptq-q8conv-f2"
HFQ_OUT="$HOME/.hipfire/quantized/qwen3.5-9b.mq4-awq-gptq-q8conv-f2-cuda.hfq"
DEVICES=("cuda:0" "cuda:1")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --alpha) ALPHA="$2"; shift 2 ;;
        --source-model) SOURCE_MODEL="$2"; shift 2 ;;
        --hessian) HESSIAN="$2"; shift 2 ;;
        --imatrix) IMATRIX="$2"; shift 2 ;;
        --manifest-out) MANIFEST_OUT="$2"; shift 2 ;;
        --hfq-out) HFQ_OUT="$2"; shift 2 ;;
        --devices) IFS=' ' read -r -a DEVICES <<< "$2"; shift 2 ;;
        -h|--help) sed -n '4,28p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Auto-detect the Qwen3.5-9B HF cache snapshot if --source-model wasn't given.
if [[ -z "$SOURCE_MODEL" ]]; then
    HF_REPO="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots"
    if [[ ! -d "$HF_REPO" ]]; then
        echo "error: --source-model not set and no HF cache at $HF_REPO" >&2
        echo "       Download: huggingface-cli download Qwen/Qwen3.5-9B" >&2
        exit 1
    fi
    SOURCE_MODEL="$(ls -td "$HF_REPO"/*/ | head -n1)"
    SOURCE_MODEL="${SOURCE_MODEL%/}"
    echo "[auto] source model: $SOURCE_MODEL"
fi

for path in "$SOURCE_MODEL" "$HESSIAN" "$IMATRIX"; do
    if [[ ! -e "$path" ]]; then
        echo "error: missing input: $path" >&2
        exit 1
    fi
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p "$(dirname "$MANIFEST_OUT")" "$(dirname "$HFQ_OUT")"

LOG="$MANIFEST_OUT.log"

echo "=== Stage 1: Python+CUDA GPTQ → manifest ==="
echo "    source:   $SOURCE_MODEL"
echo "    hessian:  $HESSIAN"
echo "    imatrix:  $IMATRIX"
echo "    alpha:    $ALPHA"
echo "    devices:  ${DEVICES[*]}"
echo "    manifest: $MANIFEST_OUT"
echo "    log:      $LOG"
echo

# Tee output so the user can watch and grep clamp counts mid-run.
./.venv-cuda/bin/python scripts/gptq_cuda.py \
    --input "$SOURCE_MODEL" \
    --hessian "$HESSIAN" \
    --imatrix "$IMATRIX" \
    --alpha "$ALPHA" \
    --output "$MANIFEST_OUT" \
    --devices "${DEVICES[@]}" \
    --verbose 2>&1 | tee "$LOG"

echo
echo "=== Stage 2: Rust → .hfq ==="
echo "    manifest: $MANIFEST_OUT"
echo "    output:   $HFQ_OUT"
echo

cargo build --release -p hipfire-quantize

./target/release/hipfire-quantize \
    --input "$SOURCE_MODEL" \
    --output "$HFQ_OUT" \
    --format mq4 \
    --precomputed-gptq-path "$MANIFEST_OUT"

echo
echo "=== Done ==="
echo "manifest: $MANIFEST_OUT"
echo "hfq:      $HFQ_OUT"
echo "log:      $LOG"
echo
echo "Next: copy the .hfq to the gfx1100 box and run eval_hipfire n=512 q8-KV"
echo "      Compare per-chunk NLL paired-t vs F2 α=0.55 anchor"
echo "      (anchor: KLD 0.1830 / NLL 2.1730 / PPL 8.79 from master-doc §1.1h/i)"
