#!/usr/bin/env bash
# audit-dispatch-coverage.sh — per-quant runtime-dispatch coverage gate.
#
# Background
# ----------
# `hipfire-quantize` produces files with a `quant_type` u32 in the header.
# `crates/hipfire-runtime/src/hfq.rs::load_weight_tensor_raw()` translates
# that into a `DType::*G256` variant attached to each weight tensor.
#
# The forward pass dispatches on `WeightTensor.gpu_dtype` at every per-layer
# call site (~28 matchers in `crates/hipfire-arch-qwen35/src/qwen35.rs`),
# selecting between rotated (MQ-family) and non-rotated (HFQ-family) paths.
# When a `DType::*G256` is missing from a matcher, the corresponding weight
# silently falls through to the default arm and is read at the wrong byte
# stride. Result: stride-mismatched arithmetic, garbage prefill state, no
# panic, no error — just nonsense tokens at the bench-measured speed.
#
# This script grep-audits both surfaces and reports the matrix gaps. It
# is the runtime-dispatch sibling of build-time kernel-source compilation
# tests (those catch *kernel* gaps; this catches *wiring* gaps).
#
# Discovery history
# -----------------
# - 2026-05-06: gemv_mq8g256 unbuildable on gfx906 (sudot4 builtin needs
#   RDNA3+ feature). Fixed in `ee0fac6`. After fix, end-to-end bench on
#   qwen3.5-9b.mq8 produced suspiciously-clean numbers — investigated and
#   found 14 `is_mq` matchers in qwen35.rs excluded `MQ8G256`. Same audit
#   surfaced two more silent-corruption-latent gaps (MQ3 missing from
#   MoE-batched matchers — upstream issue #179; MQ2G256 has 0/28 coverage).
# - This script is the automated form of plan §5.4 part 2's audit
#   methodology, so future PRs introducing new DType variants can run it
#   pre-merge instead of relying on suspicious-bench-result detection.
#
# Usage
# -----
#   ./scripts/audit-dispatch-coverage.sh          # report; exit non-zero on gaps
#   ./scripts/audit-dispatch-coverage.sh --quiet  # exit code only, no report
#   ./scripts/audit-dispatch-coverage.sh --json   # machine-readable report
#
# Exit codes
# ----------
#   0  no gaps in deployed-quant × deployed-matcher coverage
#   1  one or more silent-corruption-latent gaps found
#   2  parse error (loader or arch crate has unexpected structure)

set -u
cd "$(dirname "$0")/.."

QUIET=0
JSON=0
while [ $# -gt 0 ]; do
    case "$1" in
        --quiet) QUIET=1; shift ;;
        --json)  JSON=1; shift ;;
        -h|--help)
            sed -n '2,38p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ── Step 1 — enumerate loader-producible DTypes for per-layer rotated quants ──
#
# We're looking at the MQ-family rotated quants specifically (the ones that
# need FWHT activation rotation pre-pass). HFQ-family non-rotated quants
# don't go through `is_mq` matchers, so they're not in scope.
#
# The loader maps quant_type → DType::*G256 in two locations:
# - crates/hipfire-runtime/src/hfq.rs: runtime-side default mapping
# - crates/hipfire-arch-*/src/*.rs: per-arch loader (e.g. qwen35.rs has its
#   own quant_type → DType match for arch-specific tensor handling)
#
# Both produce WeightTensor values whose gpu_dtype reaches the per-layer
# matchers, so both need to be scanned.

LOADER_FILES=$(find crates/hipfire-runtime/src crates/hipfire-arch-*/src -name "*.rs" 2>/dev/null)
if [ -z "$LOADER_FILES" ]; then
    echo "ERR: no loader sources found — run from repo root?" >&2; exit 2
fi

LOADER_DTYPES=$(
    grep -hE 'gpu_dtype: DType::MQ[0-9]+G256' $LOADER_FILES \
        | sed -E 's/.*DType::(MQ[0-9]+G256).*/\1/' \
        | sort -u
)

if [ -z "$LOADER_DTYPES" ]; then
    echo "ERR: parsed zero MQ*G256 dtypes from loader sources — pattern changed?" >&2
    exit 2
fi

# ── Step 2 — enumerate per-layer matchers across all arch crates ──

ARCH_FILES=$(find crates/hipfire-arch-*/src -name "*.rs" 2>/dev/null)
if [ -z "$ARCH_FILES" ]; then
    echo "ERR: no arch crate sources found in crates/hipfire-arch-*/src/" >&2
    exit 2
fi

# Each matcher line has form:
#   let X_is_mq = matches!(layer.Y.gpu_dtype, DType::MQ4G256 | DType::MQ6G256 | ...);
# We extract per-line: file:line, the matcher-variable name (the X_is_mq part),
# and the set of DType::* variants the matcher includes.

# Use a temp file to hold matcher data; format (TSV):
#   file<TAB>line<TAB>matcher_name<TAB>dtypes_csv
TMPDIR_AUDIT=$(mktemp -d)
trap 'rm -rf "$TMPDIR_AUDIT"' EXIT
MATCHERS_FILE="$TMPDIR_AUDIT/matchers.tsv"
> "$MATCHERS_FILE"

for f in $ARCH_FILES; do
    grep -nE '= matches!\(layer\.[a-z_0-9]+\.gpu_dtype, DType::' "$f" \
    | while IFS=':' read -r lineno rest; do
        # rest looks like:
        #   "                let qkv_is_mq = matches!(layer.wq.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);"
        name=$(echo "$rest" | sed -E 's/.*let ([a-z_0-9]+) = matches!.*/\1/')
        # extract the | -separated DType set inside the second matches! arg
        dtypes=$(echo "$rest" \
            | sed -E 's/.*matches!\(layer\.[a-z_0-9]+\.gpu_dtype, ([^)]*)\).*/\1/' \
            | sed 's/ DType:://g; s/^DType:://; s/[ \t]//g; s/|/,/g')
        printf "%s\t%s\t%s\t%s\n" "$f" "$lineno" "$name" "$dtypes" >> "$MATCHERS_FILE"
    done
done

n_matchers=$(wc -l < "$MATCHERS_FILE")

# ── Step 3 — coverage check ──
#
# A matcher named *_is_mq (or just is_mq) is the rotation-gate matcher: it
# decides whether to apply FWHT rotation pre-pass before the GEMV. Every
# MQ*G256 dtype the loader produces MUST appear in every is_mq matcher,
# OR the matcher must not be on a rotated-quant path (which we'd need to
# audit case-by-case — but the convention is is_mq covers the lot).
#
# For now, only flag is_mq* matchers. is_6bit matchers separately gate
# 6-bit kernel selection and are independently audited downstream.

MATCHERS_IS_MQ=$(awk -F'\t' '$3 ~ /is_mq$/ { print }' "$MATCHERS_FILE")
n_is_mq=$(echo "$MATCHERS_IS_MQ" | grep -c . || echo 0)

GAPS_FILE="$TMPDIR_AUDIT/gaps.tsv"
> "$GAPS_FILE"

while IFS=$'\t' read -r f lineno name dtypes; do
    [ -z "$f" ] && continue
    for dt in $LOADER_DTYPES; do
        if ! echo ",$dtypes," | grep -q ",$dt,"; then
            printf "%s\t%s\t%s\t%s\t%s\n" "$f" "$lineno" "$name" "$dtypes" "$dt" >> "$GAPS_FILE"
        fi
    done
done <<< "$MATCHERS_IS_MQ"

n_gaps=$(wc -l < "$GAPS_FILE")

# ── Step 4 — report ──

if [ "$JSON" = "1" ]; then
    echo "{"
    echo "  \"loader_dtypes\": [$(echo "$LOADER_DTYPES" | sed 's/.*/"&"/' | paste -sd, -)],"
    echo "  \"n_matchers_total\": $n_matchers,"
    echo "  \"n_matchers_is_mq\": $n_is_mq,"
    echo "  \"n_gaps\": $n_gaps,"
    echo "  \"gaps\": ["
    awk -F'\t' '{ printf "    {\"file\": \"%s\", \"line\": %s, \"matcher\": \"%s\", \"covers\": \"%s\", \"missing\": \"%s\"}", $1, $2, $3, $4, $5; if (NR < n) printf ","; print "" }' n=$n_gaps "$GAPS_FILE"
    echo "  ]"
    echo "}"
elif [ "$QUIET" = "0" ]; then
    echo "═══ Dispatch-coverage audit ═══"
    echo ""
    echo "Loader-producible MQ-family dtypes:"
    for dt in $LOADER_DTYPES; do echo "  $dt"; done
    echo ""
    echo "Found $n_is_mq is_mq matchers across:"
    awk -F'\t' '{print $1}' "$MATCHERS_FILE" | sort -u | while read -r f; do
        n=$(awk -F'\t' -v f="$f" '$1==f && $3 ~ /is_mq$/' "$MATCHERS_FILE" | wc -l)
        [ "$n" -gt 0 ] && echo "  $n in $f"
    done
    echo ""
    if [ "$n_gaps" = "0" ]; then
        echo "✓ No coverage gaps. Every loader-producible MQ-family dtype appears"
        echo "  in every is_mq matcher across all arch crates."
        echo ""
        exit 0
    else
        echo "✗ FOUND $n_gaps COVERAGE GAPS:"
        echo ""
        awk -F'\t' '{ printf "  %s:%s  %-22s  missing %s  (covers: %s)\n", $1, $2, $3, $5, $4 }' "$GAPS_FILE"
        echo ""
        echo "Each gap is a silent-corruption-latent dispatch path: a weight"
        echo "tensor of the missing dtype will fall through to the default"
        echo "kernel arm and be read at the wrong byte stride, producing"
        echo "garbage state without any panic or error."
        echo ""
        echo "See docs/perf-checkpoints/2026-05-06-mq8-runtime-dispatch-audit.md"
        echo "and plan §5.4 part 2 for the audit methodology and historical"
        echo "context."
        exit 1
    fi
else
    [ "$n_gaps" = "0" ] && exit 0 || exit 1
fi
