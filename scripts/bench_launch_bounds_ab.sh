#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# A/B end-to-end bench: launch_bounds (32,2) vs (32,10) on gfx1151
# Runs the actual Qwen 3.6 27B model through dflash_spec_demo with both
# AR and DFlash paths, across code and prose prompts.

set -euo pipefail

TARGET="/home/kread/.hipfire/models/qwen3.6-27b.mq4"
DRAFT="/home/kread/.hipfire/models/qwen35-27b-dflash-mq4.hfq"
CODE_PROMPT="benchmarks/prompts/lru_cache_pep8_strict.txt"
PROSE_PROMPT="benchmarks/prompts/merge_sort_thinking_off.txt"
MAX_TOKENS=256
CTX=4096
KV_MODE="q8"
RUNS=3

BENCH_DIR="benchmarks/results/launch_bounds_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH_DIR"

# Verify files exist
for f in "$TARGET" "$DRAFT" "$CODE_PROMPT" "$PROSE_PROMPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found"
    exit 1
  fi
done

echo "=== launch_bounds A/B bench ==="
echo "Target: $TARGET"
echo "Draft:  $DRAFT"
echo "Code:   $CODE_PROMPT ($(md5sum "$CODE_PROMPT" | cut -c1-8))"
echo "Prose:  $PROSE_PROMPT ($(md5sum "$PROSE_PROMPT" | cut -c1-8))"
echo "Results: $BENCH_DIR"
echo ""

run_demo() {
  local label="$1"
  local extra="$2"
  local prompt_file="$3"
  local outfile="$BENCH_DIR/${label}.txt"

  for i in $(seq 1 "$RUNS"); do
    echo "--- $label run $i ---"
    ./target/release/examples/dflash_spec_demo \
      --target "$TARGET" \
      --draft "$DRAFT" \
      --prompt-file "$prompt_file" \
      --max "$MAX_TOKENS" --ctx "$CTX" --kv-mode "$KV_MODE" \
      --no-adaptive-b --no-chatml \
      $extra \
      2>&1 | tee -a "$outfile"
    echo "" >> "$outfile"
    echo "--- end run $i ---"
    # Small sleep to let DPM settle
    sleep 2
  done
}

# -------------------------------------------------------
# Step 1: Revert to ORIGINAL (32,2), build, bench
# -------------------------------------------------------
echo "========== PHASE A: ORIGINAL launch_bounds(32,2) =========="
for f in kernels/src/*.gfx1151.hip; do
  [[ "$(basename "$f")" == *setprio* || "$(basename "$f")" == *hiocc* ]] && continue
  sed -i 's/__launch_bounds__(32, 10)/__launch_bounds__(32, 2)/g' "$f"
done
rm -rf .hipfire_kernels/gfx1151
cargo build --release --example dflash_spec_demo -p hipfire-runtime 2>&1 | tail -1
echo "Binary: $(md5sum target/release/examples/dflash_spec_demo | cut -c1-8)"

# AR baseline
run_demo "orig_ar_code"   "--ar-baseline" "$CODE_PROMPT"
run_demo "orig_ar_prose"  "--ar-baseline" "$PROSE_PROMPT"

# DFlash
run_demo "orig_dflash_code"   "" "$CODE_PROMPT"
run_demo "orig_dflash_prose"  "" "$PROSE_PROMPT"

# -------------------------------------------------------
# Step 2: Apply HIOCC (32,10), rebuild, bench
# -------------------------------------------------------
echo ""
echo "========== PHASE B: HIOCC launch_bounds(32,10) =========="
for f in kernels/src/*.gfx1151.hip; do
  [[ "$(basename "$f")" == *setprio* || "$(basename "$f")" == *hiocc* ]] && continue
  sed -i 's/__launch_bounds__(32, 2)/__launch_bounds__(32, 10)/g' "$f"
done
rm -rf .hipfire_kernels/gfx1151
cargo build --release --example dflash_spec_demo -p hipfire-runtime 2>&1 | tail -1
echo "Binary: $(md5sum target/release/examples/dflash_spec_demo | cut -c1-8)"

# AR baseline
run_demo "hiocc_ar_code"   "--ar-baseline" "$CODE_PROMPT"
run_demo "hiocc_ar_prose"  "--ar-baseline" "$PROSE_PROMPT"

# DFlash
run_demo "hiocc_dflash_code"   "" "$CODE_PROMPT"
run_demo "hiocc_dflash_prose"  "" "$PROSE_PROMPT"

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "========== SUMMARY =========="
echo ""

extract_stat() {
  local file="$1"
  # Extract tok/s from the last "emitted: ... tok/s" line in each run
  grep -oP 'emitted: \d+ tokens in [\d.]+s\s+\([\d.]+ tok/s\)' "$file" | \
    grep -oP '[\d.]+(?= tok/s)' | sort -n
}

for mode in ar_code ar_prose dflash_code dflash_prose; do
  orig_file="$BENCH_DIR/orig_${mode}.txt"
  hiocc_file="$BENCH_DIR/hiocc_${mode}.txt"

  orig_vals=$(extract_stat "$orig_file" 2>/dev/null || echo "N/A")
  hiocc_vals=$(extract_stat "$hiocc_file" 2>/dev/null || echo "N/A")

  if [ "$orig_vals" != "N/A" ] && [ "$hiocc_vals" != "N/A" ]; then
    orig_med=$(echo "$orig_vals" | awk 'NR==int(('$RUNS'+1)/2)')
    hiocc_med=$(echo "$hiocc_vals" | awk 'NR==int(('$RUNS'+1)/2)')
    if [ -n "$orig_med" ] && [ -n "$hiocc_med" ] && [ "$orig_med" != "0" ]; then
      delta=$(echo "scale=1; ($hiocc_med - $orig_med) * 100 / $orig_med" | bc)
      printf "  %-20s  orig=%.1f  hiocc=%.1f  Δ=%+.1f%%\n" "$mode" "$orig_med" "$hiocc_med" "$delta"
    else
      printf "  %-20s  (insufficient data)\n" "$mode"
    fi
  else
    printf "  %-20s  (missing)\n" "$mode"
  fi
done

echo ""
echo "Raw results in: $BENCH_DIR"
