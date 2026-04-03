#!/bin/bash
# Pre-compile all HIP kernels for target GPU architectures.
# Usage: ./scripts/compile-kernels.sh [arch1 arch2 ...]
# Default: gfx1010 gfx1030 gfx1100 gfx1200
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$SCRIPT_DIR/kernels/src"
OUT_BASE="$SCRIPT_DIR/kernels/compiled"

# Default target architectures
if [ $# -gt 0 ]; then
    ARCHS=("$@")
else
    ARCHS=(gfx1010 gfx1030 gfx1100 gfx1200 gfx1201)
fi

echo "=== hipfire kernel compiler ==="
echo "Source: $SRC_DIR"
echo "Architectures: ${ARCHS[*]}"

TOTAL=0
FAILED=0

for arch in "${ARCHS[@]}"; do
    out_dir="$OUT_BASE/$arch"
    mkdir -p "$out_dir"
    echo ""
    echo "--- $arch ---"

    for src in "$SRC_DIR"/*.hip; do
        name=$(basename "$src" .hip)

        # Check for arch-specific variant: name.arch.hip overrides name.hip
        variant="$SRC_DIR/${name}.${arch}.hip"
        if [ -f "$variant" ]; then
            src="$variant"
            echo "  [variant] $name ($arch-specific)"
        fi

        out="$out_dir/${name}.hsaco"
        TOTAL=$((TOTAL + 1))

        if hipcc --genco --offload-arch="$arch" -O3 -I "$SRC_DIR" \
            -o "$out" "$src" 2>/dev/null; then
            size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null)
            echo "  ✓ $name ($(( size / 1024 )) KB)"
        else
            echo "  ✗ $name FAILED"
            FAILED=$((FAILED + 1))
            rm -f "$out"
        fi
    done
done

echo ""
echo "=== Done: $((TOTAL - FAILED))/$TOTAL compiled, $FAILED failed ==="
[ $FAILED -eq 0 ] || exit 1
