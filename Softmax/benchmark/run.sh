#!/bin/bash
# Benchmark all Softmax kernel variants
# Requires: compiled cuda_operators binary

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BINARY="$BUILD_DIR/cuda_operators"

if [ ! -f "$BINARY" ]; then
    echo "Binary not found. Building first..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$PROJECT_ROOT" && cmake --build . --parallel
    cd "$PROJECT_ROOT"
fi

echo "=== Softmax Benchmark ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

# Run cudaEvent timing for all softmax variants
# (the benchmark driver runs ALL operators; we grep softmax lines)
"$BINARY" 2>&1 | grep -E "softmax|correctness"

echo ""
echo "For ncu profiling (requires native Linux):"
echo "  ncu --set full --kernel-name 'softmax_naive*' $BINARY"
echo "  ncu --set full --kernel-name 'softmax_online*' $BINARY"
echo "  ncu --set full --kernel-name 'softmax_warp*' $BINARY"
