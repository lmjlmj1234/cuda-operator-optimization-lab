#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BINARY="$BUILD_DIR/cuda_operators"

if [ ! -f "$BINARY" ]; then
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$PROJECT_ROOT" && cmake --build . --parallel
    cd "$PROJECT_ROOT"
fi

echo "=== Fused Residual + LayerNorm Benchmark ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""
"$BINARY" 2>&1 | grep -E "fused_residual|Fused"

echo ""
echo "For ncu profiling:"
echo "  ncu --set full --kernel-name 'fused_residual_layernorm*' $BINARY"
