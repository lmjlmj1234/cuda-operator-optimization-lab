#!/bin/bash
# ============================================================
# CUDA Operator Performance Test Script (based on Nsight Compute)
#
# Profiles all kernel variants using ncu for hardware metrics:
#   - Memory Throughput
#   - SM Throughput
#   - Kernel Duration
#   - Occupancy
#
# Usage:
#   bash tests/test_performance.sh
#
# Requirements:
#   - CUDA Toolkit (ncu command)
#   - Compiled cuda_operators binary
#   - Native Linux (WSL2 GPU-PV blocks hardware counter passthrough)
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BINARY="$BUILD_DIR/cuda_operators"
NCU_DIR="$PROJECT_ROOT/reports/ncu"
NSYS_DIR="$PROJECT_ROOT/reports/nsys"
BENCHMARK_DIR="$PROJECT_ROOT/reports/benchmark"
mkdir -p "$NCU_DIR" "$NSYS_DIR" "$BENCHMARK_DIR"

NCU="ncu"
NCU_OPTS=(
    --set full
    --import-source yes
    --page details
    --csv
)

echo "=========================================="
echo " CUDA Operator Performance Profiling"
echo "=========================================="

if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "WARNING: WSL2 environment detected."
    echo "  GPU hardware performance counters unavailable (GPU-PV)."
    echo "  ncu will fail with ERR_NVGPUCTRPERM."
    echo ""
    echo "  Alternatives:"
    echo "    1. Run on native Linux (non-WSL2)"
    echo "    2. Run ./build/cuda_operators for software timing (cudaEvent)"
    echo ""
fi

if ! command -v $NCU &> /dev/null; then
    echo "WARNING: ncu not installed. Running software timing only."
    echo "  Install: apt install cuda-nsight-compute"
    NCU=""
fi

if [ ! -f "$BINARY" ]; then
    echo "Binary not found, building..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. && cmake --build . --parallel
    cd "$PROJECT_ROOT"
fi

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

# ============================================================
# WSL2: nsys lightweight profile
# ============================================================
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Using Nsight Systems for lightweight profiling"

    if command -v nsys &> /dev/null; then
        echo "Running nsys (CUDA API trace)..."
        nsys profile --trace=cuda --stats=true \
            -o "$NSYS_DIR/nsys_profile" \
            --force-overwrite=true \
            "$BINARY" 2>&1 | tail -20 || true
        echo "  nsys report: $NSYS_DIR/nsys_profile.nsys-rep"
    fi

    echo ""
    echo "Software timing benchmark results:"
    echo ""
    "$BINARY" 2>&1 | grep -E "^\s+" | head -13
    echo ""
    echo "Note: Full ncu hardware metrics (SM Throughput, DRAM Throughput, Occupancy)"
    echo "  require native Linux. WSL2 cannot pass through GPU performance counters."
    exit 0
fi

# ============================================================
# Native Linux: ncu hardware profiling
# ============================================================
if [ -z "$NCU" ]; then
    echo "ncu not installed, running software timing only."
    "$BINARY" 2>&1 | grep -E "^\s+"
    exit 0
fi

run_ncu() {
    local kernel_name="$1"
    local kernel_pattern="$2"
    local out_file="$NCU_DIR/${kernel_name}.ncu-rep"

    echo "--- Profiling: $kernel_name (pattern: $kernel_pattern) ---"

    $NCU "${NCU_OPTS[@]}" \
        --kernel-name "$kernel_pattern" \
        --export "$out_file" \
        --target-processes all \
        "$BINARY" 2>&1 | tail -5

    echo "    Report: $out_file"
    echo ""
}

echo "Per-Kernel Hardware Profiling"
echo ""

# Softmax
run_ncu "softmax_naive"       "void softmax_naive*"
run_ncu "softmax_online"      "void softmax_online*"
run_ncu "softmax_warp"        "softmax_warp*"

# LayerNorm
run_ncu "layernorm_sum_sq"    "void layernorm_sum_sq*"
run_ncu "layernorm_float4"    "void layernorm_float4*"
run_ncu "layernorm_welford"   "void layernorm_welford*"

# RMSNorm
run_ncu "rmsnorm_kernel"      "void rmsnorm_kernel*"
run_ncu "rmsnorm_float4"      "void rmsnorm_float4*"

# Fused LayerNorm
run_ncu "fused_residual_ln"   "void fused_residual_layernorm*"
run_ncu "fused_residual_ln_f4" "void fused_residual_layernorm_float4*"

echo ""
echo "Lightweight metrics collection"
echo ""

METRICS=(
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
)

for kernel_ptn in "softmax_naive" "softmax_online" \
                  "layernorm_sum_sq" "layernorm_float4" "layernorm_welford" \
                  "rmsnorm_kernel" "rmsnorm_float4" \
                  "fused_residual_layernorm" "fused_residual_layernorm_float4"; do
    echo "  [metrics] $kernel_ptn ..."
    $NCU --metrics "${METRICS[*]}" \
         --kernel-name "${kernel_ptn}*" \
         --csv \
         --target-processes all \
         "$BINARY" 2>/dev/null | grep -v "^#" || true
done

echo ""
echo ""
echo "Profiling complete!"
echo "  ncu reports:     $NCU_DIR/"
echo "  nsys reports:    $NSYS_DIR/"
echo "  benchmark logs:  $BENCHMARK_DIR/"
echo "Visualize: ncu-ui reports/ncu/*.ncu-rep"
echo ""
echo ""
