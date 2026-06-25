#!/bin/bash
# ============================================================
# CUDA 算子性能测试脚本 (基于 Nsight Compute)
#
# 使用 ncu 命令行工具抓取真实的硬件性能指标:
#   - 内存带宽利用率 (Memory Throughput)
#   - 计算吞吐量 (SM Throughput)
#   - 执行时间 (Kernel Duration)
#   - 占用率 (Occupancy)
#
# 用法:
#   bash tests/test_performance.sh
#
# 需要:
#   - CUDA Toolkit (ncu 命令)
#   - 已编译的 cuda_operators 二进制
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BINARY="$BUILD_DIR/cuda_operators"
REPORT_DIR="$PROJECT_ROOT/reports"

# ============================================================
# 配置
# ============================================================
NCU="ncu"
NCU_OPTS=(
    --set full                # 全指标集合
    --import-source yes        # 导入源码
    --page details             # 详细报告
    --csv                      # CSV 输出 (便于后续分析)
)

# ============================================================
# 1. 检查环境
# ============================================================
echo "=========================================="
echo " CUDA 算子性能测试"
echo "=========================================="

if ! command -v $NCU &> /dev/null; then
    echo "ERROR: ncu 未安装。请安装 CUDA Toolkit (Nsight Compute)"
    echo "  apt install cuda-nsight-compute"
    exit 1
fi

if [ ! -f "$BINARY" ]; then
    echo "编译二进制不存在，正在编译..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. && cmake --build . --parallel
    cd "$PROJECT_ROOT"
fi

mkdir -p "$REPORT_DIR"

NCU_VERSION=$($NCU --version 2>&1 | head -1)
echo "NCU Version: $NCU_VERSION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Report dir: $REPORT_DIR"
echo ""

# ============================================================
# 2. 各 Kernel 逐 Profiling
# ============================================================
run_ncu() {
    local kernel_name="$1"
    local kernel_pattern="$2"
    local out_file="$REPORT_DIR/${kernel_name}.ncu-rep"

    echo "--- Profiling: $kernel_name ---"

    $NCU "${NCU_OPTS[@]}" \
        --kernel-name "$kernel_pattern" \
        --export "$out_file" \
        --target-processes all \
        "$BINARY" 2>&1 | tail -5

    echo "   报告: $out_file"
    echo ""
}

# 对于使用模板的 kernel，_nv 会自动做 name mangling
# kernel-name 用正则匹配即可，例如 "softmax_naive"

# Softmax
# run_ncu "softmax_naive"       "softmax_naive*"       # 取消注释以执行
# run_ncu "softmax_online"      "softmax_online*"

# LayerNorm
# run_ncu "layernorm_sum_sq"    "layernorm_sum_sq*"
# run_ncu "layernorm_float4"    "layernorm_float4*"
# run_ncu "layernorm_welford"   "layernorm_welford*"

# RMSNorm
# run_ncu "rmsnorm_kernel"      "rmsnorm_kernel*"
# run_ncu "rmsnorm_float4"      "rmsnorm_float4*"

# Fused LayerNorm
# run_ncu "fused_residual_ln"   "fused_residual_layernorm*"
# run_ncu "fused_residual_ln_f4" "fused_residual_layernorm_float4*"

echo "=========================================="
echo " 快速吞吐量估算 (metrics mode)"
echo "=========================================="

# ============================================================
# 3. 使用 --metrics 模式做轻量级指标收集 (更快)
# ============================================================
METRICS=(
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
)

echo "收集关键指标 (每个 kernel 约 5 秒)..."
echo ""

for kernel_ptn in "softmax_naive" "softmax_online" \
                  "layernorm_sum_sq" "layernorm_float4" "layernorm_welford" \
                  "rmsnorm_kernel" "rmsnorm_float4" \
                  "fused_residual_layernorm" "fused_residual_layernorm_float4"; do
    echo "  [metrics] $kernel_ptn ..."
    $NCU --metrics "${METRICS[*]}" \
         --kernel-name "${kernel_ptn}*" \
         --csv \
         --target-processes all \
         "$BINARY" 2>/dev/null | tail -1 || true
done

echo ""
echo "=========================================="
echo " 测试完成!"
echo " 完整报告目录: $REPORT_DIR"
echo " 用 ncu-ui 打开可视化: ncu-ui reports/*.ncu-rep"
echo "=========================================="
