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
#   - 原生 Linux (WSL2 不支持 GPU 性能计数器直通)
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
    --import-source yes       # 导入源码
    --page details            # 详细报告
    --csv                     # CSV 输出 (便于后续分析)
)

# ============================================================
# 1. 检查环境
# ============================================================
echo "=========================================="
echo " CUDA 算子性能测试"
echo "=========================================="

# 检测 WSL2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "WARNING: 检测到 WSL2 环境。"
    echo "  WSL2 使用 GPU 半虚拟化 (GPU-PV)，无法直通硬件性能计数器。"
    echo "  ncu 将因 ERR_NVGPUCTRPERM 无法采集硬件指标。"
    echo ""
    echo "  替代方案:"
    echo "    1. 在原生 Linux (非 WSL2) 上运行本脚本"
    echo "    2. 直接运行 ./build/cuda_operators 获取软件计时 (cudaEvent)"
    echo "    3. 在 Windows 上用 ncu-ui 打开 .nsys-rep 文件"
    echo ""
fi

if ! command -v $NCU &> /dev/null; then
    echo "WARNING: ncu 未安装。仅运行软件计时基准测试。"
    echo "  安装: apt install cuda-nsight-compute"
    NCU=""
fi

if [ ! -f "$BINARY" ]; then
    echo "编译二进制不存在，正在编译..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. && cmake --build . --parallel
    cd "$PROJECT_ROOT"
fi

mkdir -p "$REPORT_DIR"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Report dir: $REPORT_DIR"
echo ""

# ============================================================
# 2. 在 WSL2 环境下用 nsys 做轻量级 profile
# ============================================================
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "=========================================="
    echo " 使用 Nsight Systems 做轻量级 profile"
    echo "=========================================="

    if command -v nsys &> /dev/null; then
        echo "运行 nsys (CUDA API 追踪)..."
        nsys profile --trace=cuda --stats=true \
            -o "$REPORT_DIR/nsys_profile" \
            --force-overwrite=true \
            "$BINARY" 2>&1 | tail -20 || true
        echo "  nsys 报告: $REPORT_DIR/nsys_profile.nsys-rep"
    fi

    echo ""
    echo "=========================================="
    echo " 输出软件计时基准结果"
    echo "=========================================="
    echo ""
    "$BINARY" 2>&1 | grep -E "^\s+" | head -13
    echo ""
    echo "注意: 完整的 ncu 硬件指标 (SM Throughput, DRAM Throughput, Occupancy)"
    echo "  需要在原生 Linux 上运行。WSL2 无法直通 GPU 性能计数器。"
    exit 0
fi

# ============================================================
# 3. 原生 Linux: 使用 ncu 硬件 Profiling
# ============================================================
if [ -z "$NCU" ]; then
    echo "ncu 未安装，仅运行软件计时基准。"
    "$BINARY" 2>&1 | grep -E "^\s+"
    exit 0
fi

# ============================================================
# 逐 Kernel Profiling
# ============================================================
run_ncu() {
    local kernel_name="$1"
    local kernel_pattern="$2"
    local out_file="$REPORT_DIR/${kernel_name}.ncu-rep"

    echo "--- Profiling: $kernel_name (pattern: $kernel_pattern) ---"

    $NCU "${NCU_OPTS[@]}" \
        --kernel-name "$kernel_pattern" \
        --export "$out_file" \
        --target-processes all \
        "$BINARY" 2>&1 | tail -5

    echo "   报告: $out_file"
    echo ""
}

# 对于使用模板的 kernel，nvcc 做 name mangling
# kernel-name 用正则匹配核函数名称即可
# 注意: 过滤掉非目标 kernel（如 softmax_warp 匹配 softmax_naive 等需要更精确的模式）

echo "=========================================="
echo "逐 Kernel 硬件 Profiling"
echo "=========================================="

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
echo "=========================================="
echo "轻量级快速指标收集 (metrics mode)"
echo "=========================================="

# 用 --metrics 模式快速收集关键指标
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
echo "=========================================="
echo " Profiling 完成!"
echo " 报告目录: $REPORT_DIR"
echo " 可视化: ncu-ui reports/*.ncu-rep"
echo "=========================================="
