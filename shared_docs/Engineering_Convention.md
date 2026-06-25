# CUDA Engineering Convention

> **这是一个仓库级规范。** 所有 Kernel 的开发、优化、文档都必须遵守本 Convention。
> 不遵守 = 代码不会被合并。

---

## 1. 每个 Kernel 必须交付的 9 项内容

任何新增或修改的 CUDA Kernel 必须包含以下 9 个部分：

| # | 项目 | 要求 |
|---|------|------|
| 1 | **Problem Definition** | 解决什么问题？为什么需要它？在 Transformer/LLM 中的作用？输入输出规格？时间复杂度和访存模式？ |
| 2 | **Baseline Implementation** | 先写最简单、最容易理解的正确版本。不要一开始就优化。 |
| 3 | **Correctness Verification** | CPU Reference 对比 + GPU 输出 + 误差分析 + 自动验证脚本。需包含 dump 策略说明。 |
| 4 | **Benchmark** | GPU/CUDA/Driver/Problem Size / Kernel Time / Bandwidth / GFLOPS / 重复次数 / 均值 / 标准差。 |
| 5 | **Profiling** | nsight/ncu 数据（或说明为什么拿不到 + 未来如何获取）。SM Throughput / DRAM / Occupancy / L1/L2 / Warp Stall。性能分析必须按照 [Profiling_Guide.md](Profiling_Guide.md) 中的 **Performance Analysis Workflow** 进行。 |
| 6 | **Bottleneck Analysis** | 结合 Benchmark + Memory Pattern + Instruction 分析：Memory-bound 还是 Compute-bound？证据是什么？ |
| 7 | **Optimization History** | v0→v1→v2…每步：做了什么？为什么做？量化收益？副作用？ |
| 8 | **Decision Log** | 每个关键决策：为什么选择这个方案？为什么不用其他方案？有没有更好的替代？ |
| 9 | **Interview Notes** | 面试官问"为什么这样写"时如何回答？容易误解什么？你真正学到了什么？ |

---

## 2. 每次 Commit 必须回答的问题

Commit message 必须包含（可以在 body 中）：

```
Why this optimization?      — 这个优化的动机是什么？
Why not other solutions?    — 为什么不用其他方案？
What gain?                  — 量化收益（时间/带宽/精度）
Side effects?               — 寄存器增加？精度变化？代码复杂度？
What's next?                — 还有哪些可以继续优化？
```

**Bad**: "Optimize softmax"

**Good**:
```
perf(softmax): online one-pass reduces HBM passes 5→3

Why: naive 3-pass wastes bandwidth by reading HBM 3 times
Why not: warp shuffle (only works for dim≤32), float4 (not helpful for coalesced access)
Gain: −39.7% kernel time (2.17→1.31 ms), bandwidth 82%→unchanged
Side effects: extra expf() calls per element, 31 vs 38 registers (fewer!)
What's next: fuse into QK^T matmul for FlashAttention-style elimination of S matrix
```

---

## 3. 每次实验必须保存的数据

每次调参或改 kernel 后的实验记录必须包含：

| 数据 | 格式 | 位置 |
|------|------|------|
| Environment | GPU / CUDA / Driver | docs/ 或 benchmark/ |
| Kernel Config | Grid / Block / Reg / SMEM | docs/ |
| Compiler Flags | nvcc flags, -arch, -O | CMakeLists.txt |
| Benchmark | Time / Bandwidth / GFLOPS | benchmark/ |
| Nsight | ncu metrics 或 nsys trace | benchmark/ |
| SASS / PTX | disassembly (可选) | benchmark/ |
| Reference Output | Correctness golden values | tests/ |
| Regression Result | vs previous version | benchmark/ |

---

## 4. Kernel 完成后的交付物

每个 Kernel 模块必须包含 `src/`、`benchmark/`、`tests/`、`docs/`、`reports/`、`README.md`，目录结构参见 [Documentation_Template.md](Documentation_Template.md) §1。

---

## 5. 代码规范

### 5.1 访存标注

所有 kernel 代码必须用以下标记标注访存模式：

```cuda
x[i] = y[i];        // [ HBM]    — Global Memory 访问
shared[ tid] = v;   // [🟡 SRAM] — Shared Memory 访问
                     // [ FUSED]  — 算子融合标注
```

### 5.2 头文件顺序

```cuda
#include <cuda_runtime.h>         // 标准库优先
#include <float.h>
#include "utils.cuh"              // 项目头文件
```

### 5.3 Kernel 命名

```
operator_variant<template_params>(parameters)
```

示例：`softmax_online<256, 4>` = softmax, online 算法, BLOCK=256, float4

---

## 6. 性能分析原则

> 以下原则是整个仓库每份 Profiling 和 Bottleneck Analysis 的分析准则。

### 6.1 核心原则

1. **GPU Performance Counter 是分析证据（Evidence），不是最终结论（Conclusion）。**
   - "DRAM Throughput > 80%" 只说明"显存带宽用满了"，不直接说明"这是 Memory-bound 的 Kernel"
   - 可能是非合并访问导致的虚假带宽占用

2. **不要用单一指标直接下结论。**
   - 不能只看 Occupancy 就说"延迟隐藏不够"
   - 不能只看 L1 Hit Rate 就说"数据局部性差"
   - 不能只看 SM Throughput 就说"计算是瓶颈"
   - **必须交叉验证**：DRAM Throughput + SM Throughput + Arithmetic Intensity + Occupancy + Warp Stall

3. **不要写"看到 A → 一定是 B → 一定做 C"。**
   - 错误写法："Occupancy 低 → 延迟隐藏差 → 增加 Block Size"
   - 正确思路："Occupancy 低 → 检查限制原因（REG / SMEM / BLOCK）→ 检查 Warp Stall Reason → 如果 Long Scoreboard 高且 Occupancy 确实导致延迟隐藏不足 → 再考虑调整 Block Size"

4. **分析必须采用决策树/诊断流程，不是查表。**
   - 遵循 [Profiling_Guide.md](Profiling_Guide.md) 中的 **Performance Analysis Workflow**

### 6.2 判断 Memory-bound vs Compute-bound 必须综合以下指标

| 维度 | 指标 | 用途 |
|------|------|------|
| Benchmark | Kernel Time, Bandwidth, GFLOPS | 定量基准 |
| Memory Pattern | Coalescing, Transaction Count, Sector Misses | 访存效率 |
| Occupancy | Theoretical vs Achieved | 延迟隐藏能力 |
| Warp Stall | Long Scoreboard / Short Scoreboard / Barrier | 真正阻塞原因 |
| Instruction Mix | ALU / FMA / LSU / SFU 比例 | 计算负载特征 |
| Kernel 类型 | Streaming / Tiled / Reduce / Scan | 算法特征决定指标预期 |

**禁止出现**的分析写法：
- ❌ "DRAM Throughput 高 → 只需优化访存"
- ❌ "Occupancy > 80% → 延迟隐藏好"
- ❌ "SM Throughput 高 → Compute-bound"

**正确**的分析写法：
- ✅ "DRAM Throughput = 85%，SM Throughput = 25%，Arithmetic Intensity = 0.4 FLOP/byte（远低于 Ridge Point 35.3）→ Memory-bound。但进一步检查发现 Sector Misses 偏高，说明非合并访问浪费了部分带宽 → 实际有效带宽利用率可能更低"

---

## 7. Review Checklist

Code Review 时必须逐项检查：

- [ ] 有 Baseline 吗？是否先写了简单正确版本？
- [ ] Correctness 验证通过了？
- [ ] Benchmark 有均值/标准差/带宽/GFLOPS？
- [ ] 性能分析遵循了 Performance Analysis Workflow（Profiling_Guide.md）？
- [ ] Optimization History 完整？
- [ ] Decision Log 记录了每个选择的理由？
- [ ] Interview Notes 充足？
- [ ] docs/READEME.md 是 GitHub 可读的质量？
- [ ] 代码标注了 [ HBM] / [🟡 SRAM]？
- [ ] 这个 kernel 是否还有优化空间？如果有，记录在 future_work.md

---

## 8. 版本管理

- `main` 分支：稳定，可通过所有 correctness 测试
- 每个优化步骤作为一个独立 commit
- 每个 kernel 完成时，docs/README.md 必须更新到最新状态

---

## 9. 长期维护规范 (20+ Kernel 规模)

> 以下规范在 Kernel 数量增长到 20+ 时自动生效。4-5 个 Kernel 时可有选择地遵循。

### 9.1 Deprecated / Rejected 变体标记

不推荐的 kernel 变体（如某些场景下 float4 版本更慢）必须在 `README.md` 的变体列表中标注：

```
| Variant | Status | Reason |
|---------|--------|--------|
| rmsnorm_kernel | ✅ Default | Scalar baseline, 424.6 GB/s |
| rmsnorm_float4 | ⚠️ Deprecated | No measurable gain, increased register pressure |
```

"Deprecated" 表示不做默认但不删除——作为学习记录保留。Code Review 时检查是否有新的 variant 替代了旧的 default。

### 9.2 决策去重

跨 kernel 通用的决策理由（block size 选择、float4 适用条件、warp shuffle 限制等）应当在 [Performance_Playbook.md](Performance_Playbook.md) 中记录一份权威版本，kernel 文档中通过引用链接（交叉引用）代替复述。

### 9.3 Kernel 索引

当 Kernel 数量 > 10 时，在 `docs/kernel_index.md` 中维护索引表：

```
| Operator | Variants | Time (ms) | BW % Peak | Last Verified |
|----------|----------|-----------|-----------|---------------|
| Softmax  | 3        | 1.31      | 89.9%     | 2026-06-25    |
| LayerNorm| 3        | 1.63      | 86.7%     | 2026-06-25    |
```

每次 benchmark 更新时同步更新索引表。
