# Documentation Template

> 本文档定义每个 CUDA Kernel 模块的 docs/ 目录下的文件模板。
> 所有新 Kernel 的文档必须按此结构创建。

---

## 1. 目录结构

每个 Kernel 模块必须包含以下文件：

```
KernelName/
├── README.md                          # 模块概览（3-5段）
├── src/                               # 源代码
├── benchmark/                         # 基准测试
├── tests/                             # 正确性测试
├── docs/
│   ├── problem_definition.md          # 问题定义
│   ├── correctness.md                 # 正确性验证
│   ├── benchmark.md                   # 基准测试结果
│   ├── kernel_analysis.md             # Kernel 分析
│   ├── optimization_history.md        # 优化历史
│   ├── bottleneck_analysis.md         # 瓶颈分析
│   ├── decision_log.md                # 决策日志
│   ├── nsight_analysis.md             # Nsight 分析
│   └── interview_notes.md             # 面试笔记
└── reports/
    └── latest_summary.md              # 最新实验摘要
```

---

## 2. 各文件内容要求

### README.md

3-5 段总结，每段有明确主题：

```
¶1: 算子是什么？数学定义 + 在 Transformer 中的作用
¶2: 实现哪些变体？各自的技术特点
¶3: 核心优化结果：量化数据（time / bandwidth / % peak）
¶4: 当前状态 + 下一步方向（预留一个 future work 钩子）
```

所有 README.md 按此结构对齐，确保 20+ kernel 可统一浏览。

### docs/problem_definition.md

模板：
```
## 1. Problem Definition

**What:** {算子的数学定义和公式}

**Why:** {为什么需要这个算子，在 Transformer/LLM 中的角色}

**Role in Transformer:** {具体在什么地方被调用}

**I/O Shapes:** {输入输出的维度、数据类型}

**Complexity:** {时间复杂度、访存量}

**Memory Pattern:** {Row-wise / Column-wise / Tiled / Random}
```

### docs/correctness.md

模板：
```
## Correctness Verification

### Verification Method
{CPU Reference / PyTorch Reference / Cross-validation}

### Results Table
| Variant | vs Reference | Max Diff | Status |
|---------|-------------|----------|--------|

### Cross-Validation Matrix
{哪些变体之间相互验证了}

### Edge Cases
{输入全等、全零、极端值等测试}
```

**必须包含 Cross-Validation Matrix 和 Edge Cases 两个小节**，不可省略。Edge Cases 可以简短引用测试文件（`tests/test_*.py` 中有覆盖），但必须体现验证意识。

### docs/benchmark.md

模板：
```
## Benchmark

### Environment
{GPU, CUDA, Driver, nvcc version}

### Problem Size
{N=?, D=?, float32}

### Protocol
{warmup=?, iters=?, timing method}

### Results Table
| Kernel | Time (ms) | Bandwidth (GB/s) | % Peak | GFLOPS |
|--------|-----------|-------------------|--------|--------|
```

### docs/kernel_analysis.md

模板：
```
## Kernel Analysis

### Launch Configuration
{Grid, Block, Shared Memory, Registers}

### Resource Usage
{Registers per thread, SMEM per block, Theoretical Occupancy}

### Memory Access Pattern
{Coalescing analysis, Transaction count}

### Instruction Analysis
{L1/L2 hit rate, Instruction mix, SASS count}

### Key Observations
{意外发现、反直觉结果}
```

### docs/optimization_history.md

模板：
```
## Optimization History

| Version | Technique | Gain | Side Effects | Why Chosen |
|---------|-----------|------|--------------|------------|
| v0 | {baseline} | — | — | — |
| v1 | {optimization 1} | {quantified} | {reg/accuracy} | {reason} |

### Rejected Approaches
{测试过但放弃的方案 + 原因 + 数据}
```

### docs/bottleneck_analysis.md

模板：
```
## Bottleneck Analysis

### Compute Intensity
{FLOPs / bytes, vs Ridge Point}

### Evidence
{DRAM Throughput, SM Throughput, Occupancy, Warp Stall}

### Verdict
{Memory-bound / Compute-bound} with {evidence summary}

### Next Priority
{什么优化最有希望 + 为什么}

### Future Acquisition
{WSL2 环境下需要补充的 ncu 指标}
```

### docs/decision_log.md

每个关键决策必须记录：

```
| Decision | Alternatives | Why Chosen | Tradeoffs |
|----------|-------------|------------|-----------|
| {block_size=256} | 128, 512 | {reason} | {compromise} |
| {use shared memory} | warp shuffle, atomic | {reason} | {compromise} |
| {float4 vectorized} | scalar | {reason} | {compromise} |
| {thread mapping} | alternative mapping | {reason} | {compromise} |
```

常见决策点：
- 为什么 block size 选某个值
- 为什么使用/不使用 shared memory
- 为什么使用/不使用 warp shuffle
- 为什么使用/不使用 float4
- 为什么不用 atomic
- 为什么不用其他归约方案
- 为什么用这个算法而不是另一个

**决策去重规则**：如果某个决策理由跨多个 kernel 通用（如 block size 与 occupancy 的权衡），应当在 [Performance_Playbook.md](../Performance_Playbook.md) 或 shared_docs/ 中的共享文档中记录一次，kernel 文档中直接交叉引用。避免 30 个 kernel 各维护一份相同的分析。

### docs/nsight_analysis.md

```
## Nsight Profiling

### Environment
{是否 WSL2 / 能否使用 ncu}

### Metrics (if available)
{SM Throughput, DRAM Throughput, Occupancy, L1/L2, Warp Stall}

### Analysis
{按照 Performance Analysis Workflow 分析}

### WSL2 Fallback
{如果无法获取 ncu 数据：为什么拿不到 + 未来如何获取 + 预期结果}

### Static Analysis
{cuobjdump resource usage, SASS instruction count}
```

### docs/interview_notes.md

```
## Interview Notes

### Q: {常见面试问题}?
A: {专业回答}

### Q: {另一个问题}?
A: {回答}

### Common Misconceptions
{面试官经常误解的内容}

### Key Takeaways
{实际开发中学到的东西}
```

### reports/latest_summary.md

单页实验摘要：
```
# {Operator} — Experiment Summary

Date: {YYYY-MM-DD}
Author: {name}

## Config
{关键参数}

## Results
{核心 benchmark 表}

## Key Findings
{3-5 条最重要的发现}

## Open Questions
{需要进一步验证的问题}
```
