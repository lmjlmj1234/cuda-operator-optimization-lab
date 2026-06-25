# CUDA Operator Optimization Lab — Project Overview

> 这个仓库不是 CUDA Demo Collection。
> 这是一个 AI Infra 工程实验室，目标是理解、应用和扩展 NVIDIA 工程师设计、验证、分析、优化 CUDA Kernel 的方法论。

---

## 仓库定位

每个 Kernel 的开发过程体现完整的工程链路：

```
Problem → Baseline → Correctness → Benchmark → Profiling
→ Bottleneck Analysis → Optimization History → Decision Log → Interview Notes
```

不关注"Kernel 能否运行"，关注"如何运行、为什么快、还能更快吗"。

---

## 实现的算子

| 算子 | 文件 | 变体数 | 核心优化 |
|------|------|--------|---------|
| Softmax | `Softmax/src/` | 3 | Online 1-pass, Warp shuffle |
| LayerNorm | `LayerNorm/src/` | 3 | Welford online variance |
| RMSNorm | `RMSNorm/src/` | 2 | Simplified LN (LLaMA-style) |
| Fused Residual+LN | `FusedResidualLN/src/` | 2 | Operator fusion |
| FlashAttention Softmax | `FlashAttentionSoftmax/src/` | 1 | Prototype |

## 仓库结构

```
shared_docs/           # 仓库级工程规范和方法论
  Engineering_Convention.md    # 9-point delivery standard
  Performance_Playbook.md      # 优化技术目录
  Optimization_Methodology.md  # 优化方法论
  Benchmark_Specification.md   # 基准测试协议
  Profiling_Guide.md           # Profiling 指南 + Performance Analysis Workflow
  Dump_and_Correctness_Guide.md # Dump 策略和正确性验证
  Documentation_Template.md    # 文档模板

Softmax/ LayerNorm/ RMSNorm/ FusedResidualLN/ FlashAttentionSoftmax/
  ├── README.md         # 模块概览
  ├── src/              # Kernel 实现
  ├── benchmark/        # 基准测试配置和脚本
  ├── tests/            # 正确性测试
  ├── docs/             # 9 份独立分析文档
  └── reports/          # 实验摘要

tests/                  # 顶层集成测试
include/                # 共享 CUDA 头文件
reports/                # Profiling 输出（ncu, nsys, benchmark）
docs/                   # 跨仓库文档
  project_overview.md   # 本文
  debug_log.md          # Bug 记录
  roadmap.md            # 路线图
```

---

## Transformer 中的算子链路

```
Input → Embed → [Transformer Layer × N] → LM Head
                   │
            Self-Attention
            ├── QKV Linear (GEMM)
            ├── Attention Score
            │   ├── QK^T (GEMM)
            │   ├── S * scale → softmax(S)
            │   │   └── ✦ Softmax (Online 1-pass)
            │   └── softmax(S) @ V (GEMM)
            │       └── Future: FlashAttention (tile fusion)
            └── Output Linear (GEMM)
                   │
            Residual Add + LayerNorm
            └── ✦ Fused Residual+LayerNorm
                └── ✦ LayerNorm / RMSNorm
                   │
            FFN (2-3 GEMMs + Activation)
                   │
            Residual Add + LayerNorm (same as above)
```

**✦ = 已在仓库中实现的优化算子**

### 每层理论 HBM 节省（N=4096, D=8192, 原生 float32）

| 优化 | 单次节省 | 32 层累计 |
|------|---------|-----------|
| Online Softmax (3→1 pass) | -40% HBM | ~40% × 32 |
| Fused Residual+LN | -33% HBM traffic | ~30% 端到端 |
| RMSNorm (vs LN) | ~0% (同 bound) | 精度等效但更简单 |
| Float4 向量化 | 此场景无效 | 仅在非合并访问处有效 |
| FlashAttention | O(N²) → O(N) | 长序列下极大 |

---

## 实验环境

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 (Ampere, GA106) |
| CUDA | 12.4 |
| Driver | WSL2 (GPU-PV) |
| Profiling | cudaEvent, nsys (ncu 需要原生 Linux) |

## Roadmap

参见 `docs/roadmap.md`。
