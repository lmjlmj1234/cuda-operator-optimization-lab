# CUDA 算子极致优化：内功心法与外功招式

> 本文档是 `cuda-operator-optimization-lab` 项目的深度笔记，
> 系统总结了 GPU 算子优化的核心思想与高级技巧。
> 涵盖 Softmax、LayerNorm、RMSNorm、FlashAttention 等关键算子的优化方法论。

---

## 目录

1. [核心优化哲学](#1-核心优化哲学)
2. [编码符号约定](#2-编码符号约定)
3. [优化方法全景对比表](#3-优化方法全景对比表)
4. [深度技术解析](#4-深度技术解析)
5. [Transformer 模型算子优化路线图](#5-transformer-模型算子优化路线图)
6. [延伸阅读与实践建议](#6-延伸阅读与实践建议)

---

## 1. 核心优化哲学

GPU Kernel 优化的本质，是缓解计算能力与数据供给之间的不匹配。现代 GPU 的 Tensor Core 拥有极高的计算吞吐，而 HBM 数据传输成为瓶颈，因此大量 Kernel 属于 Memory-bound。

| 硬件层次 | 带宽 | 延迟 | 容量 |
|---------|------|------|------|
| 寄存器 (Register) | ~50 TB/s | 1 cycle | 256 KB / SM |
| 共享内存 (Shared Memory) | ~20 TB/s | ~5 cycles | 48~164 KB / SM |
| L1/L2 Cache | ~10 TB/s | ~30 cycles | 几 MB |
| 全局显存 (HBM) | ~1 TB/s | ~400 cycles | 几十 GB |

Kernel 优化通常围绕三个目标展开：**减少 HBM 访问、提高数据复用、提升内存访问效率**（如 memory coalescing、shared memory、register reuse）。

---

## 2. 编码符号约定

为帮助代码阅读者直观理解每个 Kernel 的访存模式，代码中嵌入了以下标记符号：

| 标记 | 含义 | 对应硬件 |
|------|------|---------|
| `[ HBM]` | Global Memory 读写 | HBM / GDDR |
| `[🟡 SRAM]` | Shared Memory 读写 | On-chip SRAM |
| `[ FUSED]` | 算子融合标注 | 多个操作合并 |

> **使用方式**：统计代码中 `[ HBM]` 的出现次数即可快速得知该 Kernel 的 HBM 访问总量。
> 这是代码可读性的重要工具，在 CUTLASS / FlashAttention 等工业级项目中被广泛采用。

---

## 3. 优化方法全景对比表

| 优化方法 | 本质原理 | 减少的核心开销 | 关键数学公式 |
|---------|---------|--------------|------------|
| **One-Pass Softmax** (在线单遍算法) | 用数学换性能：同时在单趟扫描中追踪 max 和 sum | HBM 访存：从 3 次读取降至 2 次 | $m_{new} = \max(m_{old}, x_i)$ <br> $s_{new} = s_{old} \cdot e^{m_{old} - m_{new}} + e^{x_i - m_{new}}$ |
| **FlashAttention Softmax** (算子融合) | 用局部存储换性能：将 QK^T 和 Softmax 融合为单次扫描 | 消除 S = QK^T 中间矩阵的 HBM 读写 | $O \leftarrow O \cdot \frac{s_{old}}{s_{new}} + \frac{e^{s_{ij} - m_{new}}}{s_{new}} \cdot V_j$ |
| **Welford 在线方差** | 用算法换精度：单趟稳定计算方差 | 避免 Catastrophic Cancellation 导致的精度损失 | $\delta = x - \bar{x}_{old}$ <br> $\bar{x}_{new} = \bar{x}_{old} + \delta / n$ <br> $M_{2,new} = M_{2,old} + \delta \cdot (x - \bar{x}_{new})$ |
| **Warp Shuffle** | 用寄存器换性能：Warp 内通信不走 Shared Memory | Shared Memory 访问延迟和 Bank Conflict | $\text{val} \mathrel{+}= \text{\_\_shfl\_xor\_sync}(mask, val, offset)$ |
| **RMSNorm** (均方根归一化) | 用数学简化换性能：去掉均值计算 | 减少约 40% 归约计算量 + 省去均值广播 | $y_i = \frac{x_i}{\sqrt{\frac{1}{N}\sum_{j=1}^N x_j^2 + \epsilon}} \cdot \gamma_i$ |
| **Fused LayerNorm** (算子融合) | 用局部存储换性能：将残差加法和 LN 合并到一次 HBM 遍历 | HBM 访存：从 4 次遍历降至 2 次 (减 50%) | $\tilde{x}_i = x_i + r_i$ <br> $y_i = \frac{\tilde{x}_i - \mu}{\sigma} \cdot \gamma_i + \beta_i$ |
| **Float4 向量化** | 用指令宽度换性能：单条指令加载 16 字节 | 指令数降至 1/4，提高内存控制器利用率 | $\text{float4 } v = \text{reinterpret\_cast<float4*>(ptr)[i]}$ |
| **Register Blocking** (寄存器阻塞) | 用寄存器换性能：数据复用减少 HBM 访问 | HBM 访存：每个元素被多次读取时只需加载一次 | $C_{ij} = \sum_{k} A_{ik} \cdot B_{kj}$ <br> tile size = (BM, BK, BN) 受寄存器数量约束 |
| **Tiling** (分块) | 用局部存储换性能：分块加载到 SRAM 供多次计算复用 | HBM 访存：从 $O(MNK)$ 降至 $O(MN + NK + MK)$ | Tile size 受 Shared Memory 大小约束： <br> $S_{tile} \le 48\text{KB} / 4\text{bytes}$ |

---

## 4. 深度技术解析

### 4.1 One-Pass Softmax（在线单遍算法）

#### 传统 Softmax 的缺陷

传统的 Softmax 实现需要 **3 次全局内存遍历**：

1. 找最大值 $m = \max(x_1, ..., x_n)$
2. 求和 $s = \sum e^{x_i - m}$
3. 归一化 $y_i = e^{x_i - m} / s$

这在 GPU 上意味着同一个数据要被读取 3 次，造成大量的 HBM 带宽浪费。

#### 在线算法的洞察

核心洞察是：**max 和 sum 不需要分开计算，可以在一次扫描中同时维护**。

当我们按顺序读取数据时：

- 如果遇到新的最大值 $m_{new} > m_{old}$，之前用 $m_{old}$ 计算的所有指数值 $e^{x_i - m_{old}}$ 都需要"缩放修正"为 $e^{x_i - m_{new}}$
- 这个缩放因子就是 $e^{m_{old} - m_{new}}$
- 因此 $s_{new} = s_{old} \cdot e^{m_{old} - m_{new}} + e^{x_i - m_{new}}$

```
扫描开始:
  m = -inf, s = 0
  x = [2, 5, 1, 3]
  ↓
  x₁=2:  m = max(-inf, 2) = 2
         s = 0 * e^(-inf-2) + e^(2-2) = 1
  x₂=5:  m = max(2, 5) = 5
         s = 1 * e^(2-5) + e^(5-5) = e^(-3) + 1
  x₃=1:  m = max(5, 1) = 5
         s = (e^(-3)+1) * e^(5-5) + e^(1-5) = e^(-3) + 1 + e^(-4)
  x₄=3:  m = max(5, 3) = 5
         s = (e^(-3)+1+e^(-4)) * e^(5-5) + e^(3-5) = e^(-3) + 1 + e^(-4) + e^(-2)
```

**最终结果**：$s = e^{-3} + e^0 + e^{-4} + e^{-2}$，这正是 $\sum e^{x_i - m}$。

#### HBM 访问对比

| 版本 | HBM 读取 | HBM 写入 | 总计 |
|------|---------|---------|------|
| 朴素 3-Pass | 3× | 2× | 5 次 |
| Online 1-Pass | 2× | 1× | 3 次 |
| **节省** | **-33%** | **-50%** | **-40%** |

### 4.2 FlashAttention 中的 Softmax（算子融合）

#### 问题背景

Attention 计算的完整表达式：

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

其中 $S = QK^T$ 是一个 $N \times N$ 的中间矩阵。对于 $N = 4096$，$S$ 的大小是 $4096^2 \times 4\text{bytes} = 64\text{MB}$，远超出 SRAM 容量。

#### 融合方案

FlashAttention 的核心思想是 **将 Softmax 的计算与 QK^T 和 PV 融合**，而不是 materialize 完整的 $S$ 矩阵：

```
for each query tile Q_tile:
  for each key-value tile KV_tile:
    // 1. 计算部分 attention score (寄存器 / SRAM)
    S_partial = Q_tile @ K_tile^T

    // 2. 在线 softmax 更新: 同时维护 m 和 s
    m_new, s_new = online_softmax_update(m_old, s_old, S_partial)

    // 3. 用修正因子缩放已有的输出累加
    O = O * (s_old / s_new) + exp(S_partial - m_new) / s_new @ V_tile
```

[ FUSED] 此过程融合了：`QK^T + Online Softmax + PV`。

#### 显存节省

| 方案 | HBM 读写 | 备注 |
|------|---------|------|
| 标准 Attention | $O(N^2 + N d)$ | S 和 P 都 materialize |
| FlashAttention | $O(N d)$ | 无 N² 中间矩阵 |
| 节省比例 | $O(N/d)$ | 当 N >> d 时极为显著 |

### 4.3 Welford 算法 + Warp Shuffle

#### Welford 算法原理

计算方差的标准公式 $\text{Var}(X) = E[X^2] - E[X]^2$ 在数学上等价，但在浮点数运算中存在严重的 **Catastrophic Cancellation** 问题。

当 $X \approx 100.0$ 且 $\sigma(X) \approx 0.001$ 时：
- $E[X^2] \approx 10000.0$
- $E[X]^2 \approx 10000.0$
- $\text{Var} = 10000.0 - 10000.0 = 1 \times 10^{-6}$ ❌ (有效数字大减)

Welford 算法通过迭代更新避免此问题：

```
初始化: mean = 0, m2 = 0, count = 0
对每个 x_i:
  count += 1
  delta  = x_i - mean
  mean  += delta / count
  delta2 = x_i - mean
  m2    += delta * delta2
最终: var = m2 / count
```

**并行化**：在 CUDA 中，每个线程对其负责的元素运行 Welford，然后用 closed-form 公式合并多组统计量。

**合并公式**：
$$\begin{aligned}
n &= n_a + n_b \\
\delta &= \bar{x}_a - \bar{x}_b \\
\bar{x} &= \frac{n_a \bar{x}_a + n_b \bar{x}_b}{n} \\
M_2 &= M_{2,a} + M_{2,b} + \delta^2 \cdot \frac{n_a n_b}{n}
\end{aligned}$$

#### Warp Shuffle 加速

通过 `__shfl_xor_sync` 指令，同一个 warp 中的线程可以直接交换寄存器中的值，无需求助于 Shared Memory：

```
// 4 个线程的树形归约
t0: a,b,c,d    // 初始值
   ↓ __shfl_xor(4)
t0: a+b, c+d  // 第 1 轮
   ↓ __shfl_xor(2)
t0: a+b+c+d   // 第 2 轮
```

### 4.4 RMSNorm（均方根归一化）

RMSNorm 是 LayerNorm 的简化版本，在 Llama、Mistral 等现代 LLM 中被广泛使用。

| | LayerNorm | RMSNorm |
|--|-----------|---------|
| 计算公式 | $\frac{x - \mu}{\sigma} \odot \gamma + \beta$ | $\frac{x}{\sqrt{\frac{1}{N}\sum x_i^2 + \epsilon}} \odot \gamma$ |
| 归约量 | sum + sum_sq (或 mean + m2) | sum_sq 仅需一次归约 |
| 计算量 | 100% | ~60% |
| 额外参数 | gamma + beta | gamma 仅需 (beta 省略) |
| LLM 效果 | 传统方案 | Llama/Mistral 验证等价或更优 |

### 4.5 算子融合 — Fused LayerNorm

#### 未融合的流水线

```
1. residual = x + sub_layer(x)   // HBM 读 x, HBM 读 sub, HBM 写 residual
2. y = LayerNorm(residual)       // HBM 读 residual, HBM 写 y
                                  // 总计: 3× HBM 读 + 2× HBM 写
```

#### 融合后的流水线

```
1. y = LayerNorm(x + sub_layer(x))
   // 读 x + 读 sub → 寄存器加法 → 归一化 → 写 y
   // 总计: 2× HBM 读 + 1× HBM 写 (减少 40%)
```

[ FUSED] 融合了：残差加法 + 均值和方差统计 + 归一化变换。

### 4.6 Float4 向量化

#### 原理

现代的 GPU 内存控制器本质上是宽接口设备。RTX 3060 有 192-bit 显存总线，最佳的访问模式是连续的 128-bit (16 字节) 对齐访问。

```c
// 非向量化: 4 条加载指令
float a = ptr[0];   // 1 条 LDG
float b = ptr[1];   // 1 条 LDG
float c = ptr[2];   // 1 条 LDG
float d = ptr[3];   // 1 条 LDG

// 向量化: 1 条加载指令
float4 v = reinterpret_cast<const float4*>(ptr)[0];  // 1 条 LDG.128
```

> **Memory Coalescing vs Vectorized Load**：两者容易混淆，但作用于不同层次。Memory Coalescing 是 Warp 级别的访存优化，由硬件自动完成；Vectorized Load（如 float4）是单线程级别的访存优化，需要程序员或编译器生成宽加载指令。两者可以同时发挥作用 — 连续地址上每个线程用 float4 加载，既满足 warp 合并条件，又减少指令数。

#### 性能提升

| 指标 | 标量 | Float4 | 提升 |
|------|------|--------|------|
| 指令数 | 4 LDG | 1 LDG.128 | 4× |
| 内存带宽利用率 | ~60% | ~90%+ | 1.5× |
| 执行时间 | 基线 | ~70% | 30% |

### 4.7 Block Reduce 设计模式

#### 两阶段归约

GPU 的线程层次结构决定了归约需要两阶段：

```
阶段 1: 每个线程独立归约 -> 寄存器中一个标量
阶段 2: 跨线程归约 -> Shared Memory + Warp Shuffle

跨线程归约又分两步:
  2a: Warp 内归约 (通过 __shfl_xor_sync, 无 Shared Memory)
  2b: Warp 间归约 (通过 Shared Memory, 最后用 1 个 warp 归约)
```

#### 最佳实践

- **Warp 内**优先用 Warp Shuffle，避免 Shared Memory Bank Conflict
- **Warp 间**用 Shared Memory 中转，但只需 `num_warps` 个元素
- **Padding** Shared Memory 数组避免 Bank Conflict:
  ```c
  __shared__ float shared[32][33];  // 每行 33 个元素而非 32
  ```

---

## 5. Transformer 模型算子优化路线图

```
Transformer 层
├── Self-Attention
│   ├── QKV 投影 —— Matmul + Tiling
│   ├── Attention Score —— FlashAttention (Online Softmax 融合)
│   │   ├── Q K^T —— 分块 Matmul + 在线 Softmax
│   │   └── Softmax × V —— 融合累加
│   └── Output 投影 —— Matmul
├── 残差连接 —— 与 LayerNorm 融合 [FUSED]
├── LayerNorm / RMSNorm —— Welford + Float4
└── FFN
    ├── Gate Proj —— Matmul
    ├── SiLU 激活 —— 可以融合 [FUSED]
    ├── Up Proj —— Matmul
    └── Down Proj —— Matmul
```

### 每层的理论 HBM 节省

| 优化 | 单次节省 | 层数 | 总计 (32 层) |
|------|---------|------|-------------|
| Online Softmax | ~40% HBM | 32 | 显著 |
| FlashAttention | $O(N/d)$ | 32 | 极大 (N² → N) |
| Fused Residual+LN | ~40% HBM | 32 | ~20-30% 端到端 |
| Float4 向量化 | 30% 访存指令 | 所有层 | ~10-15% 端到端 |
| RMSNorm | 40% 计算 | 32 | 轻微 |
| Welford | 精度提升 | 32 | 精度 |

---

## 6. 延伸阅读与实践建议

### 推荐的 Profiling 工具

| 工具 | 用途 | 命令 |
|------|------|------|
| Nsight Compute (ncu) | Kernel 级别的性能指标 | `ncu --set full ./binary` |
| Nsight Systems (nsys) | 全局时间线 + Kernel Launch 开销 | `nsys profile ./binary` |
| NVIDIA Visual Profiler (nvvp) | 可视化分析 (旧) | `nvvp ./binary` |

### 关键性能指标解读

| 指标 | 好的信号 | 差的信号 |
|------|---------|---------|
| DRAM Throughput | >80% 带宽峰值 | <50%，说明被计算 bound |
| SM Throughput | >80% | <50%，说明被访存 bound |
| Occupancy | >50% | <25%，可能隐藏不足延迟 |
| L1 Hit Rate | >80% | <50%，Shared Memory 利用率低 |
| Sector Misses | 少 | 多，访存模式不连续 |

### 调试方法论

1. **先正确，再高效**：先用 Python 验证数学逻辑
2. **逐级优化**：一次只应用一个优化技巧，持续测量
3. **怀疑一切**：编译器可能做了你预期的优化，也可能没做
4. **Nsight 是真理**：直觉不可靠，硬件数据才是真相

---

> **"写 GPU 代码就像写一首诗 — 你需要同时懂数学、硬件和并行算法。"**

---

## 7. Benchmark 结果

> 实验环境：NVIDIA GeForce RTX 3060, CUDA 13.2, Ubuntu (WSL2)
> 数据规模：N=4096 rows, D=8192 dims, float32
> 计时方式：cudaEvent (50 次 warmup + 500 次测量取均值)

| 算子 | Kernel | 耗时 (ms) | vs 基线 | 备注 |
|------|--------|-----------|---------|------|
| **Softmax** | naive 3-pass (block=128) | 2.069 | — | 3 次 HBM 遍历 |
| | online 1-pass (block=256) | 1.227 | **-40.7%** | 1 次 HBM 遍历 |
| | online+float4 (block=256) | 1.220 | -41.0% | float4 无额外收益 |
| | warp (dim=32) | 0.017 | — | 小维度专用路径 |
| **LayerNorm** | sum_sq | 1.219 | — | |
| | float4 | 1.227 | ~持平 | |
| | welford | 1.257 | ~持平 | 精度更好 |
| **RMSNorm** | scalar | 1.203 | — | |
| | float4 | 1.216 | ~持平 | |
| **Fused Residual+LN** | scalar | 2.044 | — | 含残差加载 |
| | float4 | 2.029 | ~持平 | |

### 关键发现

1. **Online Softmax 获得 -41% 真实提速**（2.07→1.22 ms），验证了"合并 HBM 遍历"的理论收益。实际节省比例接近理论值 40%。

2. **Float4 向量化在此场景无收益**。原因是这些 kernel 已经是连续访存，编译器自动生成的标量加载已充分利用带宽。Float4 在更复杂的访存模式（如非对齐、跨步访问）中才有收益。

3. **Welford LayerNorm 在精度上有优势**但速度略慢（+3%）。对于大批量训练需要高精度方差时值得选择。

4. **Fused LayerNorm 耗时约 2.04 ms**，高于单独 LN 的 1.22 ms，因为它多读了 residual 输入。但端到端看，融合省掉了 `x + residual` 这个中间 kernel 的 HBM 读写，如果非融合方案需要两次完整 HBM 遍历 + 一次 LN，则融合仍更优。

5. **Nsight Compute 硬件指标**（SM Throughput、DRAM Throughput、Occupancy）因 WSL2 虚拟化环境无法直通 GPU 性能计数器，需在原生 Linux 上使用 `ncu` 获取。
