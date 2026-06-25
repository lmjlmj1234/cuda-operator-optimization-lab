# Profiling Guide: Nsight Compute & Nsight Systems

> 本文档解释如何使用 Nsight 工具分析 CUDA kernel，以及如何按照 Performance Analysis Workflow
> 解读性能指标。**GPU Performance Counter 是分析证据（Evidence），不是最终结论（Conclusion）。**
> 多个指标需要结合 Kernel 类型、算法特点一起分析。

---

## 1. 工具选择

| 工具 | 用途 | 粒度 | WSL2 支持 |
|------|------|------|-----------|
| Nsight Systems (nsys) | 应用级时间线、CUDA API trace | Kernel launch, memcpy, sync | 完全支持 |
| Nsight Compute (ncu) | Kernel 级硬件性能计数器 | SM, DRAM, L1/L2, occupancy | 不支持 (GPU-PV) |
| ncu-ui (GUI) | 可视化 ncu 报告 | 同上 + 交互式分析 | Windows 原生可用 |

---

## 2. Performance Analysis Workflow

性能分析是一个**诊断过程**，不是查表。每个指标只告诉你"发生了什么"，不直接告诉你"该做什么"。

多个指标交叉验证 + 结合算法特性 -> 定位瓶颈 -> 选择优化方向。

```
Step 1: Benchmark              确认测量本身可靠
    ↓
Step 2: 全局瓶颈判断            Memory-bound vs Compute-bound
    ↓                          (多指标交叉验证)
Step 3a: Memory-bound 诊断      Step 3b: Compute-bound 诊断
  - Memory Coalescing            - Instruction Mix
  - HBM Transaction Count        - Occupancy vs Latency Hiding
  - L1/L2 Cache Efficiency       - ILP / Instruction Dependency
  - Vectorized Load 效果         - Tensor Core 利用率
    ↓                                ↓
Step 4: Warp Stall 分析         (如果 SM 利用率低)
    ↓
Step 5: 确定优化策略
```

### Step 1: Benchmark — 判断测量本身是否可靠

在进入任何性能分析之前，首先确认下面的问题：

- **Kernel Time**: 是稳定值还是波动较大？(stddev 应在 mean 的 5% 以内)
- **Warmup**: 是否做了足够的 warmup？GPU 频率需要 10-50 次 launch 才能 boost 到稳定值
- **cudaEvent vs CPU clock**: 用 cudaEventRecord 计时（GPU 同步），不要用 CPU clock
- **Problem Size**: 是否足够大？kernel 时间 < 1 us 时，launch overhead 占主导
- **Launch Configuration**: grid/block 是否正确覆盖了所有数据？

如果 Benchmark 本身有问题（波动大、未 warmup、launch 配置错误），**先修 Benchmark**，不进后续分析。

---

### Step 2：判断全局瓶颈 — Memory-bound vs Compute-bound

**原则**：不要根据单一指标下结论。综合下面的证据判断。

#### 需要查看的指标

| 指标 | Memory-bound 倾向 | Compute-bound 倾向 |
|------|------------------|-------------------|
| **DRAM Throughput** | 接近峰值 (>80%) | 明显低于峰值 (<30%) |
| **SM Throughput** | 明显低于峰值 (<30%) | 接近峰值 (>70%) |
| **Arithmetic Intensity** | 远低于 Ridge Point | 远高于 Ridge Point |
| **Kernel Time** | 减少 HBM 访问量后可显著缩短 | 减少计算量后可显著缩短 |

#### 分析逻辑

**情况 1：DRAM Throughput > 70% 且 SM Throughput < 40%**

→ 大概率 Memory-bound。
- 但并不一定说明访存是"最优的"——可能仍有非对齐访问、非合并访问
- 需要继续分析 **Memory Coalescing、Transaction Count、Sector Misses** 等指标
- 优化方向：减少 HBM 访问量（融合、Shared Memory、Online Algorithm）

**情况 2：DRAM Throughput < 30% 且 SM Throughput > 60%**

→ 大概率 Compute-bound。
- 但并不一定说明计算是"高效的"——可能有指令依赖、ILP 不足
- 需要继续分析 **Instruction Mix、Occupancy、Warp Stall**
- 优化方向：ILP、Loop Unroll、Tensor Core、降低指令数

**情况 3：DRAM Throughput 和 SM Throughput 都低 (<30%)**

→ 延迟隐藏有问题。Occupancy 可能被资源限制。
- 结合 **Warp Stall、Occupancy、Long Scoreboard** 分析
- 进入 Step 4

**情况 4：DRAM Throughput 和 SM Throughput 都高 (>70%)**

→ 接近理论峰值。可能已经是优化上限。
- 如果仍需加速：只能通过降低精度（fp16/bf16）、Tensor Core、或者跨 kernel 融合

---

### Step 3a：如果怀疑 Memory-bound

#### 3a.1 Memory Coalescing

- 查看 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 和 `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`
- **Sector Misses** / **Requestors** 分析访问模式
- 如果一个 warp 的 32 个 thread 访问了超过 32 个 sector（4 个 cache line）→ 非合并访问

**注意**：简单的 Row-wise 访问通常是合并的。非合并访问在 Row-major 下很少发生。
如果发生，检查 Thread Mapping：`threadIdx.x` 是否映射到连续地址。

#### 3a.2 HBM Transaction Count

- 查看 `dram__sectors_read` 和 `dram__sectors_write`
- 对比**理论最小交易数**：`ceil(thread_count * bytes_per_thread / 32)` sector
- 如果实际 >> 理论，说明存在重复读取或非对齐访问

**注意**：某些算法天然需要多次读取同一数据（比如 3-pass softmax 读 3 遍）。
这种情况下"减少 HBM 交易数"不是靠改善合并度，而是靠**算法优化**（Online Softmax）。

#### 3a.3 Data Reuse / L1/L2 Cache

- `l1tex__throughput` 高 ≠ 好
- **Streaming Access**（Reduce、Scan、逐元素操作）对 L1 利用很低，因为数据只用一次
- 如果算法访问模式是 **Streaming**（每个数据只读一次），不要指望 L1 命中率提升性能
- L1 / L2 的 hit rate 需要结合算法分析：
  - 如果是 **tile-based**（GEMM、Conv），高 L1 hit rate 说明 tile 重用做得好
  - 如果是 **streaming**（LayerNorm、Softmax），L1 hit rate 低是正常的

#### 3a.4 Vectorized Load 是否有效

- 查看 `sm__inst_executed_pipe_lsu` 中的 LDG.E / LDG.E.128 数量
- Float4 (LDG.E.128) 减少了指令数但**不一定减少 HBM 交易数**
- 如果 Kernel 已经是 **DRAM 瓶颈**（DRAM Throughput > 80%），减少 LDG 指令数不会加速
- Vectorized Load 主要帮助：减少指令发射压力、改善非合并访问、减少 Sector Misses

#### 总结：Memory-bound 下的诊断思路

| 看到的现象 | 可能的根因 | 如何验证 |
|-----------|-----------|---------|
| DRAM Throughput 低，但计算 Intensity 低 | Memory Coalescing 差 | 查看 Sector Misses |
| DRAM Throughput 高但 Kernel 慢 | HBM 交易过多（多次 Pass） | 对比实际/理论交易数 |
| L1 Hit Rate 极低 | Streaming 访问模式，天然无法重用 | 检查算法是 Streaming 还是 Tiled |
| L2 Hit Rate 极低 | 数据规模远大于 L2 | 检查 Problem Size |
| Float4 没有加速 | 已经是 DRAM 瓶颈 | 对比有无 Vectorize 的 DRAM Throughput |

---

### Step 3b：如果怀疑 Compute-bound

#### 3b.1 Instruction Mix

- 查看 `sm__inst_executed_pipe_alu`, `sm__inst_executed_pipe_fma`, `sm__inst_executed_pipe_xu`
- 高 ALU 指令比例（非 FMA）→ 可能是整数寻址、循环开销占主导
- 检查 **Control Flow** 开销：分支、循环

**注意**：不要只看"总指令数"。关注**瓶颈管线**（某个 Pipe 的 throughput 接近上限）。

#### 3b.2 Occupancy 分析

**原则**：Occupancy 不是越高越好。

- 高 Occupancy（>80%）意味着 GPU 有更多 warp 可切换，但也意味着**更多的寄存器/Shared Memory 压力**
- 低 Occupancy 不一定坏——如果 Kernel 有足够的 ILP（Instruction Level Parallelism），低 Occupancy 也能达到高 SM Throughput

如何判断 Occupancy 是否合理：

1. 理论 Occupancy 受什么限制？REG / SMEM / BLOCK
2. 实际 Achieved Occupancy 与理论差多少？
3. 如果 Occupancy 低但 SM Throughput 高 → ILP 足够，Occupancy 不是瓶颈
4. 如果 Occupancy 低且 SM Throughput 低 → 检查 **Warp Stall** 原因（Step 4）

#### 3b.3 ILP / Instruction Dependency

- `sm__inst_executed_pipe_alu` 和 `sm__inst_executed_pipe_fma` 的 issue rate 低于 max
- 指令级依赖导致流水线气泡
- 通过 **Loop Unroll** 让编译器跨迭代调度指令，提高 ILP
- 使用 `--ptxas-options=-v` 查看寄存器压力

#### 3b.4 Tensor Core 利用率

- 如果硬件支持 Tensor Core（Volta+），但 kernel 只用 CUDA Core → 检查是否可以迁移
- Tensor Core 对数据类型有要求：fp16, bf16, int8, tf32
- 纯 float32 训练/推理可用 TF32（Ampere+）

---

### Step 4：Warp Stall 分析

如果 SM Throughput 低（<30%）且 Occupancy 不理想，不要直接修改 Block Size。

先分析 **Warp Stall Reason**，找到真正的阻塞原因。

#### Stall Reason 分析

| Stall Reason | 含义 | 继续查看的指标 |
|-------------|------|---------------|
| **Long Scoreboard** | 等待 global memory 数据 | DRAM Throughput、L1/L2 Hit、Coalescing |
| **Short Scoreboard** | 等待计算指令结果 | Instruction Dependency、ILP |
| **Barrier** | 等待 `__syncthreads()` | 检查 Barrier 频率；是否可以在 Barrier 间做更多工作 |
| **Wait** | Warp 未就绪（调度器无 warp 可发射） | Occupancy 是否足够；Register/SMEM 是否限制了更多 warp |
| **Not Predicated** | Warp 在 Non-active 状态 | 部分 warp 已完成；等待尾端线程 |
| **Branch Divergence** | Warp 内分支导致串行化 | 减少 Warp 级分支；用 Predication 替代 |

#### 分析逻辑

**Long Scoreboard 高**：
- 减少 latency：提高 Occupancy 或 Vectorized Load
- 减少次数：算法优化（减少 pass、用 Shared Memory）
- 不一定是 Occupancy 问题——也可能是非合并访问导致 Sector Misses 暴增

**Short Scoreboard 高**：
- 指令级依赖。尝试 Loop Unroll、手动指令重排
- 检查寄存器压力——如果编译器 spill 到 local memory，Short Scoreboard + Long Scoreboard 同时高

**Barrier 高**：
- 卡在 `__syncthreads()` 等待其他 warp
- 如果 barrier 间的计算量很小 → barrier 开销占比高
- 方案：增加 barrier 间的工作量，或减少 barrier 数量（如用 Warp Shuffle 代替 Shared Memory）

---

### Step 5：确定优化策略

根据上述诊断的最终结论，选择优化方向。

**如果问题是 Memory Coalescing 差：**
- 调整 Thread Mapping：确保 `threadIdx.x` 映射到连续地址
- Vectorized Load（float4/float2）会强制对齐，有时能自动修复非对齐访问
- Warp 级重排（Bank Conflict 优化）

**如果问题是 HBM 交易数过高（多次 Pass）：**
- Online Algorithm（如 Online Softmax 从 3 pass → 1 pass）
- Kernel Fusion（消除中间 tensor 的 HBM 读写）
- Shared Memory 缓存（tile-based GEMM、reduction tree）

**如果问题是 Instruction Dependency 高（Short Scoreboard）：**
- Loop Unroll（让编译器跨迭代调度）
- 手动 ILP（展开 + 重排 FMA 链）
- 降低寄存器压力（减少 Spill）

**如果问题是 Occupancy 受限（Wait / Long Scoreboard）：**
- 减少每线程寄存器使用（`__launch_bounds__`，限制 max_registers）
- 减少 Shared Memory 使用
- 增加 Block Size（更多线程 = 更多 warps = 更高 Occupancy）

**注意：上述优化可能互相矛盾。**
- ILP 需要更多寄存器，但提升 Occupancy 需要减少寄存器
- Loop Unroll 增加指令数但减少 dependency
- Float4 减少 LDG 指令但增加寄存器压力

**最终决策需要根据 Kernel 类型：**
- Memory-bound Kernel → 优先减少 HBM 交易数
- Compute-bound Kernel → 优先 ILP / Tensor Core / 降低指令数
- Latency-bound Kernel → 优先 Occupancy / ILP

---

## 3. 关键指标详解

### 3.1 Occupancy

**不是越高越好。**

- 高 Occupancy 提供更多 warp 隐藏 latency
- 但追求 100% Occupancy 可能需要牺牲寄存器、SMEM 或 Block Size
- 对于 Compute-bound Kernel，低 Occupancy + 高 ILP 可能更好

**如何分析：**
```
Theoretical Occupancy: 由 REG / SMEM / BLOCK 限制的最小值
Achieved Occupancy:    实际观测值（可能低于理论值）
如果 theoretical == achieved: 限制在资源
如果 achieved < theoretical: 可能存在 I-Cache 或调度问题
```

### 3.2 Memory Throughput

```
DRAM Throughput (% peak) = 已用带宽 / 理论峰值
```

- DRAM Throughput > 80%：核显存带宽利用充分（但不一定是"好"——可能因为非合并导致）
- 结合 `dram__sectors_read` 与理论最小值对比
- 如果 DRAM Throughput 高但计算 intensity 低 → 一直在等待 DRAM

### 3.3 L1/L2 Cache

**原则**：Cache Metrics 必须结合访问模式分析。

- **Streaming 访问**（Softmax, LayerNorm）：每个数据只用一次 → L1/L2 命中率天然低
- **Tiled 访问**（GEMM, Conv）：数据在一个 tile 内多次复用 → 高命中率是可实现的
- 如果 Streaming 模式的 Kernel L1 Hit 低 → **正常**，不需要优化
- 如果 Tiled 模式的 Kernel L1 Hit 低 → 考虑调整 Tile Size

### 3.4 Long Scoreboard

- 等待 global memory 加载完成
- 是 latency 问题，不是 throughput 问题（但两者相关）
- 高 Long Scoreboard + 低 DRAM Throughput：Occupancy 不足，没有足够的 warp 隐藏延迟
- 高 Long Scoreboard + 高 DRAM Throughput：HBM 是瓶颈，提高 Occupancy 无济于事

---

## 4. Roofline 分析

### 构建步骤

1. 计算 kernel 的 compute intensity = FLOPs / bytes_loaded
2. 在 roofline 图上定位 (x = intensity, y = measured throughput)
3. 如果 y 贴近带宽线 -> memory-bound
4. 如果 y 贴近算力线 -> compute-bound

### RTX 3060 Roofline 关键点

```
Ridge Point: 12.7 TFLOPS / 360 GB/s = 35.3 FLOP/byte

Compute Intensity < 35.3 -> Memory-bound
Compute Intensity > 35.3 -> Compute-bound
```

### 典型 Kernel 的 compute intensity

| Kernel | FLOP/byte | 判定 |
|--------|-----------|------|
| Softmax | ~0.4 | Strong memory-bound |
| LayerNorm | ~0.35 | Strong memory-bound |
| RMSNorm | ~0.38 | Strong memory-bound |
| Fused Residual+LN | ~0.40 | Strong memory-bound |
| GEMM (M=N=K=1024) | ~100-200 | Compute-bound |

---

## 5. 典型的分析案例

### 案例：Softmax Online vs Naive

**现象**：Online 比 Naive 快 40%，Float4 几乎无改善。

**分析过程**：
1. Step 1 Benchmark：测量稳定（stddev < 3%），50 warmup ✅
2. Step 2 全局判断：DRAM Throughput naive 54%、online 90%，SM Throughput < 30%
   → Memory-bound
3. Step 3a 进一步诊断：
   - Softmax Naive：3 次 HBM pass → 3× HBM 交易数
   - Softmax Online：1 次 HBM pass + block reduce
4. 结论：**40% 加速来自减少 HBM pass，不是来自计算优化**
5. Float4 不加速的原因：已经是 DRAM 瓶颈（online 90%），减少 LDG 指令不减少 HBM 交易

### 案例：LayerNorm Float4 没有加速

**现象**：Float4 版比 Scalar 版慢 1.6%。

**分析过程**：
1. Step 1 Benchmark：测量稳定 ✅
2. Step 2 全局判断：DRAM Throughput 两者都在 85% 附近
   → Memory-bound，但接近带宽上限
3. Step 3a 进一步诊断：
   - 两种版本的 HBM 访问量相同（读 input + gamma + beta，写 output）
   - Float4 增加的寄存器压力可能降低了 Occupancy
4. 结论：Float4 减少了 LDG 指令数，但没有减少 HBM 交易数
   - DRAM 是瓶颈时，减少 LDG 对性能无帮助
   - 寄存器压力增大反而不利

---

## 6. Nsight Systems (nsys) 使用

### 基本用法

```bash
nsys profile --trace=cuda --stats=true -o profile_name ./binary
nsys profile --trace=cuda,openacc,nvtx -o profile_name ./binary
```

### 关注指标

| 指标 | 解读 |
|------|------|
| cudaLaunchKernel 耗时 | CPU 端的 launch overhead (~5-7 us/launch) |
| cudaMemcpy 耗时 | Host-Device 数据传输时间 |
| cudaEventSynchronize 耗时 | Benchmark 模式下的计时等待 |
| cudaMalloc 次数 | 显存分配效率（频繁分配 = 差） |
| Kernel 总时长/调用次数 | 每个 kernel 的 CPU 感知时间 |

---

## 7. Nsight Compute (ncu) 使用

### 基本用法

```bash
ncu --set full --kernel-name "my_kernel*" --import-source yes ./binary

ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    --kernel-name "my_kernel*" ./binary
```

### ncu Section Table 解读

`--set full` 包含以下 Section，每个 Section 对应特定分析维度。

| Section | 用途 | 包含指标 |
|---------|------|---------|
| Memory Workload Analysis | 检查 Load/Store 指令效率 | LDG/STG count, sectors/request, replay |
| Memory Workload Analysis Chart | 可视化访问模式 | GPU memory requests chart |
| Occupancy | 理论 vs 实际 | Block Size, Registers, SMEM per block |
| Speed Of Light | 整体性能快照 | SM Throughput, DRAM Throughput, Duration |
| Speed Of Light Roofline | Roofline 图 | Compute Intensity + throughput chart |
| Scheduler Statistics | Warp 调度效率 | Issue stall, issue slot utilization |
| Source Counters | 源码级性能计数器 | Per-line executed instructions |
| Launch Statistics | Kernel launch 信息 | Grid, Block, Shared Memory config |
| Instruction Statistics | 指令混合 | ALU, FMA, LSU, SFU counts |

**如何配合 Workflow 使用：**

1. 先看 **Speed Of Light**（Step 2：全局判断）
2. 如果 Memory-bound 嫌疑大 → **Memory Workload Analysis** → **Occupancy** → **Source Counters**
3. 如果 Compute-bound 嫌疑大 → **Instruction Statistics** → **Occupancy** → **Scheduler Statistics**
4. 如果 SM Utilization 低 → **Scheduler Statistics** → **Warp Stall**（Step 4）

---

## 8. WSL2 环境下的 Profiling 策略

### 能做的

```bash
# 软件计时 (cudaEvent)
./build/cuda_operators

# Nsight Systems 应用级 profiling
nsys profile --trace=cuda --stats=true -o reports/profile ./build/cuda_operators

# cuobjdump 静态资源分析
cuobjdump --dump-resource-usage ./build/cuda_operators
```

### 不能做的

```bash
# 以下命令在 WSL2 上会返回 ERR_NVGPUCTRPERM
ncu --set full --kernel-name "softmax*" ./build/cuda_operators
```

### 解决方案

1. **在原生 Linux 上运行** `bash tests/test_performance.sh`
2. **在 Windows 上用 ncu-ui** 打开 .ncu-rep 文件
3. **购买云 GPU** (Lambda Labs / Vast.ai) 提供原生 Linux 环境

---

## 9. 常见 Profiling 错误

| 错误 | 后果 | 修正 |
|------|------|------|
| 不 warmup | GPU 频率未 boost，计时偏慢 | 50 warmup iter |
| ncu 用 --set basic | 缺少 SM/DRAM/occupancy 数据 | 用 --set full |
| ncu 不指定 kernel | profiling 所有 kernel，文件巨大 | --kernel-name 精确指定 |
| 在 WSL2 里跑 ncu | 拿不到任何硬件计数器 | 换原生 Linux |
| 只抓 1 次 launch | 数据受首次 launch 影响 | 用 --launch-skip 1 --launch-count N |
| 忽略 nsys 报告 | 错过 CPU 端瓶颈 | nsys UI 查看时间线 |
| 只看 Occupancy | 误判（高 Occupancy ≠ 高性能） | 结合 SM Throughput, Warp Stall |
| Occupancy 低就加 Block Size | 可能增加资源限制不加 | 检查 REG/SMEM 限制再调整 |
| L1 Hit 低就优化缓存 | Streaming 访问模式没意义 | 先判断访问模式再分析 Cache |
