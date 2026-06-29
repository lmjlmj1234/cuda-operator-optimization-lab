# Softmax Benchmark Report

## 核心八问

### 1. Baseline 是什么？

`softmax_naive` — 标准 3-pass 实现：

```
Pass 1: blockReduceMax(x)      → 读 HBM 1 次，写 HBM 1 次（max 写 shared memory）
Pass 2: expf(x - max), sum     → 读 HBM 1 次，写 HBM 1 次
Pass 3: x[i] = expf(...) / sum → 读 HBM 1 次，写 HBM 1 次
```

每个元素被读取 3 次、写入 1 次。理论 HBM 流量 = 4 × N × D × 4 bytes = 512 MB。
实测: 2.17 ms, 266.1 GB/s (54.3% peak)。

### 2. 优化了什么？

`softmax_online` — 在线 1-pass softmax：

```
Pass 1: 同时维护 max 和 sum，边读边算
         每访问一个元素: m_new = max(m_old, x[i])
                        s_new = s_old * expf(m_old - m_new) + expf(x[i] - m_new)
         一个循环内完成 max、exp、sum，最后统一 normalize
Pass 2: 读 x、写 y

```

三个操作融合到一个线程内循环，kernel 从 3 个 pass 降到 1 个。

额外变体:
- `+float4` vectorization: 每个线程每步加载 4 个 float，减少 LDG 指令数
- `softmax_warp`: 当 D ≤ 32 时用 warp shuffle 替代 shared memory

### 3. 为什么减少 HBM 访问？

每行 D=8192，Naive 每读一次 -> 每元素经过 DRAM 3 次。Online 1-pass 把 max/exp/sum 合并到同一个循环，每元素只需 1 次 DRAM 读取。

| 变体 | HBM reads | HBM writes | HBM总量 | 实测时间 |
|------|-----------|------------|---------|---------|
| naive (3-pass) | 3 × 128 MB | 1 × 128 MB | 512 MB | 2.17 ms |
| online (1-pass) | 1 × 128 MB | 1 × 128 MB | 256 MB | 1.31 ms |

Kernel 时间减少 **−39.7%**，与 HBM traffic 减少 50% 基本吻合（剩余开销来自 shared memory reduction 和 expf 计算）。

float4 的边际收益仅 +2.4%，因为 row-major 连续访存本身已经 128-byte coalescing 饱和，指令数不是瓶颈。

### 4. cudaEvent 测了多少？

标准流程：

- `cudaEventRecord(start)` + `cudaEventRecord(stop)`
- Warmup: 50 次迭代（GPU 频率从 idle 升到 boost）
- Measured: 500 次迭代
- 报告值: 总时间 / 500 = 单次 kernel 平均时间
- 不使用 CPU clock 计时（避免 launch overhead 干扰）

Block size sweep（128, 256, 512）各重复整套流程。

### 5. nsys 能看到什么？

仅用 `nsys profile`（无 GPU metrics 开关），可以看：

| 观测项 | 能看到 | 具体价值 |
|--------|--------|---------|
| Kernel timeline | ✓ | 明确 kernel launch 顺序，验证 3-pass → 1-pass |
| Grid/Block dims | ✓ | 确认 block size 是否正确映射到问题规模 |
| CUDA API trace | ✓ | Launch overhead = cudaLaunchKernel + cudaEvent 时间 |
| Memory ops | ✓ (nsys) | cuMemcpy / cudaMalloc 调用序列 |

### 6. WSL2 下 ncu 为什么受限？

WSL2 GPU-PV（GPU Paravirtualization）的架构限制：

```
裸机 PCIe → NVIDIA 驱动 → GPU
WSL2:     → Windows 驱动 → GPU-PV vGPU → Linux 驱动栈
```

硬件 PM（Performance Monitoring）单元在 WSL2 中不可访问，因为 GPU-PV 层不转发 PMU 寄存器访问。ncu 在 WSL2 下只能跑 `--replay-mode`（离线回放），无法获取:

- `dram__sectors_read` / `dram__sectors_write`（真实 DRAM 事务数）
- `sm__throughput`（SM 利用率）
- `lts__t_sectors`（L2 命中率）
- `stall reasons`（long scoreboard / short scoreboard / not selected）

WSL2 下 ncu 报告的性能数据是**模拟值**而非硬件测量值，不反映真实 DRAM 行为。

### 7. 如果上原生 Linux，ncu 能看哪些指标？

| 指标 | 意义 | 预期结果 |
|------|------|---------|
| `dram__throughput.avg` | 真实 DRAM 吞吐 | naive ~250 GB/s, online ~440 GB/s |
| `sm__throughput.avg_pct_of_peak_sustained_elapsed` | SM 利用率 | ~20% (memory-bound 典型值) |
| `l1tex__t_sector_hit_rate.pct` | L1 命中率 | 高（row-major 连续访问） |
| `lts__t_sector_hit_rate.pct` | L2 缓存命中率 | 对比同一 kernel 不同 block 数的影响 |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | 等待 DRAM 读返回 | 应 > 50%，确认 memory bound |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际 occupancy | 验证 block 256 的 occupancy 状态 |

**最关键**: 在 `--set full` 模式下跑一次 online softmax，确认 `dram__throughput` 是否接近 490 GB/s 的理论极限。如果实测 > 460 GB/s，说明优化已经到位；如果 < 400 GB/s，说明有未被发现的流水线气泡。

### 8. 下一步值得做的优化？

- **Half-precision** (float16/bfloat16): HBM 流量再减半，带宽利用率不变但总吞吐翻倍
- **FlashAttention fusion**: 将 softmax 与 matmul QK^T 融合，消除 logits 矩阵的 HBM 写入
- **原生 Linux ncu profiling**: 解决 WSL2 限制，获取上述硬件 counter 的真实值
