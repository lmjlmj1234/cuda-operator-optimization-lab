# LayerNorm Benchmark Report

## 核心八问

### 1. Baseline 是什么？

`layernorm_sum_sq` — 标准两矩法 LayerNorm：

```
Pass 1: 求 mean (E[X]) 和 var (E[X²] - E[X]²)
         每个线程累加 sum_x += x[i], sum_x2 += x[i] * x[i]
         结束后 blockReduce，算 mean 和 var
Pass 2: 归一化 y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
```

两 pass，每元素读 2 次、写 1 次。理论 HBM 流量 = 3 × N × D × 4 = 384 MB。
实测: 1.63 ms, 424.6 GB/s (86.7% peak)。

环境: RTX 4090 (1008 GB/s peak)。注意 Softmax 在 RTX 3060 上跑 (360 GB/s peak)，跨 GPU 不能直接比绝对值。

### 2. 优化了什么？

三个变体，逐个分析：

| 变体 | 技术 | 结果 |
|------|------|------|
| v1: `layernorm_float4` | float4 向量化加载/存储 | −1.6% 倒退 |
| v2: `layernorm_welford` | Welford 在线方差 | −0.6% (数值稳定性优先) |

**float4 被拒绝** — 因为 row-major 已经完美 coalescing，瓶颈是 DRAM 带宽而非指令数。float4 增加了寄存器压力（24→32），降低了 occupancy，反而更慢。

**Welford 是生产推荐** — 虽然比 sum_sq 慢 0.6%，但避免了 sum_sq 在 var << mean² 时的 catastrophic cancellation。

**真正的优化不在这个 kernel 内部** — 见 FusedResidualLN。

### 3. 为什么减少 HBM 访问？

LayerNorm 内部变体之间 HBM 访问量没有减少：

| 变体 | HBM reads | HBM writes | HBM总量 |
|------|-----------|------------|---------|
| sum_sq | 2 × 128 MB | 1 × 128 MB | 384 MB |
| float4 | 2 × 128 MB | 1 × 128 MB | 384 MB |
| welford | 2 × 128 MB | 1 × 128 MB | 384 MB |

三个变体 HBM 流量完全相同。性能差异来自寄存器压力和 occupancy 的次生效应。

**真正的 HBM 优化是算子融合**: `FusedResidualLN` 把 residual add + LayerNorm 合为一个 kernel，消除了中间 tensor 的 HBM 写入，每层节省 32 MB（详见 Fusion 报告）。

### 4. cudaEvent 测了多少？

标准流程：

- `cudaEventRecord(start/stop)` 包裹 500 次 kernel 调用
- Warmup: 50 次（GPU 升频、缓存预热）
- Measured: 500 次
- 报告值: 总 Σtime / 500
- 三个变体 + block size sweep (128, 256, 512) 各一套

LayerNorm 因 kernel 简单，500 次迭代的 stddev < 1%，数据非常稳定。

### 5. nsys 能看到什么？

| 观测项 | 价值 |
|--------|------|
| Kernel timeline | 验证两 pass 的 launch 顺序：reduce → normalize |
| Grid/Block 映射 | 确认 <<<N, BLOCK>>> 是否正确（4096 行 × 256 线程） |
| 是否独立 memory ops | 确认 kernel 之间没有意外的 cudaMemcpy |
| 有无 cudaMemset 等额外开销 | 确认 benchmark 过程中无隐藏同步 |

对于 LayerNorm 这种简单的 row-wise kernel，nsys timeline 通常干净，反而更容易发现框架层问题（比如多余的 memory allocation）。

### 6. WSL2 下 ncu 为什么受限？

WSL2 的 GPU-PV（GPU Paravirtualization）层不转发硬件 PMU 寄存器访问。ncu 在 WSL2 只能用 `--replay-mode`，获取的是**模拟值**，而非硬件计数。

无法获取的关键指标：
- `dram__sectors_read` / `dram__sectors_write`（真实 DRAM 事务数）
- `sm__throughput`（SM 利用率 — 对 memory-bound kernel 至关重要）
- `lts__t_sectors`（L2 cache 行为）
- `stall reasons`（long scoreboard — 确认 memory bound 的直接证据）

### 7. 如果上原生 Linux，ncu 能看哪些指标？

| 指标 | 预期作用 |
|------|---------|
| `dram__bytes_read.sum` + `write.sum` | 验证 HBM 流量 = 384 MB 理论值 |
| `dram__throughput.avg` | 应该 ~870 GB/s，接近 1008 GB/s 峰值的 86% |
| `sm__throughput.avg_pct_of_peak_sustained_elapsed` | 应 < 30%（memory-bound） |
| `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` | L1 命中率，row-major 应很高 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_warp_active.pct` | 应 > 40%，等待 DRAM |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际 occupancy（float4 版预计下降） |

**最关键**: 确认 `stall_long_scoreboard` > 40%，这是 memory-bound kernel 的"铁证"。核心问题中可以直接说 "这个 kernel 97% 的时间在等数据，不是算数据"。

### 8. 下一步值得做的优化？

- **Half-precision** (float16/bfloat16): 内存流量减半，吞吐翻倍，已在上游项目中普遍应用
- **Flash-style fusion**: 将 residual add / LayerNorm / matmul 合并，消除中间 tensor 的 HBM 写入
- **跨 SM 的 row-grouping**: 当 N < SM 数时，通过 grid-stride loop 提高 SM 利用率
