# RMSNorm Benchmark Report

## 核心八问

### 1. Baseline 是什么？

`rmsnorm_kernel` — 标准两 pass RMSNorm：

```
Pass 1: 求 sum_sq = Σ x[i]²
         每个线程累加 sum_sq += x[i] * x[i]
         结束后 blockReduce sum_sq，算 rms = rsqrt(sum_sq / D + eps)
Pass 2: 归一化 y[i] = x[i] * rms * gamma[i]
```

两 pass，每元素读 2 次、写 1 次。理论 HBM 流量 = 3 × N × D × 4 = 384 MB。
实测: 1.63 ms, 424.6 GB/s (86.7% peak)。

与 LayerNorm sum_sq 性能完全相同（少一次 mean 计算在 D=8192 时不影响 wall time）。

### 2. 优化了什么？

仅对比了一个变体：

| 变体 | 技术 | 结果 |
|------|------|------|
| v1: `rmsnorm_float4` | float4 向量化 | 无增益（1.65 ms, −0.7% 倒退） |

**float4 被拒绝** — 理由与 LayerNorm 相同：coalesced row-major 访问下，float4 不会减少 HBM 事务数，反而因寄存器压力（26→34）导致 occupancy 下降。

RMSNorm 算法本身很简单（无 mean 计算、无两个矩的合并），scalar 版本已经是 near-optimal memory-bound kernel。

### 3. 为什么减少 HBM 访问？

RMSNorm 内部没有减少 HBM 访问的优化——float4 不改变 HBM traffic：

| 变体 | HBM reads | HBM writes | HBM总量 | DRAM事务数 |
|------|-----------|------------|---------|-----------|
| kernel (scalar) | 2 × 128 MB | 1 × 128 MB | 384 MB | 等效 |
| float4 | 2 × 128 MB | 1 × 128 MB | 384 MB | 相等 |

**RMSNorm vs LayerNorm**: 理论上 RMSNorm 少一次 mean 计算和一次 mean 相关访存，但在 D=8192 时，两个 kernel 都被 HBM reduction 阶段主导（sum_sq），mean 的计算开销淹没在 DRAM 等待中。小 D 场景（D ≤ 1024）RMSNorm 才能体现出优势。

### 4. cudaEvent 测了多少？

标准流程：

- `cudaEventRecord(start/stop)` + `cudaEventSynchronize(stop)`
- Warmup: 50 次（GPU boost 频率、缓存预热）
- Measured: 500 次
- 报告值: 总时间 / 500 = 单 kernel 平均时间
- Block size sweep: 128, 256, 512 各一套

每个 kernel 500 次迭代结果：stddev 约 0.3%（RTX 4090 上非常稳定，因为 kernel short → SM 负载均匀）。

### 5. nsys 能看到什么？

| 观测项 | 价值 |
|--------|------|
| Kernel launch timeline | 验证两 pass launch：reduce → element-wise |
| 对比 LayerNorm launch | 两者 launch 次数相同（2×N），确认 RMSNorm 没有节省 launch |
| Occupancy (nsys CUDA HW) | 如果可用（WSL2 可能受限），可以看到实际 warp 占用 |
| 整体 timeline 干净度 | 验证没有意外 cudaMemcpy 或同步行为干扰 |

RMSNorm 与 LayerNorm 的 nsys trace 看起来几乎一样——kernel name 不同，但 launch 模式和 duration 一致。

### 6. WSL2 下 ncu 为什么受限？

同样的 GPU-PV 限制：

```
裸机:  Linux 驱动 →  PCIe → GPU 硬件寄存器可访问
WSL2:  Linux 驱动 → GPU-PV vGPU → Windows 驱动 → PCIe → GPU
       ↑___ PMU 访问在此层被拦截 ___↑
```

ncu 在 WSL2 下 `--set full` 不会报错，但返回的 SM/dram 计数器不是硬件实际值。无法用 ncu 验证 memory-bound 的断言，只能靠静态分析和 cudaEvent timing 间接推断。

### 7. 如果上原生 Linux，ncu 能看哪些指标？

| 指标 | 预期值 | 意义 |
|------|--------|------|
| `dram__bytes_read.sum` | ~256 MB | 两 pass × 128 MB 的读 |
| `dram__bytes_write.sum` | ~128 MB | 输出写入 |
| `dram__throughput.avg` | ~870 GB/s | 86% vs 1008 GB/s peak |
| `sm__throughput.avg_pct_of_peak_sustained_elapsed` | ~13% | 典型 memory bound（等数据 ≫ 算数据） |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 对比 float4 版 | 验证 occupancy 差异是否是性能差异的原因 |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | > 50% | 确认 50% 以上 warp 在等 DRAM |

**最关键**: `sm__throughput` vs `dram__throughput` 的对比，直接画出 workload 在 Roofline 上的位置——这比任何单指标都更有说服力。

### 8. 下一步值得做的优化？

- **Small D 优化**: D ≤ 1024 时 switch 到 grid-stride loop，减少 block 数避免 SM 空闲
- **算子融合**: 将 RMSNorm 与 matmul 或 residual add 融合，消除中间 tensor 的 HBM 读写
- **Half-precision**: RMSNorm 在 half 精度下减少一半 HBM 流量，是 LLM 推理的标准做法
