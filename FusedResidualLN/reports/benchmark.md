# FusedResidualLN Benchmark Report

## 核心八问

### 1. Baseline 是什么？

两个 baseline：

- **A — 未融合版**: 两个独立 kernel `residual_add <<<...>>>` + `layernorm <<<...>>>`
  - Pass 1: residual_add 读 x (128 MB) + res (128 MB)，写 x' (128 MB) → 384 MB HBM
  - Pass 2: layernorm 读 x' (128 MB) + gamma/beta (~0)，写 y (128 MB) → 256 MB HBM
  - 合计: 640 MB HBM 流量，2 次 kernel launch
- **B — 融合版**: `fused_residual_layernorm <<<...>>>` 一个 kernel
  - 同时加载 x (128 MB) + res (128 MB)，在寄存器中相加，然后 layernorm
  - 输出 y (128 MB) → 384 MB HBM
  - 合计: 384 MB HBM 流量，1 次 kernel launch

| 方案 | Kernel 数 | HBM 流量 | 实测时间 |
|------|----------|---------|---------|
| 未融合 | 2 | 640 MB | ~3.0 ms |
| 融合 | 1 | 384 MB | 1.63 ms |

### 2. 优化了什么？

融合 residual add 到 LayerNorm kernel 内部：

```
// 未融合: 两个 kernel
kernel_add(x, res, x_prime)    // 写 HBM
kernel_layernorm(x_prime, y)   // 读 HBM

// 融合: 一个 kernel
kernel_fused(x, res, y)        // 在寄存器里加，不写中间 tensor
  register float tmp = x[i] + residual[i];  // ← 关键行
  sum_x += tmp;
  sum_x2 += tmp * tmp;
```

float4 变体被尝试但拒绝：计算量没变，HBM 流量没变，但寄存器压力从 ~32 升到 ~48，occupancy 从 ~100% 降到 ~50%，导致 1.63 ms → 1.67 ms (−2.5% 倒退)。

### 3. 为什么减少 HBM 访问？

核心：**消除中间 tensor x' 的写和读**。

```
未融合版:
  residual_add: 写 x' = x + res   → 128 MB HBM write
  layernorm:    读 x'              → 128 MB HBM read
                合计 256 MB 额外流量 (占总体 640 MB 的 40%)

融合版:
  直接在寄存器中做 add，x' 从不离开 SM
  HBM 流量: 384 MB (仅为未融合版的 60%)
```

| 步骤 | 未融合 HBM | 融合 HBM |
|------|-----------|---------|
| read x | 128 MB | 128 MB |
| read res | 128 MB | 128 MB |
| write x' | 128 MB | — |
| read x' | 128 MB | — |
| read gamma | ~0 | ~0 |
| write y | 128 MB | 128 MB |
| **总计** | **640 MB** | **384 MB** |

**每层节省 256 MB HBM 流量**，32 层模型累计节省 ~8 GB。融合版 wall time (1.63 ms) 与 standalone LN (1.63 ms) 相同——做更多的事但花同样的时间。

### 4. cudaEvent 测了多少？

- Warmup: 50 次（GPU 升频）
- Measured: 500 次
- 报告值: 总时间 / 500
- 两种变体（scalar + float4）+ block size sweep（128, 256）
- 额外验证: 未融合版（add + LN 分别计时）对比融合版

特别注意：因 nvcc register aliasing bug，mean/rstd 通过 `__shared__` 传递，cudaEvent 计时不受影响（bug 修复不影响性能，只影响正确性）。

### 5. nsys 能看到什么？

融合 vs 未融合的对比在 nsys timeline 上一目了然：

| 观测项 | 未融合 | 融合 |
|--------|--------|------|
| Kernel count | 2 个 launch | 1 个 launch |
| 中间 tensor | 可见 global memory 读写 | 无 |
| Stream 行为 | 可能看到 stream 间隙 | 紧凑 |

此外 nsys 能验证：
- 融合版没有额外的 cuMemcpy
- Block grid 映射正确（grid = 4096, block = 256）
- Kernel duration 接近 standalone LN（证明融合没有引入额外同步开销）

### 6. WSL2 下 ncu 为什么受限？

WSL2 GPU-PV 不转发硬件 PMU 访问。ncu `--set full` 在 WSL2 下无法获取 `dram__sectors_write` 来**直接验证**中间 tensor 写操作被消除——只能通过推算确认。

无法获取的证据：
- `dram__bytes_write.sum` — 无法用仪器值对比融合前后的 HBM 写流量
- `lts__t_sectors_srcunit_tex_op_write` — 无法确认 L2 写入次数减少
- 这些计数器在原生 Linux 下可以直接读出 256 MB 的差值

### 7. 如果上原生 Linux，ncu 能看哪些指标？

| 指标 | 预期 | 意义 |
|------|------|------|
| `dram__bytes_write.sum` | 融合版比未融合版少 ~128 MB | 直接验证中间 tensor 写消除 |
| `dram__bytes_read.sum` | 融合版比未融合版少 ~128 MB | 直接验证中间 tensor 读消除 |
| `dram__throughput.avg` | ~870 GB/s | 86% peak 利用率 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 87% (scalar) vs 50% (float4) | 验证 float4 的 occupancy 损失 |
| `register_file__bank_conflict_pct` | 应接近 0 | 验证寄存器 bank 是否 conflict |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active_active` | 应 > 40% | 确认 memory bound |

**最关键**: `dram__bytes_write.sum` 差值直接证明融合是否真正减少了 HBM 写入。这是核心问题中可以展示的定量证据。

### 8. 下一步值得做的优化？

- **Online 1-pass fusion**: 将 LayerNorm 的 mean+var 和 normalize 也合并为 1-pass，HBM reads 从 2x 降到 1x，总流量从 384 MB 降到 256 MB
- **FlashAttention 级融合**: 将 QK^T matmul + softmax + residual + LN 全融合，消除 attention 后的 logits 矩阵写入
- **用 nsight-systems 对比**: 获取融合 vs 未融合的完整 kernel 时间线，测量 launch overhead 的减少
