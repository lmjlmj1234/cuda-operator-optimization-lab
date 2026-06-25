# Benchmark Specification: 统一基准测试规范

> 所有 kernel 的 Benchmark 必须遵守此规范。不遵守 = 数据不可比。

---

## 1. 计时方法

### 标准 (cudaEvent)

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

for (int w = 0; w < N_WARMUP; w++) { kernel<<<...>>>(); }
cudaEventRecord(start);
for (int i = 0; i < N_ITERS; i++) { kernel<<<...>>>(); }
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
float avg_ms = ms / N_ITERS;
```

### 参数

| 参数 | 值 | 原因 |
|------|-----|------|
| N_WARMUP | 50 | GPU 频率从 idle 提升到 boost (~210→1777 MHz)，冷缓存预热 |
| N_ITERS | 500 | 足够样本量消除单次波动，cudaEvent 计时粒度 ~1 μs |
| 报告值 | mean | cudaEvent 计时稳定，均值足够 |

### 不使用 CPU 计时

```cuda
// ❌ 错误: CPU clock 计时包含 kernel launch 延迟
auto start = clock();
kernel<<<...>>>();
cudaDeviceSynchronize();
auto end = clock();

// ✅ 正确: cudaEvent 只测量 GPU 执行时间
cudaEventRecord(start);
kernel<<<...>>>();
cudaEventRecord(stop);
```

---

## 2. 报告格式

每个 Benchmark 结果必须包含以下信息：

### Environment Block

```
GPU:           NVIDIA GeForce RTX 3060 (GA106, 28 SM)
CUDA Version:  13.2
Driver:        595.95
Platform:      WSL2 (GPU-PV, +5-15% timing variance)
```

### Problem Size Block

```
N = 4096 (rows)
D = 8192 (cols)
Data type = float32 (4 bytes)
Total data = N × D × 4 = 128 MB
```

### Results Table

| Kernel | Time (ms) | Bandwidth (GB/s) | BW % Peak | GFLOPS | vs Baseline |
|--------|-----------|-------------------|-----------|--------|-------------|
| baseline | 2.166 ± 0.15 | 295 | 82% | — | — |
| v1 | 1.305 ± 0.09 | 294 | 82% | — | −39.7% |

### 计算方式

```
Bandwidth = total_bytes / (time_seconds) × 1e-9  → GB/s
BW % Peak = Bandwidth / theoretical_peak × 100
GFLOPS = total_FLOPs / (time_seconds) × 1e-12  → TFLOPS (如适用)
```

### RTX 3060 理论峰值

- HBM Bandwidth: 360 GB/s (192-bit bus × 15 Gbps GDDR6)
- FP32 Compute: 12.7 TFLOPS (28 SM × 128 FP32 cores × 2 FMA × 1777 MHz)

---

## 3. 跨 kernel 对比规则

### 可比场景 (直接对比)

- 相同 N、D、数据类型
- 相同 GPU、CUDA 版本
- 相同 warmup/iterations 次数
- 相同计时方法

### 不可比场景 (需说明)

- 不同 GPU 型号 (需标注架构差异)
- 不同数据量 (小 N 时 kernel launch overhead 占比更大)
- 不同数据类型 (float32 vs float16 带宽不同)
- WSL2 vs 原生 Linux (WSL2 慢 5-15%)

### 相对性能的稳定性

虽然绝对值波动，但相对比例稳定:
```
online/naive = 1.305/2.166 = 0.603 → -39.7%
跨 session 波动: ±2% 以内
```

---

## 4. 带宽峰值参考表

| GPU | SM | HBM Bandwidth | FP32 TFLOPS | Ridge Point |
|-----|-----|-------------|-------------|-------------|
| RTX 3060 | 28 | 360 GB/s | 12.7 | 35.3 FLOP/byte |
| RTX 4090 | 128 | 1008 GB/s | 82.6 | 82.0 FLOP/byte |
| A100 SXM | 108 | 2039 GB/s | 19.5 | 9.6 FLOP/byte |
| H100 SXM | 132 | 3350 GB/s | 66.9 (sparse) | 20.0 FLOP/byte |
| RTX 3090 | 82 | 936 GB/s | 35.7 | 38.1 FLOP/byte |
| RTX 2080 | 46 | 448 GB/s | 11.1 | 24.8 FLOP/byte |
| T4 | 40 | 320 GB/s | 8.1 | 25.3 FLOP/byte |

---

## 5. 多 session 比较

```python
# 当需要比较不同 session 的 benchmark 数据时:
# 1. 确保 N, D, 数据类型相同
# 2. 记录 Environment Block
# 3. 只比较同一 kernel 变体间的相对变化

# 示例:
session1 = {"softmax_online": 1.305, "softmax_naive": 2.166, "platform": "WSL2"}
session2 = {"softmax_online": 1.227, "softmax_naive": 2.069, "platform": "WSL2"}

# 相对比例跨 session 稳定:
ratio1 = session1["softmax_online"] / session1["softmax_naive"]  # 0.603
ratio2 = session2["softmax_online"] / session2["softmax_naive"]  # 0.593

# ratio1 ≈ ratio2, 偏差 < 2%
```
