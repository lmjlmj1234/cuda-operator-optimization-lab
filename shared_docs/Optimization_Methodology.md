# Optimization Methodology: CUDA Kernel 优化方法论

> 本文档定义从零开始优化一个 CUDA Kernel 的完整方法论。
> 每一步都有明确的输入、输出、判定标准。

---

## 核心原则

1. **先正确，后高效** — 优化前的 version 0 必须通过所有 correctness 测试
2. **一次只改一个变量** — 不要同时应用多个优化 (不知道哪个有效)
3. **每次修改都量化** — 记下 before/after 的 time、bandwidth、occupancy
4. **失败也是结果** — 记录"为什么这个优化没效果"
5. **带宽压满就停** — 当 DRAM Throughput > 90%，停止优化 (已达硬件极限)

---

## 方法论流程

```
                ┌──────────────┐
                │  问题定义     │  ← 理解算法、输入输出、约束
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │  编写 Baseline  │  ← 最简单正确版本
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │  正确性验证    │  ← CPU Ref 对比
                └──────┬───────┘
                       ↓ PASS
                ┌──────────────┐
                │  基准测试     │  ← cudaEvent: 50 warmup + 500 iters
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │  Profiling   │  ← ncu --set full
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │  瓶颈判断     │  ← 是 memory-bound 还是 compute-bound?
                └──────┬───────┘
                       ↓
               ┌────────┴────────┐
               ↓                  ↓
        Memory-bound        Compute-bound
        ┌──────────────┐   ┌──────────────┐
        │ 减少 HBM 访问 │   │ 优化计算模式  │
        │  (Playbook 中 │   │  (Playbook 中 │
        │  上半部分)    │   │  下半部分)    │
        └──────┬───────┘   └──────┬───────┘
               ↓                  ↓
        ┌──────────────────────────────┐
        │  验证正确性 + 重新 benchmark  │
        └──────────────┬───────────────┘
                       ↓
               ┌──────────────┐
               │  记录决策日志  │  ← 为什么做？为什么不做其他方案？
               └──────────────┘
                       ↓
               ┌──────────────┐
               │  收敛判断     │  ← BW > 90%? 或 SM > 80%? 
               └──────┬───────┘
                   NO  ↓ YES
               ┌──────────────┐
               │  Benchmark   │  ← 最终数据
               │  完成文档     │  ← 9 项交付物
               └──────────────┘
```

---

## 步骤详解

### Step 1: 问题定义

**输入**: 算法描述、数学公式

**输出**:
- 这个 kernel 解决什么问题？
- 在 Transformer 流水线中的位置
- 输入/输出形状和数据类型
- 时间复杂度分析
- 访存模式描述 (连续？跨步？随机？)
- 理论 HBM 访问量 (bytes per element)

### Step 2: 编写 Baseline

**目标**: 最简单、一定正确的版本

**规则**:
- 不要用 shared memory (除了 block_reduce)
- 不要用 warp shuffle
- 不要用 float4
- 不要用算子融合
- 用一个 grid、一个 block、stride loop

**模板**:
```cuda
__global__ void my_kernel(const float* input, float* output, int N, int D) {
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        // 最简单的实现
    }
    // 最简单的 block reduce
}
```

### Step 3: 正确性验证

**方法**:
1. 编写 CPU reference function
2. 对前 8 行做逐元素对比
3. 用 PyTorch 做全量对比
4. dump GPU 输出到 .bin 文件 (固定 seed 保证可重现)

**判定**: max_abs_diff < 1e-3 (float32, D=8192)

### Step 4: 基准测试

按 [Benchmark_Specification.md](Benchmark_Specification.md) 执行：50 warmup + 500 iters、cudaEvent timing、计算 bandwidth/GFLOPS。记录 kernel time、标准差、带宽利用率。

### Step 5: Profiling

**命令**: `ncu --set full --kernel-name "kernel*" ./binary`

**关键指标**:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM 利用率
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — HBM 利用率
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed` — Occupancy
- `l1tex__t_sector_pipe_lsu_mem_global_op_ld.stall` — Load stall
- `smsp__average_warps_issue_stalled_barrier` — Barrier stall

### Step 6: 瓶颈判断

**方法 1 — Roofline Model**:
```
Compute Intensity = FLOPs / Bytes transferred

if Compute Intensity < Ridge Point → Memory-bound
if Compute Intensity > Ridge Point → Compute-bound
```

RTX 3060 Ridge Point: 12.7 TFLOPS / 360 GB/s = 35.3 FLOP/byte

**方法 2 — ncu metric**:
```
if DRAM Throughput > SM Throughput → Memory-bound
if SM Throughput > DRAM Throughput → Compute-bound
```

**方法 3 — 实验验证**:
```
如果"减少 HBM 遍历"有效 → Memory-bound
如果"减少 FMA 数量"有效 → Compute-bound
```

### Step 7: 应用优化

从 Performance Playbook 中选择对应技术。

**Memory-bound 优化优先级**:
1. Online Algorithm (减少遍历次数)
2. Operator Fusion (消除中间 HBM 读写)
3. Float4 (仅非连续访存时)

**Compute-bound 优化优先级**:
1. Tensor Core (混合精度)
2. Register Blocking (数据复用)
3. Warp Shuffle (减少 SMEM 延迟)

### Step 8: 记录决策

每个优化尝试无论成功还是失败都要记录：

```
## [2026-06-25] 尝试: Float4 向量化

做了什么: 将标量加载改为 float4 向量加载
预期收益: 减少 LDG 指令 4×
结果: Time 1.286→1.307 ms (+1.6%), 寄存器 34→40
分析: 连续访存中编译器已生成高效 LDG。float4 增加了寄存器压力
结论: 此场景不适用 float4
```

### Step 9: 收敛判断

**停止优化条件** (满足任一)：
- DRAM Throughput > 90% (memory-bound 已压满)
- SM Throughput > 80% (compute-bound 已压满)
- 距离理论极限 < 10%
- 剩余优化收益 < 5% 且实现成本过高

**停止后的工作**：
- 在 future_work.md 中记录"还有哪些可以做但不紧急的优化"
- 生成最终 Benchmark 数据
- 完成所有 9 项文档交付物

---

## 常见陷阱

| 陷阱 | 正确做法 |
|------|---------|
| 第一次就写 float4 版本 | 先写标量正确版本 |
| 优化后不重新验证正确性 | 每次优化后必须 re-run correctness |
| 只测 1 次 (start→stop) | 50 warmup + 500 iters |
| 只看 time，不看 bandwidth | bandwidth 揭示瓶颈是带宽还是计算 |
| "我感觉"哪种方案更好 | 记录数据，让数据说话 |
| 同时应用多个优化 | 一次只改一个变量 |
| WSL2 上测 ncu | 在原生 Linux 上 profiling，WSL2 只做软件计时 |
