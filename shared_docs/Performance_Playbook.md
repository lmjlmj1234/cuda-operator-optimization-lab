# Performance Playbook: CUDA 优化技术目录

> 本文档是按需查阅的优化技术目录。每种技术包含：原理、适用场景、预期收益、风险、代码模式。
> 优化一个 kernel 时，从 Baseline 出发，依次尝试 Playbook 中的技术，记录每一步的收益。

---

## 技术 1: Online Algorithm (在线算法)

### 原理

把多遍 HBM 遍历合并为一遍。核心洞察是 max 和 sum 可以在一次扫描中同时维护：
当遇到新 max 时，用 `expf(old_m - new_m)` 缩放已有 sum。

### 适用场景

- Softmax (3-pass → 1-pass)
- FlashAttention (QK^T 分块 + 在线 softmax)
- 任何需要先求 max 再求 sum 的归约

### 预期收益

- HBM 遍历: 5 次 → 3 次 (减少 40%)
- 实测: softmax 的 −39.7% 与理论值吻合

### 风险

- 额外 expf 计算（但 memory-bound kernel 中计算是免费的）
- 保持两个状态变量 (m, s) 的寄存器压力略增

### 代码模式

```cuda
float local_m = -FLT_MAX, local_s = 0.0f;
for (int i = tid; i < D; i += BLOCK) {
    float xi = x[i];
    float old_m = local_m;
    local_m = fmaxf(local_m, xi);
    local_s = local_s * expf(old_m - local_m) + expf(xi - local_m);
}
// Then: block_reduce_max for m, correct s, block_reduce_sum for s
```

---

## 技术 2: Welford Online Variance (在线方差)

### 原理

用迭代更新替代 E[X²] − E[X]²，避免 catastrophic cancellation。

### 适用场景

- LayerNorm / RMSNorm 的方差计算
- 大均值 + 小方差 (NLP 常见)
- fp16/bf16 混合精度 (精度预算紧张)

### 预期收益

- 数值稳定性: 从 ~1e-3 误差降至 ~1e-4 (10× 提升)
- 速度损失: ~2.5% (代价微小)

### 代码模式

```cuda
struct WelfordStats { float mean; float m2; int count; };

void welford_update(WelfordStats& s, float x) {
    s.count++;
    float delta = x - s.mean;
    s.mean += delta / s.count;
    s.m2 += delta * (x - s.mean);
}

WelfordStats welford_merge(WelfordStats a, WelfordStats b) {
    int count = a.count + b.count;
    float delta = a.mean - b.mean;
    float mean = (a.count * a.mean + b.count * b.mean) / count;
    float m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / count;
    return {mean, m2, count};
}
```

---

## 技术 3: Block Reduce (Shared Memory Tree)

### 原理

利用 shared memory 做树形归约。每个 thread 写入 sdata[tid]，然后分层合并。

### 适用场景

- 所有 element-wise 算子的跨线程归约
- BLOCK > warpSize (32) 时必需

### 预期收益

- 比 global atomic 快 ~100×
- 比 warp shuffle 慢 (~20-40 cycles vs ~5 cycles)

### 风险

- Bank conflict (可以通过 padding 解决：sdata[32][33])
- __syncthreads() 开销 (~40 cycles)

### 代码模式

```cuda
template <int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    return sdata[0];
}
```

---

## 技术 4: Warp Shuffle (寄存器通信)

### 原理

同一 warp 内的线程通过 `__shfl_xor_sync` 直接交换寄存器值，无需 shared memory。

### 适用场景

- BLOCK ≤ 32 (一个 warp)
- 作为 shared memory tree 的第一阶段 (先 warp shuffle，再跨 warp 合并)
- Welford 合并 (layernorm_welford)

### 预期收益

- 无 shared memory 开销 (~5 cycles vs ~20-40)
- 消除 bank conflict

### 代码模式

```cuda
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

---

## 技术 5: Float4 Vectorized Load/Store

### 原理

用 `reinterpret_cast<const float4*>(ptr)` 一次加载 16 字节，将 LDG 指令减至 1/4。

### 适用场景

- **非连续访存** (stride, 随机索引) — 减少 sector miss
- **指令瓶颈** kernel — 减少指令发射压力
- **Tensor Core 输入** — 需要 16 字节对齐

### 不适用场景

- **连续对齐访存** — 编译器已生成高效 LDG，float4 反而增加寄存器压力
- **Memory-bound kernel** — 瓶颈在 HBM 带宽，不在指令数

### 预期收益

- 连续访存: ~0-2% (可能为负!)
- 非连续访存: ~30-50%
- 代价: 寄存器 +4-6

### 代码模式

```cuda
// Bad if access is coalesced: float4 adds regs but no bandwidth benefit
// Good if access is strided: reduces sector misses by 4x

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return reinterpret_cast<const float4*>(ptr)[0];
}
```

---

## 技术 6: Operator Fusion (算子融合)

### 原理

将多个连续算子合并为单个 kernel，消除中间张量的 HBM 读写。

### 适用场景

```
Residual Add + LayerNorm → FusedResidualLN     (−33% HBM)
QK^T + Softmax + PV → FlashAttention           (O(N²) → O(N))
GEMM + Bias + ReLU → FusedGemmBiasReLU         (−25% HBM)
```

### 预期收益

- 消除中间张量的读写 (每 fusion 点 ~256 MB HBM)
- 减少 kernel launch 次数 (每次 ~7 μs)

### 风险

- 寄存器压力增大 (多输入需要更多寄存器)
- 编译器可能引入 bug (见 fused_layernorm 的 nvcc 寄存器别名 bug)

---

## 技术 7: Persistent Kernel (持久线程)

### 原理

启动更少的 block，让空闲 warp 从工作队列中取任务。

### 适用场景

- N < SM count 时 (小 batch)
- 需要跨行数据共享时

### 预期收益

- 小 batch: ~10-20%
- 大 batch: 无收益 (1-block-per-row 已足够)

---

## 技术 8: Register Blocking (寄存器阻塞)

### 原理

在寄存器中缓存数据，减少对 HBM 的重复访问。常用于 GEMM 的 tiling。

### 适用场景

- GEMM (矩阵乘)
- 卷积 (implicit GEMM)
- 任何有数据复用的操作

### 预期收益

- GEMM: ~2-5× (取决于 tile 大小)
- 代价: 极高寄存器压力 (kernel 可达 128+ regs)

---

## 技术 9: Tensor Core (张量核心)

### 原理

利用 NVIDIA Tensor Core 执行 `D = A × B + C` 的混合精度矩阵运算。

### 适用场景

- GEMM with fp16/bf16/tf32 input
- 大矩阵乘 (M/N/K > 1024)

### 限制

- 需要 sm_80+ (Ampere) 或 sm_90+ (Hopper)
- 对 fp32 无加速 (除非使用 tf32)
- 不适用 element-wise 算子

---

## 技术选择决策树

```
Kernel 慢?
│
├─ === 是否是 memory-bound? (Compute Intensity < 35.3) ===
│   YES → 减少 HBM 访问:
│   │   ├─ Online Algorithm? (多遍遍历 → 单遍)
│   │   ├─ Operator Fusion? (消除中间张量)
│   │   ├─ Persistent Kernel? (小 batch 优化)
│   │   └─ 如果已经带宽压满 → 无法再优化 (stop)
│   │
│   NO → 优化计算:
│       ├─ Tensor Core? (混合精度矩阵乘)
│       ├─ Register Blocking? (数据复用)
│       ├─ Warp Shuffle? (减少 SMEM 延迟)
│       ├─ Float4? (减少指令发射)
│       └─ 如果已经计算压满 → 无法再优化 (stop)
│
└─ === 是否是延迟隐藏不足? (Occupancy 或 Stall 高) ===
    YES → 提高并行度:
        ├─ 增大 BLOCK size
        ├─ 减少寄存器使用
        └─ 减少 shared memory
```
