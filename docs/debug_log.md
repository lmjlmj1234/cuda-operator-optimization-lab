# 调试日志 / Debug Log

> 记录了 CUDA kernel 开发和调试过程中发现并修复的 bug。
> 每一条记录包括：问题现象、根因分析、修复方案。

---

## 2026-06-24: Fused LayerNorm 寄存器别名错误

### 现象

`fused_residual_layernorm` 输出的每行均值约为 `0.042`（期望值 `0.0`），
导致与 PyTorch 参考对比的最大绝对误差高达 **0.24** (24%)。

奇怪的是：
- 开启编译器优化 `-O0` 时正常，`-O2` / `-O3` 时出错；
- 第一行 (row 0) 在 **`<<<N, 256>>>`** 时正确，在 **`<<<1, 256>>>`** 时错误；
- 多轮运行时结果在 `0.000` 和 `0.042` 之间随机跳变；
- 单独提取 kernel 编译的独立测试文件中也复现。

### 根因

**nvcc 编译器寄存器别名错误** (register aliasing / register reuse bug)。

Kernel 的结构是两遍数据遍历：
```
Pass 1: 读 x[i] + r[i], 计算 sum, sum_sq
        → block_reduce_sum → mean, rstd (存在寄存器中)
Pass 2: 再次读 x[i] + r[i], 计算 (fused - mean) * rstd
```

Pass 2 的循环体中有 `x[i] + r[i]` 访存，编译器在优化时将
`mean` 和 `rstd` 的物理寄存器**错误地复用了给** 循环内的临时变量，
导致 `mean`/`rstd` 在循环中被覆盖，产生随机错误。

核心特征：该 bug **只影响 fused kernel**（因为两遍都做 `x[i] + r[i]` 访存），
不影响标准 LayerNorm kernel（第一遍只读，第二遍也只读，编译器不会冲突）。

### 修复

通过 `__shared__` 传递 mean 和 rstd，强制编译器将统计量写入共享内存，
消除寄存器冲突：

```cuda
__shared__ float sm_mean, sm_rstd;
if (threadIdx.x == 0) {
    sm_mean = sum / dim;
    sm_rstd = rsqrtf(fmaxf(var, 0.0f) + eps);
}
__syncthreads();
float mean = sm_mean;
float rstd = sm_rstd;
```

**教训**: nvcc 的寄存器分配器不是完全可靠的。当 kernel 有"两遍结构"且
两遍都做相似的访存模式时，中间结果通过 shared memory 传递比寄存器更安全。

---

## 2026-06-24: Welford LayerNorm 缺少 Warp 缩减

### 现象

`layernorm_welford` 输出中包含 **NaN**，导致 Welford 与 sum_sq 版本的
交叉验证失败（diff = nan）。

### 根因

Kernel 的 Welford 合并流程为：
1. 每个线程独立 Welford 更新 → `(count, mean, m2)`
2. Warp shuffle 合并（`__shfl_xor_sync`）→ 每个 warp 一组统计量
3. **warp_stats** 写入 shared memory
4. Warp 0 读取 warp_stats，**再次** warp shuffle 合并 → block_stat
5. `__shfl_sync` 广播 block_stat → 所有线程

Bug 出在第 4 步：当时版本的代码中，warp 0 读取 warp_stats 后**缺少
最终的 warp shuffle 缩减**。这意味着 warp 0 除了 lane 0 以外的 31 个线程
各自持有部分 `warp_stats[w]` 但**从未合并**。

最终广播时 `__shfl_sync(..., 0)` 只从 lane 0 取值，而 lane 0 只合并了
warp_stats[0]（如果线程恰好在 warp 0 内读到自己负责的那个 slot），
或者根本没正确合并。结果 `m2` 可能为 0 → var = 0 → 除零 → NaN。

### 修复

在 warp 0 读取 warp_stats 之后增加 warp shuffle 树形归约：

```cuda
if (wid == 0) {
    int num_warps = (BLOCK_SIZE + 31) / 32;
    for (int w = threadIdx.x; w < num_warps; w += warpSize) {
        WelfordStats ws = warp_stats[w];
        block_stat = welford_merge(block_stat, ws);
    }
    // 新增: warp shuffle 缩减合并所有 lane 的结果
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordStats nb;
        nb.count = __shfl_xor_sync(0xffffffff, block_stat.count, offset);
        nb.mean  = __shfl_xor_sync(0xffffffff, block_stat.mean,  offset);
        nb.m2    = __shfl_xor_sync(0xffffffff, block_stat.m2,    offset);
        block_stat = welford_merge(block_stat, nb);
    }
    if (lane == 0) warp_stats[0] = block_stat;
}
__syncthreads();
block_stat = warp_stats[0];
```

**教训**: Welford 的并行合并需要两层缩减（warp 内 + block 级）。
只做 warp shuffle（块内）而不做 block 级缩减（warp 间）是不完整的。
