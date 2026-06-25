#include "utils.cuh"

// ============================================================
// Layer Normalization — 版本 1: 经典 sum/sum_sq 法
//
//   mean   = 1/N * sum(x_i)
//   var    = 1/N * sum(x_i^2) - mean^2
//   y_i    = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
//
// 数学等价，且利于并行（每个元素独立贡献 sum 和 sum_sq）
// HBM 访问: 2x 读 (x) + 1x 写 (y) = 3 次 + gamma/beta 读
// ============================================================
template <int BLOCK_SIZE>
__global__ void layernorm_sum_sq(
    const float* input,     // [ HBM] 输入
    float* output,          // [ HBM] 输出
    const float* gamma,     // [ HBM] 缩放参数
    const float* beta,      // [ HBM] 偏移参数
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;     // [ HBM]
    float* y = output + row * dim;          // [ HBM]

    // --- 求和与平方和 (并行归约) ---
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float xi = x[i];                    // [ HBM] 读取
        sum    += xi;
        sum_sq += xi * xi;
    }

    sum    = block_reduce_sum<BLOCK_SIZE>(sum);     // 跨 block 归约
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);  // 跨 block 归约
    __syncthreads();

    float mean   = sum / dim;
    float var    = sum_sq / dim - mean * mean;      // Var = E[X^2] - E[X]^2
    float rstd   = rsqrtf(fmaxf(var, 0.0f) + eps);  // 1 / sqrt(var + eps)

    // --- 归一化 + 仿射变换 ---
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = (x[i] - mean) * rstd                 // [ HBM] 读 x
             * gamma[i]                              // [ HBM] 读 gamma
             + beta[i];                              // [ HBM] 读 beta
                                                     // [ HBM] 写 y
    }
}

// ============================================================
// Layer Normalization — 版本 2: Float4 向量化
//
// 使用 float4 将访存指令数降低 4 倍
// 全局合并访存 (coalesced memory access) 更高效
// HBM 访问: 2x 读 x (向量化) + 1x 写 y (向量化) + gamma/beta
// ============================================================
template <int BLOCK_SIZE>
__global__ void layernorm_float4(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;     // [ HBM]
    float* y = output + row * dim;          // [ HBM]

    // Float4 方式求和: 每 4 个元素一组加载
    float sum = 0.0f, sum_sq = 0.0f;
    int vec_dim = dim / 4;
    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 v = load_float4(x + 4 * i);          // [ HBM] 向量化加载
        sum     += v.x + v.y + v.z + v.w;
        sum_sq  += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    // 处理剩余元素 (dim % 4 != 0)
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float xi = x[i];                            // [ HBM] 标量加载
        sum    += xi;
        sum_sq += xi * xi;
    }

    sum    = block_reduce_sum<BLOCK_SIZE>(sum);
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    float mean   = sum / dim;
    float var    = fmaxf(sum_sq / dim - mean * mean, 0.0f);
    float rstd   = rsqrtf(var + eps);

    // Float4 方式写出
    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 v = load_float4(x + 4 * i);          // [ HBM] 第二次读取 x
        float4 g = load_float4(gamma + 4 * i);      // [ HBM] 读取 gamma
        float4 b = load_float4(beta + 4 * i);       // [ HBM] 读取 beta
        float4 out;
        out.x = (v.x - mean) * rstd * g.x + b.x;
        out.y = (v.y - mean) * rstd * g.y + b.y;
        out.z = (v.z - mean) * rstd * g.z + b.z;
        out.w = (v.w - mean) * rstd * g.w + b.w;
        store_float4(y + 4 * i, out);              // [ HBM] 向量化写入
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = (x[i] - mean) * rstd                // [ HBM] 标量
             * gamma[i] + beta[i];
    }
}

// ============================================================
// Layer Normalization — 版本 3: Welford 在线算法 + Warp Shuffle
//
// Welford 算法: 在线计算均值和方差
//   单线程在线更新: delta = x - mean; mean += delta / count; m2 += delta * (x - mean)
//   并行合并: 两个统计量通过 closed-form 公式合并
//
// 优势: 数值稳定性远超 sum/sum_sq 法 (避免 catastrophic cancellation)
// 代价: 需要两阶段 (每线程 Welford + 跨线程 merge)，实现更复杂
//
// Warp Shuffle 优势: 无需 shared memory 即可完成 warp 内通信
// ============================================================
template <int BLOCK_SIZE>
__global__ void layernorm_welford(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;     // [ HBM]
    float* y = output + row * dim;          // [ HBM]

    // --- 阶段 1: 每个线程独立 Welford 更新 ---
    WelfordStats local = {0.0f, 0.0f, 0};
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        welford_update(local, x[i]);        // [ HBM] 读取
    }

    // --- 阶段 2: 跨线程 Welford 合并 ---
    // 先 warp 内合并: 每个 warp 产生一组统计量
    // 使用 __shfl_xor_sync 做树形归约
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordStats neighbor;
        // 用 shuffle 获取兄弟线程的统计量
        neighbor.count = __shfl_xor_sync(0xffffffff, local.count, offset);
        neighbor.mean  = __shfl_xor_sync(0xffffffff, local.mean,  offset);
        neighbor.m2    = __shfl_xor_sync(0xffffffff, local.m2,    offset);
        local = welford_merge(local, neighbor);
    }

    // lane 0 将 warp 结果写入 shared memory
    __shared__ WelfordStats warp_stats[32];         // [🟡 SRAM]
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) warp_stats[wid] = local;          // [🟡 SRAM] 写

    __syncthreads();

    // 第一个 warp 做最终的 Welford 合并
    WelfordStats block_stat = {0.0f, 0.0f, 0};
    if (wid == 0) {
        int num_warps = (BLOCK_SIZE + 31) / 32;
        for (int w = threadIdx.x; w < num_warps; w += warpSize) {
            WelfordStats ws = warp_stats[w];         // [🟡 SRAM] 读
            block_stat = welford_merge(block_stat, ws);
        }
        // Warp shuffle reduction to merge across warp 0 lanes
        for (int offset = 16; offset > 0; offset >>= 1) {
            WelfordStats nb;
            nb.count = __shfl_xor_sync(0xffffffff, block_stat.count, offset);
            nb.mean  = __shfl_xor_sync(0xffffffff, block_stat.mean,  offset);
            nb.m2    = __shfl_xor_sync(0xffffffff, block_stat.m2,    offset);
            block_stat = welford_merge(block_stat, nb);
        }
        if (lane == 0) warp_stats[0] = block_stat;   // [🟡 SRAM] 写
    }
    __syncthreads();
    block_stat = warp_stats[0];                       // [🟡 SRAM] 广播

    // 计算最终统计量
    float mean  = block_stat.mean;
    float var   = block_stat.m2 / block_stat.count;
    float rstd  = rsqrtf(var + eps);

    // --- 阶段 3: 归一化 + 仿射变换 ---
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = (x[i] - mean) * rstd                 // [ HBM] 读 + [ HBM] 写
             * gamma[i] + beta[i];
    }
}

// 显式实例化
template __global__ void layernorm_sum_sq<128>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_sum_sq<256>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_sum_sq<512>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_float4<128>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_float4<256>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_welford<128>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_welford<256>(const float*, float*, const float*, const float*, int, float);
