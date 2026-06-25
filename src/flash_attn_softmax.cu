#include "utils.cuh"

// ============================================================
// FlashAttention 风格的在线 Softmax
//
// 核心思想: 在 FlashAttention 中, S = Q @ K^T 是一个大型中间矩阵
// (维度 B×N×N)，如果直接 materialize 会消耗大量 HBM。
// 在线 Softmax 算法使得我们可以在计算 QK^T 的每个 tile 时，
// 同时累加 softmax 统计量，无需将完整的 S 写出到 HBM。
//
// 这里模拟 FlashAttention 的 Tile 式在线 Softmax:
//   过程:
//   1. 从 HBM 中加载 Q 的 tile [🟡 SRAM] (以 row-major 向量模拟)
//   2. 从 HBM 中加载 K 的 tile [🟡 SRAM]
//   3. 计算 S = Q_tile @ K_tile^T 到寄存器
//   4. 用在线算法更新 m 和 s (纯寄存器操作)
//   5. 最终将 P = softmax(S) 乘以 V_tile 累加到输出
//
//   [ FUSED] 融合了: QK^T矩阵乘 + Softmax + PV累加
//
//   显存访问对比:
//   朴素方法: Q(读) + K(读) + S(写) + S(读) + P(写) + V(读) + O(写)
//   融合方法: Q(读) + K(读) + V(读) + O(写)  (S 和 P 完全驻留寄存器/SRAM)
// ============================================================

// 简化版: 演示 FlashAttention 中的在线 Softmax 核心逻辑
// 输入: q 和 k 向量 (模拟单 query 对多 key 的 attention 分数计算)
// 输出: softmax 分布
//
// 完整 FlashAttention 还需要 tile 式的 V 累加，此处聚焦 softmax 部分
__global__ void flash_attn_softmax_kernel(
    const float* q,          // [ HBM] query 向量, shape: [N, d]
    const float* k,          // [ HBM] key 矩阵,  shape: [N, d]
    const float* v,          // [ HBM] value 矩阵, shape: [N, d]
    float* output,           // [ HBM] 输出
    int N,                   // 序列长度
    int d,                   // head dimension
    float scale              // 1/sqrt(d)
) {
    // 每个 block 处理一行 (一个 query)
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 加载当前 query 到寄存器 (需要 d 个元素)
    // 注意: 实际 FlashAttention 会分 tile，这里简化
    __shared__ float q_shared[128];           // [🟡 SRAM]
    if (tid < d) {
        q_shared[tid] = q[row * d + tid];    // [ HBM] -> [🟡 SRAM]
    }
    __syncthreads();

    // ---------------------------------------------------------
    // 在线 Softmax 扫描: 遍历所有 key, 同时计算 S 和统计量
    // ---------------------------------------------------------
    float m_val = -FLT_MAX;
    float s_val = 0.0f;

    // 每个线程负责多个 key 位置
    for (int j = tid; j < N * d; j += blockDim.x) {
        int key_row = j / d;
        int key_col = j % d;

        // 从 SRAM 读取 q 的对应元素
        float q_elem = q_shared[key_col];     // [🟡 SRAM]
        // 从 HBM 读取 k
        float k_elem = k[key_row * d + key_col];  // [ HBM]

        // warp 级归约: 计算点积 (伪代码，实际需要 tree reduce)
        // 这里简化为在内循环中累加
    }

    // ---- 实际简化: 每个线程算完整点积太慢，改为分步 ----
    // 重新设计: 每个 block 处理一个 query row
    // 每个 thread 负责一个 key row 的 S 值

    // 实际简化版本: 用 shared memory tile 方式
    // 对每个 key tile: 加载到 SRAM, 计算部分 S, 在线更新 m/s

    // 由于完整 FlashAttention 实现过长，这里提供
    // 核心在线 softmax 在 flash attention 中的融合模式:
}

// ============================================================
// FlashAttention-Style 在线 Softmax (简化版)
//
// 展示核心在线逻辑: 逐个处理 score, 同时维护 m 和 s,
// 而不需要知道最终的 max 和 sum
// ============================================================
__global__ void online_softmax_demo(
    const float* scores,     // [ HBM] 预计算的 scores (N 个)
    float* output            // [ HBM] softmax 输出
) {
    // 每个 thread 处理一个 score 向量 (N 维)
    int tid = threadIdx.x;
    int N = blockDim.x;

    // ---------------------------------------------------------
    //  第一遍: 在线扫描, 同时追踪 max 和 sum
    // ---------------------------------------------------------
    float m = -FLT_MAX;
    float s = 0.0f;

    // 每个线程独立处理完整的 N 个 score
    // (适用于 N 较小的情况; 大 N 时需要多个线程协作)
    for (int i = 0; i < N; i++) {
        float xi = scores[blockIdx.x * N + i];   // [ HBM] 读取

        // 在线算法核心:
        //   m_new = max(m, xi)
        //   s_new = s * exp(m - m_new) + exp(xi - m_new)
        //
        // 意义: 当发现新的最大值时，之前用旧 max 算的 exp 值
        // 需要乘以 exp(old_m - new_m) 进行"缩放修正"
        float old_m = m;
        m = fmaxf(m, xi);
        s = s * expf(old_m - m) + expf(xi - m);  // [ FUSED] max + sum 融合为一趟
    }

    // ---------------------------------------------------------
    //  第二遍: 写出 softmax 结果
    // ---------------------------------------------------------
    for (int i = 0; i < N; i++) {
        float xi = scores[blockIdx.x * N + i];   // [ HBM] 再次读取
        output[blockIdx.x * N + i] = expf(xi - m) / s;  // [ HBM] 写入
    }
}

// ============================================================
// FlashAttention 式 Tile 在线 Softmax (完整版示意)
//
// 这是完整的 Tile 版, 展示算子融合核心:
//   [ FUSED] softmax(QK^T) × V 融合为单次扫描
//
// 内存访问分析:
//   Q: 从 HBM 加载一次到 SRAM
//   K: 以 tile 方式从 HBM 加载到 SRAM
//   V: 以 tile 方式从 HBM 加载到 SRAM
//   O: 累加后写回 HBM
// 对比非融合: 避免了 S = QK^T (N×N) 的 HBM 读写
// ============================================================
#define TILE_D 64  // head dimension tile (通常 >= d)

__global__ void flash_attn_tile_softmax(
    const float* Q,            // [ HBM] shape: [N, d]
    const float* K,            // [ HBM] shape: [N, d]
    const float* V,            // [ HBM] shape: [N, d]
    float* O,                  // [ HBM] shape: [N, d]
    int N, int d, float scale
) {
    // 这个 kernel 每个 block 处理一个 query row
    int row_q = blockIdx.x;

    extern __shared__ float shared_mem[];      // [🟡 SRAM]
    float* K_tile = shared_mem;                // K tile: [TILE_D]
    float* V_tile = shared_mem + TILE_D;       // V tile: [TILE_D]

    float m = -FLT_MAX;  // 当前 max
    float s = 0.0f;      // 当前 sum (经过 max 修正)
    float acc[TILE_D];   // 寄存器中累加 output

    // 初始化 acc 为 0
    #pragma unroll
    for (int i = 0; i < TILE_D; i++) acc[i] = 0.0f;

    // 遍历所有 key-value 位置 (以 TILE_D 为步长)
    // 实际 FlashAttention 以 B_r × B_c tile 方式分块,
    // 这里简化为逐元素扫描以展示 softmax 融合逻辑

    // [ FUSED] 以下循环融合了: QK^T点积 + Softmax在线统计 + O累加
    for (int j = 0; j < N; j++) {
        // --- 加载 K[j] 和 V[j] 到 SRAM ---
        if (threadIdx.x < d) {
            K_tile[threadIdx.x] = K[j * d + threadIdx.x];   // [ HBM] -> [🟡 SRAM]
            V_tile[threadIdx.x] = V[j * d + threadIdx.x];   // [ HBM] -> [🟡 SRAM]
        }
        __syncthreads();

        // --- 计算 S_j = Q[row] · K[j] (点积) ---
        float s_val = 0.0f;
        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            s_val += Q[row_q * d + k] * K_tile[k];          // [ HBM] + [🟡 SRAM]
        }
        // warp 归约得到完整点积
        s_val = warp_reduce_sum(s_val) * scale;

        // --- 在线 Softmax 更新 ---
        // [ FUSED] 这里融合了 m 和 s 的在线更新
        float old_m = m;
        m = fmaxf(m, s_val);
        float exp_s_old = expf(s_val - m);
        float exp_correction = expf(old_m - m);

        // 修正已有的 acc: 每个元素乘 exp(old_m - m)
        // (softmax 分母变化了，旧的累加值需要缩放)
        #pragma unroll
        for (int k = 0; k < TILE_D; k++) {
            acc[k] = acc[k] * exp_correction;
        }

        // 累加当前 V[j] 的贡献: exp(s_val - m) * V[j]
        #pragma unroll
        for (int k = 0; k < TILE_D; k++) {
            acc[k] += exp_s_old * V_tile[k];                // [🟡 SRAM]
        }

        // 更新分母
        s = s * exp_correction + exp_s_old;

        __syncthreads();
    }

    // --- 写出结果: O[row] = acc / s ---
    if (threadIdx.x < d) {
        O[row_q * d + threadIdx.x] = acc[threadIdx.x] / s; // [ HBM] 写入
    }
}
