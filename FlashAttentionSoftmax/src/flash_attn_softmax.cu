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
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float q_shared[128];
    if (tid < d) {
        q_shared[tid] = q[row * d + tid];
    }
    __syncthreads();

    float m_val = -FLT_MAX;
    float s_val = 0.0f;

    for (int j = tid; j < N * d; j += blockDim.x) {
        int key_row = j / d;
        int key_col = j % d;
        float q_elem = q_shared[key_col];
        float k_elem = k[key_row * d + key_col];
    }
}

// ============================================================
// FlashAttention-Style 在线 Softmax (简化版)
// ============================================================
__global__ void online_softmax_demo(
    const float* scores,     // [ HBM] 预计算的 scores (N 个)
    float* output            // [ HBM] softmax 输出
) {
    int tid = threadIdx.x;
    int N = blockDim.x;

    float m = -FLT_MAX;
    float s = 0.0f;

    for (int i = 0; i < N; i++) {
        float xi = scores[blockIdx.x * N + i];
        float old_m = m;
        m = fmaxf(m, xi);
        s = s * expf(old_m - m) + expf(xi - m);
    }

    for (int i = 0; i < N; i++) {
        float xi = scores[blockIdx.x * N + i];
        output[blockIdx.x * N + i] = expf(xi - m) / s;
    }
}

// ============================================================
// FlashAttention 式 Tile 在线 Softmax (完整版示意)
// ============================================================
#define TILE_D 64

__global__ void flash_attn_tile_softmax(
    const float* Q,            // [ HBM] shape: [N, d]
    const float* K,            // [ HBM] shape: [N, d]
    const float* V,            // [ HBM] shape: [N, d]
    float* O,                  // [ HBM] shape: [N, d]
    int N, int d, float scale
) {
    int row_q = blockIdx.x;

    extern __shared__ float shared_mem[];
    float* K_tile = shared_mem;
    float* V_tile = shared_mem + TILE_D;

    float m = -FLT_MAX;
    float s = 0.0f;
    float acc[TILE_D];

    #pragma unroll
    for (int i = 0; i < TILE_D; i++) acc[i] = 0.0f;

    for (int j = 0; j < N; j++) {
        if (threadIdx.x < d) {
            K_tile[threadIdx.x] = K[j * d + threadIdx.x];
            V_tile[threadIdx.x] = V[j * d + threadIdx.x];
        }
        __syncthreads();

        float s_val = 0.0f;
        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            s_val += Q[row_q * d + k] * K_tile[k];
        }
        s_val = warp_reduce_sum(s_val) * scale;

        float old_m = m;
        m = fmaxf(m, s_val);
        float exp_s_old = expf(s_val - m);
        float exp_correction = expf(old_m - m);

        #pragma unroll
        for (int k = 0; k < TILE_D; k++) {
            acc[k] = acc[k] * exp_correction;
        }

        #pragma unroll
        for (int k = 0; k < TILE_D; k++) {
            acc[k] += exp_s_old * V_tile[k];
        }

        s = s * exp_correction + exp_s_old;

        __syncthreads();
    }

    if (threadIdx.x < d) {
        O[row_q * d + threadIdx.x] = acc[threadIdx.x] / s;
    }
}
