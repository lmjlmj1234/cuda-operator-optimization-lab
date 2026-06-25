#include "utils.cuh"

// ============================================================
// RMSNorm (Root Mean Square Normalization)
//
// 公式: y_i = x_i / sqrt( 1/N * sum(x_j^2) + eps ) * gamma_i
//
// 与 LayerNorm 的区别: 不需要减均值 (不中心化)
//   1. 省去了 mean 的计算和广播
//   2. 省去了 x_i - mean 的减法
//   3. 计算量约为 LayerNorm 的 60%
//
// 实验发现: RMSNorm 在许多 LLM 任务中效果与 LayerNorm 相当甚至更好
// (如 Llama 系列全部使用 RMSNorm)
//
// HBM 访问: 2x 读 + 1x 写 = 3 次 (比 LayerNorm 少一次 mean 广播)
// ============================================================
template <int BLOCK_SIZE>
__global__ void rmsnorm_kernel(
    const float* input,     // [ HBM]
    float* output,          // [ HBM]
    const float* gamma,     // [ HBM]
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;     // [ HBM]
    float* y = output + row * dim;          // [ HBM]

    // --- 计算平方和 ---
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float xi = x[i];                    // [ HBM] 读取
        sum_sq += xi * xi;
    }
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    // RMS = sqrt( mean(x^2) + eps )
    float rms = sqrtf(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;

    // --- 归一化 + gamma 缩放 ---
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = x[i] * inv_rms              // [ HBM] 读 x + [ HBM] 写 y
             * gamma[i];                    // [ HBM] 读 gamma
    }
}

// ============================================================
// RMSNorm — Float4 向量化版本
// ============================================================
template <int BLOCK_SIZE>
__global__ void rmsnorm_float4(
    const float* input, float* output,
    const float* gamma, int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;     // [ HBM]
    float* y = output + row * dim;          // [ HBM]

    float sum_sq = 0.0f;
    int vec_dim = dim / 4;
    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 v = load_float4(x + 4 * i);          // [ HBM] 向量化加载
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float xi = x[i];                            // [ HBM] 标量补丁
        sum_sq += xi * xi;
    }

    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    float inv_rms = rsqrtf(sum_sq / dim + eps);

    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 v = load_float4(x + 4 * i);          // [ HBM] 第二次读取 x
        float4 g = load_float4(gamma + 4 * i);      // [ HBM] 读 gamma
        float4 out;
        out.x = v.x * inv_rms * g.x;
        out.y = v.y * inv_rms * g.y;
        out.z = v.z * inv_rms * g.z;
        out.w = v.w * inv_rms * g.w;
        store_float4(y + 4 * i, out);              // [ HBM] 向量化写入
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = x[i] * inv_rms * gamma[i];           // [ HBM] 标量
    }
}

template __global__ void rmsnorm_kernel<128>(const float*, float*, const float*, int, float);
template __global__ void rmsnorm_kernel<256>(const float*, float*, const float*, int, float);
template __global__ void rmsnorm_kernel<512>(const float*, float*, const float*, int, float);
template __global__ void rmsnorm_float4<128>(const float*, float*, const float*, int, float);
template __global__ void rmsnorm_float4<256>(const float*, float*, const float*, int, float);
