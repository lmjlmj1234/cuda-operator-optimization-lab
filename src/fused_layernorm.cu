#include "utils.cuh"

// ============================================================
// Fused LayerNorm — 融合残差连接 + Layer Normalization
//
//   [ FUSED] 融合了: Residual Add + LayerNorm
//
// 标准流水线:
//   residual = x + sub_layer(x)    // 残差连接 (HBM 读+写)
//   y = LayerNorm(residual)         // 归一化 (HBM 读+写)
//   总计: 4 次 HBM 遍历
//
// 融合后:
//   y = LayerNorm(x + sub_layer(x)) // 一次性完成
//   总计: 2 次 HBM 遍历 (读 x, 读 sub, 写 y)
//   节省: 一次残差结果的 HBM 写 + 一次 LayerNorm 输入的 HBM 读
//
// 在 Transformer 中，每层都有残差+LN，
// 融合后可以节省约 33% 的 HBM 带宽
// ============================================================
template <int BLOCK_SIZE>
__global__ void fused_residual_layernorm(
    const float* input,         // [ HBM] x: 主路径输入
    const float* residual_add,  // [ HBM] sub_layer(x): 子层输出 (与 x 相加)
    float* output,              // [ HBM] y: 归一化输出
    const float* gamma,         // [ HBM] 仿射参数
    const float* beta,          // [ HBM] 仿射参数
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;             // [ HBM]
    const float* r = residual_add + row * dim;      // [ HBM]
    float* y = output + row * dim;                  // [ HBM]

    // [ FUSED] 融合残差: 读取 x 和 residual 的同时做加法
    // 不需要中间存储

    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float fused = x[i] + r[i];                  // [ HBM] 读 x + [ HBM] 读 r, 融合加法
        sum    += fused;
        sum_sq += fused * fused;
    }

    sum    = block_reduce_sum<BLOCK_SIZE>(sum);
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    // 通过 shared memory 传递 mean/rstd 避免编译器寄存器别名错误
    __shared__ float sm_mean, sm_rstd;
    if (threadIdx.x == 0) {
        sm_mean = sum / dim;
        float var = sum_sq / dim - sm_mean * sm_mean;
        sm_rstd = rsqrtf(fmaxf(var, 0.0f) + eps);
    }
    __syncthreads();
    float mean = sm_mean;
    float rstd = sm_rstd;

    // 再次读取并融合写出
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float fused = x[i] + r[i];                  // [ HBM] 再次读 x 和 r
        y[i] = (fused - mean) * rstd                // [ HBM] 写 y
             * gamma[i] + beta[i];                  // [ HBM] 读 gamma, beta
    }
}

// ============================================================
// Fused LayerNorm — Float4 向量化 + 残差融合
//
// 最极致版本: 同时使用 Float4 和残差融合
// ============================================================
template <int BLOCK_SIZE>
__global__ void fused_residual_layernorm_float4(
    const float* input, const float* residual_add,
    float* output, const float* gamma, const float* beta,
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    const float* r = residual_add + row * dim;
    float* y = output + row * dim;

    // [ FUSED] 融合: Residual Add + Float4 Load + LayerNorm

    float sum = 0.0f, sum_sq = 0.0f;
    int vec_dim = dim / 4;

    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 vx = load_float4(x + 4 * i);                 // [ HBM] x 向量化
        float4 vr = load_float4(r + 4 * i);                 // [ HBM] r 向量化
        float vf0 = vx.x + vr.x;
        float vf1 = vx.y + vr.y;
        float vf2 = vx.z + vr.z;
        float vf3 = vx.w + vr.w;
        sum    += vf0 + vf1 + vf2 + vf3;
        sum_sq += vf0*vf0 + vf1*vf1 + vf2*vf2 + vf3*vf3;
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float fused = x[i] + r[i];
        sum    += fused;
        sum_sq += fused * fused;
    }

    sum    = block_reduce_sum<BLOCK_SIZE>(sum);
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    // 通过 shared memory 传递 mean/rstd 避免编译器寄存器别名错误
    __shared__ float sm_mean, sm_rstd;
    if (threadIdx.x == 0) {
        sm_mean = sum / dim;
        float var = fmaxf(sum_sq / dim - sm_mean * sm_mean, 0.0f);
        sm_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = sm_mean;
    float rstd = sm_rstd;

    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 vx = load_float4(x + 4 * i);                 // [ HBM]
        float4 vr = load_float4(r + 4 * i);                 // [ HBM]
        float4 vg = load_float4(gamma + 4 * i);             // [ HBM]
        float4 vb = load_float4(beta + 4 * i);              // [ HBM]
        float4 out;
        out.x = ((vx.x + vr.x) - mean) * rstd * vg.x + vb.x;
        out.y = ((vx.y + vr.y) - mean) * rstd * vg.y + vb.y;
        out.z = ((vx.z + vr.z) - mean) * rstd * vg.z + vb.z;
        out.w = ((vx.w + vr.w) - mean) * rstd * vg.w + vb.w;
        store_float4(y + 4 * i, out);                       // [ HBM]
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float fused = x[i] + r[i];
        y[i] = (fused - mean) * rstd * gamma[i] + beta[i];  // [ HBM]
    }
}

template __global__ void fused_residual_layernorm<128>(const float*, const float*, float*, const float*, const float*, int, float);
template __global__ void fused_residual_layernorm<256>(const float*, const float*, float*, const float*, const float*, int, float);
template __global__ void fused_residual_layernorm_float4<128>(const float*, const float*, float*, const float*, const float*, int, float);
template __global__ void fused_residual_layernorm_float4<256>(const float*, const float*, float*, const float*, const float*, int, float);
