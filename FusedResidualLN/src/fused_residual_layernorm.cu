#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// Fused Residual + LayerNorm: scalar version
// Fuses: y = LN(x + residual)
// Eliminates intermediate tensor write/read
template <int BLOCK_SIZE>
__global__ void fused_residual_layernorm(
    const float* input, const float* residual_add,
    float* output, const float* gamma, const float* beta,
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    const float* r = residual_add + row * dim;
    float* y = output + row * dim;

    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float fused = x[i] + r[i];
        sum    += fused;
        sum_sq += fused * fused;
    }

    sum    = block_reduce_sum<BLOCK_SIZE>(sum);
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    __shared__ float sm_mean, sm_rstd;
    if (threadIdx.x == 0) {
        sm_mean = sum / dim;
        float var = sum_sq / dim - sm_mean * sm_mean;
        sm_rstd = rsqrtf(fmaxf(var, 0.0f) + eps);
    }
    __syncthreads();
    float mean = sm_mean;
    float rstd = sm_rstd;

    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float fused = x[i] + r[i];
        y[i] = (fused - mean) * rstd * gamma[i] + beta[i];
    }
}

template __global__ void fused_residual_layernorm<128>(const float*, const float*, float*, const float*, const float*, int, float);
template __global__ void fused_residual_layernorm<256>(const float*, const float*, float*, const float*, const float*, int, float);
