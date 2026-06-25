#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// LayerNorm: float4 vectorized version
// Uses float4 loads to reduce LDG instruction count by 4x
template <int BLOCK_SIZE>
__global__ void layernorm_float4(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    float sum = 0.0f, sum_sq = 0.0f;
    int vec_dim = dim / 4;
    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 v = load_float4(x + 4 * i);
        sum     += v.x + v.y + v.z + v.w;
        sum_sq  += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float xi = x[i];
        sum    += xi;
        sum_sq += xi * xi;
    }

    sum    = block_reduce_sum<BLOCK_SIZE>(sum);
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    float mean   = sum / dim;
    float var    = fmaxf(sum_sq / dim - mean * mean, 0.0f);
    float rstd   = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 v = load_float4(x + 4 * i);
        float4 g = load_float4(gamma + 4 * i);
        float4 b = load_float4(beta + 4 * i);
        float4 out;
        out.x = (v.x - mean) * rstd * g.x + b.x;
        out.y = (v.y - mean) * rstd * g.y + b.y;
        out.z = (v.z - mean) * rstd * g.z + b.z;
        out.w = (v.w - mean) * rstd * g.w + b.w;
        store_float4(y + 4 * i, out);
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

template __global__ void layernorm_float4<128>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_float4<256>(const float*, float*, const float*, const float*, int, float);
