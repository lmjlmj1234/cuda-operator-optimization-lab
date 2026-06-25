#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// LayerNorm: classic sum/sum_sq method
// mean = 1/N * sum(x_i)
// var  = 1/N * sum(x_i^2) - mean^2
// y_i  = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
template <int BLOCK_SIZE>
__global__ void layernorm_sum_sq(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float xi = x[i];
        sum    += xi;
        sum_sq += xi * xi;
    }

    sum    = block_reduce_sum<BLOCK_SIZE>(sum);
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    float mean   = sum / dim;
    float var    = sum_sq / dim - mean * mean;
    float rstd   = rsqrtf(fmaxf(var, 0.0f) + eps);

    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

template __global__ void layernorm_sum_sq<128>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_sum_sq<256>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_sum_sq<512>(const float*, float*, const float*, const float*, int, float);
