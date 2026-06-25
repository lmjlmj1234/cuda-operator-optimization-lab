#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// RMSNorm: scalar version
// y_i = x_i / sqrt( 1/N * sum(x_j^2) + eps ) * gamma_i
// No mean subtraction, no beta parameter
template <int BLOCK_SIZE>
__global__ void rmsnorm_kernel(
    const float* input, float* output,
    const float* gamma, int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float xi = x[i];
        sum_sq += xi * xi;
    }
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    float inv_rms = rsqrtf(sum_sq / dim + eps);

    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

template __global__ void rmsnorm_kernel<128>(const float*, float*, const float*, int, float);
template __global__ void rmsnorm_kernel<256>(const float*, float*, const float*, int, float);
template __global__ void rmsnorm_kernel<512>(const float*, float*, const float*, int, float);
