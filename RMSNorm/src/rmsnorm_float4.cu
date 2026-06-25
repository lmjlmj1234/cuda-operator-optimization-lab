#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// RMSNorm: float4 vectorized version
template <int BLOCK_SIZE>
__global__ void rmsnorm_float4(
    const float* input, float* output,
    const float* gamma, int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    float sum_sq = 0.0f;
    int vec_dim = dim / 4;
    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 v = load_float4(x + 4 * i);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float xi = x[i];
        sum_sq += xi * xi;
    }

    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    float inv_rms = rsqrtf(sum_sq / dim + eps);

    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 v = load_float4(x + 4 * i);
        float4 g = load_float4(gamma + 4 * i);
        float4 out;
        out.x = v.x * inv_rms * g.x;
        out.y = v.y * inv_rms * g.y;
        out.z = v.z * inv_rms * g.z;
        out.w = v.w * inv_rms * g.w;
        store_float4(y + 4 * i, out);
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

template __global__ void rmsnorm_float4<128>(const float*, float*, const float*, int, float);
template __global__ void rmsnorm_float4<256>(const float*, float*, const float*, int, float);
