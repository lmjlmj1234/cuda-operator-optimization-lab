#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// Softmax: warp-level (dim <= 32)
// No shared memory needed -- all communication via warp shuffle
__global__ void softmax_warp(const float* input, float* output, int dim) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    float m_val = -FLT_MAX;
    float s_val = 0.0f;

    for (int i = threadIdx.x; i < dim; i += warpSize) {
        float xi = x[i];
        float old_m = m_val;
        m_val = fmaxf(m_val, xi);
        s_val = s_val * expf(old_m - m_val) + expf(xi - m_val);
    }

    float global_m = warp_reduce_max(m_val);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += warpSize) {
        sum += expf(x[i] - global_m);
    }
    sum = warp_reduce_sum(sum);

    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < dim; i += warpSize) {
        y[i] = expf(x[i] - global_m) * inv_sum;
    }
}
