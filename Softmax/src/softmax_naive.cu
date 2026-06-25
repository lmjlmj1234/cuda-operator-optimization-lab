#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// Softmax: naive 3-pass implementation (baseline)
// Pass 1: find row max
// Pass 2: exp(x-max) and sum
// Pass 3: normalize
template <int BLOCK_SIZE>
__global__ void softmax_naive(const float* input, float* output, int dim) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float v = x[i];
        max_val = fmaxf(max_val, v);
    }
    max_val = block_reduce_max<BLOCK_SIZE>(max_val);
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float e = expf(x[i] - max_val);
        y[i] = e;
        sum += e;
    }
    sum = block_reduce_sum<BLOCK_SIZE>(sum);
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = y[i] / sum;
    }
}

template __global__ void softmax_naive<128>(const float*, float*, int);
template __global__ void softmax_naive<256>(const float*, float*, int);
template __global__ void softmax_naive<512>(const float*, float*, int);
