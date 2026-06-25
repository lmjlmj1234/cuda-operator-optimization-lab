#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// Softmax: online one-pass algorithm
// Single pass tracks both max and sum simultaneously
// When new max found, retroactively correct sum via exp(old_m - new_m)
template <int BLOCK_SIZE, int VEC_SIZE = 1>
__global__ void softmax_online(const float* input, float* output, int dim) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    float local_m = -FLT_MAX;
    float local_s = 0.0f;

    if constexpr (VEC_SIZE == 4) {
        for (int i = threadIdx.x * 4; i < dim; i += BLOCK_SIZE * 4) {
            float4 v = load_float4(x + i);
            float vals[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float old_m = local_m;
                local_m = fmaxf(local_m, vals[k]);
                local_s = local_s * expf(old_m - local_m) + expf(vals[k] - local_m);
            }
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
            float xi = x[i];
            float old_m = local_m;
            local_m = fmaxf(local_m, xi);
            local_s = local_s * expf(old_m - local_m) + expf(xi - local_m);
        }
    }

    float global_m = block_reduce_max<BLOCK_SIZE>(local_m);
    __syncthreads();

    local_s = local_s * expf(local_m - global_m);
    float global_s = block_reduce_sum<BLOCK_SIZE>(local_s);
    __syncthreads();

    float inv_s = 1.0f / global_s;
    if constexpr (VEC_SIZE == 4) {
        for (int i = threadIdx.x * 4; i < dim; i += BLOCK_SIZE * 4) {
            float4 v = load_float4(x + i);
            float4 out;
            out.x = expf(v.x - global_m) * inv_s;
            out.y = expf(v.y - global_m) * inv_s;
            out.z = expf(v.z - global_m) * inv_s;
            out.w = expf(v.w - global_m) * inv_s;
            store_float4(y + i, out);
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
            y[i] = expf(x[i] - global_m) * inv_s;
        }
    }
}

template __global__ void softmax_online<128, 1>(const float*, float*, int);
template __global__ void softmax_online<256, 1>(const float*, float*, int);
template __global__ void softmax_online<256, 4>(const float*, float*, int);
template __global__ void softmax_online<512, 1>(const float*, float*, int);
