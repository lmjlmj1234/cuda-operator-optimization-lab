#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// LayerNorm: Welford online algorithm
// Numerically stable variance via iterative update
// Uses warp shuffle + shared memory for 2-stage reduction
template <int BLOCK_SIZE>
__global__ void layernorm_welford(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    WelfordStats local = {0.0f, 0.0f, 0};
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        welford_update(local, x[i]);
    }

    // Stage 1: warp shuffle merge
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordStats neighbor;
        neighbor.count = __shfl_xor_sync(0xffffffff, local.count, offset);
        neighbor.mean  = __shfl_xor_sync(0xffffffff, local.mean,  offset);
        neighbor.m2    = __shfl_xor_sync(0xffffffff, local.m2,    offset);
        local = welford_merge(local, neighbor);
    }

    __shared__ WelfordStats warp_stats[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) warp_stats[wid] = local;

    __syncthreads();

    // Stage 2: warp 0 does final merge
    WelfordStats block_stat = {0.0f, 0.0f, 0};
    if (wid == 0) {
        int num_warps = (BLOCK_SIZE + 31) / 32;
        for (int w = threadIdx.x; w < num_warps; w += warpSize) {
            WelfordStats ws = warp_stats[w];
            block_stat = welford_merge(block_stat, ws);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            WelfordStats nb;
            nb.count = __shfl_xor_sync(0xffffffff, block_stat.count, offset);
            nb.mean  = __shfl_xor_sync(0xffffffff, block_stat.mean,  offset);
            nb.m2    = __shfl_xor_sync(0xffffffff, block_stat.m2,    offset);
            block_stat = welford_merge(block_stat, nb);
        }
        if (lane == 0) warp_stats[0] = block_stat;
    }
    __syncthreads();
    block_stat = warp_stats[0];

    float mean  = block_stat.mean;
    float var   = block_stat.m2 / block_stat.count;
    float rstd  = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

template __global__ void layernorm_welford<128>(const float*, float*, const float*, const float*, int, float);
template __global__ void layernorm_welford<256>(const float*, float*, const float*, const float*, int, float);
