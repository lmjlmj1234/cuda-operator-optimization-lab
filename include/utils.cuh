#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

// [ HBM]       -- Global Memory 读写
// [YELLOW SRAM] -- Shared Memory 读写
// [ FUSED]     -- 算子融合标注

// Float4 vectorized load/store
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return reinterpret_cast<const float4*>(ptr)[0];
}
__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    reinterpret_cast<float4*>(ptr)[0] = val;
}

// Warp-level reduce
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Block-level reduce (shared memory tree)
template <int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    return sdata[0];
}

template <int BLOCK_SIZE>
__device__ float block_reduce_max(float val) {
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    return sdata[0];
}

// Welford online statistics
struct WelfordStats {
    float mean;
    float m2;
    int count;
};

__device__ __forceinline__ void welford_update(WelfordStats& s, float x) {
    s.count++;
    float delta = x - s.mean;
    s.mean += delta / s.count;
    float delta2 = x - s.mean;
    s.m2 += delta * delta2;
}

__device__ __forceinline__ WelfordStats welford_merge(WelfordStats a, WelfordStats b) {
    if (b.count == 0) return a;
    if (a.count == 0) return b;
    int count = a.count + b.count;
    float delta = a.mean - b.mean;
    float mean = (a.count * a.mean + b.count * b.mean) / count;
    float m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / count;
    return {mean, m2, count};
}

#endif // UTILS_CUH
