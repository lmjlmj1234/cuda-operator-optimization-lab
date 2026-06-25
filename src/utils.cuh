#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

// ============================================================
// 显存/共享内存读写标记 (仅供代码阅读标注，编译时被移除)
// ============================================================
// [ HBM]       — Global Memory 读写
// [🟡 SRAM]   — Shared Memory 读写
// [ FUSED]     — 算子融合标注

// ============================================================
// Float4 向量化加载/存储工具
// ============================================================
// 确保地址是 16 字节对齐
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return reinterpret_cast<const float4*>(ptr)[0];  // [ HBM]
}
__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    reinterpret_cast<float4*>(ptr)[0] = val;         // [ HBM]
}

// ============================================================
// Warp 级 Reduce: Sum / Max (使用 __shfl_down_sync)
// ============================================================
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

// ============================================================
// Block 级 Reduce (不使用 warp shuffle, 纯 Shared Memory)
// ============================================================
template <int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    __shared__ float sdata[BLOCK_SIZE];              // [🟡 SRAM]
    sdata[threadIdx.x] = val;                        // [🟡 SRAM]
    __syncthreads();
    // Tree reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];  // [🟡 SRAM]
        }
        __syncthreads();
    }
    return sdata[0];                                 // [🟡 SRAM]
}

template <int BLOCK_SIZE>
__device__ float block_reduce_max(float val) {
    __shared__ float sdata[BLOCK_SIZE];              // [🟡 SRAM]
    sdata[threadIdx.x] = val;                        // [🟡 SRAM]
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]); // [🟡 SRAM]
        }
        __syncthreads();
    }
    return sdata[0];                                 // [🟡 SRAM]
}

// ============================================================
// Welford 在线统计合并 (用于 LayerNorm / RMSNorm)
// ============================================================
struct WelfordStats {
    float mean;
    float m2;    // 二阶矩之和: sum((x - mean)^2)
    int count;
};

// 单个线程的 Welford 在线更新
__device__ __forceinline__ void welford_update(WelfordStats& s, float x) {
    s.count++;
    float delta = x - s.mean;
    s.mean += delta / s.count;
    float delta2 = x - s.mean;
    s.m2 += delta * delta2;
}

// 合并两组 Welford 统计量 (并行归约核心)
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
