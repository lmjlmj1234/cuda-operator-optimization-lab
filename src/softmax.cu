#include "utils.cuh"

// ============================================================
// Softmax — 版本 1: 朴素 3-Pass 实现 (基线对比用)
//
// Pass 1: 找行最大值    — 1× HBM 读
// Pass 2: exp(x-m) 求和 — 1× HBM 读 + 1× HBM 写 (中间结果)
// Pass 3: 归一化        — 1× HBM 读 + 1× HBM 写
// 总计: 3× HBM 读 + 2× HBM 写 = 5 次 HBM 访问
// ============================================================
template <int BLOCK_SIZE>
__global__ void softmax_naive(const float* input, float* output, int dim) {
    int row = blockIdx.x;
    const float* x = input + row * dim;     // [ HBM]
    float* y = output + row * dim;          // [ HBM]

    // --- Pass 1: 找最大值 ---
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float v = x[i];                     // [ HBM] 读取
        max_val = fmaxf(max_val, v);
    }
    max_val = block_reduce_max<BLOCK_SIZE>(max_val);
    __syncthreads();

    // --- Pass 2: exp 求和 ---
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float e = expf(x[i] - max_val);     // [ HBM] 读取
        y[i] = e;                           // [ HBM] 写入中间结果
        sum += e;
    }
    sum = block_reduce_sum<BLOCK_SIZE>(sum);
    __syncthreads();

    // --- Pass 3: 归一化 ---
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y[i] = y[i] / sum;                  // [ HBM] 读 + [ HBM] 写
    }
}

// ============================================================
// Softmax — 版本 2: 在线单遍算法 (Online One-Pass Softmax)
//
// 核心洞察: 在线追踪 max 和 sum, 避免一次 HBM 读取
// 对每个局部 x_i:
//   m_new = max(m_old, x_i)
//   s_new = s_old * exp(m_old - m_new) + exp(x_i - m_new)
//
// 流程:
//   1. 每个线程在线扫描自己负责的元素 → (m_t, s_t)  [1x HBM 读]
//   2. Block Reduce Max → global_m
//   3. 每个线程修正自己的 sum: s_t *= exp(m_t - global_m)  [纯寄存器]
//   4. Block Reduce Sum → global_s
//   5. 写出 y[i] = exp(x[i] - global_m) / global_s    [1x HBM 读 + 1x HBM 写]
//
// HBM 访问: 2x 读 + 1x 写 = 3 次
// 对比朴素: 3x 读 + 2x 写 = 5 次, 节省 40%
// 对比传统: 去掉了一次独立的"找最大值"遍历
// ============================================================
template <int BLOCK_SIZE, int VEC_SIZE = 1>
__global__ void softmax_online(const float* input, float* output, int dim) {
    int row = blockIdx.x;
    const float* x = input + row * dim;     // [ HBM]
    float* y = output + row * dim;          // [ HBM]

    // ---------------------------------------------------------
    // 第一遍: 在线扫描同时追踪 max 和 sum
    // 每个线程处理自己的子集, 维护 (m_t, s_t)
    // ---------------------------------------------------------
    float local_m = -FLT_MAX;
    float local_s = 0.0f;

    if constexpr (VEC_SIZE == 4) {
        for (int i = threadIdx.x * 4; i < dim; i += BLOCK_SIZE * 4) {
            float4 v = load_float4(x + i);                  // [ HBM]
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
            float xi = x[i];                                // [ HBM]
            float old_m = local_m;
            local_m = fmaxf(local_m, xi);
            local_s = local_s * expf(old_m - local_m) + expf(xi - local_m);
        }
    }

    // ---------------------------------------------------------
    // 归约: 跨线程合并 (m_t, s_t)
    //   global_m = max_t(m_t)
    //   每个线程修正: s_corrected = s_t * exp(m_t - global_m)
    //   global_s = sum_t(s_corrected)
    // ---------------------------------------------------------
    float global_m = block_reduce_max<BLOCK_SIZE>(local_m);
    __syncthreads();

    // 修正: s_t 是用 local_m 计算的, 需要缩放到 global_m
    local_s = local_s * expf(local_m - global_m);
    float global_s = block_reduce_sum<BLOCK_SIZE>(local_s);
    __syncthreads();

    // ---------------------------------------------------------
    // 第二遍: 写出 softmax 结果 (HBM 读 + HBM 写)
    // ---------------------------------------------------------
    float inv_s = 1.0f / global_s;
    if constexpr (VEC_SIZE == 4) {
        for (int i = threadIdx.x * 4; i < dim; i += BLOCK_SIZE * 4) {
            float4 v = load_float4(x + i);                  // [ HBM] 读
            float4 out;
            out.x = expf(v.x - global_m) * inv_s;
            out.y = expf(v.y - global_m) * inv_s;
            out.z = expf(v.z - global_m) * inv_s;
            out.w = expf(v.w - global_m) * inv_s;
            store_float4(y + i, out);                       // [ HBM] 写
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
            y[i] = expf(x[i] - global_m) * inv_s;           // [ HBM] 读 + [ HBM] 写
        }
    }
}

// ============================================================
// Softmax — 版本 3: 纯 Warp 级 One-Pass (无 Shared Memory)
//
// 当整个行可以放入一个 warp (dim <= 32) 时，
// 完全不需要 shared memory，所有通信通过 warp shuffle 完成
// ============================================================
__global__ void softmax_warp(const float* input, float* output, int dim) {
    int row = blockIdx.x;
    const float* x = input + row * dim;     // [ HBM]
    float* y = output + row * dim;          // [ HBM]

    float m_val = -FLT_MAX;
    float s_val = 0.0f;

    // 在线算法, 每个线程处理 1 个元素
    for (int i = threadIdx.x; i < dim; i += warpSize) {
        float xi = x[i];                    // [ HBM]
        float old_m = m_val;
        m_val = fmaxf(m_val, xi);
        s_val = s_val * expf(old_m - m_val) + expf(xi - m_val);
    }

    // Warp 级归约合并 (m, s)
    // 注意: (m, s) 是成对非线性量，我们用两步:
    // 1. Warp max 得到全局 m
    float global_m = warp_reduce_max(m_val);

    // 2. 各线程用 global_m 重新算局部 sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += warpSize) {
        sum += expf(x[i] - global_m);       // [ HBM]
    }
    sum = warp_reduce_sum(sum);

    // 写出
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < dim; i += warpSize) {
        y[i] = expf(x[i] - global_m) * inv_sum;   // [ HBM]
    }
}

// ============================================================
// 显式实例化: 编译器为这些模板参数生成代码
// ============================================================
template __global__ void softmax_naive<128>(const float*, float*, int);
template __global__ void softmax_naive<256>(const float*, float*, int);
template __global__ void softmax_naive<512>(const float*, float*, int);

template __global__ void softmax_online<128, 1>(const float*, float*, int);
template __global__ void softmax_online<256, 1>(const float*, float*, int);
template __global__ void softmax_online<256, 4>(const float*, float*, int);
template __global__ void softmax_online<512, 1>(const float*, float*, int);
