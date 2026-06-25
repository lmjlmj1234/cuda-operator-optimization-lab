#include <cuda_runtime.h>
#include <float.h>
#include "utils.cuh"

// Fused Residual + LayerNorm: float4 vectorized version
// Most advanced variant: fusion + vectorization
template <int BLOCK_SIZE>
__global__ void fused_residual_layernorm_float4(
    const float* input, const float* residual_add,
    float* output, const float* gamma, const float* beta,
    int dim, float eps
) {
    int row = blockIdx.x;
    const float* x = input + row * dim;
    const float* r = residual_add + row * dim;
    float* y = output + row * dim;

    float sum = 0.0f, sum_sq = 0.0f;
    int vec_dim = dim / 4;

    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 vx = load_float4(x + 4 * i);
        float4 vr = load_float4(r + 4 * i);
        float vf0 = vx.x + vr.x;
        float vf1 = vx.y + vr.y;
        float vf2 = vx.z + vr.z;
        float vf3 = vx.w + vr.w;
        sum    += vf0 + vf1 + vf2 + vf3;
        sum_sq += vf0*vf0 + vf1*vf1 + vf2*vf2 + vf3*vf3;
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float fused = x[i] + r[i];
        sum    += fused;
        sum_sq += fused * fused;
    }

    sum    = block_reduce_sum<BLOCK_SIZE>(sum);
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    __syncthreads();

    __shared__ float sm_mean, sm_rstd;
    if (threadIdx.x == 0) {
        sm_mean = sum / dim;
        float var = fmaxf(sum_sq / dim - sm_mean * sm_mean, 0.0f);
        sm_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = sm_mean;
    float rstd = sm_rstd;

    for (int i = threadIdx.x; i < vec_dim; i += BLOCK_SIZE) {
        float4 vx = load_float4(x + 4 * i);
        float4 vr = load_float4(r + 4 * i);
        float4 vg = load_float4(gamma + 4 * i);
        float4 vb = load_float4(beta + 4 * i);
        float4 out;
        out.x = ((vx.x + vr.x) - mean) * rstd * vg.x + vb.x;
        out.y = ((vx.y + vr.y) - mean) * rstd * vg.y + vb.y;
        out.z = ((vx.z + vr.z) - mean) * rstd * vg.z + vb.z;
        out.w = ((vx.w + vr.w) - mean) * rstd * vg.w + vb.w;
        store_float4(y + 4 * i, out);
    }
    for (int i = vec_dim * 4 + threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float fused = x[i] + r[i];
        y[i] = (fused - mean) * rstd * gamma[i] + beta[i];
    }
}

template __global__ void fused_residual_layernorm_float4<128>(const float*, const float*, float*, const float*, const float*, int, float);
template __global__ void fused_residual_layernorm_float4<256>(const float*, const float*, float*, const float*, const float*, int, float);
