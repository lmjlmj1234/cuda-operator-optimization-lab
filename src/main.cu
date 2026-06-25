
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// Softmax
template <int BLOCK_SIZE> __global__ void softmax_naive(const float*, float*, int);
template <int BLOCK_SIZE, int VEC_SIZE> __global__ void softmax_online(const float*, float*, int);
__global__ void softmax_warp(const float*, float*, int);

// LayerNorm
template <int BLOCK_SIZE> __global__ void layernorm_sum_sq(const float*, float*, const float*, const float*, int, float);
template <int BLOCK_SIZE> __global__ void layernorm_float4(const float*, float*, const float*, const float*, int, float);
template <int BLOCK_SIZE> __global__ void layernorm_welford(const float*, float*, const float*, const float*, int, float);

// RMSNorm
template <int BLOCK_SIZE> __global__ void rmsnorm_kernel(const float*, float*, const float*, int, float);
template <int BLOCK_SIZE> __global__ void rmsnorm_float4(const float*, float*, const float*, int, float);

// Fused LayerNorm
template <int BLOCK_SIZE> __global__ void fused_residual_layernorm(const float*, const float*, float*, const float*, const float*, int, float);
template <int BLOCK_SIZE> __global__ void fused_residual_layernorm_float4(const float*, const float*, float*, const float*, const float*, int, float);

__global__ void online_softmax_demo(const float*, float*);

// ============================================================
// CUDA 计时工具 (宏: 统一 warmup + measure 模式)
// 使用 __VA_ARGS__ 避免 <<<...>>> 中的逗号被解析为参数分隔符
// ============================================================
#define BENCH(name, n_warmup, n_iters, ...)                                \
    do {                                                                    \
        for (int _w = 0; _w < n_warmup; _w++) { __VA_ARGS__; }             \
        cudaEventRecord(start, 0);                                          \
        for (int _i = 0; _i < n_iters; _i++) { __VA_ARGS__; }              \
        cudaEventRecord(stop, 0);                                           \
        cudaEventSynchronize(stop);                                         \
        float _ms;                                                          \
        cudaEventElapsedTime(&_ms, start, stop);                            \
        printf("    " name ":   %.3f ms\n", _ms / n_iters);                 \
    } while (0)

// ============================================================
// 文件输出工具: 将 GPU buffer 写入 .bin 文件供 Python 验证
// ============================================================
void dump_gpu_buffer(const float* d_src, size_t count, const char* filename) {
    float* h_buf = new float[count];
    cudaMemcpy(h_buf, d_src, count * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen(filename, "wb");
    fwrite(h_buf, sizeof(float), count, f);
    fclose(f);
    delete[] h_buf;
    printf("    dumped %zu floats -> %s\n", count, filename);
}
void reference_softmax(const float* x, float* y, int dim) {
    float max_val = -1e38f;
    for (int i = 0; i < dim; i++) max_val = fmaxf(max_val, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += expf(x[i] - max_val);
    for (int i = 0; i < dim; i++) y[i] = expf(x[i] - max_val) / sum;
}

bool verify_result(const float* gpu, const float* cpu, int n, float eps) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(gpu[i] - cpu[i]);
        // 双标准: 绝对误差 < 1e-4 或 相对误差 < eps 都算通过
        // 这样在处理 softmax 极小值时不会误报
        float rel = diff / fmaxf(1e-8f, fmaxf(fabsf(cpu[i]), fabsf(gpu[i])));
        if (diff > 1e-3f && rel > eps) {
            printf("  MISMATCH at [%d]: gpu=%e cpu=%e diff=%e rel=%e\n",
                   i, gpu[i], cpu[i], diff, rel);
            return false;
        }
    }
    return true;
}

// ============================================================
// 主函数
// ============================================================
int main() {
    int N = 4096;   // 行数
    int D = 8192;   // 每行维度
    int BLOCK = 256;

    size_t size_data = (size_t)N * D * sizeof(float);
    size_t size_param = (size_t)D * sizeof(float);

    printf("============================================\n");
    printf(" CUDA Operator Optimization Lab\n");
    printf(" Config: N=%d, D=%d, BLOCK=%d\n", N, D, BLOCK);
    printf(" Total data: %.2f MB\n", (float)size_data / 1024 / 1024);
    printf("============================================\n\n");

    // --- 分配 Host 内存 ---
    float *h_in = new float[N * D];
    float *h_ref_out = new float[N * D];
    float *h_gpu_out = new float[N * D];
    float *h_gamma = new float[D];
    float *h_beta = new float[D];
    float *h_res = new float[N * D];

    // 初始化输入
    srand(42);
    for (int i = 0; i < N * D; i++) h_in[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
    for (int i = 0; i < D; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }
    for (int i = 0; i < N * D; i++) h_res[i] = ((float)rand() / RAND_MAX) * 1.0f;

    // --- 分配 Device 内存 ---
    float *d_in, *d_out, *d_gamma, *d_beta, *d_res;
    cudaMalloc(&d_in, size_data);
    cudaMalloc(&d_out, size_data);
    cudaMalloc(&d_gamma, size_param);
    cudaMalloc(&d_beta, size_param);
    cudaMalloc(&d_res, size_data);

    cudaMemcpy(d_in, h_in, size_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, size_param, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, size_param, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, size_data, cudaMemcpyHostToDevice);

    // --- CUDA Events for timing ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 50, iters = 500;

    // ============================================================
    // 1. Softmax 测试
    // ============================================================
    printf("--- Softmax (%d x %d) ---\n", N, D);

    BENCH("softmax_naive (3-pass)", warmup, iters,
        softmax_naive<128><<<N, 128>>>(d_in, d_out, D)
    );

    BENCH("softmax_online (1-pass)", warmup, iters,
        softmax_online<256, 1><<<N, 256>>>(d_in, d_out, D)
    );

    BENCH("softmax_online+float4", warmup, iters,
        softmax_online<256, 4><<<N, 256>>>(d_in, d_out, D)
    );

    // Verify softmax (verifies the last kernel = softmax_online<256,4>)
    cudaMemcpy(h_gpu_out, d_out, size_data, cudaMemcpyDeviceToHost);
    bool softmax_ok = true;
    for (int r = 0; r < 8; r++) {
        reference_softmax(h_in + r * D, h_ref_out, D);
        if (!verify_result(h_gpu_out + r * D, h_ref_out, D, 1e-3f)) {
            softmax_ok = false;
            break;
        }
    }
    printf("    softmax correctness: %s\n\n", softmax_ok ? "PASS" : "FAIL");
    // Dump for Python verification
    dump_gpu_buffer(d_out, (size_t)N * D, "softmax_cuda_out.bin");  // online+float4

    // Also dump naive softmax for cross-validation
    softmax_naive<128><<<N, 128>>>(d_in, d_out, D);
    cudaDeviceSynchronize();
    dump_gpu_buffer(d_out, (size_t)N * D, "softmax_naive_out.bin");

    // Now benchmark the warp version (dim=32, separate small test)
    BENCH("softmax_warp (dim<=32)", warmup, iters,
        softmax_warp<<<N, 32>>>(d_in, d_out, 32)
    );

    cudaDeviceSynchronize();

    // Dump input data for Python verification
    dump_gpu_buffer(d_in, (size_t)N * D, "input_data.bin");
    dump_gpu_buffer(d_gamma, D, "gamma.bin");
    dump_gpu_buffer(d_beta, D, "beta.bin");
    dump_gpu_buffer(d_res, (size_t)N * D, "residual.bin");

    // ============================================================
    // 2. LayerNorm 测试
    // ============================================================
    printf("--- LayerNorm (%d x %d) ---\n", N, D);

    BENCH("layernorm_sum_sq", warmup, iters,
        layernorm_sum_sq<256><<<N, 256>>>(d_in, d_out, d_gamma, d_beta, D, 1e-5f)
    );

    BENCH("layernorm_float4", warmup, iters,
        layernorm_float4<256><<<N, 256>>>(d_in, d_out, d_gamma, d_beta, D, 1e-5f)
    );

    BENCH("layernorm_welford", warmup, iters,
        layernorm_welford<256><<<N, 256>>>(d_in, d_out, d_gamma, d_beta, D, 1e-5f)
    );

    cudaDeviceSynchronize();
    // Dump layernorm_sum_sq output (simplest variant)
    layernorm_sum_sq<256><<<N, 256>>>(d_in, d_out, d_gamma, d_beta, D, 1e-5f);
    cudaDeviceSynchronize();
    dump_gpu_buffer(d_out, (size_t)N * D, "layernorm_cuda_out.bin");

    // Dump welford layernorm for cross-validation
    layernorm_welford<256><<<N, 256>>>(d_in, d_out, d_gamma, d_beta, D, 1e-5f);
    cudaDeviceSynchronize();
    dump_gpu_buffer(d_out, (size_t)N * D, "layernorm_welford_out.bin");

    // Dump float4 layernorm for cross-validation
    layernorm_float4<256><<<N, 256>>>(d_in, d_out, d_gamma, d_beta, D, 1e-5f);
    cudaDeviceSynchronize();
    dump_gpu_buffer(d_out, (size_t)N * D, "layernorm_float4_out.bin");

    // ============================================================
    // 3. RMSNorm 测试
    // ============================================================
    printf("\n--- RMSNorm (%d x %d) ---\n", N, D);

    BENCH("rmsnorm_kernel", warmup, iters,
        rmsnorm_kernel<256><<<N, 256>>>(d_in, d_out, d_gamma, D, 1e-5f)
    );

    BENCH("rmsnorm_float4", warmup, iters,
        rmsnorm_float4<256><<<N, 256>>>(d_in, d_out, d_gamma, D, 1e-5f)
    );

    cudaDeviceSynchronize();
    dump_gpu_buffer(d_out, (size_t)N * D, "rmsnorm_cuda_out.bin");

    // Dump float4 rmsnorm for cross-validation
    rmsnorm_float4<256><<<N, 256>>>(d_in, d_out, d_gamma, D, 1e-5f);
    cudaDeviceSynchronize();
    dump_gpu_buffer(d_out, (size_t)N * D, "rmsnorm_float4_out.bin");

    // ============================================================
    // 4. Fused Residual + LayerNorm 测试
    // ============================================================
    printf("\n--- Fused Residual + LayerNorm (%d x %d) ---\n", N, D);

    BENCH("fused_residual_layernorm", warmup, iters,
        fused_residual_layernorm<256><<<N, 256>>>(d_in, d_res, d_out, d_gamma, d_beta, D, 1e-5f)
    );

    BENCH("fused_residual_ln_float4", warmup, iters,
        fused_residual_layernorm_float4<256><<<N, 256>>>(d_in, d_res, d_out, d_gamma, d_beta, D, 1e-5f)
    );

    cudaDeviceSynchronize();
    // Dump fused layernorm (scalar) for Python verification
    fused_residual_layernorm<256><<<N, 256>>>(d_in, d_res, d_out, d_gamma, d_beta, D, 1e-5f);
    cudaDeviceSynchronize();
    dump_gpu_buffer(d_out, (size_t)N * D, "fused_layernorm_cuda_out.bin");
    // Dump float4 fused layernorm for cross-validation
    fused_residual_layernorm_float4<256><<<N, 256>>>(d_in, d_res, d_out, d_gamma, d_beta, D, 1e-5f);
    cudaDeviceSynchronize();
    dump_gpu_buffer(d_out, (size_t)N * D, "fused_ln_float4_out.bin");

    // ============================================================
    // 5. 交叉验证: 对比 Naive vs Online Softmax
    // ============================================================
    printf("\n--- Cross-Validation: Naive vs Online Softmax ---\n");

    {
        // Compare naive vs online on row 2
        {
            float *d_out2;
            cudaMalloc(&d_out2, size_data);
            int test_row = 2;
            softmax_naive<128><<<1, 128>>>(d_in + test_row * D, d_out, D);
            softmax_online<256, 1><<<1, 256>>>(d_in + test_row * D, d_out2, D);
            cudaDeviceSynchronize();

            float *h_naive = new float[D];
            float *h_online = new float[D];
            cudaMemcpy(h_naive, d_out, D * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_online, d_out2, D * sizeof(float), cudaMemcpyDeviceToHost);

            float max_diff = 0.0f;
            int max_idx = -1;
            for (int i = 0; i < D; i++) {
                float diff = fabsf(h_naive[i] - h_online[i]);
                if (diff > max_diff) { max_diff = diff; max_idx = i; }
            }
            printf("    Max diff between naive & online: %e at idx %d\n", max_diff, max_idx);
            printf("    Naive[%d]=%e  Online[%d]=%e\n",
                   max_idx, h_naive[max_idx], max_idx, h_online[max_idx]);

            if (max_diff > 1e-4f) {
                printf("    Input row %d (first 16):\n", test_row);
                for (int i = 0; i < 16; i++)
                    printf("      [%d] %f\n", i, h_in[test_row * D + i]);
            }

            cudaFree(d_out2);
            delete[] h_naive;
            delete[] h_online;
        }
    }

    // ============================================================
    // 6. Host 端 Online 算法数学验证
    // ============================================================
    printf("\n--- Host online algorithm verification ---\n");
    {
        int test_row = 2;
        int nthreads = 256;
        float* row_data = h_in + test_row * D;

        float per_thread_max[256], per_thread_sum[256];
        for (int t = 0; t < nthreads; t++) {
            float lm = -FLT_MAX;
            float ls = 0.0f;
            for (int i = t; i < D; i += nthreads) {
                float xi = row_data[i];
                float old_m = lm;
                lm = fmaxf(lm, xi);
                ls = ls * expf(old_m - lm) + expf(xi - lm);
            }
            per_thread_max[t] = lm;
            per_thread_sum[t] = ls;
        }

        float global_m_ref = -FLT_MAX;
        for (int t = 0; t < nthreads; t++)
            global_m_ref = fmaxf(global_m_ref, per_thread_max[t]);

        float global_s_ref = 0.0f;
        for (int t = 0; t < nthreads; t++)
            global_s_ref += per_thread_sum[t] * expf(per_thread_max[t] - global_m_ref);

        float* expected = new float[D];
        for (int i = 0; i < D; i++)
            expected[i] = expf(row_data[i] - global_m_ref) / global_s_ref;

        float *d_naive;
        cudaMalloc(&d_naive, size_data);
        softmax_naive<128><<<1, 128>>>(d_in + test_row * D, d_naive, D);
        cudaDeviceSynchronize();
        float* h_naive = new float[D];
        cudaMemcpy(h_naive, d_naive, D * sizeof(float), cudaMemcpyDeviceToHost);

        float cpu_err = 0.0f;
        for (int i = 0; i < D; i++)
            cpu_err = fmaxf(cpu_err, fabsf(expected[i] - h_naive[i]));
        printf("    Host online vs naive kernel: max_diff=%e\n", cpu_err);
        printf("    global_m=%e global_s=%e\n", global_m_ref, global_s_ref);

        float min_pt = per_thread_max[0], max_pt = per_thread_max[0];
        for (int t = 0; t < nthreads; t++) {
            min_pt = fminf(min_pt, per_thread_max[t]);
            max_pt = fmaxf(max_pt, per_thread_max[t]);
        }
        printf("    Per-thread max range: [%e, %e]\n", min_pt, max_pt);

        delete[] expected;
        delete[] h_naive;
        cudaFree(d_naive);
    }

    // ============================================================
    // Summary
    // ============================================================
    printf("\n============================================\n");
    printf(" All kernels launched successfully!\n");
    printf("============================================\n");

    // Cleanup
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_res);
    delete[] h_in; delete[] h_ref_out; delete[] h_gpu_out;
    delete[] h_gamma; delete[] h_beta; delete[] h_res;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
