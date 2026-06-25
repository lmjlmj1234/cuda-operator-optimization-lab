# Optimization History

## Summary

RMSNorm is simple enough that the scalar version is already near-optimal for memory-bound kernels. Unlike operators with complex indexing patterns or multiple fused operations, the row-wise reduction leaves little room for algorithmic optimization.

## Version Log

| Version | Technique | Gain | Side Effects |
|---------|-----------|------|--------------|
| v0: rmsnorm_kernel | Scalar baseline (BLOCK=256) | Baseline | -- |
| v1: rmsnorm_float4 | Float4 vectorized loads/stores | No measurable gain | Same pattern as LayerNorm float4 attempt |

## Analysis

### v0 -- Scalar Baseline

The first implementation follows the textbook RMSNorm algorithm:
1. Threads cooperatively load D elements and compute sum_sq via parallel reduction.
2. One thread computes `rsqrtf(sumsq / D + eps)`.
3. Threads cooperatively load x and gamma, compute y = x * rms_denom * gamma, and store.

BLOCK = 256 was chosen as the default because it provides sufficient parallelism (4096 rows / 256 threads = 16 SMs worth of thread blocks) while keeping per-block register pressure low.

### v1 -- Float4 Vectorization

Float4 packed loads (`LDG.128`) were introduced to reduce instruction count by 4x in the inner loops. Unlike strided-access patterns where vectorization materially improves throughput, RMSNorm's row-wise reduction is already perfectly coalesced. The bottleneck is DRAM bandwidth, not instruction issue, so reducing instruction count has no measurable effect on kernel time.

## Key Lesson

Not all vectorization opportunities matter. Float4 helps when (a) the kernel is instruction-bound, or (b) access patterns are non-coalesced and the larger transaction size improves bus utilization. For bandwidth-bound kernels with perfect coalescing, instruction count reduction is irrelevant.
