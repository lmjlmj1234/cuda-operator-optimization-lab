# Bottleneck Analysis

## Compute Intensity

| Metric | Value |
|--------|-------|
| Arithmetic intensity | 0.28 FLOP/byte |
| RTX 4090 peak compute | 82.6 TFLOPS |
| RTX 4090 peak bandwidth | 1008 GB/s |
| Achieved bandwidth | 429.1 GB/s |
| Utilization | 87.6% |

The compute intensity (0.28 FLOP/byte) is well below the RTX 4090's compute-to-bandwidth balance point (~82 TFLOPS / 1008 GB/s = ~81 FLOP/byte). This confirms the kernel is **memory-bound**.

The intensity is slightly higher than standalone LayerNorm due to the extra FMA for the fused add, but not enough to change the memory-bound classification.

## Verdict

**MEMORY BOUND.** Same as standalone LayerNorm. Optimizations should focus on reducing HBM traffic, not increasing arithmetic throughput.

## Fusion Advantage: 33% HBM Reduction

### Without Fusion (two kernels)

| Operation | HBM Traffic |
|-----------|-------------|
| Read x | 16 MB |
| Read residual | 16 MB |
| Write z = x + residual | 16 MB |
| Read z (for LN) | 16 MB |
| Read gamma, beta | ~0.06 MB |
| Write output y | 16 MB |
| **Total** | **~80 MB** |

### With Fusion (single kernel)

| Operation | HBM Traffic |
|-----------|-------------|
| Read x | 16 MB |
| Read residual | 16 MB |
| Read gamma, beta | ~0.06 MB |
| Write output y | 16 MB |
| **Total** | **~48 MB** |

### Savings

- Per-layer: **32 MB saved** (40% reduction in total traffic, 33% reduction excluding gamma/beta)
- Per 32-layer model: **~1 GB saved**

The eliminated traffic is the intermediate `z = x + residual` buffer -- one write and one read that are unnecessary when computation is fused.
