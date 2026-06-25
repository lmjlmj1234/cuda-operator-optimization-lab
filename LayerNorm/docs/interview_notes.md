# Interview Notes

## Q: Why is LayerNorm memory-bound but the float4 version is slower?

Row-wise reduction is already fully coalesced — each thread loads a contiguous block of elements. Float4 reduces LDG instruction count but the bottleneck is DRAM bandwidth, not instruction issue rate. Worse, float4 increases register pressure (need 4x input + gamma + beta registers), which can reduce occupancy and hide latency less effectively.

## Q: When does catastrophic cancellation happen in sum_sq?

When the mean is much larger than the standard deviation. For example, x = [1000.0, 1000.001]:
sum = 2000.001 -> mean = 1000.0005, sum_sq = 2,000,001, var = 1,000,000.5 - 1,000,001.00000025 = 0.49999975.
Subtracting two ~1e6 numbers to get ~0.5 loses ~7 digits of precision. Welford avoids this entirely.

## Q: What is the fused residual + layernorm optimization?

Instead of writing `x + residual` to an intermediate tensor then reading it back for LN, fuse: load `x` and `residual`, add in registers, normalize, write final output. Eliminates one HBM write+read of NxD floats (256 MB at our problem size). See `FusedResidualLN/docs/README.md`.

## Q: How would you optimize further if this were compute-bound?

Use warp shuffle instead of shared memory for the reduction (reduces latency), use `__expf()` intrinsic instead of `expf()` (lower precision, 2x throughput), preload gamma/beta into registers if dim is small.

## Key Takeaways

1. **Memory access patterns dominate**: For row-wise reductions on large D, bandwidth utilization is the only metric that matters. Instruction count optimization (float4) is irrelevant when the bottleneck is DRAM.
2. **Numerical stability is a correctness concern**: sum_sq's catastrophic cancellation is not a performance issue but can produce wrong gradients during training. Welford should be the default for production.
3. **Fusion is the next frontier**: With all variants hitting ~85% bandwidth, the only way to go faster is to eliminate HBM traffic entirely through operator fusion.
