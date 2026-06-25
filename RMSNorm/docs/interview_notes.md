# Interview Notes

## Q: Why is RMSNorm preferred over LayerNorm in modern LLMs?

**A:** Two reasons:

1. **Computational savings:** RMSNorm removes the mean computation, saving approximately 25% of the variance computation logic (one fewer reduction pass and one fewer element-wise subtraction).
2. **No quality loss:** Empirical findings show that mean subtraction does not improve model quality for most tasks. The LLaMA paper demonstrated that RMSNorm matches LayerNorm perplexity across a range of model sizes while being faster to compute.

## Q: How much faster is RMSNorm than LayerNorm on GPU?

**A:** At our problem size (D = 8192), the difference is less than 1%. The mean computation is a simple FMA per element in the reduction loop -- it adds approximately 2-3 instructions out of hundreds. The bottleneck is HBM traffic, and both RMSNorm and LayerNorm read and write the same number of bytes from HBM (both read x + gamma, both write y; LayerNorm reads x twice but that is cached in L1/L2).

## Q: When would RMSNorm be meaningfully faster than LayerNorm?

**A:** On small dimensions (D < 512) where reduction overhead dominates, or on compute-bound kernels where every arithmetic operation counts. At D = 128, RMSNorm can be approximately 10% faster than LayerNorm because the reduction overhead is a larger fraction of total runtime, and the arithmetic savings become visible.

## Q: Why doesn't float4 help here when it helps in other contexts?

**A:** Vectorization helps when access is non-coalesced (strided, cross-element patterns). Row-wise reduction is perfectly coalesced -- 32 threads * 4 bytes/thread = 128 bytes (one cache line). Float4 does not change the DRAM transaction pattern; it only reduces the LDG instruction count. The bottleneck remains DRAM bandwidth, not instruction issue throughput, so reducing instruction count has no measurable effect.

## Q: What is the next optimization step for RMSNorm?

**A:** The only way to speed up RMSNorm beyond its current performance is to fuse it with adjacent operators. For example:
- **Fuse with the preceding linear layer** (matmul + RMSNorm): The matmul output would be consumed immediately, avoiding a round-trip through HBM.
- **Fuse with the following element-wise operations** (e.g., SwiGLU or residual add): Combine the normalization write with the next operation's read.

In isolation, the current RMSNorm kernel is already within 15% of the DRAM bandwidth roofline.
