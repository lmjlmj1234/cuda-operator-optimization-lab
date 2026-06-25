# LayerNorm

## Module Overview

Layer Normalization normalizes activations across the hidden dimension: `LN(x)_i = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i`. It is applied after every attention and feed-forward sub-layer in the standard Transformer architecture, stabilizing training by reducing internal covariate shift. Unlike BatchNorm, LayerNorm computes per-sample statistics (mean, variance) across the feature dimension, making it independent of batch size and suitable for online inference with batch size 1.

Three kernel variants are implemented in this module. The baseline **sum_sq** variant uses the classic two-moment method (E[X] and E[X^2]) in a single fused loop with a shared-memory tree reduction. The **float4** variant applies 128-bit vectorized loads and stores to reduce LDG instruction count. The **welford** variant uses the Welford online algorithm for numerically stable variance computation, employing a two-stage reduction (warp shuffle within each warp, then shared memory for cross-warp merge).

All three variants are **memory-bound** at 84-87% of peak DRAM bandwidth utilization. Compute intensity is approximately 0.27 FLOP/byte, far below the ridge point. The sum_sq variant achieves 1.63 ms (424.6 GB/s, 86.7% utilization) as the fastest option. Float4 vectorization provides no benefit — it is actually 1.6% slower (1.65 ms) because row-wise access is already fully coalesced and the bottleneck is DRAM bandwidth, not instruction issue rate. Welford trades 0.6% performance (1.68 ms, 84.1%) for numerical stability, avoiding the catastrophic cancellation that can occur in sum_sq when variance is much smaller than the squared mean.

With per-layer bandwidth utilization already near the hardware limit, the only path to meaningful improvement is operator fusion. The recommended next step is to fuse the residual addition into the LayerNorm kernel, eliminating the intermediate tensor write and read that occurs in the standard `x = LN(x + sublayer_output)` pattern. A prototype is available in the `FusedResidualLN/` module.
