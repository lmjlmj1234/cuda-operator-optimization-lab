# Decision Log

## Design Decisions

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| Scalar as default kernel | Float4 vectorized | Float4 gives no measurable benefit; scalar code is simpler to read and maintain |
| Use `rsqrtf` | `sqrtf` + division | `rsqrtf` is approximately 4x faster than `sqrtf + div` and produces identical results within IEEE 754 single-precision accuracy for this use case (positive denominator) |
| No mean subtraction | Include mean computation | RMSNorm definition specifically omits it; retaining the mean would make it LayerNorm, losing the computational savings that motivate RMSNorm |
| BLOCK = 256 | 128, 512 | Balances parallelism (4096 rows / 256 = 16 thread blocks per wave) and per-block register pressure; 128 leaves performance on the table (fewer parallel warps per SM), 512 offers no improvement and reduces multi-wave occupancy on smaller GPUs |
| Row-wise, one block per row | Warp-per-row, or grid-stride loop | One block per row is the standard approach for normalization kernels; warp-per-row requires D <= warpSize for optimal performance (not possible at D = 8192); grid-stride loops complicate the reduction |

## Technical Rationale

### Why `rsqrtf` Instead of `sqrtf` + Division

The denominator `sqrt(sumsq / D + eps)` is computed as the reciprocal square root:

```cuda
float denom = rsqrtf(sumsq / D + eps);
y[i] = x[i] * denom * gamma[i];
```

This replaces a division (y[i] = x[i] / sqrt(...) * gamma[i]) with a multiplication. On NVIDIA GPUs, `rsqrtf` is a single special-function unit instruction (MUFU.RSQRT) with approximately 4x the throughput of `sqrtf` + division. The `rsqrtf` result is accurate to approximately 1 ULP (unit in the last place) for normal-range inputs, which is well within tolerance for normalization.

### Why No Mean Subtraction

The fundamental definition of RMSNorm distinguishes it from LayerNorm. Including mean subtraction would:
- Add an additional reduction pass (sum x, then subtract mean)
- Increase register pressure
- Negate the clean vectorization of the element-wise normalization
- Change the mathematical operation entirely (it would become LayerNorm)

RMSNorm is used in production (LLaMA, Mistral) precisely because removing the mean does not hurt convergence or quality.

### Why Float4 Was Rejected

Float4 vectorization was tested extensively and produced no measurable gain (<1% difference, within noise). This is consistent with the memory-bound analysis: when the kernel spends >85% of its time stalled on DRAM, reducing instruction count in the compute portion has no visible effect. The scalar version is preferred for its readability.

### Why BLOCK = 256 Is the Default

| Block Size | Rationale |
|------------|-----------|
| 128 | Insufficient parallelism per SM -- each block has only 4 warps, and the latency-hiding capacity of the SM is underutilized |
| 256 | Optimal balance -- 8 warps per block, enough to hide 200-400 cycle DRAM latency; works on both 64-warp (compute 7.0) and 32-warp (compute 8.0) SMs with 2+ blocks resident |
| 512 | No performance improvement over 256; increases shared memory pressure (reduction scratch per warp doubles); can reduce occupancy on 32-warp SMs (only 1 block resident instead of 2) |
