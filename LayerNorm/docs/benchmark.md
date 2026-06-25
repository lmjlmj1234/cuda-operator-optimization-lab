# Benchmark

## Configuration

| Parameter | Value |
|-----------|-------|
| N | 4096 |
| D | 8192 |
| Data type | float32 |
| Warmup | 50 iterations |
| Measured | 500 iterations |
| eps | 1e-5 |
| Default block size | 256 |

## Results

| Kernel | Time (ms) | Bandwidth (GB/s) | Peak Utilization |
|--------|-----------|-------------------|------------------|
| layernorm_sum_sq (BLOCK=256) | 1.63 | 424.6 | 86.7% |
| layernorm_float4 (BLOCK=256) | 1.65 | 419.5 | 85.6% |
| layernorm_welford (BLOCK=256) | 1.68 | 412.0 | 84.1% |

## Key Finding

Float4 does NOT help — it is ~1.6% slower than the scalar sum_sq baseline. All three variants are bandwidth-bound at 84-87% utilization.

## Block Size Sweep

The sum_sq kernel is instantiated for block sizes 128, 256, and 512. Block 256 was selected as the default (see `decision_log.md`).

## Environment

GPU: Determined by `nvidia-smi` at benchmark time. All runs under the same device.
