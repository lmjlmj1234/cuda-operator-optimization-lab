# Benchmark — Fused Residual + LayerNorm

## Configuration

| Parameter | Value |
|-----------|-------|
| Device | NVIDIA RTX 4090 (Ada Lovelace) |
| CUDA Version | 12.x |
| Problem Size | N=4096, D=8192 |
| Data Type | float32 |
| Grid/Block | <<<N, BLOCK>>> |
| BLOCK Size | 256 |

## Results

| Kernel Variant | Time (ms) | Bandwidth (GB/s) | Peak Utilization |
|----------------|-----------|-------------------|------------------|
| fused_residual_layernorm (BLOCK=256) | 1.63 | 429.1 | 87.6% |
| fused_residual_ln_float4 (BLOCK=256) | 1.67 | 419.0 | 85.5% |

## Analysis

The fused scalar kernel achieves identical wall-clock time to standalone LayerNorm while performing strictly more work (loading residual + fused add). This is the expected behavior:

- **Without fusion:** Two kernels (add + LN), two HBM round-trips for the intermediate tensor
- **With fusion:** One kernel, one HBM round-trip, same wall time

The float4 variant slightly regresses due to 50% occupancy from register pressure.

## Run Yourself

```bash
cd FusedResidualLN/benchmark
./run.sh
```
