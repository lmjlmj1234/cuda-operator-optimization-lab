# RMSNorm -- CUDA Kernel Implementation

## Overview

This module implements Root Mean Square Normalization (RMSNorm) as a fused CUDA kernel. RMSNorm normalizes activations by their root mean square without mean subtraction, making it a simpler and faster alternative to LayerNorm. It is the standard normalization used in LLaMA, Mistral, and most modern open-source LLMs. Two variants are provided: a scalar baseline (`rmsnorm_kernel`) and a float4 vectorized version (`rmsnorm_float4`).

## Performance

At the default problem size (N = 4096, D = 8192, float32), the scalar kernel achieves 1.63 ms kernel time and 424.6 GB/s bandwidth (86.7% of peak DRAM). The float4 variant shows no measurable improvement (1.65 ms, 419.5 GB/s) because row-wise reduction is already perfectly coalesced, making the kernel entirely memory-bound. Vectorization reduces instruction count but does not affect DRAM transaction patterns, so the bottleneck remains unchanged.

## Key Results

The analysis confirms RMSNorm is memory-bound with a compute intensity of 0.25 FLOP/byte. At D = 8192, RMSNorm and LayerNorm perform within 1% of each other since both are dominated by the same HBM traffic. Block size tuning shows BLOCK = 256 is optimal, balancing per-SM parallelism and register pressure. Future optimization work should focus on kernel fusion with adjacent operators (matmul or residual add) rather than further isolated kernel tuning.

## Documentation

Detailed reports covering each aspect of the 9-point engineering standard are in the `docs/` directory:

| File | Content |
|------|---------|
| `docs/problem_definition.md` | Problem formulation, I/O shapes, complexity, memory pattern |
| `docs/correctness.md` | Verification against host reference, cross-variant agreement |
| `docs/benchmark.md` | Benchmark config, results table, bandwidth calculation |
| `docs/kernel_analysis.md` | Static analysis via cuobjdump, instruction count, occupancy |
| `docs/nsight_analysis.md` | Profiling results (WSL2 limitation, future ncu commands) |
| `docs/bottleneck_analysis.md` | Compute intensity, roofline analysis, memory-bound verdict |
| `docs/optimization_history.md` | Version log, technique evaluation, key lessons |
| `docs/decision_log.md` | Design decisions with expanded rationale |
| `docs/interview_notes.md` | Engineering interview Q&A on RMSNorm vs. LayerNorm |
