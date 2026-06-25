# Fused Residual Add + LayerNorm

## Overview

This module implements a fused CUDA kernel that combines the Transformer sub-layer pattern `y = LN(x + residual)` into a single kernel launch. Instead of computing the intermediate `z = x + residual`, writing it to HBM, and reading it back for LayerNorm, the fused kernel loads `x` and `residual` simultaneously, adds them in registers, and normalizes -- eliminating the intermediate buffer entirely.

## Key Benefit

The fusion saves **33% of HBM traffic per layer** (32 MB at N=4096, D=8192). For a 32-layer Transformer, this eliminates approximately 1 GB of unnecessary memory traffic. Crucially, the fused kernel achieves the same wall-clock time as a standalone LayerNorm kernel (1.63 ms) while doing strictly more work -- the fusion removes the bandwidth bottleneck from the intermediate tensor.

## nvcc Register Aliasing Bug

During development, a subtle compiler bug was discovered: nvcc at `-O3` incorrectly aliased the `mean`/`rstd` registers with loop temporaries in the second pass, producing wrong outputs with errors up to 24%. The fix uses `__shared__` memory as a compiler barrier, a pattern also used in CUTLASS for similar issues. A detailed account of the bug diagnosis and fix is in [docs/kernel_analysis.md](docs/kernel_analysis.md).

## Current Status

The fused scalar implementation is the default and recommended variant. It passes correctness verification (max diff ~1e-7 against PyTorch) and achieves 87.6% of peak HBM bandwidth. The float4 variant showed a slight regression due to register pressure and is not recommended. Future work includes exploring a one-pass kernel with online softmax and extending the FlashAttention prototype with tiled matmul support for end-to-end attention fusion.

## Documentation

- [Problem Definition](docs/problem_definition.md)
- [Correctness Verification](docs/correctness.md)
- [Benchmark Results](docs/benchmark.md)
- [Kernel Analysis](docs/kernel_analysis.md)
- [Optimization History](docs/optimization_history.md)
- [Bottleneck Analysis](docs/bottleneck_analysis.md)
- [Nsight Analysis](docs/nsight_analysis.md)
- [Interview Notes](docs/interview_notes.md)
- [Decision Log](docs/decision_log.md)
