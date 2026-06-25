# Kernel Analysis

## Environment

Profiling is performed in WSL2 GPU-PV (GPU Paravirtualization) which does not expose hardware performance counters. ncu profiling is unavailable in this environment.

**Future acquisition command:**
```bash
ncu --set full --kernel-name "fused_residual_layernorm*" --export fused_ln ./build/cuda_operators
```

## Static Analysis (cuobjdump)

### fused_residual_layernorm (scalar)

- **Structure:** Two-pass loop
- **Pass 1:** 2x LDG (x + residual) + FMA for fused add + FMA for square
- **Pass 2:** 2x LDG (x + residual) + 2x LDG (gamma + beta) + FMA chain + STG
- **Total instructions:** ~35

### fused_residual_layernorm_float4

- Fewer LDG instructions due to vectorized loads
- Requires 6x float4 registers (x, residual, gamma, beta, output) -> REG ~= 48
- **Occupancy:** ~50% (vs ~100% for scalar version)

## SASS Analysis

*Pending: requires ncu on native Linux or Windows with GPU-PV disabled.*

## nvcc Register Aliasing Bug

### Symptom

The `fused_residual_layernorm` kernel output had per-row mean of approximately `0.042` instead of the expected `0.0`, causing a max absolute error of **0.24 (24%)** versus the PyTorch reference.

Key observations:
- Correct under `-O0`, incorrect under `-O2` / `-O3`
- Row 0 correct at `<<<N, 256>>>` but incorrect at `<<<1, 256>>>`
- Results randomly varied between `0.000` and `0.042` across runs
- Reproduced in a standalone test file extracted from the main build

### Root Cause

**nvcc compiler register aliasing bug** (register reuse across kernel passes).

The kernel's two-pass structure:

```
Pass 1: load x[i] + residual[i], compute sum, sum_sq
        -> block_reduce_sum -> mean, rstd (held in registers)
Pass 2: reload x[i] + residual[i], normalize with mean/rstd
```

In Pass 2's loop, the `x[i] + residual[i]` load uses induction variables and temporaries. The compiler (at `-O3`) assigned `mean` and `rstd` to **the same physical registers** as these loop temporaries. When Pass 2's loop ran, it overwrote `mean` / `rstd` before they were consumed.

This bug only affects the fused kernel because both passes perform the same `x[i] + residual[i]` access pattern. Standard LayerNorm does not have this pattern (Pass 1 reads input only, Pass 2 reads input only -- the compiler's live-range analysis does not conflict).

### Fix

Pass `mean` and `rstd` through `__shared__` memory as a compiler barrier:

```cuda
__shared__ float sm_mean, sm_rstd;
if (threadIdx.x == 0) {
    sm_mean = sum / dim;
    sm_rstd = rsqrtf(fmaxf(var, 0.0f) + eps);
}
__syncthreads();
float mean = sm_mean;
float rstd = sm_rstd;
```

The `__syncthreads()` forces a compiler ordering fence. The shared memory write creates a true data dependency that prevents register aliasing. This pattern follows the approach used in CUTLASS for similar compiler ordering issues.

### Lesson

nvcc's register allocator is not fully reliable. When a kernel has a "two-pass" structure where both passes access similar memory patterns, intermediate results should be passed through shared memory rather than relying on register liveness analysis.
