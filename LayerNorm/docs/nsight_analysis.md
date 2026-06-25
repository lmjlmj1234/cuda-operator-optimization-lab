# Nsight Analysis

## Environment Limitation

**WSL2 GPU-PV** — ncu hardware counters are unavailable in this environment. All profiling data below is from static analysis only.

## Static Analysis (cuobjdump)

The kernel SASS was inspected via `cuobjdump`:
- **sum_sq**: Loop 1 (sum + sum_sq reduction, fused), Loop 2 (normalize with gamma/beta)
- **float4**: Same structure but vectorized LDG/STG for 4x fewer memory instructions, at cost of more registers
- **welford**: Welford update in loop, warp shuffle reduction (first stage), shared memory merge (second stage)

## Metrics Requiring ncu

The following metrics are needed for a complete analysis but cannot be collected in WSL2:

- Achieved occupancy (vs theoretical)
- DRAM throughput and sector misses
- L1/L2 cache hit rates
- Warp stall reasons (Long Scoreboard, Short Scoreboard, Barrier)
- SM throughput

## Future Acquisition

Once a native Linux environment or GPU-PV with PMU support is available:

```bash
# Full metric set for sum_sq kernel
ncu --set full --kernel-name "layernorm_sum_sq*" --export layernorm_sum_sq ./build/cuda_operators

# Welford kernel
ncu --set full --kernel-name "layernorm_welford*" --export layernorm_welford ./build/cuda_operators

# Float4 kernel
ncu --set full --kernel-name "layernorm_float4*" --export layernorm_float4 ./build/cuda_operators
```

See `shared_docs/Profiling_Guide.md` for the full Performance Analysis Workflow.
