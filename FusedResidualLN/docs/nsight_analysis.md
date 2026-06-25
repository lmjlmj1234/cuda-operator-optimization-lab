# Nsight Analysis

## Current Environment Limitation

**WSL2 GPU-PV (GPU Paravirtualization)** does not expose hardware performance counters to user-space tools. ncu profiling is unavailable in this environment.

This means the following metrics cannot be measured directly:

- **Memory bandwidth utilization** (achieved vs. peak)
- **Cache hit rates** (L1/L2)
- **SASS instruction counts**
- **Occupancy and stall reasons**
- **LDG/STG transaction counts**
- **Sector utilization**
- **Warp stall cycles** (waiting for data, synchronization, etc.)

## Workaround: Static Analysis

Without hardware counters, we rely on:

- **cuobjdump** for SASS-level instruction inspection
- **Manual occupancy calculation** based on register/shared memory usage
- **FLOP/byte analysis** to classify as compute-bound or memory-bound

The results of this analysis are documented in [kernel_analysis.md](kernel_analysis.md).

## Future Acquisition

To get full profiling data, run on a native Linux environment or Windows with GPU-PV disabled:

```bash
# Full metric collection
ncu --set full \
    --kernel-name "fused_residual_layernorm*" \
    --export fused_ln \
    ./build/cuda_operators

# Focus on memory metrics
ncu --set memory \
    --kernel-name "fused_residual_layernorm*" \
    --export fused_ln_memory \
    ./build/cuda_operators

# Focus on occupancy and stall reasons
ncu --set occupancy \
    --kernel-name "fused_residual_layernorm*" \
    --export fused_ln_occupancy \
    ./build/cuda_operators
```

## Key Questions for Future Profiling

1. What is the actual achieved bandwidth percentage? (Our estimate of 87.6% is calculated from wall-clock time, not hardware counters.)
2. How do cache hit rates differ between Pass 1 and Pass 2?
3. What are the primary stall reasons (L1 tag, L2, synchronization)?
4. What is the sector utilization for the float4 variant?
5. Does the `__shared__` barrier for the nvcc bug workaround introduce measurable latency?
