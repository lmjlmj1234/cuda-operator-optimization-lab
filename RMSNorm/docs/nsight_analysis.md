# Profiling / Nsight Analysis

## Environment Limitation

This work was conducted inside **WSL2 with GPU-PV (GPU Paravirtualization)**. Under this configuration, NVIDIA's Nsight Compute (`ncu`) cannot access hardware performance counters. Kernel profiling data (occupancy, memory throughput, instruction mix, stall reasons) is unavailable.

## Static Analysis (cuobjdump)

Without hardware counters, we analyze the compiled SASS via `cuobjdump --dump-sass`.

### rmsnorm_kernel

Approximately **25 instructions**:

1. **Reduction loop**: LDG (load x[i]), FMA (x[i]^2 accumulator), loop branching.
2. **Block reduction**: Shared memory reduce, final warp shuffle, compute `rsqrtf(sumsq/D + eps)`.
3. **Normalization loop**: LDG (x[i] + gamma[i]), FMA (scale), STG (store y[i]).

### rmsnorm_float4

Approximately **20 instructions**. Same structure but `LDG.128` reduces loop overhead.

### Theoretical Occupancy

| Metric | Value |
|--------|-------|
| Registers per thread | 24-32 |
| Shared memory per block | Minimal (reduction scratch, ~1 KB) |
| Theoretical occupancy | 100% (no register or SMEM bottleneck) |

## Command to Acquire Full Profile (on native Linux or non-WSL2 system)

```bash
ncu --set full --kernel-name "rmsnorm_kernel*" --export rmsnorm_kernel ./build/cuda_operators
```

## Metrics of Interest (Future Work)

When hardware counters become accessible, collect:

| Metric | What It Reveals |
|--------|-----------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM utilization |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bandwidth utilization |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1 cache utilization |
| `smsp__average_warps_issue_stalled_barrier` | Reduction synchronization overhead |
| `smsp__average_warps_issue_stalled_long_scoreboard` | Load-to-use latency hiding |
