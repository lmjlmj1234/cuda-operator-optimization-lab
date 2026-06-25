# Nsight Compute (ncu) Acquisition Plan

**Environment Note:** WSL2 GPU-PV does not support ncu hardware performance counters. The commands below are prepared for execution on a bare-metal Linux/Windows system with an RTX 3060.

## Acquisition Command

```bash
ncu --set full --kernel-name "softmax_online*" --export softmax_online ./build/cuda_operators
```

## Expected Metrics

| Metric | Expected Value | Rationale |
|--------|---------------|-----------|
| SM Throughput | ~30% | Low arithmetic intensity per byte loaded |
| DRAM Throughput | ~90% | Kernel is memory-bound, should saturate HBM |
| Occupancy | ~100% | Low register count (~32 regs), no shared memory |

## How to Run

1. Copy the binary (`./build/cuda_operators`) to a bare-metal Linux or Windows system with an RTX 3060.
2. Install NVIDIA Nsight Compute CLI (`ncu`).
3. Run the command above.
4. Import the `softmax_online.ncu-rep` file into Nsight Compute GUI for visual analysis.

## What to Look For

- **DRAM vs L1/L2 cache hit rate:** Confirm that the majority of traffic goes to DRAM (expected for row-wise streaming access with no reuse between rows).
- **Sector utilization:** Verify that each 128-byte cache line sector is fully utilized by coalesced loads.
- **Stall reasons:** Look for "long scoreboard" stalls (waiting on memory) as the dominant stall reason, confirming the memory-bound classification.
