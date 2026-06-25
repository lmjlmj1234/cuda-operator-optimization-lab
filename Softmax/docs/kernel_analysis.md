# Profiling Analysis -- Static Kernel Analysis

**Environment:** WSL2 GPU-PV right-arrow ncu hardware counters unavailable.

## Static Analysis (cuobjdump)

- softmax_naive: ~40 instructions (3x LDG, EXP, FMA, STG cycles)
- softmax_online: ~30 instructions (1x LDG pass, EXP, FMA for online merge, 1x STG pass)

## Theoretical Occupancy

- REG count right-arrow 32 right-arrow 100% occupancy
- SMEM usage right-arrow 0 bytes right-arrow no SMEM limitation
- Bottleneck: DRAM bandwidth
