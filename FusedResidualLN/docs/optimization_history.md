# Optimization History

## Version Timeline

| Version | Technique | Gain | Side Effects |
|---------|-----------|------|--------------|
| v0: fused scalar | Baseline fusion (load x+r, normalize, write) | Equivalent to standalone LN time | `__shared__` workaround needed for nvcc register aliasing bug |
| v1: fused_float4 | Float4 vectorized loads + fusion | No gain (slight regression: 1.67 ms vs 1.63 ms) | 50% occupancy from register pressure (REG ~= 48) |

## Notes

### v0: Baseline Fused Scalar

- Two-pass kernel: sum/sum_sq in pass 1, normalize in pass 2
- Block size = 256, one block per row
- Shared memory used only for block reduction and the nvcc bug workaround
- Achieves 87.6% of peak bandwidth on RTX 4090

### v1: Float4 Vectorized Loads

- Replaced scalar LDG with `float4` loads (4x fewer instructions)
- Required additional registers for the vectorized access pattern
- Register count jumped, reducing occupancy from ~100% to ~50%
- The occupancy drop offset the instruction reduction, resulting in a net regression

### Key Insight

For a memory-bound kernel at 87.6% bandwidth utilization, instruction-level optimizations (vectorized loads) have limited impact. The bottleneck is HBM traffic, not instruction throughput. Future gains require algorithmic changes (e.g., one-pass with online softmax), not micro-optimizations.
