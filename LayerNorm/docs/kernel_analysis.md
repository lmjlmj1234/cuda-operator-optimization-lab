# Kernel Analysis

## Static Analysis (cuobjdump)

### layernorm_sum_sq
- Sum + sum_sq fused in single loop, then 2x MUL + ADD + RSQRT for mean/var, then normalize loop with 2x LDG (x, gamma), 1x STG
- Total: approximately 24 instructions in the hot path

### layernorm_float4
- 4x fewer LDG instructions compared to sum_sq (vectorized loads)
- Higher register pressure: needs 4x input + 4x gamma + 4x output registers
- Must handle tail elements (D not always divisible by 4) with scalar fallback loop

### layernorm_welford
- Welford online variance update in each iteration
- Two-stage reduction: warp shuffle within warp (first 16 iterations), then shared memory for cross-warp merge
- More instructions per element: 2 subtracts + 2 adds + 1 multiply for welford_update vs 1 add + 1 multiply-add for sum_sq

## Theoretical Occupancy

| Variant | Registers | Occupancy |
|---------|-----------|-----------|
| sum_sq | ~24 | 100% |
| float4 | ~32 | 100% |
| welford | ~40 | 83% |

Occupancy computed assuming no shared memory usage during reduction (shared memory is only used transiently). The welford variant's higher register count (due to WelfordStats struct with mean/m2/count plus shuffle loop state) reduces occupancy from 100% to 83%.

## Instruction Mix

- **sum_sq**: Predominantly FMA (sum_sq update) and ADD (sum update), with 1 RSQRT per row
- **float4**: Same arithmetic as sum_sq but with fewer address computation instructions due to vectorized access
- **welford**: More arithmetic per element due to Welford update formula; additional shuffle instructions for warp-level merge
