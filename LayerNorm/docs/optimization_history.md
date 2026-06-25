# Optimization History

| Version | Technique | Gain | Side Effects |
|---------|-----------|------|--------------|
| v0: layernorm_sum_sq | Two-moment baseline | — | Numeric risk: catastrophic cancellation |
| v1: layernorm_float4 | Float4 vectorization | -1.6% regression | Higher register pressure (24 -> 32) |
| v2: layernorm_welford | Welford online variance | -0.6% vs v0 | Numerically stable; more registers (40, 83% occupancy) |

## Key Insight

Welford is the right choice for numerical stability even though it is slightly slower. Float4 was rejected because coalesced row access already hits full bandwidth — vectorization only adds register pressure without reducing HBM transactions.

## Summary

- v0 established the correctness baseline with the simplest possible implementation
- v1 attempted to reduce LDG instruction count via float4 vectorization but regressed due to register pressure
- v2 prioritized numerical stability for production use cases where sum_sq's catastrophic cancellation is unacceptable
