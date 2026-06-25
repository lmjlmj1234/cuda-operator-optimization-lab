# Optimization History

| Version | Technique | Gain | Side Effects |
|---------|-----------|------|--------------|
| v0: softmax_naive | 3-pass baseline | -- | 54.3% bandwidth util |
| v1: softmax_online | Online 1-pass | -39.7% time | Slightly different numerics (float associativity) |
| v2: softmax_online+float4 | Float4 vectorization | -2.4% from v1 | Negligible gain for coalesced access pattern |
| v3: softmax_warp | Warp shuffle (dim<=32) | Reduced block size | Only applicable for small dimensions |

**Key Insight:** Float4 provides minimal benefit for row-wise coalesced access because each thread's loop stride already achieves full memory coalescing. Float4 mainly helps when vectorization changes the access pattern (e.g., strided or cross-query patterns).
