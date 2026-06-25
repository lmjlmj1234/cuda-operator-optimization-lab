# Interview Notes — Fused Residual + LayerNorm

## Q: What's the expected speedup from fusion?

The primary benefit is **memory savings, not speed**. For an end-to-end model, fusing add + LN saves ~256 MB of HBM traffic per layer (write + read of 4096 x 8192 floats). This compounds across layers: 32 layers x 256 MB = 8 GB saved. The per-layer kernel time stays the same because we added work but removed the intermediate buffer bottleneck.

## Q: Describe the nvcc register aliasing bug that was found.

In the two-pass fused kernel, both passes use loop induction variables and temporaries. The compiler (at `-O3`) assigned `mean` and `rstd` to the same physical registers as loop temporaries from pass 1. Since pass 2's loop runs _before_ reading `mean`/`rstd`, the compiler's live-range analysis failed and the loop overwrote `mean`/`rstd`. Fix: use `__shared__ float` as a compiler barrier. This follows the pattern used in CUTLASS for similar issues.

## Q: Can the fusion be extended to include the attention computation?

Yes -- FlashAttention fuses QK^T matmul + softmax + PV matmul into a single tile-wise kernel. Our `FlashAttentionSoftmax/` directory has a prototype demonstrating the online softmax aspect of this fusion. Full FlashAttention requires tiled matmul support which is a significant extension.

## Q: What's the trade-off between fusion scope and kernel complexity?

Each fused operation adds:
1. More register pressure
2. More shared memory usage
3. Harder debugging

The rule of thumb: fuse at natural pipeline boundaries where intermediate tensors are large enough that HBM traffic dominates. Add + LN is a clean boundary (no cross-row data dependencies).

## Q: Why did float4 vectorized loads not help?

The kernel is already memory-bound at 87.6% bandwidth utilization. Reducing instruction count via float4 cannot significantly improve performance when the bottleneck is HBM traffic. The float4 variant actually regressed slightly (1.67 ms vs 1.63 ms) because the 6x float4 registers reduced occupancy to ~50%.

## Q: What optimization would you try next?

If this kernel needed further improvement, the next steps would be:
1. **One-pass with online softmax** (like FlashAttention) -- compute mean/variance incrementally and normalize in a single pass, avoiding the second HBM read.
2. **Persistent kernel** with thread block scheduling for better load balancing.
3. **Tiling across rows** if D is small enough to fit in shared memory (not applicable at D=8192).

## Key Takeaways for Interviewers

- The candidate identified a compiler bug through systematic debugging (optimization level, grid size, reproducibility)
- Applied a principled fix (shared memory barrier) from production codebases (CUTLASS)
- Quantified the fusion benefit precisely (33% HBM reduction, 256 MB per layer)
- Made correct trade-off decisions (scalar over float4, two-pass over one-pass)
- Documented environment limitations and planned next steps
