# Decision Log

## Core Decisions

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| sum_sq as default | Welford | Simpler code; switch to Welford if numerical issues appear |
| Float4 removed from default | Float4 always | 1.6% regression; higher reg pressure |
| Welford as stable option | None | Needed for large-dim/low-var regimes |
| gamma/beta in global memory | Shared mem preload | D is too large for shared mem caching |

## Expanded Reasoning

### Why block size 256 (vs 128/512)?

The sum_sq kernel is instantiated for BLOCK=128, 256, and 512. Block 256 was chosen as the default:

- **128**: Requires 64 iterations per thread for D=8192. More loop overhead and fewer threads in flight per SM (lower occupancy potential for hiding latency).
- **256**: Sweet spot. 8192/256 = 32 iterations per thread. 100% theoretical occupancy at ~24 registers. Leaves room for register pressure without dropping to a lower occupancy tier.
- **512**: Reduces iterations to 16 but increases shared memory usage for the tree reduction (512 floats = 2 KB). More importantly, 512 threads per block reduces the number of blocks that can run concurrently per SM, potentially reducing achieved occupancy.

### Why sum_sq as default (vs Welford)?

- **Code simplicity**: sum_sq is a straightfoward single-loop reduction with two accumulators. Welford requires a struct, update/merge functions, and a two-stage reduction with warp shuffle.
- **Register pressure**: sum_sq at ~24 registers achieves 100% occupancy. Welford at ~40 registers drops to 83%.
- **Performance**: sum_sq is 0.6% faster at 86.7% bandwidth vs 84.1%.
- **Trade-off**: sum_sq has catastrophic cancellation risk when var << mean^2. For inference or well-scaled activations this is acceptable. For training with low-variance layers, switch to Welford.

### Why float4 was rejected (1.6% regression)

Float4 vectorization reduces LDG instruction count by 4x. However:
- Row-wise access is already fully coalesced — bandwidth is saturated at the DRAM level, so reducing instruction count does not improve throughput.
- Higher register pressure (24 -> 32) does not drop occupancy (still 100%) but adds register allocation pressure that can cause compiler spillage or reduce latency-hiding capacity.
- The measured 1.6% regression (1.63 ms -> 1.65 ms) confirmed the hypothesis: **when memory-bound with coalesced access, vectorization does not help.**

### Why gamma/beta stay in global memory (D too large for SMEM)

- D=8192 means gamma and beta are each 8192 floats = 32 KB.
- Total: 64 KB for both weight vectors.
- Shared memory per SM is typically 48 KB (configurable up to 164 KB on Ampere+, at cost of L1 cache).
- Caching 64 KB of weights in shared memory would consume all available SMEM, leaving nothing for reduction.
- Alternative: preload per-thread segments of gamma/beta into registers. This would require each thread to hold D/BLOCK_SIZE gamma + beta values = 8192/256 * 2 = 64 floats = 256 bytes per thread = 256 registers, which is infeasible.

### Why shared memory tree reduction (vs warp shuffle only)

- Warp shuffle only works within a single warp (32 threads). With BLOCK=256, we have 8 warps.
- Cross-warp communication requires shared memory or global memory.
- The current approach uses a shared memory tree reduction (`block_reduce_sum` in `utils.cuh`): each warp does a warp-level reduce within its warp first, but the final cross-warp merge goes through shared memory.
- A pure shuffle approach would require all 256 threads to be in the same warp, which is not possible.
- The welford variant uses a hybrid: warp shuffle within each warp, then shared memory for cross-warp combining, then a final shuffle on warp 0. This minimizes shared memory bank conflicts while still supporting arbitrary block sizes.
