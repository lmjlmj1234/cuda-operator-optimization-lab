# Decision Log

## Decision Table

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| Block reduce over shared memory | warp shuffle (for dim>32) | Shared memory reduction handles arbitrary BLOCK_SIZE; shuffle limited to 32 threads |
| BLOCK=256 as default | 128/512 | Best balance of parallelism vs per-thread work for D=8192 |
| Template BLOCK_SIZE | Runtime param | Compile-time loop unrolling; no perf penalty for branching |
| Online 1-pass as primary | 3-pass naive | 40% speedup from HBM pass reduction; numerically stable |
| Float4 optional | Default float4 | +2.4% gain doesn't justify code complexity; kept as template param |

## Expanded Rationale

### Why BLOCK_SIZE=256 vs 128/512

With D=8192 columns and one block per row:
- **BLOCK_SIZE=128:** 128 threads per block, 64 elements per thread. Lower thread count means more serial work per thread, increasing the reduction tree depth and loop overhead. The GPU can schedule more blocks per SM (up to 1024 threads / 128 = 8 blocks), but the increased per-thread work increases register pressure and reduces the benefit of thread-level parallelism.
- **BLOCK_SIZE=512:** 512 threads per block, 16 elements per thread. Higher thread count reduces per-thread work but limits SM occupancy (1024 threads max per SM allows only 2 blocks). Many threads sit idle during the tree reduction, and the 512-thread reduction requires 9 rounds of `__syncthreads` (log2(512)=9) vs 8 rounds for 256 threads.
- **BLOCK_SIZE=256:** 256 threads per block, 32 elements per thread. Each SM can hold 4 blocks (1024/256), providing good occupancy. The 8-round tree reduction is efficient. 32 elements per thread gives enough work to amortize loop overhead without excessive serialization.

On RTX 3060 (Ampere GA106), BLOCK=256 maximizes throughput by balancing per-SM thread count against reduction overhead.

### Why Shared Memory Tree for Reduction

Shared memory is the only on-chip mechanism for inter-thread communication across thread blocks larger than one warp (32 threads). The alternatives are:
- **Warp shuffle (`__shfl_xor_sync`):** Only works within a single warp. For BLOCK_SIZE=256, we would need 8 independent warp reductions followed by a shared memory stage to combine them, which adds complexity with no benefit over a direct shared memory tree.
- **Global memory atomic:** Would serialize the reduction and add HBM latency, completely destroying performance.
- **Separate kernel launch:** Would add kernel launch overhead (~5-10 us) and require an extra HBM pass.

A tree reduction in shared memory is the standard pattern: each round halves the active threads, threads write partial results to `__shared__` arrays, and a `__syncthreads()` barrier ensures visibility. This is simple, fast, and scales to arbitrary BLOCK_SIZE.

### Why Warp Shuffle Only for Dim <= 32

When D <= 32:
- D=32: each thread loads exactly 1 element. A single warp (32 threads) handles the entire row. Thread IDs 0..31 correspond directly to element indices 0..31.
- D=16: 16 threads handle 1 element each; the warp is under-filled but shuffle still works.
- The reduction can be done entirely with `__shfl_xor_sync` within one warp, avoiding shared memory entirely and eliminating `__syncthreads()` barriers.

When D > 32:
- Multiple warps are needed. Cross-warp communication requires shared memory.
- The simplicity of the warp-shuffle approach breaks down: you need shared memory staging, multiple shuffle rounds per warp, and then a warp-level reduction across warp results.

The warp variant exists as a specialized optimization for the small-row case. It is not a general replacement for the shared-memory reduction.

### Why Online Algorithm Over 3-Pass

The 3-pass naive algorithm requires three global memory passes per row:
1. Find row max (read input)
2. Compute exp(x-max) and sum (read input again)
3. Normalize (read max/sum from registers, write output)

Each pass reads the entire input row from HBM. This is 3x HBM reads + 1x HBM write.

The online 1-pass algorithm merges all three into a single pass:
- Maintain running max and running sum as we stream through the row.
- When a new max is found, retroactively correct the sum using the identity `sum_new = exp(old_m - new_m) * sum_old + exp(x_new - new_m)`.
- At the end of the row, write normalized results in the same pass (using a second shared memory buffer for the reduced max/sum).

This reduces HBM reads from 3x to 1x, yielding a ~40% speedup (2.17 ms to 1.31 ms). The only cost is slightly different floating-point summation order, which produces results within 1e-7 of the naive version -- well within float32 precision.

### Why Float4 Is Optional (Not Default)

Float4 vectorization groups 4 consecutive float loads into a single `LDG.128` instruction. For the softmax kernel:
- **Coalesced row-major access:** Each thread already loads sequential elements (e.g., thread 0 loads indices 0, 256, 512, ... with stride = BLOCK_SIZE). The hardware already coalesces these into 128-byte cache line transactions (32 bytes per sector x 4 sectors = 128 bytes). Each 128-byte transaction serves 32 consecutive float elements.
- **What float4 changes:** Reduces the number of load instructions from 32 (for 32 elements per thread) to 8 (for 8 float4 loads). This saves instruction decode/issue bandwidth but does NOT reduce DRAM transaction count.
- **Measured gain:** 2.4% (1.31 ms to 1.27 ms). The kernel was already 89.9% bandwidth-efficient; float4 pushed it to 92.8%.

The gain is real but marginal. Float4 is kept as a template parameter so it can be enabled/disabled without code duplication. It would matter more for strided or non-coalesced access patterns where manual vectorization changes the transaction size.
