# Decision Log

## Decision Table

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| Shared memory for mean/rstd | Local variables (buggy) | nvcc register aliasing bug -- shared memory forces correct ordering via `__syncthreads()` barrier |
| Scalar fusion as default | Float4 fusion | Float4 causes 50% occupancy from register pressure; no perf gain (1.67 ms vs 1.63 ms) |
| Two-pass (reload data) | One-pass (cache in registers) | D=8192 is too large for register caching (each thread processes 32 elements, cannot keep all in registers for a single pass) |
| Fuse only add + LN | Also fuse QK^T or FFN | Current scope; FlashAttention fusion is future work |
| Block size BLOCK=256 | BLOCK=128 or BLOCK=512 | 256 threads per block provides optimal occupancy for this workload given register pressure |

## Detailed Rationale

### Shared Memory for mean/rstd (nvcc Bug Workaround)

**Problem:** The compiler aliased `mean`/`rstd` registers with loop temporaries in Pass 2, producing wrong results at `-O2`/`-O3`.

**Why shared memory:** Writing to `__shared__` and reading back via `__syncthreads()`:
- Creates a true data dependency the compiler cannot optimize away
- Forces a memory fence between compute and consumption
- Only adds ~4 bytes of shared memory (negligible cost)
- Follows CUTLASS convention for compiler ordering issues

**Why not `volatile`:** `volatile` prevents some optimizations but does not create the same strong ordering fence. Shared memory + `__syncthreads()` is the standard pattern.

### Two-Pass vs One-Pass

**Why two passes** (reload `x + residual` in Pass 2):
- D=8192 means each thread processes 8192 / 256 = 32 elements
- Keeping 32 elements of `x[i] + residual[i]` in registers would require 64 registers just for the fused sum
- Plus gamma, beta, accumulator, mean, rstd, etc.
- Total would exceed the register budget, causing spilling to L1/local memory
- Reloading from HBM is faster than spilling to L1 (which still counts as a memory access)

**One-pass alternative (online softmax):** Would require computing mean and variance incrementally while normalizing. This is possible (used in FlashAttention) but adds algorithmic complexity. Deferred to future work.

### Float4 Rejection

**Why rejected:**
- 6x float4 registers for the two-pass structure: `float4 x`, `float4 residual`, `float4 gamma`, `float4 beta`, `float4 output`, `float4 fused`
- Plus accumulators for sum, sum_sq
- Total register count ~= 48
- Occupancy drops from ~100% to ~50%
- Result: 1.67 ms (regression from 1.63 ms)

**When float4 would help:** If the kernel were compute-bound or had lower register pressure. At 87.6% bandwidth utilization, the bottleneck is HBM traffic, not instruction count.

### Why Not Fuse QK^T as Well

**Scope decision:** QK^T fusion requires tiled matrix multiplication support (like FlashAttention). This is a fundamentally different kernel architecture:
- Cross-row data dependencies (softmax across key rows)
- Tiled loading of Q and K
- Online softmax computation
- Split into a separate project: `FlashAttentionSoftmax/`

**Rule of thumb applied:** Fuse at natural pipeline boundaries where intermediate tensors are large and no cross-row dependencies exist. Add + LN is such a boundary.

### Why BLOCK = 256

- 256 threads per block: one block per row of 8192 elements
- Each thread processes 32 elements (8192 / 256)
- 32 elements per thread balances register pressure with loop overhead
- Lower block sizes (128) would double elements per thread, increasing register pressure
- Higher block sizes (512) reduce elements per thread but may cause scheduler issues on the RTX 4090's 128 SM count
