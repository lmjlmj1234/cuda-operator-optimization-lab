# Interview Notes

## Q: Why does online softmax work mathematically?

A: The key identity is `sum_j exp(x_j) = exp(m_new) * [exp(old_m - m_new) * sum_old + exp(x_new - m_new)]`. When we find a new max, the old sum's per-element `exp(x_i - old_m)` values would need to be recomputed with `exp(x_i - new_m)`. Since `exp(x_i - new_m) = exp(x_i - old_m) * exp(old_m - new_m)`, we can retroactively correct the sum with a single multiplication. This reduces 3 HBM passes to 1.

## Q: Why is softmax memory-bound?

A: Each element requires 1 load and 1 store (4+4=8 bytes) but only O(1) arithmetic (1 FMA + 1 EXP per element). The ratio of bytes to FLOPs is ~5:1, which is far below the GPU's compute-to-bandwidth equilibrium point (35.3 FLOP/byte on RTX 3060). The pipeline is always waiting on DRAM.

## Q: When would softmax be compute-bound?

A: For very small rows (dim < 128), the overhead of reductions dominates and the kernel becomes latency-bound. For larger rows, it's always memory-bound because the arithmetic intensity doesn't increase with dimension size.

## Q: Why does float4 not help much for softmax?

A: With `grid(N) block(BLOCK)` row-major access, each thread already loads `dim/BLOCK` contiguous elements. The hardware already coalesces these into 128-byte cache line transactions. Float4 groups 4 loads into 1 instruction, reducing LDG instruction count but not DRAM transaction count -- the bottleneck remains DRAM bandwidth, not instruction issue.
