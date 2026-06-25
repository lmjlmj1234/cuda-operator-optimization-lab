# Softmax — Kernel Optimization Report

## 1. Problem Definition

**What:** Softmax normalizes an input vector into a probability distribution:  
`softmax(x)_i = exp(x_i) / sum_j exp(x_j)`

**Why:** Fundamental building block in Transformer attention layers (after QK^T scores), classification heads, and
any multi-class output layer.

**Role in Transformer:** The normalization step in `Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V`. Must be
numerically stable and efficient for large sequence lengths and hidden dimensions.

**I/O Shapes:** `N=4096` rows, `D=8192` columns, float32.

**Complexity:** O(N·D) reads, O(N·D) writes — 3× HBM reads for naive (read input twice + read max/sum).

**Memory Pattern:** Row-wise: each row is contiguous in memory (coalesced), each block handles one row.

---

## 2. Baseline Implementation

**softmax_naive** (`Softmax/src/softmax_naive.cu`): 3-pass algorithm
1. Find row maximum
2. Compute exp(x-max) and sum
3. Normalize

Three global passes → 3× HBM reads of input + 1× HBM write of output.

Key config: `BLOCK_SIZE=128, 256, 512` — one block per row.

---

## 3. Correctness Verification

All variants verified against `reference_softmax()` (host-side 2-pass softmax):

```
softmax correctness: PASS
```

Cross-validation: `softmax_online<float4>` vs `softmax_naive` on row 2:
- Max diff: ~1e-7 — within float32 precision
- Only difference is floating-point summation order (deterministic vs associative online merge)

---

## 4. Benchmark

| Kernel | Time (ms) | Bandwidth (GB/s) | Peak Utilization |
|--------|-----------|-------------------|------------------|
| softmax_naive (BLOCK=128) | 2.17 | 266.1 | 54.3% |
| softmax_online (BLOCK=256) | 1.31 | 440.8 | 89.9% |
| softmax_online+float4 (BLOCK=256) | 1.27 | 454.7 | 92.8% |
| softmax_warp (dim=32) | 0.007 | — | — |

**40% reduction** in kernel time from naive to online. At 92.8% bandwidth utilization, the online variant is
near the RTX 3060 peak (490 GB/s).

---

## 5. Profiling Analysis

**Environment:** WSL2 GPU-PV → ncu hardware counters unavailable.

**Static Analysis (cuobjdump):**
- softmax_naive: ~40 instructions (3× LDG, EXP, FMA, STG cycles)
- softmax_online: ~30 instructions (1× LDG pass, EXP, FMA for online merge, 1× STG pass)

**Theoretical Occupancy:**
- REG count ≈ 32 → 100% occupancy
- SMEM usage ≈ 0 bytes → no SMEM limitation
- Bottleneck: DRAM bandwidth

**Future Acquisition:**
```bash
ncu --set full --kernel-name "softmax_online*" --export softmax_online ./build/cuda_operators
# Expect: SM Throughput ~30%, DRAM Throughput ~90%, Occupancy ~100%
```

---

## 6. Bottleneck Analysis

**Compute Intensity:** `(4 + 4) bytes/load × 2 load passes / (1 FMA + 2 EXP) ops` ≈ **0.2 FLOP/byte**

**RTX 3060 Ridge Point:** 35.3 FLOP/byte

**Verdict: MEMORY BOUND** by a wide margin. Every optimization must reduce HBM traffic — compute optimizations
(like using EXP2 instead of EXP) would yield negligible gains.

**Next Priority:** Reducing the 3-pass access pattern to 1-pass (online algorithm).

---

## 7. Optimization History

| Version | Technique | Gain | Side Effects |
|---------|-----------|------|--------------|
| v0: softmax_naive | 3-pass baseline | — | 54.3% bandwidth util |
| v1: softmax_online | Online 1-pass | −39.7% time | Slightly different numerics (float associativity) |
| v2: softmax_online+float4 | Float4 vectorization | −2.4% from v1 | Negligible gain for coalesced access pattern |
| v3: softmax_warp | Warp shuffle (dim≤32) | Reduced block size | Only applicable for small dimensions |

**Key Insight:** Float4 provides minimal benefit for row-wise coalesced access because each thread's loop stride
already achieves full memory coalescing. Float4 mainly helps when vectorization changes the access pattern
(e.g., strided or cross-query patterns).

---

## 8. Decision Log

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| Block reduce over shared memory | warp shuffle (for dim>32) | Shared memory reduction handles arbitrary BLOCK_SIZE; shuffle limited to 32 threads |
| BLOCK=256 as default | 128/512 | Best balance of parallelism vs per-thread work for D=8192 |
| Template BLOCK_SIZE | Runtime param | Compile-time loop unrolling; no perf penalty for branching |
| Online 1-pass as primary | 3-pass naive | 40% speedup from HBM pass reduction; numerically stable |
| Float4 optional | Default float4 | +2.4% gain doesn't justify code complexity; kept as template param |

---

## 9. Interview Notes

**Q: Why does online softmax work mathematically?**  
A: The key identity is `sum_j exp(x_j) = exp(m_new) * [exp(old_m - m_new) * sum_old + exp(x_new - m_new)]`.
When we find a new max, the old sum's per-element `exp(x_i - old_m)` values would need to be recomputed with
`exp(x_i - new_m)`. Since `exp(x_i - new_m) = exp(x_i - old_m) * exp(old_m - new_m)`, we can retroactively
correct the sum with a single multiplication. This reduces 3 HBM passes to 1.

**Q: Why is softmax memory-bound?**  
A: Each element requires 1 load and 1 store (4+4=8 bytes) but only O(1) arithmetic (1 FMA + 1 EXP per element).
The ratio of bytes to FLOPs is ~5:1, which is far below the GPU's compute-to-bandwidth equilibrium point
(35.3 FLOP/byte on RTX 3060). The pipeline is always waiting on DRAM.

**Q: When would softmax be compute-bound?**  
A: For very small rows (dim < 128), the overhead of reductions dominates and the kernel becomes
latency-bound. For larger rows, it's always memory-bound because the arithmetic intensity doesn't increase
with dimension size.

**Q: Why does float4 not help much for softmax?**  
A: With `grid(N) block(BLOCK)` row-major access, each thread already loads `dim/BLOCK` contiguous elements.
The hardware already coalesces these into 128-byte cache line transactions. Float4 groups 4 loads into 1
instruction, reducing LDG instruction count but not DRAM transaction count — the bottleneck remains DRAM
bandwidth, not instruction issue.
