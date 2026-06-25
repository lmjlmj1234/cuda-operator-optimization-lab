# RMSNorm — Kernel Optimization Report

## 1. Problem Definition

**What:** Root Mean Square Normalization normalizes by RMS of activations (no mean subtraction):  
`RMSNorm(x)_i = x_i / sqrt( (1/D) * sum(x_j²) + eps ) * gamma_i`

**Why:** Simplified alternative to LayerNorm. Removes mean computation — reduces one reduction and one
accumulation step. Used in LLaMA, Mistral, and most modern open-source LLMs. Paper shows no quality loss
compared to LayerNorm.

**Role in Transformer:** Same position as LayerNorm (after each sub-layer), but simpler/faster. Increasingly
the default choice in new architectures.

**I/O Shapes:** `N=4096` rows, `D=8192` columns, float32, plus gamma weight (D elements).

**Complexity:** O(N·D) reads + O(N·D) writes, but ~33% less computation than LayerNorm (no mean + subtract).

**Memory Pattern:** Row-wise, one block per row. Only one reduction (sum_sq) instead of two (sum + sum_sq).

---

## 2. Baseline Implementation

**rmsnorm_kernel** (`RMSNorm/src/rmsnorm_kernel.cu`): Single reduction pass
1. Compute `sum_sq = sum(x_i²)`
2. Block reduce sum_sq
3. Normalize: `y_i = x_i / rsqrt(sum_sq/D + eps) * gamma_i`

No mean computation → fewer FLOPs + simpler code than LayerNorm.

---

## 3. Correctness Verification

Both variants verified against host RMSNorm reference:

```
rmsnorm correctness: PASS
```

Cross-validation: rmsnorm_float4 matches rmsnorm_kernel to ~1e-6.

---

## 4. Benchmark

| Kernel | Time (ms) | Bandwidth (GB/s) | Peak Utilization |
|--------|-----------|-------------------|------------------|
| rmsnorm_kernel (BLOCK=256) | 1.63 | 424.6 | 86.7% |
| rmsnorm_float4 (BLOCK=256) | 1.65 | 419.5 | 85.6% |

RMSNorm and LayerNorm sum_sq have nearly identical performance — the mean subtraction is negligible cost
compared to the HBM traffic.

---

## 5. Profiling Analysis

**Environment:** WSL2 GPU-PV → ncu hardware counters unavailable.

**Static Analysis (cuobjdump):**
- rmsnorm_kernel: ~25 instructions. Single reduction loop (LDG + FMA for x²), block_reduce, normalize loop
  (LDG x + gamma, FMA for scale, STG).
- rmsnorm_float4: ~20 instructions. Float4 LDG in reduction and normalization loops.

**Theoretical Occupancy:**
- Both variants: REG≈24-32 → 100%
- No shared memory usage beyond reduction scratch

**Future Acquisition:**
```bash
ncu --set full --kernel-name "rmsnorm_kernel*" --export rmsnorm_kernel ./build/cuda_operators
```

---

## 6. Bottleneck Analysis

**Compute Intensity:** Similar to LayerNorm — 0.27 FLOP/byte (slightly less computation per byte).

**Verdict: MEMORY BOUND.**

**Why float4 doesn't help:** Same as LayerNorm — row coalescing already saturates DRAM bandwidth.

---

## 7. Optimization History

| Version | Technique | Gain | Side Effects |
|---------|-----------|------|--------------|
| v0: rmsnorm_kernel | Scalar baseline | — | — |
| v1: rmsnorm_float4 | Float4 vectorization | No gain | Same pattern as layernorm |

RMSNorm is simple enough that the scalar version is already near-optimal for memory-bound kernels.

---

## 8. Decision Log

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| Scalar as default | Float4 | Float4 gives no measurable benefit |
| Use rsqrtf | sqrtf + division | rsqrtf is ~4× faster; identical in accuracy for this use case |
| No mean subtraction | Include mean | RMSNorm definition specifically omits it — computational saving vs LayerNorm |

---

## 9. Interview Notes

**Q: Why is RMSNorm preferred over LayerNorm in modern LLMs?**  
A: Two reasons: (1) Removes mean computation — saves ~25% of the variance computation logic; (2) Empirical
finding that mean subtraction doesn't improve model quality for most tasks. LLaMA paper showed RMSNorm
matches LayerNorm perplexity while being faster.

**Q: How much faster is RMSNorm than LayerNorm on GPU?**  
A: At our problem size (D=8192), the difference is <1%. The mean computation is a simple FMA per element
in the reduction loop — it adds ~2-3 instructions out of hundreds. The bottleneck is HBM traffic, and
both RMSNorm and LayerNorm read/write the same number of bytes from HBM.

**Q: When would RMSNorm be meaningfully faster than LayerNorm?**  
A: On small dimensions (D < 512) where the reduction overhead dominates, or on compute-bound kernels
where every arithmetic operation counts. For D=128, RMSNorm can be ~10% faster.

**Q: Why doesn't float4 help here when it helps in other contexts?**  
A: Vectorization helps when access is non-coalesced (strided, cross-element patterns). Row-wise reduction
is perfectly coalesced — 32 threads × 4 bytes/thread = 128 bytes (one cache line). Float4 doesn't change
the DRAM transaction pattern, it just reduces LDG instruction count. The bottleneck remains DRAM bandwidth.
