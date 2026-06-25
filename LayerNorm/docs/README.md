# LayerNorm — Kernel Optimization Report

## 1. Problem Definition

**What:** Layer Normalization normalizes activations across the hidden dimension:  
`LN(x)_i = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i`

**Why:** Stabilizes training in Transformer models by reducing internal covariate shift. Applied after every
sub-layer (attention, FFN) in the standard Transformer architecture.

**Role in Transformer:** Essential normalization after each attention and FFN block. Per-layer statistics
(mean, var) make it different from BatchNorm — no dependency on batch dimension, works with batch size 1.

**I/O Shapes:** `N=4096` rows, `D=8192` columns, float32, plus gamma/beta weight vectors (D elements each).

**Complexity:** O(N·D) reads + O(N·D) writes. Must compute mean + variance through row-wide reduction.

**Memory Pattern:** Row-wise with gamma/beta element-wise broadcast.

---

## 2. Baseline Implementation

**layernorm_sum_sq** (`LayerNorm/src/layernorm_sum_sq.cu`): Classic two-moment method
- `mean = sum(x_i) / D`
- `var = sum(x_i²) / D - mean²`
- One pass computing sum and sum of squares simultaneously

Numeric risk: catastrophic cancellation when var ≪ mean² (variance is the difference of large numbers).

---

## 3. Correctness Verification

All three variants verified against PyTorch `F.layer_norm` (via `tests/test_correctness.py`).

| Variant | Max Diff vs PyTorch | Status |
|---------|---------------------|--------|
| layernorm_sum_sq | ~1e-6 | PASS |
| layernorm_float4 | ~1e-6 | PASS |
| layernorm_welford | ~1e-6 | PASS |

Float4 variant cross-validated against sum_sq output — max diff < 1e-6.

---

## 4. Benchmark

| Kernel | Time (ms) | Bandwidth (GB/s) | Peak Utilization |
|--------|-----------|-------------------|------------------|
| layernorm_sum_sq (BLOCK=256) | 1.63 | 424.6 | 86.7% |
| layernorm_float4 (BLOCK=256) | 1.65 | 419.5 | 85.6% |
| layernorm_welford (BLOCK=256) | 1.68 | 412.0 | 84.1% |

**Key finding:** Float4 does NOT help — it's ~1.6% slower. All three variants are bandwidth-bound at 84-87% utilization.

---

## 5. Profiling Analysis

**Environment:** WSL2 GPU-PV → ncu hardware counters unavailable.

**Static Analysis (cuobjdump):**
- layernorm_sum_sq: sum + sum_sq fused in single loop, then 2× MUL + ADD + RSQRT for mean/var, then
  normalize loop with 2× LDG (x, gamma), 1× STG
- layernorm_float4: 4× fewer LDG instructions but higher register pressure (need 4× input + 4× gamma + 4× output)

**Theoretical Occupancy:**
- sum_sq: REG≈24 → 100%
- float4: REG≈32 → 100%
- welford: REG≈40 → 83%

**Future Acquisition:**
```bash
ncu --set full --kernel-name "layernorm_sum_sq*" --export layernorm_sum_sq ./build/cuda_operators
```

---

## 6. Bottleneck Analysis

**Compute Intensity:** Input Read (4B) + Gamma/Beta Read (8B per thread for reduction) ≈ 0.27 FLOP/byte

**Verdict: MEMORY BOUND.** All three variants sit at 84-87% bandwidth utilization — there's ~13-16% headroom
but all improvements in this regime require reducing HBM traffic.

**Next Priority:** Operator fusion (combine with preceding residual add to eliminate intermediate tensor I/O).

---

## 7. Optimization History

| Version | Technique | Gain | Side Effects |
|---------|-----------|------|--------------|
| v0: layernorm_sum_sq | Two-moment baseline | — | Numeric risk: catastrophic cancellation |
| v1: layernorm_float4 | Float4 vectorization | −1.6% regression | Higher register pressure |
| v2: layernorm_welford | Welford online variance | −0.6% vs v0 | Numerically stable; more registers |

**Key Insight:** Welford is the right choice for numerical stability even though it's slightly slower.
Float4 was rejected because coalesced row access already hits full bandwidth — vectorization only adds
register pressure.

---

## 8. Decision Log

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| sum_sq as default | Welford | Simpler code; switch to Welford if numerical issues appear |
| Float4 removed from default | Float4 always | 1.6% regression; higher reg pressure |
| Welford as stable option | None | Needed for large-dim/low-var regimes |
| gamma/beta in global memory | Shared mem preload | D is too large for shared mem caching |

---

## 9. Interview Notes

**Q: Why is LayerNorm memory-bound but the float4 version is slower?**  
A: Row-wise reduction is already fully coalesced — each thread loads a contiguous block of elements.
Float4 reduces LDG instruction count but the bottleneck is DRAM bandwidth, not instruction issue rate.
Worse, float4 increases register pressure (need 4× input + gamma + beta registers), which can reduce
occupancy and hide latency less effectively.

**Q: When does catastrophic cancellation happen in sum_sq?**  
A: When the mean is much larger than the standard deviation. For example, x = [1000.0, 1000.001]:
sum = 2000.001 → mean = 1000.0005, sum_sq = 2,000,001, var = 1,000,000.5 - 1,000,001.00000025 = 0.49999975.
Subtracting two ~1e6 numbers to get ~0.5 loses ~7 digits of precision. Welford avoids this entirely.

**Q: What is the fused residual + layernorm optimization?**  
A: Instead of writing `x + residual` to an intermediate tensor then reading it back for LN, fuse: load `x`
and `residual`, add in registers, normalize, write final output. Eliminates one HBM write+read of N×D floats
(256 MB at our problem size). See `FusedResidualLN/docs/README.md`.

**Q: How would you optimize further if this were compute-bound?**  
A: Use warp shuffle instead of shared memory for the reduction (reduces latency), use __expf() intrinsic
instead of expf() (lower precision, 2× throughput), preload gamma/beta into registers if dim is small.
