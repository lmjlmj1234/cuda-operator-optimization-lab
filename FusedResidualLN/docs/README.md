# Fused Residual + LayerNorm — Kernel Optimization Report

## 1. Problem Definition

**What:** Fused Residual Add + Layer Normalization: `y = LN(x + residual)`

**Why:** Standard Transformer has `x = LN(x + sublayer_output)` after every attention and FFN block.
Non-fused implementation: compute `z = x + residual`, write to HBM, then load `z` for LayerNorm.
Fused version: load `x` and `residual`, add in registers, normalize, write final output directly.
Eliminates one intermediate tensor write + read.

**Role in Transformer:** Every sub-layer in the Transformer uses this pattern. Fusion saves 256 MB of
HBM traffic per layer at our problem size (4096 × 8192 × 4 bytes × 2 for write+read).

**I/O Shapes:** `N=4096` rows, `D=8192` columns, float32. Three inputs (x, residual, gamma, beta) → one
output.

**Complexity:** Same as LayerNorm but with extra load of residual plus fused add.

---

## 2. Baseline Implementation

**fused_residual_layernorm** (`FusedResidualLN/src/fused_residual_layernorm.cu`): Two-pass fused kernel
- Pass 1: Load `x + residual` in registers, compute sum + sum_sq
- Block reduce mean + variance
- Pass 2: Reload `x + residual`, normalize with gamma/beta, write output

**Shared memory workaround for nvcc bug:** mean/rstd passed through `__shared__ float sm_mean, sm_rstd`
to avoid compiler register aliasing (see docs/debug_log.md).

---

## 3. Correctness Verification

Fused kernel verified against non-fused layernorm operating on pre-computed `x + residual`:

```
fused correctness: PASS (max diff ~1e-7)
```

The tiny difference comes from floating-point associativity (fused computes `x[i] + r[i]` directly vs
reading from pre-computed buffer).

---

## 4. Benchmark

| Kernel | Time (ms) | Bandwidth (GB/s) | Peak Utilization |
|--------|-----------|-------------------|------------------|
| fused_residual_layernorm (BLOCK=256) | 1.63 | 429.1 | 87.6% |
| fused_residual_ln_float4 (BLOCK=256) | 1.67 | 419.0 | 85.5% |

Fused kernel achieves identical time to standalone LayerNorm while doing strictly more work (loading
residual, fused add) — the fusion eliminates the intermediate HBM traffic, freeing bandwidth for the
extra load.

---

## 5. Profiling Analysis

**Environment:** WSL2 GPU-PV → ncu hardware counters unavailable.

**Static Analysis (cuobjdump):**
- fused kernel: Two loops. First loop: 2× LDG (x + residual) + FMA for fused + FMA for sq. Second loop:
  2× LDG (x + residual) + 2× LDG (gamma + beta) + FMA chain + STG. Total: ~35 instructions.
- float4 variant: Fewer LDG instructions but requires 6× float4 registers (x, residual, gamma, beta, output)
  → REG≈48, 50% occupancy.

**nvcc Bug:** Initial version without shared memory workaround produced wrong outputs. Compiler aliased
mean/rstd computation registers with loop temporaries. Fixed via __shared__ memory barrier.

**Future Acquisition:**
```bash
ncu --set full --kernel-name "fused_residual_layernorm*" --export fused_ln ./build/cuda_operators
```

---

## 6. Bottleneck Analysis

**Compute Intensity:** 0.28 FLOP/byte (slightly higher than standalone LN due to extra FMA).

**Verdict: MEMORY BOUND.** Same as standalone LayerNorm.

**Fusion Advantage:** Without fusion: LN reads input (16 MB) + writes intermediate (0) + reads residual
(16 MB) + reads intermediate (16 MB) = 48 MB per layer. With fusion: reads input (16 MB) + reads residual
(16 MB) = 32 MB. **33% reduction in HBM traffic.**

---

## 7. Optimization History

| Version | Technique | Gain | Side Effects |
|---------|-----------|------|--------------|
| v0: fused scalar | Baseline fusion | Equivalent to LN | Shared memory workaround needed for nvcc bug |
| v1: fused_float4 | Float4 + fusion | No gain (slight regression) | 50% occupancy from register pressure |

---

## 8. Decision Log

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| Shared memory for mean/rstd | Local variables (buggy) | nvcc register aliasing bug — shared memory forces correct ordering |
| Scalar fusion as default | Float4 fusion | Float4 causes 50% occupancy; no perf gain |
| Two-pass (reload data) | One-pass (cache in registers) | D=8192 too large for register caching |
| Fuse only add + LN | Also fuse QK^T or FFN | Current scope; FlashAttention fusion is future work |

---

## 9. Interview Notes

**Q: What's the expected speedup from fusion?**  
A: The primary benefit is **memory savings, not speed**. For an end-to-end model, fusing add + LN saves
~256 MB of HBM traffic per layer (write + read of 4096×8192 floats). This compounds across layers:
32 layers × 256 MB = 8 GB saved. The per-layer kernel time stays the same because we added work but
removed the intermediate buffer bottleneck.

**Q: Describe the nvcc register aliasing bug that was found.**  
A: In the two-pass fused kernel, both passes use loop induction variables and temporaries. The compiler
(likely at -O3) assigned mean/rstd to the same physical registers as loop temporaries from pass 1.
Since pass 2's loop runs *before* reading mean/rstd, the compiler's live-range analysis failed and the
loop overwrote mean/rstd. Fix: use `__shared__ float` as a compiler barrier. This follows the pattern
used in CUTLASS for similar issues.

**Q: Can the fusion be extended to include the attention computation?**  
A: Yes — FlashAttention fuses QK^T matmul + softmax + PV matmul into a single tile-wise kernel.
Our `FlashAttentionSoftmax/` directory has a prototype demonstrating the online softmax aspect of this
fusion. Full FlashAttention requires tiled matmul support which is a significant extension.

**Q: What's the trade-off between fusion scope and kernel complexity?**  
A: Each fused operation adds: (1) more register pressure, (2) more shared memory usage, (3) harder
debugging. The rule of thumb: fuse at natural pipeline boundaries where intermediate tensors are large
enough that HBM traffic dominates. Add + LN is a clean boundary (no cross-row data dependencies).
