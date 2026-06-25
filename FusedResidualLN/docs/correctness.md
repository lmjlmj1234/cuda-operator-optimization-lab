# Correctness Verification

## Method

The fused kernel output `y = LN(x + residual)` is compared against a two-step PyTorch reference:

```python
z = x + residual
y_ref = F.layer_norm(z, normalized_shape=[D], weight=gamma, bias=beta)
```

The reference operates on the pre-computed sum `x + residual`, which introduces minor floating-point associativity differences from the fused kernel computing `x[i] + residual[i]` directly in registers.

## Result

```
fused correctness: PASS (max diff ~1e-7)
```

## Error Analysis

The ~1e-7 max absolute difference is well within float32 rounding tolerance and is explained by:

- **Floating-point associativity:** `(x[i] + residual[i])` computed in fused kernel differs at the bit level from reading a previously-written `z[i] = x[i] + residual[i]` buffer.
- **No algorithmic errors:** mean, variance, and normalization all match.
- **Compiler optimizations:** Verified correct under `-O0`, `-O2`, and `-O3` after the nvcc register aliasing bug fix (see [kernel_analysis.md](kernel_analysis.md)).

## Reference

```
Tests: FusedResidualLN/tests/test_fused_residual_layernorm.py
```
