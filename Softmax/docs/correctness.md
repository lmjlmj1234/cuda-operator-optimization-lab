# Correctness Verification

All variants verified against `reference_softmax()` (host-side 2-pass softmax):

```
softmax correctness: PASS
```

**Cross-validation:** `softmax_online<float4>` vs `softmax_naive` on row 2:
- Max diff: ~1e-7 -- within float32 precision
- Only difference is floating-point summation order (deterministic vs associative online merge)
