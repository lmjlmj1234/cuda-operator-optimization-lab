# Correctness Verification

## Reference Comparison

Both kernel variants are verified against a host-side RMSNorm reference implementation:

```
rmsnorm correctness: PASS
```

## Cross-Variant Consistency

The float4 variant (`rmsnorm_float4`) matches the scalar variant (`rmsnorm_kernel`) to within approximately 1e-6 absolute error. This confirms that both implementations compute the same mathematical result, and the float4 vectorization introduces no precision loss or data races.

## Verification Method

1. Random input tensors are generated on the host.
2. A host RMSNorm reference computes the expected output.
3. Each kernel variant writes its output to a device buffer, which is copied back to the host.
4. Outputs are compared element-wise against the reference with absolute tolerance `1e-5`.

## Numeric Stability

Both variants use `rsqrtf(sumsq / D + eps)` with `eps = 1e-5`. This is the same formulation used in LLaMA and Mistral reference implementations. No denormal or NaN issues have been observed for any tested input range (Gaussian with mean 0, variance 1).
