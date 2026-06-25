# Correctness Verification

All three variants verified against PyTorch `F.layer_norm` (via `tests/test_layernorm.py`).

| Variant | Max Diff vs PyTorch | Status |
|---------|---------------------|--------|
| layernorm_sum_sq | ~1e-6 | PASS |
| layernorm_float4 | ~1e-6 | PASS |
| layernorm_welford | ~1e-6 | PASS |

### Cross-Validation Matrix

```
               sum_sq  float4  welford
sum_sq          —      ✓       ✓
float4          ✓      —       ✓
welford         ✓      ✓       —
```

All cross-variant max diffs < 1e-6.

### Edge Cases

尚未编写专用的 edge case 测试。标准 input (N=4096, D=8192, [0,1) uniform) 已通过所有 variant。

需要补充的 edge cases：
- All-identical rows (x = 1.0 for all elements) — 验证 var ≈ 0 时的数值稳定性
- Zero input (x = 0.0) — 验证 mean=0, var=0 输出 gamma
- Extreme values (large positive/negative) — 验证 exp/sqrt 不溢出
- Single-element row (D=1)

这些 case 应在 `tests/test_layernorm.py` 中补充。

## Test Setup

- Reference: `torch.layer_norm(x, [D], weight=gamma, bias=beta, eps=1e-5)`
- Inputs are saved to `.bin` files by the CUDA benchmark harness
- Comparison uses `numpy` binary load + `torch` reference computation
- Cross-validation ensures sum_sq, float4, and welford all agree within tolerance
