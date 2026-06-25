# Dump and Correctness Verification Guide

> 本文档定义仓库内所有 CUDA Kernel 的 .bin dump 策略和正确性验证规范。

---

## 1. 为什么需要 Dump

CUDA Kernel 在 GPU 上运行，无法直接访问其计算结果。Dump 策略定义：

- 什么时候需要将 GPU buffer 保存到磁盘？
- 保存的数据用于什么验证？
- 如何保证不同 kernel 之间的 cross-validation？

**每个 .bin dump 必须有明确的用途。** 不允许"先 dump 再说"。

---

## 2. Dump 策略

### 2.1 通用规则

1. **每个 Kernel 变体必须至少 dump 一个输出**，用于验证正确性
2. **Input 数据只 dump 一次**（seed 固定），所有变体共享同一个 input dump
3. **Dump 在 cudaDeviceSynchronize 之后**，确保 GPU kernel 已完成
4. **Dump 文件命名规则**：`{operator}_{variant}.bin`
5. **避免不必要的 dump**——每次 dump 都是 128 MB (N=4096, D=8192)

### 2.2 文件命名

```
input_data.bin               # 公共输入 (所有算子共享)
gamma.bin                    # LayerNorm/RMSNorm weight
beta.bin                     # LayerNorm bias
residual.bin                 # Fused residual add 输入

softmax_cuda_out.bin         # Softmax online+float4 输出 (primary variant)
softmax_naive_out.bin        # Softmax naive 输出 (baseline cross-validation)

layernorm_cuda_out.bin       # LayerNorm sum_sq 输出
layernorm_welford_out.bin    # LayerNorm welford 输出 (variant cross-validation)
layernorm_float4_out.bin     # LayerNorm float4 输出

rmsnorm_cuda_out.bin         # RMSNorm scalar 输出
rmsnorm_float4_out.bin       # RMSNorm float4 输出

fused_layernorm_cuda_out.bin # Fused Residual+LN scalar 输出
fused_ln_float4_out.bin      # Fused Residual+LN float4 输出
```

### 2.3 哪些文件需要 commit？

单个 .bin 文件可达 128 MB，**不应当 commit 到 git**（已在 .gitignore 中忽略）。

生成方式：执行 `./build/cuda_operators` 或 `python -m pytest tests/` 时自动生成。

---

## 3. 正确性验证规范

### 3.1 验证层级

```
Level 1: Unit Test (per-operator)
  - 每个算子的每个变体 vs PyTorch reference
  - 不同变体之间 cross-validation
  - pytest: Softmax/tests/test_softmax.py

Level 2: Integration Test (top-level)
  - 所有算子联合验证
  - Cross-operator alignment (如 naives vs online 的一致性)
  - pytest: tests/test_cuda_operators.py

Level 3: Host Reference (in C++)
  - main.cu 内置的 reference_softmax() + verify_result()
  - 构建阶段快速验证：编译后立即检查
  - 与 PyTorch 验证互为补充
```

### 3.2 验证标准

| Kernel | 验证方法 | 误差阈值 |
|--------|---------|---------|
| Softmax | vs `torch.softmax` | max diff < 1e-3 |
| LayerNorm (sum_sq) | vs `torch.layer_norm` | max diff < 0.05 |
| LayerNorm (welford) | vs `torch.layer_norm` | max diff < 0.05 |
| LayerNorm (float4) | vs `torch.layer_norm` | max diff < 0.05 |
| RMSNorm | vs `x / sqrt(mean(x²)) * gamma` | max diff < 1e-4 |
| Fused LN | vs `torch.layer_norm(x + residual)` | max diff < 0.05 |

**为什么 LayerNorm 阈值为 0.05？**  
sum_sq 方法存在 catastrophic cancellation 风险。D=8192 的累加误差可达 ~5%。  
这不是实现 bug，而是 float32 精度的固有局限。Welford 可缓解此问题。

### 3.3 Cross-Validation 矩阵

```
                 naive  online  welford  float4  fused  fused_f4
softmax_naive     —      ✓  
softmax_online    ✓      —     (n/a)    ✓
layernorm_sum_sq  —      ✓     ✓       ✓      (n/a)  (n/a)
layernorm_welford ✓      —     ✓
layernorm_float4  ✓      ✓     —
rmsnorm_scalar    —                          ✓
rmsnorm_float4    ✓                          —
fused_ln_scalar   —                                  ✓
fused_ln_f4       ✓                                  —
```

### 3.4 版本间 Regression 测试

每次修改 kernel 后：
1. 重新运行 `./build/cuda_operators` 生成新 output
2. 运行 `python -m pytest tests/` 验证所有正确性
3. 保存旧 output 与新 output 对比，确认 regression 范围

---

## 4. Python 测试框架

### 4.1 conftest.py

根目录 `conftest.py` 提供 session 级别的 fixture：

```python
@pytest.fixture(scope="session")
def bin_dir():
    """build/ 目录，存放所有 .bin 文件"""
    return BUILD_DIR  # -> /project/build/

@pytest.fixture(scope="session")
def data_dir():
    """项目根目录"""
    return PROJECT_ROOT  # -> /project/
```

### 4.2 测试文件组织

```
tests/test_cuda_operators.py        # 顶层集成测试（11 tests）
Softmax/tests/test_softmax.py       # Softmax 专用测试
LayerNorm/tests/test_layernorm.py   # LayerNorm 专用测试
RMSNorm/tests/test_rmsnorm.py       # RMSNorm 专用测试
FusedResidualLN/tests/test_fused_layernorm.py  # Fused LN 专用测试
```

### 4.3 运行方式

```bash
# 全部测试
python -m pytest

# 单算子测试
python -m pytest Softmax/tests/

# 单测试
python -m pytest Softmax/tests/test_softmax.py::test_softmax_online_vs_torch -v
```
