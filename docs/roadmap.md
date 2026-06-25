# Roadmap

> 当前状态和后续开发方向。

---

## 已完成

### Shared Infrastructure
- [x] Engineering_Convention.md — 9-point delivery standard
- [x] Performance_Playbook.md — 优化技术目录
- [x] Optimization_Methodology.md — 优化方法论
- [x] Benchmark_Specification.md — 基准测试协议
- [x] Profiling_Guide.md — Profiling + Performance Analysis Workflow
- [x] Dump_and_Correctness_Guide.md — Dump 策略和正确性验证
- [x] Documentation_Template.md — 文档模板

### Softmax
- [x] Naive 3-pass baseline
- [x] Online 1-pass (template with VEC_SIZE)
- [x] Warp shuffle variant (dim ≤ 32)
- [x] Float4 vectorization (minor gain)
- [x] Correctness + Benchmark + 9-point docs

### LayerNorm
- [x] sum_sq two-moment
- [x] Float4 vectorization (rejected — regression)
- [x] Welford online variance
- [x] Correctness + Benchmark + 9-point docs

### RMSNorm
- [x] Scalar version
- [x] Float4 vectorization (no gain)
- [x] Correctness + Benchmark + 9-point docs

### Fused Residual + LayerNorm
- [x] Scalar fusion
- [x] Float4 fusion (minor regression)
- [x] Shared memory workaround for nvcc register aliasing bug
- [x] Correctness + Benchmark + 9-point docs

### FlashAttention Softmax
- [x] Tile-based online softmax pattern (prototype)
- [ ] Correctness verification
- [ ] Benchmark

---

## 短期

### Profiling 数据采集
- [ ] 在原生 Linux 上用 ncu `--set full` 采集所有 kernel 的硬件指标
- [ ] 填入每个 kernel 的 `docs/nsight_analysis.md`
- [ ] 构建实际 roofline 图（非理论值）
- [ ] 确认 memory-bound 判定

### Persistent Kernel
- [ ] 实现 persistent kernel 版本的 reduction（消除 launch overhead）
- [ ] 适用于 small N / large D 场景

### Cross-Kernel 融合
- [ ] FlashAttention full implementation（tile matmul + online softmax + PV）
- [ ] FFN 激活融合（SiLU / GELU fuse into GEMM epilogue）

---

## 中期

### 新增算子
- [ ] GEGLU / SwiGLU（LLaMA-style FFN activation）
- [ ] RoPE (Rotary Position Embedding)
- [ ] Cross-entropy loss with reduction
- [ ] Top-K sampling / Top-P sampling

### 自动化
- [ ] CI pipeline: build + correctness + benchmark regression
- [ ] 自动 ncu profile on push（需要原生 Linux runner）
- [ ] 自动 roofline 图生成

### FP16 / BF16 支持
- [ ] 将关键 kernel 迁移到 half / bfloat16
- [ ] 对比 float32 vs float16 的精度和速度

---

## 长期

### Tensor Core 集成
- [ ] 使用 WMMA API 实现 GEMM kernel
- [ ] TF32 精度探索

### 端到端模型推理
- [ ] Transformer layer 级融合（Attention + FFN + Residual + LN 全部融合）
- [ ] Mini model inference pipeline

### 职业级产出
- [ ] 每个优化步骤都有对应的 Medium/Blog 文章
- [ ] 面试题 Quiz 集合
- [ ] 与 CUTLASS / FlashAttention 源码的 diff 分析
