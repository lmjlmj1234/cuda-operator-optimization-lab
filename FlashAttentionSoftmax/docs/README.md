# FlashAttention Softmax — Prototype

> **状态：Prototype / unverified**
> 这不是一个完整的实现。仅供参考和后续开发。

---

## 完成内容

- `flash_attn_tile_softmax` kernel：演示 FlashAttention 风格的 tile 式在线 softmax 核心逻辑
  - tile-based QK^T matmul 模拟（简化版）
  - 在线 softmax 更新 + exp 修正因子
  - tile-based V 累加
- `online_softmax_demo` kernel：独立的在线 softmax 算法演示

## 未完成

- 未验证正确性（no PyTorch `scaled_dot_product_attention` comparison）
- 未 benchmark（无性能数据）
- Tile 大小写死为 `TILE_D=64`，未泛化
- 未处理 `d < TILE_D` 或 `d > TILE_D` 的情况
- 无 block reduce 实现（当前假设一个 block 内一个 query）
- 未处理 mask / causal attention
- 未处理 dropout

## 后续计划

1. 实现完整的 tile-based QK^T matmul（shared memory + block reduce）
2. 验证正确性 vs PyTorch `F.scaled_dot_product_attention`
3. Benchmark vs 标准 attention 实现
4. 验证 N=4096, d=64/128 场景的加速比
5. 合并到完整的 FlashAttention kernel

## 核心参考

- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- `Softmax/src/softmax_online.cu` — 在线 softmax 算法的完整实现
- `Softmax/docs/` — softmax kernel 的完整分析文档
