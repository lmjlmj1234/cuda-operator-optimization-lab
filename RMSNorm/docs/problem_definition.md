# Problem Definition

## What

Root Mean Square Normalization normalizes by the RMS of activations with no mean subtraction:

```
RMSNorm(x)_i = x_i / sqrt( (1/D) * sum(x_j^2) + eps ) * gamma_i
```

## Why

Simplified alternative to LayerNorm. Removes mean computation -- reduces one reduction and one accumulation step. Used in LLaMA, Mistral, and most modern open-source LLMs. The RMSNorm paper demonstrates no quality loss compared to LayerNorm across a range of NLP tasks.

## Role in Transformer

Same position as LayerNorm (applied after each sub-layer), but simpler and faster. Increasingly the default choice in new architectures (LLaMA family, Mistral, GPT-NeoX).

## I/O Shapes

- `N = 4096` rows (batch * sequence length)
- `D = 8192` columns (hidden dimension)
- Data type: float32
- Gamma weight: D elements (one per hidden dimension)

## Complexity

- **Computational:** O(N * D) reads + O(N * D) writes, but approximately 33% less computation than LayerNorm (no mean or subtraction pass).
- **Arithmetic:** One FMA per element for the square accumulation, one rsqrtf + FMA per element for normalization.

## Memory Pattern

- Row-wise: one block per row.
- Only one reduction (sum_sq) instead of two (sum + sum_sq for LayerNorm).
- Perfectly coalesced reads/writes within each row.
