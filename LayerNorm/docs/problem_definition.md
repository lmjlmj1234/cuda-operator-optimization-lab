# Problem Definition

## What

Layer Normalization normalizes activations across the hidden dimension:

```
LN(x)_i = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
```

## Why

Stabilizes training in Transformer models by reducing internal covariate shift. Applied after every sub-layer (attention, FFN) in the standard Transformer architecture.

## Role in Transformer

Essential normalization after each attention and FFN block. Per-layer statistics (mean, var) make it different from BatchNorm — no dependency on batch dimension, works with batch size 1.

## I/O Shapes

- `N=4096` rows, `D=8192` columns, float32
- gamma/beta weight vectors (D elements each)

## Complexity

O(N-D) reads + O(N-D) writes. Must compute mean + variance through row-wide reduction.

## Memory Pattern

Row-wise with gamma/beta element-wise broadcast.
