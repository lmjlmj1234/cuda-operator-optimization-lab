# Problem Definition — Fused Residual Add + LayerNorm

## What

Fused Residual Add + Layer Normalization: `y = LN(x + residual)`

Single kernel that loads `x` and `residual`, adds them in registers, normalizes, and writes the final output -- all without an intermediate HBM round-trip.

## Why

Standard Transformer blocks contain the pattern `x = LN(x + sublayer_output)` after every attention and FFN sub-layer. A non-fused implementation must:

1. Compute `z = x + residual` and write `z` to HBM
2. Load `z` back from HBM for LayerNorm
3. Write the final normalized output

The fused version eliminates step 1's write and step 2's read, saving **two HBM transactions** per element per layer.

## Role in Transformer

Every sub-layer in the Transformer uses this pattern. For our problem size:

- Shape: `N=4096` rows, `D=8192` columns
- Data type: float32 (4 bytes)
- Per-layer savings: 4096 x 8192 x 4 bytes x 2 (write + read) = **256 MB of HBM traffic**
- Across 32 layers: 32 x 256 MB = **8 GB saved**

## I/O Shapes

| Tensor | Shape | Size (MB) |
|--------|-------|-----------|
| x (input) | N x D | 128 |
| residual | N x D | 128 |
| gamma | D | 0.03 |
| beta | D | 0.03 |
| y (output) | N x D | 128 |

## Computational Complexity

Identical to standalone LayerNorm plus one extra FMA per element for the fused add. The dominant cost is HBM bandwidth, not arithmetic.
