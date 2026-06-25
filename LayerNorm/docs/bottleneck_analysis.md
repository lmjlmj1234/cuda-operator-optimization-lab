# Bottleneck Analysis

## Compute Intensity

Input Read (4B) + Gamma/Beta Read (8B per thread for reduction) yields approximately **0.27 FLOP/byte**.

Breakdown per row (D=8192):
- Memory: Each row reads D floats (32 KB) from HBM, plus gamma (32 KB) and beta (32 KB)
- Compute: D adds + D multiply-adds (sum + sum_sq), 2 multiply + 1 add + 1 rsqrt (mean/var), D multiply-adds (normalize)
- Ratio: ~0.27 FLOP/byte

## Verdict

**MEMORY BOUND.** All three variants sit at 84-87% bandwidth utilization — there is ~13-16% headroom but all improvements in this regime require reducing HBM traffic.

### Evidence Summary

| Metric | Value | Indication |
|--------|-------|------------|
| Bandwidth utilization | 84-87% | Near DRAM bandwidth limit |
| Compute intensity | 0.27 FLOP/byte | Far below typical ridge point (~35 FLOP/byte for A100) |
| Kernel type | Row-wise streaming reduction | Inherently memory-bound for large D |

## Next Priority

**Operator fusion** — combine with preceding residual add to eliminate intermediate tensor I/O. See `FusedResidualLN/` for prototype.
