# Bottleneck Analysis

## Compute Intensity

```
Arithmetic intensity = total FLOPs / total bytes transferred
                     = 3 * N * D  /  3 * N * D * 4
                     = 0.25 FLOP/byte
```

Each element requires:
- 1 multiply (x^2 during reduction)
- 1 multiply-add (scale during normalization)
- 1 rsqrtf

Total: approximately 3 FLOP per element (the rsqrtf is counted as a single special-function operation).

Total bytes: 3 * N * D * 4 (read x + gamma, write y).

**Compute intensity = 0.25 FLOP/byte**, far below the roofline ridge point for any modern GPU (typically 8-16 FLOP/byte for compute 7.0+).

## Verdict

**MEMORY BOUND.** The kernel spends the vast majority of time waiting on DRAM transactions, not executing arithmetic instructions.

## Why Float4 Does Not Help

Float4 vectorization helps in two scenarios:
1. **Instruction-bound kernels:** Reducing LDG instruction count improves throughput when the issue bottleneck is the scheduler.
2. **Non-coalesced access patterns:** Larger transactions improve bus utilization when individual threads access scattered addresses.

RMSNorm's row-wise reduction is perfectly coalesced: 32 threads * 4 bytes/thread = 128 bytes (one cache line). Float4 changes nothing about the DRAM transaction pattern -- it only reduces the number of LDG instructions, which is irrelevant when the pipeline is already stalled on memory.

## Implications

- The only way to speed up this kernel is to **reduce HBM traffic**.
- Possible approaches: kernel fusion (e.g., fuse with the preceding matmul or the following element-wise operation), or exploiting sparsity to skip zero elements.
- Within a single isolated normalization kernel, the current implementation is near-optimal.
