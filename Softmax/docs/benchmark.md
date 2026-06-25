# Benchmark

| Kernel | Time (ms) | Bandwidth (GB/s) | Peak Utilization |
|--------|-----------|-------------------|------------------|
| softmax_naive (BLOCK=128) | 2.17 | 266.1 | 54.3% |
| softmax_online (BLOCK=256) | 1.31 | 440.8 | 89.9% |
| softmax_online+float4 (BLOCK=256) | 1.27 | 454.7 | 92.8% |
| softmax_warp (dim=32) | 0.007 | -- | -- |

**40% reduction** in kernel time from naive to online. At 92.8% bandwidth utilization, the online variant is near the RTX 3060 peak (490 GB/s).
