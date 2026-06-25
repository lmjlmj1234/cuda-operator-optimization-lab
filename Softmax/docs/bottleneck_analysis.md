# Bottleneck Analysis

## Compute Intensity

`(4 + 4) bytes/load x 2 load passes / (1 FMA + 2 EXP) ops` right-arrow **0.2 FLOP/byte**

## RTX 3060 Ridge Point

35.3 FLOP/byte

## Verdict

**MEMORY BOUND** by a wide margin. Every optimization must reduce HBM traffic -- compute optimizations (like using EXP2 instead of EXP) would yield negligible gains.

**Next Priority:** Reducing the 3-pass access pattern to 1-pass (online algorithm).
