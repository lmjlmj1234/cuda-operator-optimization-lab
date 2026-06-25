# CUDA Operator Optimization Lab

A hands-on engineering lab for designing, verifying, profiling, and optimizing CUDA kernels commonly used in Transformer-based LLM inference. Each operator follows the [9-Point Kernel Engineering Standard](shared_docs/Engineering_Convention.md).

## Operators

| Operator | Variants | Key Optimization |
|---|---|---|
| [Softmax](Softmax/) | 3 | Online 1-pass, warp shuffle |
| [LayerNorm](LayerNorm/) | 3 | Welford online variance |
| [RMSNorm](RMSNorm/) | 2 | Simplified LN (LLaMA-style) |
| [Fused Residual + LN](FusedResidualLN/) | 2 | Operator fusion (→−33% HBM traffic) |
| [FlashAttention Softmax](FlashAttentionSoftmax/) | 1 | Tile-based online softmax (prototype) |

## Project Structure

```
├── shared_docs/              # Engineering conventions & methodology
├── Softmax/ LayerNorm/ ...   # Per-operator module (src/ benchmark/ tests/ docs/ reports/)
├── include/                  # Shared CUDA headers (utils.cuh)
├── tests/                    # Integration tests (pytest)
├── reports/                  # Profiling output (ncu, nsys, benchmark)
└── docs/                     # Cross-cutting docs (roadmap, debug log)
```

## Quick Start

```bash
cmake -S . -B build
cmake --build build
./build/cuda_operators
python3 -m pytest
```

## Environment

- GPU: NVIDIA GeForce RTX 3060 (Ampere, CUDA 12.4)
- Platform: WSL2 (GPU-PV)
- Profiling: cudaEvent, nsys (ncu requires native Linux for hardware counters)

## Goals

- Apply professional AI Infra engineering workflows from CUTLASS / FlashAttention / vLLM
- Quantify every optimization: "faster" is not acceptable — only "−39.7% kernel time"
- Memory-bound analysis first: count HBM transactions, compute intensity vs ridge point
- Full documentation per kernel: problem definition, correctness, benchmark, profiling, bottleneck analysis, decision log, interview notes

See [docs/project_overview.md](docs/project_overview.md) and [docs/roadmap.md](docs/roadmap.md) for more.
