# CUDA Operator Optimization Lab — AI Infra Engineering Standards

## Core Principle

Every CUDA kernel in this project MUST follow the **9-Point Kernel Engineering Standard** defined in
[shared_docs/Engineering_Convention.md](shared_docs/Engineering_Convention.md).

The goal is not just "kernel runs." The goal is understanding how NVIDIA engineers at CUTLASS, FlashAttention,
and vLLM design, verify, analyze, and optimize CUDA kernels — building a professional AI Infra engineering
mindset and workflow.

## Project Structure

```
cuda-operator-optimization-lab/
├── shared_docs/               # Cross-cutting engineering knowledge (MUST READ)
│   ├── Engineering_Convention.md         # The 9-point standard
│   ├── Performance_Playbook.md           # Optimization techniques catalog
│   ├── Optimization_Methodology.md       # Step-by-step methodology
│   ├── Benchmark_Specification.md        # Standardized benchmark protocol
│   ├── Profiling_Guide.md               # Performance Analysis Workflow + nsight
│   ├── Dump_and_Correctness_Guide.md     # Dump strategy & correctness verification
│   └── Documentation_Template.md         # Per-kernel doc template
│
├── Softmax/ LayerNorm/ RMSNorm/ FusedResidualLN/ FlashAttentionSoftmax/
│   ├── README.md              # Module overview
│   ├── src/                   # Kernel implementations
│   ├── benchmark/             # config.json + run.sh
│   ├── tests/                 # Correctness tests
│   ├── docs/                  # 9 individual analysis docs
│   └── reports/               # Experiment summaries
│
├── tests/                     # Top-level integration tests
├── include/                   # Shared CUDA headers (utils.cuh)
├── reports/                   # Profiling output (ncu/, nsys/, benchmark/)
├── docs/                      # Project-level docs
└── build/                     # Build output (gitignored)
```

## When Starting a New Kernel

1. Read `shared_docs/Engineering_Convention.md` — understand the 9-point standard
2. Read `shared_docs/Optimization_Methodology.md` — follow the step-by-step process
3. Read `shared_docs/Performance_Playbook.md` — review applicable techniques
4. Read `shared_docs/Documentation_Template.md` — understand required doc layout
5. Create the operator directory with `src/`, `benchmark/`, `tests/`, `docs/`, `reports/`
6. Follow the 9 points in order — do not skip to optimization
7. Every commit must answer: why this optimization, why not others, quantified gains, side effects, what's next
8. Every experiment must save: environment, kernel config, compiler flags, nsight/SASS, reference output, regression results

## Critical Rules

1. **Baseline first.** Always write the simplest correct implementation before optimizing.
2. **No empty sections.** If profiling data is unavailable (WSL2 GPU-PV), document why + how to get it later.
3. **Proactive gap-filling.** If experimental data is missing, note it and suggest how to fill it.
4. **Quantify everything.** "Faster" is not acceptable. Use: "−39.7% kernel time (2.17→1.31 ms)."
5. **Memory access is king.** Count HBM transactions. Memory-bound kernels dominate deep learning.
6. **Performance Counter is Evidence, not Conclusion.** Follow the Performance Analysis Workflow in
   [Profiling_Guide.md](shared_docs/Profiling_Guide.md) — never draw conclusions from a single metric.
