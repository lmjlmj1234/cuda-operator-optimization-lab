# Kernel Analysis

## rmsnorm_kernel (Scalar)

### Instruction Flow (from cuobjdump disassembly)

Approximately 25 instructions. The kernel consists of three phases:

1. **Reduction loop** (compute sum_sq):
   - `LDG` (load x[i])
   - `FMA` (x[i] * x[i] + accumulator)
   - Loop over D elements

2. **Block reduction**:
   - Shared memory shuffle + warp-level reduction
   - Compute `rms_denom = rsqrtf(sumsq / D + eps)`

3. **Normalization loop**:
   - `LDG` (load x[i] + gamma[i])
   - `FMA` (scale: x[i] * rms_denom * gamma[i])
   - `STG` (store y[i])

### Register Usage

Approximately 24-32 registers per thread, yielding 100% theoretical occupancy on modern GPUs (compute capability 7.0+).

### Shared Memory

Minimal usage: only the block reduction scratch buffer (one float per warp).

## rmsnorm_float4 (Vectorized)

### Instruction Flow

Approximately 20 instructions. Same structure as scalar but uses `LDG.128` (float4) for coalesced 16-byte loads in both the reduction and normalization loops.

### Register Usage

Same range (24-32 registers), yielding 100% theoretical occupancy.

### Why Fewer Instructions

Float4 reduces the loop trip count by 4x, which eliminates loop overhead (address arithmetic, branch instructions) proportionally. Each `LDG.128` replaces four scalar `LDG` instructions.
