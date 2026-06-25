cmake_minimum_required(VERSION 3.18)
project(cuda-operator-optimization-lab LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# ============================================================
# Collect all per-operator source files
# ============================================================
file(GLOB_RECURSE KERNEL_SOURCES
    Softmax/src/*.cu
    LayerNorm/src/*.cu
    RMSNorm/src/*.cu
    FusedResidualLN/src/*.cu
    FlashAttentionSoftmax/src/*.cu
)

# ============================================================
# Benchmark driver executable
# ============================================================
add_executable(cuda_operators
    main.cu
    ${KERNEL_SOURCES}
)

target_include_directories(cuda_operators PRIVATE include)

set_target_properties(cuda_operators PROPERTIES
    CUDA_ARCHITECTURES "86"
)
