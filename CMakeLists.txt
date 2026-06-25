cmake_minimum_required(VERSION 3.18)
project(cuda-operator-optimization-lab LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA (CMake 内置 FindCUDAToolkit)
find_package(CUDAToolkit REQUIRED)

# ============================================================
# 可执行目标: main — 所有内核的 benchmark
# ============================================================
add_executable(cuda_operators
    src/main.cu
    src/softmax.cu
    src/layernorm.cu
    src/rmsnorm.cu
    src/fused_layernorm.cu
    src/flash_attn_softmax.cu
)

target_include_directories(cuda_operators PRIVATE src)

# CUDA 架构: RTX 3060 是 Ampere (sm_86)
# 可根据你的 GPU 调整
set_target_properties(cuda_operators PROPERTIES
    CUDA_ARCHITECTURES "86"
)
