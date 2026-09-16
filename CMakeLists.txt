cmake_minimum_required(VERSION 3.18)
project(LPA_ICI_GPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags (Warnings and Optimization).   add -Wextra for extra 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# Build the CPU Version
add_executable(v0_0_lpa_ici_2D_naive src/v0_0_lpa_ici_2D_naive.cpp)
add_executable(v1_0_lpa_ici_2D_fused src/v1_0_lpa_ici_2D_fused.cpp)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # Compile the code for 75 NVIDIA GPU (like T4) architectures
    # change to ALL for more compatibility
    set(CMAKE_CUDA_ARCHITECTURES 75)
    
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    # OTHER FLAGS
    # Suppress unused variable warning: -diag-suppress 550
    # Debug: -g for host code -G for device code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -diag-suppress 550 -Xptxas=-v")
    
    
    add_executable(v0_0_lpa_ici_2D_naive_gpu src/v0_0_lpa_ici_2D_naive.cu)
    add_executable(v0_1_lpa_ici_2D_naive_single_kernel_alloc_gpu src/v0_1_lpa_ici_2D_naive_single_kernel_alloc.cu)
    add_executable(v1_0_lpa_ici_2D_fused_gpu src/v1_0_lpa_ici_2D_fused.cu)
    add_executable(v1_1_lpa_ici_2D_fused_aggregation_gpu src/v1_1_lpa_ici_2D_fused_aggregation.cu)
    add_executable(v1_2_lpa_ici_2D_fused_aggregation_constant_gpu src/v1_2_lpa_ici_2D_fused_aggregation_constant.cu)
    add_executable(v2_0_lpa_ici_2D_fused_streams_gpu src/v2_0_lpa_ici_2D_fused_streams.cu)
    add_executable(v3_0_lpa_ici_2D_fused_ac_sparse_gpu src/v3_0_lpa_ici_2D_fused_ac_sparse.cu)
    add_executable(v3_0_2_lpa_ici_2D_fused_ac_sparse_coarsened_gpu src/v3_0_2_lpa_ici_2D_fused_ac_sparse_coarsened.cu)
    add_executable(v3_1_lpa_ici_2D_fused_ac_sparse_coalesced_gpu src/v3_1_lpa_ici_2D_fused_ac_sparse_coalesced.cu)
    add_executable(v4_0_lpa_ici_2D_sparse_scale_fused_gpu src/v4_0_lpa_ici_2D_sparse_scale_fused.cu)
    add_executable(v4_1_lpa_ici_2D_sparse_shared_gpu src/v4_1_lpa_ici_2D_sparse_shared.cu)

else()
    message(WARNING "CUDA not found! Only the CPU version will be built.")
endif()
