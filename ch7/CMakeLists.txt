cmake_minimum_required(VERSION 3.10)
project(performance_strategies CUDA)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -arch=sm_75 -rdc=false")

add_executable(matrix_multiply_float matrix_multiply_float.cu)