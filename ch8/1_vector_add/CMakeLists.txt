cmake_minimum_required(VERSION 3.10)
project(overlay_things CUDA)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

add_executable(vector_add vector_add.cu)
