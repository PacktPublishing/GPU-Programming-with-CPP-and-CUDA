cmake_minimum_required(VERSION 3.10)
project(exposing_functions CUDA)

add_library(vector_add SHARED src/vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(vector_add PUBLIC include)
