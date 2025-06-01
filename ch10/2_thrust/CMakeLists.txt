cmake_minimum_required(VERSION 3.12)
project(ThrustSortExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

add_executable(thrust_sort main.cu)

set_target_properties(thrust_sort PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)
