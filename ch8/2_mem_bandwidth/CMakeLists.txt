cmake_minimum_required(VERSION 3.10)
project(overlay_things CUDA)

add_executable(mem_bandwidth mem_bandwidth.cu)
add_executable(memcpy_benchmark memcpy_benchmark.cu)