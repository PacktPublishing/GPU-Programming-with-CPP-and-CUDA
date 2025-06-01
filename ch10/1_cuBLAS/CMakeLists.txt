cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplyCUBLAS)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(matrix_mul main.cpp)

target_link_libraries(matrix_mul ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
