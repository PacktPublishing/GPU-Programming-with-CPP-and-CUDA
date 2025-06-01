cmake_minimum_required(VERSION 3.10)
project(VectorAddTest LANGUAGES CXX CUDA)

enable_testing()

find_package(GTest REQUIRED)

add_executable(vector_add_test test/test_vector_add.cu src/main.cu)
target_link_libraries(vector_add_test GTest::GTest GTest::Main cuda)

add_test(NAME VectorAddTest COMMAND vector_add_test)

add_library(vector_add SHARED src/main.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(vector_add PUBLIC include)

