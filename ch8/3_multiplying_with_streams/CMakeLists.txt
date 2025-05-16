cmake_minimum_required(VERSION 3.10)
project(overlay_things CUDA)


add_executable(vec_mat_mul_streams vec_mat_mul_streams.cu)
add_executable(vec_mat_mul_nostreams vec_mat_mul_nostreams.cu)