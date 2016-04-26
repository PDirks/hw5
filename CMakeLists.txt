cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

cuda_add_executable(
        hw5
        src/hw5.cu)