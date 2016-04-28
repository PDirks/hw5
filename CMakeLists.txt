# Brendan Niebruegge
# Peter Dirks
# Homework 5
# CMakeLists.txt
# April 26, 2016

cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)

#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Werror -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)

# not sure what this does, but it keeps the CXX flags from breaking everything
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_executable( hw5 src/hw5.cpp src/hw5.cu include/hw5.cuh include/sdkHelper.h include/util.hpp )