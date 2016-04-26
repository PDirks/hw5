cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)

#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)


cuda_add_executable( hw5 src/hw5.cpp src/hw5.cu include/hw5.cuh include/sdkHelper.h )