cmake_minimum_required(VERSION 3.13)
project(cuda_prac2)
set(CMAKE_CXX_STANDARD 14)

include_directories(SYSTEM /Library/Developer/NVIDIA/CUDA-10.0/include)
set(SOURCE_FILES square_average.cu)