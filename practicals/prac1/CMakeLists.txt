cmake_minimum_required(VERSION 3.13)
project(cuda_prac1)
set(CMAKE_CXX_STANDARD 14)

include_directories(SYSTEM /Library/Developer/NVIDIA/CUDA-10.0/include)
set(SOURCE_FILES prac1a.cu prac1b.cu prac1c.cu)