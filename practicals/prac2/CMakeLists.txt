cmake_minimum_required(VERSION 3.13)  # CMake version check
project(cuda)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard

include_directories(SYSTEM /Library/Developer/NVIDIA/CUDA-10.0/include)
set(
        SOURCE_FILES
        addmat/addmat.cu
        prac2/square_average.cu
)