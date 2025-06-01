cmake_minimum_required(VERSION 3.30.4)
project(NVBenchLocalDemo LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(NVBench_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(NVBench_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# show the compiling verbose information
# set(CMAKE_VERBOSE_MAKEFILE ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")


add_subdirectory(nvbench)

#detect_supported_standards(NVBench CXX 17 20)
#detect_supported_standards(NVBench CUDA 17 20)
detect_supported_standards(NVBench CXX 17)
detect_supported_standards(NVBench CUDA 17)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/nvbench/cmake")
#include(NVBenchConfigTarget.cmake)

add_subdirectory(src/matmul)

add_executable(example_bench example_bench.cu)

target_link_libraries(example_bench nvbench::main)

