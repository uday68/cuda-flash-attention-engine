cmake_minimum_required(VERSION 3.18)
project(cuda_flash_attention_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Automatically register tests and benchmarks as executables if present
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/*.cu")
foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
endforeach()

