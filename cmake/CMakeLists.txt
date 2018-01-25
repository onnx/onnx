# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(onnxir C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
set(ONNXIR_ROOT ${PROJECT_SOURCE_DIR}/..)
include(zlib)
include(protobuf)
include(onnx.cmake)

if (WIN32)
    # parallel build
    add_compile_options(/MP)
    add_compile_options(/WX)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WX")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /WX")
endif()

include_directories(
    ${ONNXIR_ROOT}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${PROTOBUF_INCLUDE_DIRS}
)