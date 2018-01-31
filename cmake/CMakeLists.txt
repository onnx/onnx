# Minimum CMake required
cmake_minimum_required(VERSION 3.1)

# Project
project(onnxir C CXX)

# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
set(ONNXIR_ROOT ${PROJECT_SOURCE_DIR}/..)
if (WIN32)
    # parallel build
    add_compile_options(/MP)
    add_compile_options(/WX)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WX")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /WX")
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
FIND_PACKAGE(Protobuf REQUIRED)
include_directories(
    ${ONNXIR_ROOT}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${PROTOBUF_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
)
include(onnx.cmake)
