# Minimum CMake required
cmake_minimum_required(VERSION 3.26)
# Project
project(onnx_test LANGUAGES CXX)

find_package(ONNX REQUIRED CONFIG)

if(ONNX_ML)
  add_definitions(-DONNX_ML=1)
endif()
add_executable(main main.cc)
target_link_libraries(main ONNX::onnx)
