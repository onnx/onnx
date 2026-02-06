# This minimal CMake file is used to test finding and linking ONNX from a main project
cmake_minimum_required(VERSION 3.26)
project(onnx_test LANGUAGES CXX)

find_package(ONNX REQUIRED CONFIG)

add_executable(main main.cc)
find_package(Protobuf REQUIRED CONFIG)
target_link_libraries(main ONNX::onnx protobuf::libprotobuf)
