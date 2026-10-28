# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)
project(onnx_ci_protobuf LANGUAGES C CXX)

# Keep the external-Protobuf CI version independent of the bundled version.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Installation library directory")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static dependencies")
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Use the parent C++ standard")
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "Install Abseil")
set(ABSL_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Use the dynamic MSVC runtime")
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Use the dynamic MSVC runtime")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Skip dependency tests")
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "Skip dependency examples")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Avoid an optional system dependency")

include(FetchContent)
FetchContent_Declare(
  absl
  URL https://github.com/abseil/abseil-cpp/releases/download/20250127.0/abseil-cpp-20250127.0.tar.gz
  URL_HASH SHA256=16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8
)
FetchContent_MakeAvailable(absl)
FetchContent_Declare(
  protobuf
  URL https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protobuf-31.1.tar.gz
  URL_HASH SHA256=12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82
)
FetchContent_MakeAvailable(protobuf)
