# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# gen_proto.py runs on the build host, even when the toolchain searches for
# target programs only. Keep the caller's search mode for subsequent lookups.
if(DEFINED CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
  set(_onnx_python_program_mode "${CMAKE_FIND_ROOT_PATH_MODE_PROGRAM}")
else()
  unset(_onnx_python_program_mode)
endif()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
find_package(Python REQUIRED COMPONENTS Interpreter)
if(DEFINED _onnx_python_program_mode)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM "${_onnx_python_program_mode}")
  unset(_onnx_python_program_mode)
else()
  unset(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
endif()
