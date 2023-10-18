# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
# initialization
#
# defines dlpack_SOURCE_DIR dlpack_BINARY_DIR

#
# dlpack
#

set(dlpack_TAG "v0.8")

include(FetchContent)
FetchContent_Declare(
  dlpack
  GIT_REPOSITORY https://github.com/dmlc/dlpack
  GIT_TAG ${dlpack_TAG})

FetchContent_GetProperties(dlpack)
if(NOT dlpack_POPULATED)
  FetchContent_Populate(dlpack)
else()
  message(FATAL_ERROR "dlpack was not found.")
endif()

set(dlpack_VERSION ${dlpack_TAG})
set(dlpack_INCLUDE_DIR "${dlpack_SOURCE_DIR}/include")
message(STATUS "dlpack_INCLUDE_DIR=${dlpack_INCLUDE_DIR}")
message(STATUS "dlpack_VERSION=${dlpack_VERSION}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LocalDLPack
  VERSION_VAR dlpack_VERSION
  REQUIRED_VARS dlpack_SOURCE_DIR dlpack_BINARY_DIR dlpack_INCLUDE_DIR)
