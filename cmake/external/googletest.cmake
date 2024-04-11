# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

include (ExternalProject)

set(googletest_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include)
set(googletest_URL https://github.com/google/googletest.git)
set(googletest_BUILD ${CMAKE_CURRENT_BINARY_DIR}/googletest/)
# https://github.com/google/googletest/commit/530d5c8c84abd2a46f38583ee817743c9b3a42b4
set(googletest_TAG 530d5c8c84abd2a46f38583ee817743c9b3a42b4)

if(WIN32 AND CMAKE_BUILD_TYPE MATCHES Debug)
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/Debug/gtestd.lib)
elseif(WIN32 AND NOT CMAKE_BUILD_TYPE MATCHES Debug)
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/Release/gtest.lib)
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/../lib/libgtest.a)
else()
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/../lib/libgtest.a)
endif()

if(ONNX_USE_MSVC_STATIC_RUNTIME)
  set(ONNX_USE_MSVC_SHARED_RUNTIME OFF)
else()
  set(ONNX_USE_MSVC_SHARED_RUNTIME ON)
endif()
ExternalProject_Add(googletest
    PREFIX googletest
    GIT_REPOSITORY ${googletest_URL}
    GIT_TAG ${googletest_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target gtest
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_GMOCK:BOOL=OFF
        -DBUILD_GTEST:BOOL=ON
        -Dgtest_force_shared_crt:BOOL=${ONNX_USE_MSVC_SHARED_RUNTIME}
    BUILD_BYPRODUCTS ${googletest_STATIC_LIBRARIES}
)
