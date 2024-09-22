# SPDX-License-Identifier: Apache-2.0

# Prints accumulated ONNX configuration summary
function (onnx_print_configuration_summary)
  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "  CMake version                     : ${CMAKE_VERSION}")
  message(STATUS "  CMake command                     : ${CMAKE_COMMAND}")
  message(STATUS "  System                            : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler                      : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version              : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  CXX flags                         : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  Build type                        : ${CMAKE_BUILD_TYPE}")
  get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
  message(STATUS "  Compile definitions               : ${tmp}")
  message(STATUS "  CMAKE_PREFIX_PATH                 : ${CMAKE_PREFIX_PATH}")
  message(STATUS "  CMAKE_INSTALL_PREFIX              : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "  CMAKE_MODULE_PATH                 : ${CMAKE_MODULE_PATH}")
  message(STATUS "")
  message(STATUS "  ONNX version                      : ${ONNX_VERSION}")
  message(STATUS "  ONNX NAMESPACE                    : ${ONNX_NAMESPACE}")
  message(STATUS "  ONNX_USE_LITE_PROTO               : ${ONNX_USE_LITE_PROTO}")
  message(STATUS "  USE_PROTOBUF_SHARED_LIBS          : ${ONNX_USE_PROTOBUF_SHARED_LIBS}")
  message(STATUS "  Protobuf_USE_STATIC_LIBS          : ${Protobuf_USE_STATIC_LIBS}")
  message(STATUS "  ONNX_DISABLE_EXCEPTIONS           : ${ONNX_DISABLE_EXCEPTIONS}")
  message(STATUS "  ONNX_DISABLE_STATIC_REGISTRATION  : ${ONNX_DISABLE_STATIC_REGISTRATION}")
  message(STATUS "  ONNX_WERROR                       : ${ONNX_WERROR}")
  message(STATUS "  ONNX_BUILD_TESTS                  : ${ONNX_BUILD_TESTS}")
  message(STATUS "  ONNX_BUILD_SHARED_LIBS            : ${ONNX_BUILD_SHARED_LIBS}")
  message(STATUS "  BUILD_SHARED_LIBS                 : ${BUILD_SHARED_LIBS}")
  message(STATUS "")
  message(STATUS "  Protobuf compiler                 : ${PROTOBUF_PROTOC_EXECUTABLE}")
  message(STATUS "  Protobuf includes                 : ${PROTOBUF_INCLUDE_DIRS}")
  message(STATUS "  Protobuf libraries                : ${PROTOBUF_LIBRARIES}")
  message(STATUS "  BUILD_ONNX_PYTHON                 : ${BUILD_ONNX_PYTHON}")
  if (${BUILD_ONNX_PYTHON})
    message(STATUS "    Python version                : ${Python_VERSION}")
    message(STATUS "    Python executable             : ${Python_EXECUTABLE}")
    message(STATUS "    Python includes               : ${Python_INCLUDE_DIRS}")
  endif()
endfunction()
