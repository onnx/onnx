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
  message(STATUS "  Build type                        : ${CMAKE_BUILD_TYPE}")
  message(STATUS "  CMAKE_INSTALL_PREFIX              : ${CMAKE_INSTALL_PREFIX}")
  if(CMAKE_MODULE_PATH)
    message(STATUS "  CMAKE_MODULE_PATH                 : ${CMAKE_MODULE_PATH}")
  endif()
  message(STATUS "")
  message(STATUS "  ONNX version                      : ${ONNX_VERSION}")
  message(STATUS "  ONNX NAMESPACE                    : ${ONNX_NAMESPACE}")
  message(STATUS "  ONNX_USE_LITE_PROTO               : ${ONNX_USE_LITE_PROTO}")
  message(STATUS "  ONNX_USE_PROTOBUF_SHARED_LIBS     : ${ONNX_USE_PROTOBUF_SHARED_LIBS}")
  message(STATUS "  ONNX_DISABLE_EXCEPTIONS           : ${ONNX_DISABLE_EXCEPTIONS}")
  message(STATUS "  ONNX_DISABLE_STATIC_REGISTRATION  : ${ONNX_DISABLE_STATIC_REGISTRATION}")
  message(STATUS "  ONNX_WERROR                       : ${ONNX_WERROR}")
  message(STATUS "  ONNX_BUILD_TESTS                  : ${ONNX_BUILD_TESTS}")
  message(STATUS "  BUILD_SHARED_LIBS                 : ${BUILD_SHARED_LIBS}")
  message(STATUS "")

  get_target_property(tmp onnx COMPILE_OPTIONS)
  message(STATUS "  onnx compile options              : ${tmp}")
  get_target_property(tmp onnx_proto COMPILE_OPTIONS)
  message(STATUS "  onnx_proto compile options        : ${tmp}")
  get_target_property(tmp onnx COMPILE_DEFINITIONS)
  message(STATUS "  onnx compile definitions          : ${tmp}")
  get_target_property(tmp onnx_proto COMPILE_DEFINITIONS)
  message(STATUS "  onnx_proto COMPILE definitions    : ${tmp}")

  message(STATUS "")
  message(STATUS "  Protobuf verson                   : ${Protobuf_VERSION}")
  if(EXISTS "${ONNX_PROTOC_EXECUTABLE}")
    message(STATUS "  Protobuf compiler                 : ${ONNX_PROTOC_EXECUTABLE}")
  else()
    if(TARGET protobuf::protoc)
      get_target_property(tmp protobuf::protoc IMPORTED_LOCATION)
      if(tmp)
        message(STATUS "  Protobuf compiler                 : ${tmp}")
      endif()
    endif()
  endif()
  get_target_property(tmp ${LINKED_PROTOBUF_TARGET} IMPORTED_LOCATION)
  if(tmp)
    message(STATUS "  Protobuf libraries                : ${tmp}")
  endif()
  message(STATUS "  ONNX_BUILD_PYTHON                 : ${ONNX_BUILD_PYTHON}")
  if(ONNX_BUILD_PYTHON)
    message(STATUS "    Python version                : ${Python3_VERSION}")
    message(STATUS "    Python executable             : ${Python3_EXECUTABLE}")
    message(STATUS "    Python includes               : ${Python3_INCLUDE_DIRS}")
    if(Python3_PyPy_VERSION)
      message(STATUS "    Python3 PyPy version          : ${Python3_PyPy_VERSION}")
    endif()
    message(STATUS "    Python3 interpreter ID        : ${Python3_INTERPRETER_ID}")
    if(Python3_SOABI)
      message(STATUS "    Python3 SOABI                 : ${Python3_SOABI}")
    endif()
  endif()
endfunction()
