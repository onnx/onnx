function(RELATIVE_PROTOBUF_GENERATE_CPP SRCS HDRS ROOT_DIR)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    message(${FIL})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} DIRECTORY)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR} -I ${ROOT_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIRS} -I ${ROOT_DIR}/${REL_DIR}
      DEPENDS ${ABS_FIL} protobuf
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ${ONNXIR_ROOT} onnx/onnx.proto
)

file(GLOB_RECURSE onnx_src
    "${ONNXIR_ROOT}/onnx/*.h"
    "${ONNXIR_ROOT}/onnx/*.cc"
)

file(GLOB_RECURSE onnx_exclude_src
    "${ONNXIR_ROOT}/onnx/py_utils.h"
    "${ONNXIR_ROOT}/onnx/onnx.pb.h"
    "${ONNXIR_ROOT}/onnx/onnx.pb.cc"
    "${ONNXIR_ROOT}/onnx/onnx-operators.pb.h"
    "${ONNXIR_ROOT}/onnx/onnx-operators.pb.cc"
    "${ONNXIR_ROOT}/onnx/proto_utils.h"
    "${ONNXIR_ROOT}/onnx/cpp2py_export.cc"
)

list(REMOVE_ITEM onnx_src ${onnx_exclude_src})

add_library(onnxir ${PROTO_SRCS} ${PROTO_HDRS} ${onnx_src})

if (WIN32)
    set(onnx_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(onnxir PROPERTIES
        STATIC_LIBRARY_FLAGS "${onnx_static_library_flags}")
endif()
