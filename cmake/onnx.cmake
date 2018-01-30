PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ${ONNXIR_ROOT}/onnx/onnx.proto
    ${ONNXIR_ROOT}/onnx/onnx-operators.proto
)

file(GLOB_RECURSE onnx_src
    "${ONNXIR_ROOT}/onnx/*.h"
    "${ONNXIR_ROOT}/onnx/*.cc"
)

file(GLOB_RECURSE onnx_exclude_src
    "${ONNXIR_ROOT}/onnx/onnx.pb.h"
    "${ONNXIR_ROOT}/onnx/onnx.pb.cc"
    "${ONNXIR_ROOT}/onnx/onnx-operators.pb.h"
    "${ONNXIR_ROOT}/onnx/onnx-operators.pb.cc"
)
list(LENGTH onnx_exclude_src list_len)
if (list_len GREATER 1)
    list(REMOVE_ITEM onnx_src ${onnx_exclude_src})
endif()

add_library(onnxir ${PROTO_SRCS} ${PROTO_HDRS} ${onnx_src})

if (WIN32)
    target_compile_options(onnxir PRIVATE
        /wd4800 # disable warning type' : forcing value to bool 'true' or 'false' (performance warning)
        /wd4503 # identifier' : decorated name length exceeded, name was truncated
    )
    set(onnx_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(onnxir PROPERTIES
        STATIC_LIBRARY_FLAGS "${onnx_static_library_flags}")
endif()
