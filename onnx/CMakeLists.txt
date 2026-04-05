# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

# Sources directly in onnx/
# Note: cpp2py_export.cc is excluded — it is compiled separately into the
# onnx_cpp2py_export Python extension module.
target_sources(onnx_core PRIVATE
    checker.cc
    checker.h
    onnx_pb.h
    onnx-operators_pb.h
    proto_utils.h
    py_utils.h
    string_utils.h
)

add_subdirectory(common)
add_subdirectory(defs)
add_subdirectory(inliner)
add_subdirectory(shape_inference)
add_subdirectory(version_converter)
