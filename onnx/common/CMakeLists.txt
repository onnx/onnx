# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

configure_file(
    version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.h
    @ONLY
)

target_sources(onnx_core PRIVATE
    array_ref.h
    assertions.cc
    assertions.h
    common.h
    constants.h
    file_utils.h
    graph_node_list.h
    interned_strings.cc
    interned_strings.h
    ir_pb_converter.cc
    ir_pb_converter.h
    ir.h
    model_helpers.cc
    model_helpers.h
    path.h
    platform_helpers.h
    proto_util.h
    scoped_resource.h
    status.cc
    status.h
    tensor.h
    visitor.h
    ${CMAKE_CURRENT_BINARY_DIR}/version.h
)
