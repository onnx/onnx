# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

target_sources(onnx_core PRIVATE
    BaseConverter.h
    convert.cc
    convert.h
    helper.cc
    helper.h
)

add_subdirectory(adapters)
