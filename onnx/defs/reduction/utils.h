/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> ReduceDocGenerator_opset13_20(
    const char* name,
    bool supports_8bit_datatypes = false,
    bool axes_input = false,
    const char* func_body = nullptr,
    ContextDependentFunctionBodyBuilder function_builder = nullptr,
    bool supports_boolean_datatype = false);
} // namespace ONNX_NAMESPACE
