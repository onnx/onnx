// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/quantization/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(QuantizeLinear, 28, generated::ai_onnx::QuantizeLinear_ver28());

ONNX_OPERATOR_SET_SCHEMA(DequantizeLinear, 28, generated::ai_onnx::DequantizeLinear_ver28());

ONNX_OPERATOR_SET_SCHEMA(DynamicQuantizeLinear, 11, generated::ai_onnx::DynamicQuantizeLinear_ver11());

} // namespace ONNX_NAMESPACE
