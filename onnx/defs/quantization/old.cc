// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/quantization/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(QuantizeLinear, 25, generated::ai_onnx::QuantizeLinear_ver25());

ONNX_OPERATOR_SET_SCHEMA(DequantizeLinear, 25, generated::ai_onnx::DequantizeLinear_ver25());

ONNX_OPERATOR_SET_SCHEMA(QuantizeLinear, 24, generated::ai_onnx::QuantizeLinear_ver24());

ONNX_OPERATOR_SET_SCHEMA(DequantizeLinear, 24, generated::ai_onnx::DequantizeLinear_ver24());

ONNX_OPERATOR_SET_SCHEMA(QuantizeLinear, 23, generated::ai_onnx::QuantizeLinear_ver23());

ONNX_OPERATOR_SET_SCHEMA(DequantizeLinear, 23, generated::ai_onnx::DequantizeLinear_ver23());

ONNX_OPERATOR_SET_SCHEMA(QuantizeLinear, 21, generated::ai_onnx::QuantizeLinear_ver21());

ONNX_OPERATOR_SET_SCHEMA(DequantizeLinear, 21, generated::ai_onnx::DequantizeLinear_ver21());

ONNX_OPERATOR_SET_SCHEMA(QuantizeLinear, 19, generated::ai_onnx::QuantizeLinear_ver19());

ONNX_OPERATOR_SET_SCHEMA(DequantizeLinear, 19, generated::ai_onnx::DequantizeLinear_ver19());

ONNX_OPERATOR_SET_SCHEMA(QuantizeLinear, 13, generated::ai_onnx::QuantizeLinear_ver13());

ONNX_OPERATOR_SET_SCHEMA(DequantizeLinear, 13, generated::ai_onnx::DequantizeLinear_ver13());

ONNX_OPERATOR_SET_SCHEMA(QuantizeLinear, 10, generated::ai_onnx::QuantizeLinear_ver10());

ONNX_OPERATOR_SET_SCHEMA(DequantizeLinear, 10, generated::ai_onnx::DequantizeLinear_ver10());

} // namespace ONNX_NAMESPACE
