// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::quantization::utils {

void QuantizeLinearShapeInference_opset28(ONNX_NAMESPACE::InferenceContext& ctx);
void DequantizeLinearShapeInference_opset28(ONNX_NAMESPACE::InferenceContext& ctx);
void DynamicQuantizeLinearShapeInference_opset11(ONNX_NAMESPACE::InferenceContext& ctx);
void QuantizeLinearShapeInference_opset25(ONNX_NAMESPACE::InferenceContext& ctx);
void DequantizeLinearShapeInference_opset25(ONNX_NAMESPACE::InferenceContext& ctx);
void QuantizeLinearShapeInference_opset24(ONNX_NAMESPACE::InferenceContext& ctx);
void DequantizeLinearShapeInference_opset24(ONNX_NAMESPACE::InferenceContext& ctx);
void QuantizeLinearShapeInference_opset23(ONNX_NAMESPACE::InferenceContext& ctx);
void DequantizeLinearShapeInference_opset23(ONNX_NAMESPACE::InferenceContext& ctx);
void QuantizeLinearShapeInference_opset21(ONNX_NAMESPACE::InferenceContext& ctx);
void DequantizeLinearShapeInference_opset21(ONNX_NAMESPACE::InferenceContext& ctx);
void QuantizeLinearShapeInference_opset19(ONNX_NAMESPACE::InferenceContext& ctx);
void DequantizeLinearShapeInference_opset19(ONNX_NAMESPACE::InferenceContext& ctx);
void QuantizeLinearShapeInference_opset13(ONNX_NAMESPACE::InferenceContext& ctx);
void DequantizeLinearShapeInference_opset13(ONNX_NAMESPACE::InferenceContext& ctx);
void QuantizeLinearShapeInference_opset10(ONNX_NAMESPACE::InferenceContext& ctx);
void DequantizeLinearShapeInference_opset10(ONNX_NAMESPACE::InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::quantization::utils
