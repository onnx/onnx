// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::object_detection::utils {

void RoiAlignShapeInference_opset22(InferenceContext& ctx);
void NonMaxSuppressionShapeInference_opset11(InferenceContext& ctx);
void RoiAlignShapeInference_opset16(InferenceContext& ctx);
void RoiAlignShapeInference_opset10(InferenceContext& ctx);
void NonMaxSuppressionShapeInference_opset10(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::object_detection::utils
