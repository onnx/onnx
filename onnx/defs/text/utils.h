// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::text::utils {

void StringConcatShapeInference_opset20(InferenceContext& ctx);
void RegexFullMatchShapeInference_opset20(InferenceContext& ctx);
void StringSplitShapeInference_opset20(InferenceContext& ctx);
void StringNormalizerShapeInference_opset10(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::text::utils
