// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::rnn::utils {

void RNNShapeInference(InferenceContext& ctx);
void RNNShapeInference_opset14(InferenceContext& ctx);
void RNNShapeInference_opset1_to_6(InferenceContext& ctx);
void RNNShapeInference_opset7_to_13(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::rnn::utils
