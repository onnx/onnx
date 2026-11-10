// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::training::utils {

void AdagradShapeInference_opset1(InferenceContext& ctx);
void MomentumShapeInference_opset1(InferenceContext& ctx);
void AdamShapeInference_opset1(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::training::utils
