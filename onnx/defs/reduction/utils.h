// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE::defs::reduction::utils {

void ReduceShapeInference(InferenceContext& ctx);
void ArgReduceShapeInference(InferenceContext& ctx);
void ReduceShapeInference_opset12(InferenceContext& ctx);
void ArgReduceShapeInference_opset12(InferenceContext& ctx);
void ReduceShapeInference_opset1(InferenceContext& ctx);
void ArgReduceShapeInference_opset1(InferenceContext& ctx);
void ArgReduceShapeInference_opset11(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::reduction::utils
