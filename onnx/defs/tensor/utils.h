// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

#include <cmath>

namespace ONNX_NAMESPACE {
void resizeShapeInference(InferenceContext& ctx, bool is_resize_op);

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape);

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& sizes_data,
    TensorShapeProto* output_shape);
} // namespace ONNX_NAMESPACE
