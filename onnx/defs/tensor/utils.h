/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

#include <cmath>

namespace ONNX_NAMESPACE {
// The below is called by ops after opset 11, inclusively.
void resizeShapeInference(InferenceContext& ctx);

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape);

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& sizes_data,
    TensorShapeProto* output_shape);

// The below is called by ops between opset 7 and opset 10, inclusively.
void resizeShapeInference_opset7_to_10(InferenceContext& ctx);

void resizeShapeInferenceHelper_opset7_to_10(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape);

extern const char* NonZero_ver9_doc;
} // namespace ONNX_NAMESPACE
