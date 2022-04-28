/*
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

#include <cmath>

namespace ONNX_NAMESPACE {
// The below is called by ops after opset 11, inclusively.
void resizeShapeInference(InferenceContext& ctx, bool is_resize_op);

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

static const char* NonZero_ver9_doc = R"DOC(
    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
    NonZero behaves similar to numpy.nonzero:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
    but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
)DOC";
} // namespace ONNX_NAMESPACE
