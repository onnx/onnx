// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static void RoiAlignShapeInference_opset16(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  size_t input_param = 0, rois_param = 1, batch_index_param = 2;

  checkInputRank(ctx, input_param, 4);
  checkInputRank(ctx, rois_param, 2);
  checkInputRank(ctx, batch_index_param, 1);

  // Output dimensions, initialized to an unknown-dimension-value
  Dim num_rois, C, ht, width;

  // Get value of C from dim 1 of input_param, if available
  unifyInputDim(ctx, input_param, 1, C);

  // Get value of num_rois from dim 0 of rois_param, if available
  unifyInputDim(ctx, rois_param, 0, num_rois);
  // ... or from dim 0 of batch_index_param, if available
  unifyInputDim(ctx, batch_index_param, 0, num_rois);

  // Get height from attribute, using default-value of 1
  unifyDim(ht, getAttribute(ctx, "output_height", 1));

  // Get width from attribute, using default-value of 1
  unifyDim(width, getAttribute(ctx, "output_width", 1));

  // set output shape:
  updateOutputShape(ctx, 0, {num_rois, C, ht, width});
}

ONNX_OPERATOR_SET_SCHEMA(
    RoiAlign,
    16,
    generated::ai_onnx::RoiAlign_ver16().TypeAndShapeInferenceFunction(RoiAlignShapeInference_opset16));

static void RoiAlignShapeInference_opset10(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  size_t input_param = 0, rois_param = 1, batch_index_param = 2;

  checkInputRank(ctx, input_param, 4);
  checkInputRank(ctx, rois_param, 2);
  checkInputRank(ctx, batch_index_param, 1);

  // Output dimensions, initialized to an unknown-dimension-value
  Dim num_rois, C, ht, width;

  // Get value of C from dim 1 of input_param, if available
  unifyInputDim(ctx, input_param, 1, C);

  // Get value of num_rois from dim 0 of rois_param, if available
  unifyInputDim(ctx, rois_param, 0, num_rois);
  // ... or from dim 0 of batch_index_param, if available
  unifyInputDim(ctx, batch_index_param, 0, num_rois);

  // Get height from attribute, using default-value of 1
  unifyDim(ht, getAttribute(ctx, "output_height", 1));

  // Get width from attribute, using default-value of 1
  unifyDim(width, getAttribute(ctx, "output_width", 1));

  // set output shape:
  updateOutputShape(ctx, 0, {num_rois, C, ht, width});
}

ONNX_OPERATOR_SET_SCHEMA(
    RoiAlign,
    10,
    generated::ai_onnx::RoiAlign_ver10().TypeAndShapeInferenceFunction(RoiAlignShapeInference_opset10));

static void NonMaxSuppressionShapeInference_opset10(InferenceContext& ctx) {
  auto selected_indices_type = ctx.getOutputType(0)->mutable_tensor_type();
  selected_indices_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64);
}

ONNX_OPERATOR_SET_SCHEMA(
    NonMaxSuppression,
    10,
    generated::ai_onnx::NonMaxSuppression_ver10().TypeAndShapeInferenceFunction(
        NonMaxSuppressionShapeInference_opset10));

} // namespace ONNX_NAMESPACE
