// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/object_detection/utils.h"

namespace ONNX_NAMESPACE::defs::object_detection::utils {

void RoiAlignShapeInference_opset22(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  size_t input_param = 0, rois_param = 1, batch_index_param = 2;

  // X: [N, C, H, W], rois: [num_rois, 4], batch_indices: [num_rois]
  Dim N, C, H, W, num_rois, four;
  four.set_dim_value(4);

  ctx.unifyInputShape(input_param, {N, C, H, W});
  ctx.unifyInputShape(rois_param, {num_rois, four});
  ctx.unifyInputShape(batch_index_param, {num_rois});

  Dim ht, width;
  unifyDim(ht, getAttribute(ctx, "output_height", 1));
  unifyDim(width, getAttribute(ctx, "output_width", 1));

  // output: [num_rois, C, output_height, output_width]
  updateOutputShape(ctx, 0, {num_rois, C, ht, width});
}

void NonMaxSuppressionShapeInference_opset11(InferenceContext& ctx) {
  // Type inference - Output is always of type INT64
  auto selected_indices_type = ctx.getOutputType(0)->mutable_tensor_type();
  selected_indices_type->set_elem_type(TensorProto_DataType::TensorProto_DataType_INT64);

  // Shape inference
  // The exact shape cannot be determined as it depends on the input and
  // other input configurations for the op But part of the shape can be
  // established

  auto selected_indices_shape = getOutputShape(ctx, 0);
  selected_indices_shape->clear_dim();

  // Output is 2D always

  // The value of the first dim is determined by input data
  // hence its value cannot be determined statically
  selected_indices_shape->add_dim();

  // The value of the second dim is 3
  selected_indices_shape->add_dim()->set_dim_value(3);
}

void RoiAlignShapeInference_opset16(InferenceContext& ctx) {
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

void RoiAlignShapeInference_opset10(InferenceContext& ctx) {
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

void NonMaxSuppressionShapeInference_opset10(InferenceContext& ctx) {
  auto selected_indices_type = ctx.getOutputType(0)->mutable_tensor_type();
  selected_indices_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64);
}

} // namespace ONNX_NAMESPACE::defs::object_detection::utils
