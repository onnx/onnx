// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/text/utils.h"

namespace ONNX_NAMESPACE::defs::text::utils {

void StringConcatShapeInference_opset20(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void RegexFullMatchShapeInference_opset20(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

void StringSplitShapeInference_opset20(InferenceContext& ctx) {
  if (!hasInputShape(ctx, 0)) {
    return;
  }
  const TypeProto* input_type = ctx.getInputType(0);
  if (input_type == nullptr || !input_type->has_tensor_type() ||
      input_type->tensor_type().elem_type() != TensorProto::STRING) {
    return;
  }

  // We produce a string tensor per input element. Therefore we have one additional rank with a runtime
  // dependent number of elements. All except the final dimension of the output shape can be inferred directly
  // from the input.
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  propagateShapeFromInputToOutput(ctx, 0, 0);
  getOutputShape(ctx, 0)->add_dim();

  // The output tensor containing the number of substrings has identical shape to the input but produces int32
  // results.
  ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  propagateShapeFromInputToOutput(ctx, 0, 1);
}

void StringNormalizerShapeInference_opset10(InferenceContext& ctx) {
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_elem_type->set_elem_type(TensorProto::STRING);
  if (!hasInputShape(ctx, 0)) {
    return;
  }
  TensorShapeProto output_shape;
  auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto dim_size = input_shape.dim_size();
  // Last axis dimension is unknown if we have stop-words since we do
  // not know how many stop-words are dropped
  if (dim_size == 1) {
    // Unknown output dimension
    output_shape.add_dim();
  } else if (dim_size == 2) {
    // Copy B-dim
    auto& b_dim = input_shape.dim(0);
    if (!b_dim.has_dim_value() || b_dim.dim_value() != 1) {
      fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
    }
    *output_shape.add_dim() = b_dim;
    output_shape.add_dim();
  } else {
    fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
  }
  updateOutputShape(ctx, 0, output_shape);
}

} // namespace ONNX_NAMESPACE::defs::text::utils
