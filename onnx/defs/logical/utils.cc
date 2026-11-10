// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/logical/utils.h"

namespace ONNX_NAMESPACE::defs::logical::utils {

void unaryLogicalOpInference(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void binaryLogicalOpInference(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void BitShiftShapeInference_opset28(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void BinaryBitwiseShapeInference(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void BinaryLogicShapeInference_opset12(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void logicalOpInference_opset1(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void BinaryLogicShapeInference_opset7(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void BitShiftShapeInference_opset11(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

} // namespace ONNX_NAMESPACE::defs::logical::utils
