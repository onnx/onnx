// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#include <string>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset12() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      updateOutputElemType(ctx, 0, TensorProto::BOOL);
      // Shape inference
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Greater, 9, generated::ai_onnx::Greater_ver9().FillUsing(BinaryLogicDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(Less, 9, generated::ai_onnx::Less_ver9().FillUsing(BinaryLogicDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(Equal, 11, generated::ai_onnx::Equal_ver11().FillUsing(BinaryLogicDocGenerator_opset12()));

inline static void logicalOpInference_opset1(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

static std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset1() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(logicalOpInference_opset1); };
}

static std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset7() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      updateOutputElemType(ctx, 0, TensorProto::BOOL);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(And, 1, generated::ai_onnx::And_ver1().FillUsing(BinaryLogicDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(Or, 1, generated::ai_onnx::Or_ver1().FillUsing(BinaryLogicDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(Xor, 1, generated::ai_onnx::Xor_ver1().FillUsing(BinaryLogicDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(Greater, 1, generated::ai_onnx::Greater_ver1().FillUsing(BinaryLogicDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(Less, 1, generated::ai_onnx::Less_ver1().FillUsing(BinaryLogicDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(Equal, 1, generated::ai_onnx::Equal_ver1().FillUsing(BinaryLogicDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(Equal, 7, generated::ai_onnx::Equal_ver7().FillUsing(BinaryLogicDocGenerator_opset7()));

ONNX_OPERATOR_SET_SCHEMA(Greater, 7, generated::ai_onnx::Greater_ver7().FillUsing(BinaryLogicDocGenerator_opset7()));

ONNX_OPERATOR_SET_SCHEMA(Less, 7, generated::ai_onnx::Less_ver7().FillUsing(BinaryLogicDocGenerator_opset7()));

// Shares same doc generator as newer opset 16 version.
extern std::function<void(OpSchema&)> BinaryLogicDocGenerator();

ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual,
    12,
    generated::ai_onnx::LessOrEqual_ver12()
        .FillUsing(BinaryLogicDocGenerator())
        .TypeAndShapeInferenceFunction(InferenceFunction()));

ONNX_OPERATOR_SET_SCHEMA(
    GreaterOrEqual,
    12,
    generated::ai_onnx::GreaterOrEqual_ver12()
        .FillUsing(BinaryLogicDocGenerator())
        .TypeAndShapeInferenceFunction(InferenceFunction()));

ONNX_OPERATOR_SET_SCHEMA(Equal, 13, generated::ai_onnx::Equal_ver13().FillUsing(BinaryLogicDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    BitShift,
    11,
    generated::ai_onnx::BitShift_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // Shape inference
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

} // namespace ONNX_NAMESPACE
