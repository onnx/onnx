// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

inline static void unaryLogicalOpInference(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

static void binaryLogicalOpInference(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::function<void(OpSchema&)> BinaryLogicDocGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(binaryLogicalOpInference); };
}

ONNX_OPERATOR_SET_SCHEMA(And, 7, generated::ai_onnx::And_ver7().FillUsing(BinaryLogicDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Or, 7, generated::ai_onnx::Or_ver7().FillUsing(BinaryLogicDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Xor, 7, generated::ai_onnx::Xor_ver7().FillUsing(BinaryLogicDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Greater, 13, generated::ai_onnx::Greater_ver13().FillUsing(BinaryLogicDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Less, 13, generated::ai_onnx::Less_ver13().FillUsing(BinaryLogicDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Equal, 19, generated::ai_onnx::Equal_ver19().FillUsing(BinaryLogicDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Not, 1, generated::ai_onnx::Not_ver1().TypeAndShapeInferenceFunction(unaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    BitShift,
    28,
    generated::ai_onnx::BitShift_ver28().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // Shape inference
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual,
    16,
    generated::ai_onnx::LessOrEqual_ver16()
        .FillUsing(BinaryLogicDocGenerator())
        .TypeAndShapeInferenceFunction(binaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    GreaterOrEqual,
    16,
    generated::ai_onnx::GreaterOrEqual_ver16()
        .FillUsing(BinaryLogicDocGenerator())
        .TypeAndShapeInferenceFunction(binaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    BitwiseNot,
    18,
    generated::ai_onnx::BitwiseNot_ver18().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static std::function<void(OpSchema&)> BinaryBitwiseDocGenerator() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // Shape inference
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(BitwiseAnd, 18, generated::ai_onnx::BitwiseAnd_ver18().FillUsing(BinaryBitwiseDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(BitwiseOr, 18, generated::ai_onnx::BitwiseOr_ver18().FillUsing(BinaryBitwiseDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(BitwiseXor, 18, generated::ai_onnx::BitwiseXor_ver18().FillUsing(BinaryBitwiseDocGenerator()));

} // namespace ONNX_NAMESPACE
