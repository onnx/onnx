// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    28,
    generated::ai_onnx::QuantizeLinear_ver28().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      // y_zero_point's type is known only if it is present as an input *and* its type could be inferred
      // (e.g. it is not simply an unresolved formal parameter of an enclosing function).
      auto const zp_type = ctx.hasInput(2) ? ctx.getInputType(2) : nullptr;
      auto const output_dtype =
          static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
      if (zp_type != nullptr) {
        auto const zp_elem_type = static_cast<TensorProto_DataType>(getTensorElementType(*zp_type));
        if (output_dtype != TensorProto::UNDEFINED && output_dtype != zp_elem_type) {
          fail_type_inference(
              "output_dtype ",
              TensorProto_DataType_Name(output_dtype),
              " does not match y_zero_point type ",
              TensorProto_DataType_Name(zp_elem_type),
              ".");
        }
        propagateElemTypeFromInputToOutput(ctx, 2, 0);
      } else if (output_dtype != TensorProto::UNDEFINED) {
        propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
      } else if (!ctx.hasInput(2)) {
        // y_zero_point is not provided: the output type defaults to uint8.
        updateOutputElemType(ctx, 0, TensorProto::UINT8);
      }
      // Otherwise, y_zero_point is provided but its type could not be inferred, and no output_dtype
      // attribute was specified: the output type cannot be determined, so it is left uninferred.
      if (!hasInputShape(ctx, 0)) {
        return;
      }

      auto& input_shape = getInputShape(ctx, 0);
      updateOutputShape(ctx, 0, input_shape);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    28,
    generated::ai_onnx::DequantizeLinear_ver28().TypeAndShapeInferenceFunction(
        [](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto const output_dtype =
              static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
          if (output_dtype != TensorProto::UNDEFINED) {
            propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
          } else {
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    DynamicQuantizeLinear,
    11,
    generated::ai_onnx::DynamicQuantizeLinear_ver11().TypeAndShapeInferenceFunction(
        [](ONNX_NAMESPACE::InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::UINT8);
          updateOutputElemType(ctx, 1, TensorProto::FLOAT);
          updateOutputElemType(ctx, 2, TensorProto::UINT8);

          ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

} // namespace ONNX_NAMESPACE
