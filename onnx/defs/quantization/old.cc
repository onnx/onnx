// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    25,
    generated::ai_onnx::QuantizeLinear_ver25().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
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
        updateOutputElemType(ctx, 0, TensorProto::UINT8);
      }
      if (!hasInputShape(ctx, 0)) {
        return;
      }

      auto& input_shape = getInputShape(ctx, 0);
      updateOutputShape(ctx, 0, input_shape);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    25,
    generated::ai_onnx::DequantizeLinear_ver25().TypeAndShapeInferenceFunction(
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
    QuantizeLinear,
    24,
    generated::ai_onnx::QuantizeLinear_ver24().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
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
    24,
    generated::ai_onnx::DequantizeLinear_ver24().TypeAndShapeInferenceFunction(
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
    QuantizeLinear,
    23,
    generated::ai_onnx::QuantizeLinear_ver23().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
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
    23,
    generated::ai_onnx::DequantizeLinear_ver23().TypeAndShapeInferenceFunction(
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
    QuantizeLinear,
    21,
    generated::ai_onnx::QuantizeLinear_ver21().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
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
    21,
    generated::ai_onnx::DequantizeLinear_ver21().TypeAndShapeInferenceFunction(
        [](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    19,
    generated::ai_onnx::QuantizeLinear_ver19().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      if (ctx.hasInput(2)) {
        if (ctx.getInputType(2) != nullptr) {
          propagateElemTypeFromInputToOutput(ctx, 2, 0);
        }
        // Otherwise, y_zero_point is provided but its type could not be inferred: the output
        // type cannot be determined, so it is left uninferred.
      } else {
        // y_zero_point is not provided: the output type defaults to uint8.
        updateOutputElemType(ctx, 0, TensorProto::UINT8);
      }
      if (!hasInputShape(ctx, 0)) {
        return;
      }

      auto& input_shape = getInputShape(ctx, 0);
      updateOutputShape(ctx, 0, input_shape);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    19,
    generated::ai_onnx::DequantizeLinear_ver19().TypeAndShapeInferenceFunction(
        [](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    13,
    generated::ai_onnx::QuantizeLinear_ver13().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      if (ctx.hasInput(2)) {
        if (ctx.getInputType(2) != nullptr) {
          propagateElemTypeFromInputToOutput(ctx, 2, 0);
        }
        // Otherwise, y_zero_point is provided but its type could not be inferred: the output
        // type cannot be determined, so it is left uninferred.
      } else {
        // y_zero_point is not provided: the output type defaults to uint8.
        updateOutputElemType(ctx, 0, TensorProto::UINT8);
      }
      if (!hasInputShape(ctx, 0)) {
        return;
      }
      auto& input_shape = getInputShape(ctx, 0);
      updateOutputShape(ctx, 0, input_shape);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    13,
    generated::ai_onnx::DequantizeLinear_ver13().TypeAndShapeInferenceFunction(
        [](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto y_type = ctx.getOutputType(0);
          // only float is supported
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    10,
    generated::ai_onnx::QuantizeLinear_ver10().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      if (ctx.hasInput(2)) {
        if (ctx.getInputType(2) != nullptr) {
          propagateElemTypeFromInputToOutput(ctx, 2, 0);
        }
        // Otherwise, y_zero_point is provided but its type could not be inferred: the output
        // type cannot be determined, so it is left uninferred.
      } else {
        // y_zero_point is not provided: the output type defaults to uint8.
        updateOutputElemType(ctx, 0, TensorProto::UINT8);
      }
      if (!hasInputShape(ctx, 0)) {
        return;
      }

      auto& input_shape = getInputShape(ctx, 0);
      updateOutputShape(ctx, 0, input_shape);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    10,
    generated::ai_onnx::DequantizeLinear_ver10().TypeAndShapeInferenceFunction(
        [](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto y_type = ctx.getOutputType(0);
          // only float is supported
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

} // namespace ONNX_NAMESPACE
