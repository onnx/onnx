// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/generator/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    Bernoulli,
    15,
    generated::ai_onnx::Bernoulli_ver15()
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr)
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          else
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
        })
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              if (ctx.getInputType(0) == nullptr) {
                // we cannot create a correct function body without knowing the input type
                return false;
              }
              auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
              auto dtype = ctx.getAttribute("dtype") != nullptr
                  ? static_cast<TensorProto_DataType>(ctx.getAttribute("dtype")->i())
                  : input_type;
              FunctionBuilder builder(functionProto);
              builder
                  .Add(
                      "X_random = RandomUniformLike <low = 0.0, high = 1.0, seed = @seed> (input)", "dtype", input_type)
                  .Add("X_greater = Greater (X_random, input)")
                  .Add("output = Cast (X_greater)", "to", dtype);
              schema.BuildFunction(functionProto);
              return true;
            }));

ONNX_OPERATOR_SET_SCHEMA(
    Multinomial,
    7,
    generated::ai_onnx::Multinomial_ver7().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto dtype = ctx.getAttribute("dtype");
      auto dataType = TensorProto_DataType::TensorProto_DataType_INT32;
      if (dtype != nullptr) {
        dataType = static_cast<TensorProto_DataType>(dtype->i());
        if (dataType != TensorProto_DataType::TensorProto_DataType_INT32 &&
            dataType != TensorProto_DataType::TensorProto_DataType_INT64) {
          fail_type_inference("Output type must be int32 or int64");
        }
      }
      updateOutputElemType(ctx, 0, dataType);

      TensorShapeProto::Dimension batch_size, sample_size;
      if (hasInputShape(ctx, 0)) {
        auto& input_shape = getInputShape(ctx, 0);
        if (input_shape.dim_size() != 2) {
          fail_shape_inference("Input tensor must have rank 2");
        }
        batch_size = input_shape.dim(0);
      } // else statically-unknown batch-size
      sample_size.set_dim_value(getAttribute(ctx, "sample_size", 1));
      updateOutputShape(ctx, 0, {batch_size, sample_size});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormalLike,
    1,
    generated::ai_onnx::RandomNormalLike_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("dtype") != nullptr)
        propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
      else
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniformLike,
    1,
    generated::ai_onnx::RandomUniformLike_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("dtype") != nullptr)
        propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
      else
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormal,
    1,
    generated::ai_onnx::RandomNormal_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
      propagateShapeFromAttributeToOutput(ctx, "shape", 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniform,
    1,
    generated::ai_onnx::RandomUniform_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
      propagateShapeFromAttributeToOutput(ctx, "shape", 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    EyeLike,
    9,
    generated::ai_onnx::EyeLike_ver9().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("dtype") != nullptr) {
        propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
      } else {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      }
      if (hasInputShape(ctx, 0)) {
        auto& input_shape = getInputShape(ctx, 0);
        if (input_shape.dim_size() != 2) {
          fail_shape_inference("Input tensor must be 2-dimensional");
        }
      }
      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    24,
    generated::ai_onnx::Constant_ver24().TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    23,
    generated::ai_onnx::Constant_ver23().TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    21,
    generated::ai_onnx::Constant_ver21().TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    19,
    generated::ai_onnx::Constant_ver19().TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    13,
    generated::ai_onnx::Constant_ver13().TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    12,
    generated::ai_onnx::Constant_ver12().TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    1,
    generated::ai_onnx::Constant_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto attr_proto = ctx.getAttribute("value");
      if (nullptr == attr_proto)
        return; // attribute not present
      if (!attr_proto->has_t())
        return; // attribute has no tensor value
      const TensorProto& tensor_proto = attr_proto->t();
      updateOutputElemType(ctx, 0, tensor_proto.data_type());
      updateOutputShape(ctx, 0, tensor_proto);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    9,
    generated::ai_onnx::Constant_ver9().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto attr_proto = ctx.getAttribute("value");
      if (nullptr == attr_proto || !attr_proto->has_t())
        fail_shape_inference("Attribute 'value' of Constant node must exist with 'Tensor' data.");
      const TensorProto& tensor_proto = attr_proto->t();
      updateOutputElemType(ctx, 0, tensor_proto.data_type());
      updateOutputShape(ctx, 0, tensor_proto);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    11,
    generated::ai_onnx::Constant_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto value = ctx.getAttribute("value");
      auto sparse_value = ctx.getAttribute("sparse_value");

      if ((nullptr != value) && (nullptr != sparse_value))
        fail_shape_inference(
            "Only one of the attributes 'value' or 'sparse_value' must be specified for a Constant node.");

      if (nullptr != value) {
        // OpSchema::Verify check ensures that the attribute value has_t():
        const TensorProto& tensor_proto = value->t();
        updateOutputElemType(ctx, 0, tensor_proto.data_type());
        updateOutputShape(ctx, 0, tensor_proto);
        return;
      }

      if (nullptr != sparse_value) {
        // OpSchema::Verify check ensures that the attribute value
        // has_sparse_tensor():
        const SparseTensorProto& sparse = sparse_value->sparse_tensor();
        // checker.cc::check_sparse_tensor checks that the sparse-value is
        // well-formed
        updateOutputElemType(ctx, 0, sparse.values().data_type());
        auto output_shape = getOutputShape(ctx, 0);
        for (int i = 0; i < sparse.dims_size(); ++i)
          appendDim(output_shape, sparse.dims(i));
        return;
      }
      fail_shape_inference("One of the attributes 'value' or 'sparse_value' must be specified for a Constant node.");
    }));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    24,
    generated::ai_onnx::ConstantOfShape_ver24().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("value") != nullptr) {
        propagateElemTypeFromDtypeToOutput(ctx, ctx.getAttribute("value"), 0);
      } else {
        propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
      }

      bool found = false;
      TensorShapeProto output_shape = getShapeInput(ctx, 0, true, found);
      if (found) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = output_shape;
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    23,
    generated::ai_onnx::ConstantOfShape_ver23().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("value") != nullptr) {
        propagateElemTypeFromDtypeToOutput(ctx, ctx.getAttribute("value"), 0);
      } else {
        propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
      }

      bool found = false;
      TensorShapeProto output_shape = getShapeInput(ctx, 0, true, found);
      if (found) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = output_shape;
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    21,
    generated::ai_onnx::ConstantOfShape_ver21().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("value") != nullptr) {
        propagateElemTypeFromDtypeToOutput(ctx, ctx.getAttribute("value"), 0);
      } else {
        propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
      }

      bool found = false;
      TensorShapeProto output_shape = getShapeInput(ctx, 0, true, found);
      if (found) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = output_shape;
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    20,
    generated::ai_onnx::ConstantOfShape_ver20().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("value") != nullptr) {
        propagateElemTypeFromDtypeToOutput(ctx, ctx.getAttribute("value"), 0);
      } else {
        propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
      }

      bool found = false;
      TensorShapeProto output_shape = getShapeInput(ctx, 0, true, found);
      if (found) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = output_shape;
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    9,
    generated::ai_onnx::ConstantOfShape_ver9().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("value") != nullptr) {
        propagateElemTypeFromDtypeToOutput(ctx, ctx.getAttribute("value"), 0);
      } else {
        propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
      }

      bool found = false;
      TensorShapeProto output_shape = getShapeInput(ctx, 0, true, found);
      if (found) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = output_shape;
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Range,
    11,
    generated::ai_onnx::Range_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference
      const auto start_initializer = ctx.getInputData(0);
      const auto limit_initializer = ctx.getInputData(1);
      const auto delta_initializer = ctx.getInputData(2);

      // Output is always 1-D
      auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();

      // If any of Range's inputs are not initializers, the output dimension
      // value would remain unknown.
      if (start_initializer != nullptr && limit_initializer != nullptr && delta_initializer != nullptr) {
        // Make sure the input types are homogeneous
        if ((start_initializer->data_type() != limit_initializer->data_type()) ||
            (start_initializer->data_type() != delta_initializer->data_type())) {
          fail_shape_inference("All inputs to 'Range' op must be of the same type");
        }

        // Explicitly compute the output dimension if Range's inputs are
        // stored in initializer list.
        if (start_initializer->data_type() == TensorProto::FLOAT) {
          output_dim->set_dim_value(
              compute_output_dim_for_range<float>(start_initializer, limit_initializer, delta_initializer));
        } else if (start_initializer->data_type() == TensorProto::INT32) {
          output_dim->set_dim_value(
              compute_output_dim_for_range<int32_t>(start_initializer, limit_initializer, delta_initializer));
        } else if (start_initializer->data_type() == TensorProto::INT64) {
          output_dim->set_dim_value(
              compute_output_dim_for_range<int64_t>(start_initializer, limit_initializer, delta_initializer));
        } else if (start_initializer->data_type() == TensorProto::DOUBLE) {
          output_dim->set_dim_value(
              compute_output_dim_for_range<double>(start_initializer, limit_initializer, delta_initializer));
        } else {
          // int16 has no explicit shape computation here -
          // stop with rank inference, no action here
        }

        return;
      }
    }));

} // namespace ONNX_NAMESPACE
