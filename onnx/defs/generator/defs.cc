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
    Constant,
    25,
    generated::ai_onnx::Constant_ver25().TypeAndShapeInferenceFunction(ConstantOpInference));

static void ConstantOfShapeShapeInference_opset25(InferenceContext& ctx) {
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
}

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    25,
    generated::ai_onnx::ConstantOfShape_ver25().TypeAndShapeInferenceFunction(ConstantOfShapeShapeInference_opset25));

static void EyeLikeShapeInference_opset22(InferenceContext& ctx) {
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
}

ONNX_OPERATOR_SET_SCHEMA(
    EyeLike,
    22,
    generated::ai_onnx::EyeLike_ver22().TypeAndShapeInferenceFunction(EyeLikeShapeInference_opset22));

static void RandomUniformShapeInference_opset22(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
  propagateShapeFromAttributeToOutput(ctx, "shape", 0);
}

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniform,
    22,
    generated::ai_onnx::RandomUniform_ver22().TypeAndShapeInferenceFunction(RandomUniformShapeInference_opset22));

static void RandomNormalShapeInference_opset22(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
  propagateShapeFromAttributeToOutput(ctx, "shape", 0);
}

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormal,
    22,
    generated::ai_onnx::RandomNormal_ver22().TypeAndShapeInferenceFunction(RandomNormalShapeInference_opset22));

static void RandomUniformLikeShapeInference_opset22(InferenceContext& ctx) {
  if (ctx.getAttribute("dtype") != nullptr)
    propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
  else
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniformLike,
    22,
    generated::ai_onnx::RandomUniformLike_ver22().TypeAndShapeInferenceFunction(
        RandomUniformLikeShapeInference_opset22));

static void RandomNormalLikeShapeInference_opset22(InferenceContext& ctx) {
  if (ctx.getAttribute("dtype") != nullptr)
    propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
  else
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormalLike,
    22,
    generated::ai_onnx::RandomNormalLike_ver22().TypeAndShapeInferenceFunction(RandomNormalLikeShapeInference_opset22));

static void MultinomialShapeInference_opset22(InferenceContext& ctx) {
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
}

ONNX_OPERATOR_SET_SCHEMA(
    Multinomial,
    22,
    generated::ai_onnx::Multinomial_ver22().TypeAndShapeInferenceFunction(MultinomialShapeInference_opset22));

static bool
BuildFunctionBodyRange27(const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    return false;
  }
  int64_t T = ctx.getInputType(0)->tensor_type().elem_type();
  bool needs_stash =
      (T == static_cast<int64_t>(TensorProto_DataType_FLOAT16) ||
       T == static_cast<int64_t>(TensorProto_DataType_BFLOAT16));

  int64_t stash_type = T;
  if (needs_stash) {
    const auto* stash_attr = ctx.getAttribute("stash_type");
    stash_type = (stash_attr != nullptr) ? stash_attr->i() : static_cast<int64_t>(TensorProto_DataType_FLOAT);
    if (stash_type != static_cast<int64_t>(TensorProto_DataType_FLOAT) &&
        stash_type != static_cast<int64_t>(TensorProto_DataType_DOUBLE))
      // return false so no inline function body is emitted; the model checker
      // will catch the invalid attribute value during validation.
      return false;
  }

  FunctionBuilder builder(functionProto);
  if (needs_stash && stash_type != T) {
    // Cast inputs to stash_type for higher-precision loop accumulation,
    // then cast the collected output back to T.
    builder.Add("start_s = Cast (start)", "to", stash_type)
        .Add("limit_s = Cast (limit)", "to", stash_type)
        .Add("delta_s = Cast (delta)", "to", stash_type)
        .Add("sub_result = Sub (limit_s, start_s)")
        .Add("div_result = Div (sub_result, delta_s)")
        .Add("ceil_result = Ceil (div_result)")
        .Add("ceil_relu = Relu (ceil_result)")
        .Add("n = Cast (ceil_relu)", "to", static_cast<int64_t>(TensorProto_DataType_INT64))
        .Add("loop_cond = Cast (ceil_relu)", "to", static_cast<int64_t>(TensorProto_DataType_BOOL))
        .Add(R"ONNX(variadic_output, output_s = Loop (n, loop_cond, start_s)
          <body = loop_body (int64 i, bool cond_in, prev) => (cond_out, current, range) {
            cond_out = Identity (cond_in)
            current = Add (prev, delta_s)
            range = Identity (prev)
          }>)ONNX")
        .Add("output = Cast (output_s)", "to", T);
  } else {
    builder.Add("sub_result = Sub (limit, start)")
        .Add("sub_result_casted = Cast (sub_result)", "to", static_cast<int64_t>(TensorProto_DataType_FLOAT))
        .Add("delta_casted = Cast (delta)", "to", static_cast<int64_t>(TensorProto_DataType_FLOAT))
        .Add("div_result = Div (sub_result_casted, delta_casted)")
        .Add("ceil_result = Ceil (div_result)")
        .Add("ceil_result_relu = Relu (ceil_result)")
        .Add("ceil_result_relu_int = Cast (ceil_result_relu)", "to", static_cast<int64_t>(TensorProto_DataType_INT64))
        .Add("ceil_result_relu_bool = Cast (ceil_result_relu)", "to", static_cast<int64_t>(TensorProto_DataType_BOOL))
        .Add(R"ONNX(variadic_output, output = Loop (ceil_result_relu_int, ceil_result_relu_bool, start)
          <body = loop_body_attribute (int64 i, bool cond, prev) => (cond_out, current, range) {
            cond_out = Identity (cond)
            current = Add (prev, delta)
            range = Identity (prev)
          }>)ONNX");
  }
  schema.BuildFunction(functionProto);
  return true;
}

static void RangeShapeInference_opset27(InferenceContext& ctx) {
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
      // float16, bfloat16, int16, and other types without a native C++
      // range-computation type — stop with rank inference, no action here
    }

    return;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Range,
    27,
    generated::ai_onnx::Range_ver27()
        .SetContextDependentFunctionBodyBuilder(BuildFunctionBodyRange27)
        .TypeAndShapeInferenceFunction(RangeShapeInference_opset27));

static bool BernoulliFunctionBody_opset22(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    // we cannot create a correct function body without knowing the input type
    return false;
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  auto dtype = ctx.getAttribute("dtype") != nullptr ? static_cast<TensorProto_DataType>(ctx.getAttribute("dtype")->i())
                                                    : input_type;
  FunctionBuilder builder(functionProto);
  builder
      .Add(
          "X_random = RandomUniformLike <low = 0.0, high = 1.0, seed = @seed> (input)",
          "dtype",
          static_cast<int64_t>(input_type))
      .Add("X_greater = Greater (X_random, input)")
      .Add("output = Cast (X_greater)", "to", static_cast<int64_t>(dtype));
  schema.BuildFunction(functionProto);
  return true;
}

static void BernoulliShapeInference_opset22(ONNX_NAMESPACE::InferenceContext& ctx) {
  if (ctx.getAttribute("dtype") != nullptr)
    propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
  else
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

ONNX_OPERATOR_SET_SCHEMA(
    Bernoulli,
    22,
    generated::ai_onnx::Bernoulli_ver22()
        .TypeAndShapeInferenceFunction(BernoulliShapeInference_opset22)
        .SetContextDependentFunctionBodyBuilder(BernoulliFunctionBody_opset22));
} // namespace ONNX_NAMESPACE
