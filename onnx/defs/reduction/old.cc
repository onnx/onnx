// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/reduction/utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static void ReduceShapeInference_opset12(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  int64_t input_ndim = input_shape.dim_size();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  std::vector<int64_t> axes;
  const auto* const axes_proto = ctx.getAttribute("axes");
  if (axes_proto)
    axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

  for (int64_t& axe : axes) {
    if (axe < -input_ndim || axe >= input_ndim) {
      fail_shape_inference("axis must be in [-rank, rank-1]. input rank was ", input_ndim);
    }
    if (axe < 0)
      axe += input_ndim;
  }
  // do we need handle negative axis?
  for (int i = 0; i < input_ndim; ++i) {
    // axes empty means reduce all dim
    if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}

static std::function<void(OpSchema&)> ReduceDocGenerator_opset12() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(ReduceShapeInference_opset12); };
}

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 12, generated::ai_onnx::ReduceMax_ver12().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 12, generated::ai_onnx::ReduceMin_ver12().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 11, generated::ai_onnx::ReduceSum_ver11().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceSumSquare,
    11,
    generated::ai_onnx::ReduceSumSquare_ver11().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMean,
    11,
    generated::ai_onnx::ReduceMean_ver11().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceProd,
    11,
    generated::ai_onnx::ReduceProd_ver11().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSum,
    11,
    generated::ai_onnx::ReduceLogSum_ver11().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    11,
    generated::ai_onnx::ReduceLogSumExp_ver11().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 11, generated::ai_onnx::ReduceL1_ver11().FillUsing(ReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 11, generated::ai_onnx::ReduceL2_ver11().FillUsing(ReduceDocGenerator_opset12()));

static void ArgReduceShapeInference_opset12(InferenceContext& ctx) {
  // set output element type to int64
  updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  int64_t input_ndim = input_shape.dim_size();
  int64_t axis = 0; // default to 0
  const auto* const axis_proto = ctx.getAttribute("axis");
  if (axis_proto) {
    axis = axis_proto->i();
    if (axis < -input_ndim || axis >= input_ndim) {
      fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
    }
    if (axis < 0)
      axis += input_ndim;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  // do we need handle negative axis?
  for (int i = 0; i < input_ndim; ++i) {
    if (i != axis) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}

static std::function<void(OpSchema&)> ArgReduceDocGenerator_opset12() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(ArgReduceShapeInference_opset12); };
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 12, generated::ai_onnx::ArgMax_ver12().FillUsing(ArgReduceDocGenerator_opset12()));

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 12, generated::ai_onnx::ArgMin_ver12().FillUsing(ArgReduceDocGenerator_opset12()));

static void ReduceShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  int64_t input_ndim = input_shape.dim_size();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  std::vector<int64_t> axes;
  const auto* const axes_proto = ctx.getAttribute("axes");
  if (axes_proto)
    axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

  for (int64_t& axe : axes) {
    if (axe < 0)
      axe += input_ndim;
  }
  // do we need handle negative axis?
  for (int i = 0; i < input_ndim; ++i) {
    // axes empty means reduce all dim
    if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}

static std::function<void(OpSchema&)> ReduceDocGenerator_opset1() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(ReduceShapeInference_opset1); };
}

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 1, generated::ai_onnx::ReduceMax_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 1, generated::ai_onnx::ReduceMin_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 1, generated::ai_onnx::ReduceSum_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceSumSquare,
    1,
    generated::ai_onnx::ReduceSumSquare_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 1, generated::ai_onnx::ReduceMean_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 1, generated::ai_onnx::ReduceProd_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSum,
    1,
    generated::ai_onnx::ReduceLogSum_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    1,
    generated::ai_onnx::ReduceLogSumExp_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 1, generated::ai_onnx::ReduceL1_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 1, generated::ai_onnx::ReduceL2_ver1().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 11, generated::ai_onnx::ReduceMax_ver11().FillUsing(ReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 11, generated::ai_onnx::ReduceMin_ver11().FillUsing(ReduceDocGenerator_opset1()));

static void ArgReduceShapeInference_opset1(InferenceContext& ctx) {
  // set output element type to int64
  updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  int64_t input_ndim = input_shape.dim_size();
  int64_t axis = 0; // default to 0
  const auto* const axis_proto = ctx.getAttribute("axis");
  if (axis_proto) {
    axis = axis_proto->i();
    if (axis < 0)
      axis += input_ndim;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  // do we need handle negative axis?
  for (int i = 0; i < input_ndim; ++i) {
    if (i != axis) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}

static std::function<void(OpSchema&)> ArgReduceDocGenerator_opset1() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(ArgReduceShapeInference_opset1); };
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 1, generated::ai_onnx::ArgMax_ver1().FillUsing(ArgReduceDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 1, generated::ai_onnx::ArgMin_ver1().FillUsing(ArgReduceDocGenerator_opset1()));

static void ArgReduceShapeInference_opset11(InferenceContext& ctx) {
  // set output element type to int64
  updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  int64_t input_ndim = input_shape.dim_size();
  int64_t axis = 0; // default to 0
  const auto* const axis_proto = ctx.getAttribute("axis");
  if (axis_proto) {
    axis = axis_proto->i();
    if (axis < -input_ndim || axis >= input_ndim) {
      fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
    }
    if (axis < 0)
      axis += input_ndim;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  // do we need handle negative axis?
  for (int i = 0; i < input_ndim; ++i) {
    if (i != axis) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}

static std::function<void(OpSchema&)> ArgReduceDocGenerator_opset11() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(ArgReduceShapeInference_opset11); };
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 11, generated::ai_onnx::ArgMax_ver11().FillUsing(ArgReduceDocGenerator_opset11()));
ONNX_OPERATOR_SET_SCHEMA(ArgMin, 11, generated::ai_onnx::ArgMin_ver11().FillUsing(ArgReduceDocGenerator_opset11()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 13, generated::ai_onnx::ReduceMax_ver13().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 13, generated::ai_onnx::ReduceMin_ver13().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceSumSquare,
    13,
    generated::ai_onnx::ReduceSumSquare_ver13().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 13, generated::ai_onnx::ReduceMean_ver13().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 13, generated::ai_onnx::ReduceProd_ver13().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 13, generated::ai_onnx::ReduceLogSum_ver13().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    13,
    generated::ai_onnx::ReduceLogSumExp_ver13().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 13, generated::ai_onnx::ReduceL1_ver13().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 13, generated::ai_onnx::ReduceL2_ver13().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 18, generated::ai_onnx::ReduceMax_ver18().FillUsing(ReduceOpGenerator()));
ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 18, generated::ai_onnx::ReduceMin_ver18().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 18, generated::ai_onnx::ReduceLogSum_ver18().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    18,
    generated::ai_onnx::ReduceLogSumExp_ver18().FillUsing(ReduceOpGenerator()));
} // namespace ONNX_NAMESPACE
