// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <string>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/reduction/utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 20, generated::ai_onnx::ReduceMax_ver20().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 20, generated::ai_onnx::ReduceMin_ver20().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 13, generated::ai_onnx::ReduceSum_ver13().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceSumSquare,
    18,
    generated::ai_onnx::ReduceSumSquare_ver18().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 18, generated::ai_onnx::ReduceMean_ver18().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 18, generated::ai_onnx::ReduceProd_ver18().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 28, generated::ai_onnx::ReduceLogSum_ver28().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    28,
    generated::ai_onnx::ReduceLogSumExp_ver28().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 18, generated::ai_onnx::ReduceL1_ver18().FillUsing(ReduceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 18, generated::ai_onnx::ReduceL2_ver18().FillUsing(ReduceOpGenerator()));

static std::function<void(OpSchema&)> ArgReduceDocGenerator() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 13, generated::ai_onnx::ArgMax_ver13().FillUsing(ArgReduceDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 13, generated::ai_onnx::ArgMin_ver13().FillUsing(ArgReduceDocGenerator()));

} // namespace ONNX_NAMESPACE
