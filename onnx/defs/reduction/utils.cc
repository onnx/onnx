// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/reduction/utils.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ONNX_NAMESPACE {

static void ReduceShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t keep_dims = 1, noop_with_empty_axes = 0;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  const auto* const noop_attr_proto = ctx.getAttribute("noop_with_empty_axes");
  if (noop_attr_proto) {
    noop_with_empty_axes = noop_attr_proto->i();
  }
  std::vector<int64_t> axes;
  if (ctx.hasInput(1)) { // axes is input
    if (ctx.getAttribute("axes")) {
      fail_shape_inference("axes as an input and attribute cannot be specified at the same time.");
    }

    const TensorProto* axesInitializer = ctx.getInputData(1);
    if (axesInitializer == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    std::vector<int64_t> axes_values = ParseData<int64_t>(axesInitializer);
    axes.assign(axes_values.begin(), axes_values.end());
  } else { // axes is attribute
    const auto* const axes_proto = ctx.getAttribute("axes");
    if (axes_proto)
      axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (noop_with_empty_axes && axes.empty()) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
    return;
  }
  int64_t input_ndim = input_shape.dim_size();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int64_t& axe : axes) {
    if (axe < -input_ndim || axe >= input_ndim) {
      fail_shape_inference("axis must be in [-rank, rank-1]. Input rank was ", input_ndim);
    }
    if (axe < 0)
      axe += input_ndim;
  }
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

std::function<void(OpSchema&)> ReduceOpGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(ReduceShapeInference); };
}
} // namespace ONNX_NAMESPACE
