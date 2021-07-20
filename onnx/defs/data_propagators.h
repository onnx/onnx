/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

inline void appendDimToTensorShapeProto(TensorShapeProto& tsp, const TensorShapeProto_Dimension& dim) {
  if (dim.has_dim_value()) {
    tsp.mutable_dim()->Add()->set_dim_value(dim.dim_value());
  } else if (dim.has_dim_param()) {
    tsp.mutable_dim()->Add()->set_dim_param(dim.dim_param());
  }
}

inline bool axisIsZero(DataPropagationContext& ctx, bool defaultZero = false) {
  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr && !defaultZero) {
    fail_shape_inference("Required attribute axis is missing");
    return false;
  } else if (defaultZero) {
    return true;
  }
  int axis = static_cast<int>(axisAttr->i());
  auto input_data_0 = ctx.getInputData(0);
  if (input_data_0 == nullptr) {
    return false;
  }
  int rank = input_data_0->dim_size();
  if (axis < -rank || axis >= rank) {
    fail_shape_inference("axis must be in [-rank, rank-1].");
    return false;
  }
  if (axis < 0) {
    axis += rank;
  }
  // Only supports axis = 0 since the data comes from Shape
  return axis == 0;
}

inline void PropagateShapeDataFromInputToOutput(DataPropagationContext& ctx, int idx) {
  // propogate input data
  const auto input_data = ctx.getInputData(idx);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.CopyFrom(*input_data);
    ctx.addOutputData(0, std::move(tsp));
  }
}

}