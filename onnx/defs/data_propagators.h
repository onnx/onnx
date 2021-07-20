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
  auto rank = input_data_0->dim_size();
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

// Data propagation function for Shape op
// Propagates input shape to output shape
inline void ShapeOpDataPropagator(DataPropagationContext& ctx) {
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    auto input_shape = ctx.getInputType(0)->tensor_type().shape();
    TensorShapeProto tsp;
    tsp.CopyFrom(input_shape);
    ctx.addOutputData(0, std::move(tsp));
  }
}

inline void SizeOpDataPropagator(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

inline int MathOpTwoIntegers(std::string op_type, int a, int b) {
  if (op_type == "Add") {
    return a + b;
  } else if (op_type == "Sub") {
    return a - b;
  } else if (op_type == "Mul") {
    return a * b;
  }
  fail_shape_inference("Wrong op_type name for running propagation: ", op_type);
}

inline void MathOpDataPropagator(DataPropagationContext& ctx, std::string op_type) {
  const auto input_0 = ctx.getInputData(0);
  const auto input_1 = ctx.getInputData(1);
  if (input_0 == nullptr || input_1 == nullptr) {
      return;
  }
  TensorShapeProto tsp;
  if (input_0->dim_size() < input_1->dim_size()) {
    if (input_0->dim(0).has_dim_param()) {
      return;
    }
    int input_0_val = input_0->dim(0).dim_value();
    for (int i = 0; i < input_1->dim_size(); ++i) {
      if (input_1->dim(i).has_dim_param()) {
        return;
      }
      tsp.mutable_dim()->Add()->set_dim_value(
        MathOpTwoIntegers(op_type, input_0_val, input_1->dim(i).dim_value()));
    }
  } else if (input_0->dim_size() > input_1->dim_size()) {
    if (input_1->dim(0).has_dim_param()) {
      return;
    }
    int input_1_val = input_1->dim(0).dim_value();
    for (int i = 0; i < input_0->dim_size(); ++i) {
      if (input_0->dim(i).has_dim_param()) {
        return;
      }
      tsp.mutable_dim()->Add()->set_dim_value(
        MathOpTwoIntegers(op_type, input_0->dim(i).dim_value(), input_1_val));
    }
  } else {
    for (int i = 0; i < input_0->dim_size(); ++i) {
      if (input_0->dim(i).has_dim_param() || input_1->dim(i).has_dim_param()) {
        return;
      }
      tsp.mutable_dim()->Add()->set_dim_value(
          MathOpTwoIntegers(op_type, input_0->dim(i).dim_value(), input_1->dim(i).dim_value()));
    }
  }
  ctx.addOutputData(0, std::move(tsp));
}

inline void ConcatOpDataPropagator(DataPropagationContext& ctx) {
  if (!axisIsZero(ctx)) {
    return;
  }
  TensorShapeProto tsp;
  for (size_t i = 0; i < ctx.getNumInputs(); ++i) {
    const auto input_data = ctx.getInputData(i);
    if (input_data == nullptr) {
      return;
    }
    for (int j = 0; j < input_data->dim_size(); ++j) {
      appendDimToTensorShapeProto(tsp, input_data->dim(j));
    }
  }
  if (tsp.dim_size() > 0) {
    ctx.addOutputData(0, std::move(tsp));
  }
}

inline void GatherOpDataPropagator(DataPropagationContext& ctx) {
  if (!axisIsZero(ctx, true)) {
    return;
  }
  const auto input_data = ctx.getInputData(0);
  const auto input_indices = ctx.getInputData(1);
  if (input_data == nullptr || input_indices == nullptr) {
    return;
  }
  TensorShapeProto tsp;
  for (int i = 0; i < input_indices->dim_size(); ++i) {
    if (input_indices->dim(i).has_dim_value()) {
      int index = input_indices->dim(i).dim_value();
      if (index < input_data->dim_size()) {
        appendDimToTensorShapeProto(tsp, input_data->dim(index));
      }
    }
  }
  if (tsp.dim_size() > 0) {
    ctx.addOutputData(0, std::move(tsp));
  }
}

inline void SliceOpDataPropagator(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  const auto starts = ctx.getInputData(1);
  const auto ends = ctx.getInputData(2);
  const auto axes = ctx.getNumInputs() >= 4 ? ctx.getInputData(3) : nullptr;
  const auto steps = ctx.getNumInputs() >= 5 ? ctx.getInputData(4) : nullptr;

  if (input_data == nullptr || starts == nullptr || ends == nullptr) {
    return;
  }
  if (starts->dim_size() != ends->dim_size()) {
    fail_shape_inference("Input rank for starts and ends should be the same: (",
    starts->dim_size(), ") vs (", ends->dim_size(), ").");
  }
  // Only supports axis = 0 since the data comes from Shape
  if((axes == nullptr || (axes->dim_size() == 1 && axes->dim(0).dim_value() == 0))
    && starts->dim_size () == 1 && ends->dim_size() == 1) {
    int step = 1; // Default step is 1
    if (steps != nullptr) {
      if (steps->dim_size() != 1) {
        return;
      }
      step = steps->dim(0).dim_value();
      if (step == 0) {
        fail_shape_inference("Step cannot be 0 for Slice");
      }
    }
    TensorShapeProto tsp;
    for (int i = starts->dim(0).dim_value(); i < ends->dim(0).dim_value(); i += step) {
      appendDimToTensorShapeProto(tsp, input_data->dim(i));
    }
    if (tsp.dim_size() > 0) {
      ctx.addOutputData(0, std::move(tsp));
    }
  }
}

}