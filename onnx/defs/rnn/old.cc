// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static void RNNShapeInference_opset14(InferenceContext& ctx) {
  TensorShapeProto::Dimension num_directions, seq_length, batch_size, hidden_size;

  auto direction = getAttribute(ctx, "direction", "forward");
  if ((direction == "forward") || (direction == "reverse"))
    num_directions.set_dim_value(1);
  else if (direction == "bidirectional")
    num_directions.set_dim_value(2);
  // else leave num_directions unknown in case of incorrect attribute value

  auto hidden_size_value = getAttribute(ctx, "hidden_size", -1);
  if (hidden_size_value > 0)
    hidden_size.set_dim_value(hidden_size_value);

  auto layout_value = getAttribute(ctx, "layout", 0);

  if (hasInputShape(ctx, 0)) {
    const auto& first_input_shape = getInputShape(ctx, 0);
    if (first_input_shape.dim_size() != 3) {
      fail_shape_inference("First input tensor must have rank 3");
    }
    seq_length = first_input_shape.dim((layout_value == 0) ? 0 : 1);
    batch_size = first_input_shape.dim((layout_value == 0) ? 1 : 0);
  }

  auto num_outputs = ctx.getNumOutputs();

  if (num_outputs > 0) {
    // Y
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    if (layout_value == 0) {
      auto dims = {seq_length, num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 0, dims);
    } else {
      auto dims = {batch_size, seq_length, num_directions, hidden_size};
      updateOutputShape(ctx, 0, dims);
    }
  }

  if (num_outputs > 1) {
    // Y_h
    propagateElemTypeFromInputToOutput(ctx, 0, 1);

    if (layout_value == 0) {
      auto dims = {num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 1, dims);
    } else {
      auto dims = {batch_size, num_directions, hidden_size};
      updateOutputShape(ctx, 1, dims);
    }
  }

  if (num_outputs > 2) {
    // Y_c : only in the case of LSTM
    propagateElemTypeFromInputToOutput(ctx, 0, 2);

    if (layout_value == 0) {
      auto dims = {num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 2, dims);
    } else {
      auto dims = {batch_size, num_directions, hidden_size};
      updateOutputShape(ctx, 2, dims);
    }
  }
}
static std::function<void(OpSchema&)> RNNDocGenerator_opset14() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(RNNShapeInference_opset14); };
}

ONNX_OPERATOR_SET_SCHEMA(GRU, 14, generated::ai_onnx::GRU_ver14().FillUsing(RNNDocGenerator_opset14()));

ONNX_OPERATOR_SET_SCHEMA(LSTM, 14, generated::ai_onnx::LSTM_ver14().FillUsing(RNNDocGenerator_opset14()));

ONNX_OPERATOR_SET_SCHEMA(RNN, 14, generated::ai_onnx::RNN_ver14().FillUsing(RNNDocGenerator_opset14()));

ONNX_OPERATOR_SET_SCHEMA(GRU, 1, generated::ai_onnx::GRU_ver1());

// Versions 1 to 6 of RNN/LSTM and versions 3 to 6 of GRU:

static void RNNShapeInference_opset1_to_6(InferenceContext& ctx) {
  TensorShapeProto::Dimension num_directions, seq_length, batch_size, hidden_size;

  auto direction = getAttribute(ctx, "direction", "forward");
  if ((direction == "forward") || (direction == "reverse"))
    num_directions.set_dim_value(1);
  else if (direction == "bidirectional")
    num_directions.set_dim_value(2);
  // else leave num_directions unknown in case of incorrect attribute value

  auto hidden_size_value = getAttribute(ctx, "hidden_size", -1);
  if (hidden_size_value > 0)
    hidden_size.set_dim_value(hidden_size_value);

  if (hasInputShape(ctx, 0)) {
    const auto& first_input_shape = getInputShape(ctx, 0);
    if (first_input_shape.dim_size() != 3) {
      fail_shape_inference("First input tensor must have rank 3");
    } else {
      seq_length = first_input_shape.dim(0);
      batch_size = first_input_shape.dim(1);
    }
  }

  // The treatment of outputs is a bit complicated because of the combination of
  // optional outputs and the output_sequence attribute.
  bool output_sequence = (getAttribute(ctx, "output_sequence", 0) != 0);
  auto num_outputs = ctx.getNumOutputs();

  if (num_outputs == 0)
    return; // Unlikely, but seems legal.

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (num_outputs > 1)
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
  if (num_outputs > 2)
    propagateElemTypeFromInputToOutput(ctx, 0, 2);

  if (output_sequence) {
    // No ambiguity in spec
    updateOutputShape(ctx, 0, {seq_length, num_directions, batch_size, hidden_size}); // Y
    if (num_outputs > 1)
      updateOutputShape(ctx, 1, {num_directions, batch_size, hidden_size}); // Y_h
    if (num_outputs > 2)
      updateOutputShape(ctx, 2, {num_directions, batch_size, hidden_size}); // Y_c
  } else {
    // Documentation suggests that the output Y is absent in this case
    // Different tests seem to disagree on whether Y_h and Y_c, if present,
    // should be in positions 0 & 1 or 1 & 2. updateOutputShape(ctx, 0,
    // {num_directions, batch_size, hidden_size});  // Y_h if (num_outputs > 1)
    // updateOutputShape(ctx, 1, {num_directions, batch_size, hidden_size});  //
    // Y_c
  }
}

static std::function<void(OpSchema&)> RNNDocGenerator_opset1_to_6() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(RNNShapeInference_opset1_to_6); };
}

ONNX_OPERATOR_SET_SCHEMA(RNN, 1, generated::ai_onnx::RNN_ver1().FillUsing(RNNDocGenerator_opset1_to_6()));

ONNX_OPERATOR_SET_SCHEMA(GRU, 3, generated::ai_onnx::GRU_ver3().FillUsing(RNNDocGenerator_opset1_to_6()));

ONNX_OPERATOR_SET_SCHEMA(LSTM, 1, generated::ai_onnx::LSTM_ver1().FillUsing(RNNDocGenerator_opset1_to_6()));

} // namespace ONNX_NAMESPACE

// Versions 7 to 13 of RNN/LSTM/GRU

namespace ONNX_NAMESPACE {
static void RNNShapeInference_opset7_to_13(InferenceContext& ctx) {
  TensorShapeProto::Dimension num_directions, seq_length, batch_size, hidden_size;

  auto direction = getAttribute(ctx, "direction", "forward");
  if ((direction == "forward") || (direction == "reverse"))
    num_directions.set_dim_value(1);
  else if (direction == "bidirectional")
    num_directions.set_dim_value(2);
  // else leave num_directions unknown in case of incorrect attribute value

  auto hidden_size_value = getAttribute(ctx, "hidden_size", -1);
  if (hidden_size_value > 0)
    hidden_size.set_dim_value(hidden_size_value);

  if (hasInputShape(ctx, 0)) {
    const auto& first_input_shape = getInputShape(ctx, 0);
    if (first_input_shape.dim_size() != 3) {
      fail_shape_inference("First input tensor must have rank 3");
    }
    seq_length = first_input_shape.dim(0);
    batch_size = first_input_shape.dim(1);
  }

  auto num_outputs = ctx.getNumOutputs();

  if (num_outputs > 0) {
    // Y
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    updateOutputShape(ctx, 0, {seq_length, num_directions, batch_size, hidden_size});
  }

  if (num_outputs > 1) {
    // Y_h
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
    updateOutputShape(ctx, 1, {num_directions, batch_size, hidden_size});
  }

  if (num_outputs > 2) {
    // Y_c : only in the case of LSTM
    propagateElemTypeFromInputToOutput(ctx, 0, 2);
    updateOutputShape(ctx, 2, {num_directions, batch_size, hidden_size});
  }
}

static std::function<void(OpSchema&)> RNNDocGenerator_opset7_to_13() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(RNNShapeInference_opset7_to_13); };
}

ONNX_OPERATOR_SET_SCHEMA(RNN, 7, generated::ai_onnx::RNN_ver7().FillUsing(RNNDocGenerator_opset7_to_13()));

ONNX_OPERATOR_SET_SCHEMA(GRU, 7, generated::ai_onnx::GRU_ver7().FillUsing(RNNDocGenerator_opset7_to_13()));

ONNX_OPERATOR_SET_SCHEMA(LSTM, 7, generated::ai_onnx::LSTM_ver7().FillUsing(RNNDocGenerator_opset7_to_13()));
} // namespace ONNX_NAMESPACE
