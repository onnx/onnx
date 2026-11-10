// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
ONNX_API void RNNShapeInference(InferenceContext& ctx) {
  Dim num_directions, seq_length, batch_size, hidden_size, input_size;

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

  // X: [seq_length, batch_size, input_size] (layout=0) or [batch_size, seq_length, input_size] (layout=1)
  if (layout_value == 0) {
    ctx.unifyInputShape(0, {seq_length, batch_size, input_size});
  } else {
    ctx.unifyInputShape(0, {batch_size, seq_length, input_size});
  }

  auto num_outputs = ctx.getNumOutputs();

  if (num_outputs > 0) {
    // Y
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    if (layout_value == 0) {
      updateOutputShape(ctx, 0, {seq_length, num_directions, batch_size, hidden_size});
    } else {
      updateOutputShape(ctx, 0, {batch_size, seq_length, num_directions, hidden_size});
    }
  }

  if (num_outputs > 1) {
    // Y_h
    propagateElemTypeFromInputToOutput(ctx, 0, 1);

    if (layout_value == 0) {
      updateOutputShape(ctx, 1, {num_directions, batch_size, hidden_size});
    } else {
      updateOutputShape(ctx, 1, {batch_size, num_directions, hidden_size});
    }
  }

  if (num_outputs > 2) {
    // Y_c : only in the case of LSTM
    propagateElemTypeFromInputToOutput(ctx, 0, 2);

    if (layout_value == 0) {
      updateOutputShape(ctx, 2, {num_directions, batch_size, hidden_size});
    } else {
      updateOutputShape(ctx, 2, {batch_size, num_directions, hidden_size});
    }
  }
}

static std::function<void(OpSchema&)> RNNDocGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(RNNShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(RNN, 22, generated::ai_onnx::RNN_ver22().FillUsing(RNNDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(GRU, 22, generated::ai_onnx::GRU_ver22().FillUsing(RNNDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(LSTM, 22, generated::ai_onnx::LSTM_ver22().FillUsing(RNNDocGenerator()));
} // namespace ONNX_NAMESPACE
