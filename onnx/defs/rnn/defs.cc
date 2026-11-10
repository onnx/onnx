// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/rnn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
// Kept for backward compatibility of the public API.
ONNX_API void RNNShapeInference(InferenceContext& ctx) {
  defs::rnn::utils::RNNShapeInference(ctx);
}

ONNX_OPERATOR_SET_SCHEMA(RNN, 22, generated::ai_onnx::RNN_ver22());

ONNX_OPERATOR_SET_SCHEMA(GRU, 22, generated::ai_onnx::GRU_ver22());

ONNX_OPERATOR_SET_SCHEMA(LSTM, 22, generated::ai_onnx::LSTM_ver22());
} // namespace ONNX_NAMESPACE
