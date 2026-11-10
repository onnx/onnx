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

ONNX_OPERATOR_SET_SCHEMA(GRU, 14, generated::ai_onnx::GRU_ver14());

ONNX_OPERATOR_SET_SCHEMA(LSTM, 14, generated::ai_onnx::LSTM_ver14());

ONNX_OPERATOR_SET_SCHEMA(RNN, 14, generated::ai_onnx::RNN_ver14());

ONNX_OPERATOR_SET_SCHEMA(GRU, 1, generated::ai_onnx::GRU_ver1());

// Versions 1 to 6 of RNN/LSTM and versions 3 to 6 of GRU:

ONNX_OPERATOR_SET_SCHEMA(RNN, 1, generated::ai_onnx::RNN_ver1());

ONNX_OPERATOR_SET_SCHEMA(GRU, 3, generated::ai_onnx::GRU_ver3());

ONNX_OPERATOR_SET_SCHEMA(LSTM, 1, generated::ai_onnx::LSTM_ver1());

} // namespace ONNX_NAMESPACE

// Versions 7 to 13 of RNN/LSTM/GRU

namespace ONNX_NAMESPACE {
ONNX_OPERATOR_SET_SCHEMA(RNN, 7, generated::ai_onnx::RNN_ver7());

ONNX_OPERATOR_SET_SCHEMA(GRU, 7, generated::ai_onnx::GRU_ver7());

ONNX_OPERATOR_SET_SCHEMA(LSTM, 7, generated::ai_onnx::LSTM_ver7());
} // namespace ONNX_NAMESPACE
