// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/rnn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
// Kept for backward compatibility of the public API.
ONNX_API void RNNShapeInference(InferenceContext& ctx) {
  defs::rnn::utils::RNNShapeInference(ctx);
}

} // namespace ONNX_NAMESPACE
