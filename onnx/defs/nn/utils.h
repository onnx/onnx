// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace defs {
namespace nn {
namespace utils {

/**
 * Reads and validates the 'strides' attribute for Conv/Pool shape inference.
 * Returns the attribute value or a default value if the attribute is not present.
 */
std::vector<int64_t> getConvPoolStrides(InferenceContext& ctx, size_t n_input_dims);

/** Implements shape and type propagation for Attention (23-). */
void AttentionPropagateElemTypeFromInputToOutput(InferenceContext& ctx);

/** Implements CausalMask for Attention. */
bool AttentionAppendFunctionCausalMask(const FunctionBodyBuildContext& ctx, FunctionBuilder& builder, bool padding);

} // namespace utils
} // namespace nn
} // namespace defs
} // namespace ONNX_NAMESPACE
