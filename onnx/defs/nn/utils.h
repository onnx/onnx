/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace defs {
namespace nn {
namespace utils {

/** Implements shape and type propagation for Attention (23-). */
void AttentionPropagateElemTypeFromInputToOutput(InferenceContext& ctx);

/** Implements CausalMask for Attention. */
bool AttentionAppendFunctionCausalMask(const FunctionBodyBuildContext& ctx, FunctionBuilder& builder, bool padding);

} // namespace utils
} // namespace nn
} // namespace defs
} // namespace ONNX_NAMESPACE
