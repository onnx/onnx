/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <optional>

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/preview/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

// ---------------------------------------------------------------------------
// Operator Schema Registration
// ---------------------------------------------------------------------------

ONNX_PREVIEW_OPERATOR_SET_SCHEMA(
    FlexAttention,
    1,
    generated::ai_onnx_preview::FlexAttention_ver1()
        .TypeAndShapeInferenceFunction(defs::preview::utils::FlexAttentionShapeInference)
        .SetContextDependentFunctionBodyBuilder(defs::preview::utils::BuildFlexAttentionFunctionBody));

} // namespace ONNX_NAMESPACE
