// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::preview::utils {

void FlexAttentionShapeInference(InferenceContext& ctx);
bool BuildFlexAttentionFunctionBody(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);

} // namespace ONNX_NAMESPACE::defs::preview::utils
