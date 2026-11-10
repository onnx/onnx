// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/object_detection/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    RoiAlign,
    22,
    generated::ai_onnx::RoiAlign_ver22().TypeAndShapeInferenceFunction(
        defs::object_detection::utils::RoiAlignShapeInference_opset22));

ONNX_OPERATOR_SET_SCHEMA(
    NonMaxSuppression,
    11,
    generated::ai_onnx::NonMaxSuppression_ver11().TypeAndShapeInferenceFunction(
        defs::object_detection::utils::NonMaxSuppressionShapeInference_opset11));

} // namespace ONNX_NAMESPACE
