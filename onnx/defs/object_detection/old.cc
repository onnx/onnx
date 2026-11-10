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
    16,
    generated::ai_onnx::RoiAlign_ver16().TypeAndShapeInferenceFunction(
        defs::object_detection::utils::RoiAlignShapeInference_opset16));

ONNX_OPERATOR_SET_SCHEMA(
    RoiAlign,
    10,
    generated::ai_onnx::RoiAlign_ver10().TypeAndShapeInferenceFunction(
        defs::object_detection::utils::RoiAlignShapeInference_opset10));

ONNX_OPERATOR_SET_SCHEMA(
    NonMaxSuppression,
    10,
    generated::ai_onnx::NonMaxSuppression_ver10().TypeAndShapeInferenceFunction(
        defs::object_detection::utils::NonMaxSuppressionShapeInference_opset10));

} // namespace ONNX_NAMESPACE
