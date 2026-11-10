// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/training/utils.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(Gradient, 1, generated::ai_onnx_preview_training::Gradient_ver1());

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Adagrad,
    1,
    generated::ai_onnx_preview_training::Adagrad_ver1().TypeAndShapeInferenceFunction(
        defs::training::utils::AdagradShapeInference_opset1));

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Momentum,
    1,
    generated::ai_onnx_preview_training::Momentum_ver1().TypeAndShapeInferenceFunction(
        defs::training::utils::MomentumShapeInference_opset1));

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Adam,
    1,
    generated::ai_onnx_preview_training::Adam_ver1().TypeAndShapeInferenceFunction(
        defs::training::utils::AdamShapeInference_opset1));

} // namespace ONNX_NAMESPACE
