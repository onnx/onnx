// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/optional/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    15,
    generated::ai_onnx::OptionalHasElement_ver15().TypeAndShapeInferenceFunction(
        defs::optional::utils::OptionalHasElementShapeInference_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    15,
    generated::ai_onnx::OptionalGetElement_ver15().TypeAndShapeInferenceFunction(
        defs::optional::utils::OptionalGetElementShapeInference_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    Optional,
    15,
    generated::ai_onnx::Optional_ver15().FillUsing(defs::optional::utils::OptionalOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    18,
    generated::ai_onnx::OptionalHasElement_ver18().FillUsing(defs::optional::utils::OptionalHasElementOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    18,
    generated::ai_onnx::OptionalGetElement_ver18().FillUsing(defs::optional::utils::OptionalGetElementOpGenerator()));

} // namespace ONNX_NAMESPACE
