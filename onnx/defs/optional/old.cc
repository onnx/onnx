// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/optional/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
ONNX_OPERATOR_SET_SCHEMA(OptionalHasElement, 15, generated::ai_onnx::OptionalHasElement_ver15());

ONNX_OPERATOR_SET_SCHEMA(OptionalGetElement, 15, generated::ai_onnx::OptionalGetElement_ver15());

ONNX_OPERATOR_SET_SCHEMA(Optional, 15, generated::ai_onnx::Optional_ver15());

ONNX_OPERATOR_SET_SCHEMA(OptionalHasElement, 18, generated::ai_onnx::OptionalHasElement_ver18());

ONNX_OPERATOR_SET_SCHEMA(OptionalGetElement, 18, generated::ai_onnx::OptionalGetElement_ver18());

} // namespace ONNX_NAMESPACE
