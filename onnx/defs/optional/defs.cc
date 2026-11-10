// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/optional/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(Optional, 28, generated::ai_onnx::Optional_ver28());

ONNX_OPERATOR_SET_SCHEMA(OptionalHasElement, 28, generated::ai_onnx::OptionalHasElement_ver28());

ONNX_OPERATOR_SET_SCHEMA(OptionalGetElement, 28, generated::ai_onnx::OptionalGetElement_ver28());
} // namespace ONNX_NAMESPACE
