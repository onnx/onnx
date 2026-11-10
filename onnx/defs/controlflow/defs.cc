// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <string>
#include <vector>

#include "onnx/defs/controlflow/utils.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
ONNX_OPERATOR_SET_SCHEMA(If, 25, generated::ai_onnx::If_ver25());

ONNX_OPERATOR_SET_SCHEMA(Loop, 25, generated::ai_onnx::Loop_ver25());

ONNX_OPERATOR_SET_SCHEMA(Scan, 25,
                         generated::ai_onnx::Scan_ver25()); // Shares same shape inference as opset 11

} // namespace ONNX_NAMESPACE
