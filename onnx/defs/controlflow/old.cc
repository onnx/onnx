// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/controlflow/utils.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(If, 24, generated::ai_onnx::If_ver24());

ONNX_OPERATOR_SET_SCHEMA(If, 23, generated::ai_onnx::If_ver23());

ONNX_OPERATOR_SET_SCHEMA(If, 21, generated::ai_onnx::If_ver21());

ONNX_OPERATOR_SET_SCHEMA(If, 19, generated::ai_onnx::If_ver19());

ONNX_OPERATOR_SET_SCHEMA(If, 16, generated::ai_onnx::If_ver16());

ONNX_OPERATOR_SET_SCHEMA(Loop, 24, generated::ai_onnx::Loop_ver24());

ONNX_OPERATOR_SET_SCHEMA(Loop, 23, generated::ai_onnx::Loop_ver23());

ONNX_OPERATOR_SET_SCHEMA(Loop, 21, generated::ai_onnx::Loop_ver21());

ONNX_OPERATOR_SET_SCHEMA(Loop, 19, generated::ai_onnx::Loop_ver19());

ONNX_OPERATOR_SET_SCHEMA(Loop, 16, generated::ai_onnx::Loop_ver16());

ONNX_OPERATOR_SET_SCHEMA(Scan, 24,
                         generated::ai_onnx::Scan_ver24()); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(Scan, 23,
                         generated::ai_onnx::Scan_ver23()); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(Scan, 21,
                         generated::ai_onnx::Scan_ver21()); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(Scan, 19,
                         generated::ai_onnx::Scan_ver19()); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(Scan, 16,
                         generated::ai_onnx::Scan_ver16()); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(Scan, 8, generated::ai_onnx::Scan_ver8());

ONNX_OPERATOR_SET_SCHEMA(Loop, 1, generated::ai_onnx::Loop_ver1());

ONNX_OPERATOR_SET_SCHEMA(Loop, 11, generated::ai_onnx::Loop_ver11());

ONNX_OPERATOR_SET_SCHEMA(Scan, 9, generated::ai_onnx::Scan_ver9());

ONNX_OPERATOR_SET_SCHEMA(If, 1, generated::ai_onnx::If_ver1());

ONNX_OPERATOR_SET_SCHEMA(If, 11, generated::ai_onnx::If_ver11());

ONNX_OPERATOR_SET_SCHEMA(If, 13, generated::ai_onnx::If_ver13());

ONNX_OPERATOR_SET_SCHEMA(Loop, 13, generated::ai_onnx::Loop_ver13());

ONNX_OPERATOR_SET_SCHEMA(Scan, 11, generated::ai_onnx::Scan_ver11());

} // namespace ONNX_NAMESPACE
