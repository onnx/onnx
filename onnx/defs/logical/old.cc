// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#include <string>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/logical/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(Greater, 9, generated::ai_onnx::Greater_ver9());

ONNX_OPERATOR_SET_SCHEMA(Less, 9, generated::ai_onnx::Less_ver9());

ONNX_OPERATOR_SET_SCHEMA(Equal, 11, generated::ai_onnx::Equal_ver11());

ONNX_OPERATOR_SET_SCHEMA(And, 1, generated::ai_onnx::And_ver1());

ONNX_OPERATOR_SET_SCHEMA(Or, 1, generated::ai_onnx::Or_ver1());

ONNX_OPERATOR_SET_SCHEMA(Xor, 1, generated::ai_onnx::Xor_ver1());

ONNX_OPERATOR_SET_SCHEMA(Greater, 1, generated::ai_onnx::Greater_ver1());

ONNX_OPERATOR_SET_SCHEMA(Less, 1, generated::ai_onnx::Less_ver1());

ONNX_OPERATOR_SET_SCHEMA(Equal, 1, generated::ai_onnx::Equal_ver1());

ONNX_OPERATOR_SET_SCHEMA(Equal, 7, generated::ai_onnx::Equal_ver7());

ONNX_OPERATOR_SET_SCHEMA(Greater, 7, generated::ai_onnx::Greater_ver7());

ONNX_OPERATOR_SET_SCHEMA(Less, 7, generated::ai_onnx::Less_ver7());

// Shares same doc generator as newer opset 16 version.
extern std::function<void(OpSchema&)> BinaryLogicDocGenerator();

ONNX_OPERATOR_SET_SCHEMA(LessOrEqual, 12, generated::ai_onnx::LessOrEqual_ver12());

ONNX_OPERATOR_SET_SCHEMA(GreaterOrEqual, 12, generated::ai_onnx::GreaterOrEqual_ver12());

ONNX_OPERATOR_SET_SCHEMA(Equal, 13, generated::ai_onnx::Equal_ver13());

ONNX_OPERATOR_SET_SCHEMA(BitShift, 11, generated::ai_onnx::BitShift_ver11());

} // namespace ONNX_NAMESPACE
