/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/text/utils.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(StringConcat, 20, generated::ai_onnx::StringConcat_ver20());

ONNX_OPERATOR_SET_SCHEMA(RegexFullMatch, 20, generated::ai_onnx::RegexFullMatch_ver20());

ONNX_OPERATOR_SET_SCHEMA(StringSplit, 20, generated::ai_onnx::StringSplit_ver20());

ONNX_OPERATOR_SET_SCHEMA(StringNormalizer, 10, generated::ai_onnx::StringNormalizer_ver10());
} // namespace ONNX_NAMESPACE
