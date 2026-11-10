// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/generator/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
ONNX_OPERATOR_SET_SCHEMA(Constant, 25, generated::ai_onnx::Constant_ver25());

ONNX_OPERATOR_SET_SCHEMA(ConstantOfShape, 25, generated::ai_onnx::ConstantOfShape_ver25());

ONNX_OPERATOR_SET_SCHEMA(EyeLike, 22, generated::ai_onnx::EyeLike_ver22());

ONNX_OPERATOR_SET_SCHEMA(RandomUniform, 22, generated::ai_onnx::RandomUniform_ver22());

ONNX_OPERATOR_SET_SCHEMA(RandomNormal, 22, generated::ai_onnx::RandomNormal_ver22());

ONNX_OPERATOR_SET_SCHEMA(RandomUniformLike, 22, generated::ai_onnx::RandomUniformLike_ver22());

ONNX_OPERATOR_SET_SCHEMA(RandomNormalLike, 22, generated::ai_onnx::RandomNormalLike_ver22());

ONNX_OPERATOR_SET_SCHEMA(Multinomial, 22, generated::ai_onnx::Multinomial_ver22());

ONNX_OPERATOR_SET_SCHEMA(Range, 27, generated::ai_onnx::Range_ver27());

ONNX_OPERATOR_SET_SCHEMA(Bernoulli, 22, generated::ai_onnx::Bernoulli_ver22());
} // namespace ONNX_NAMESPACE
