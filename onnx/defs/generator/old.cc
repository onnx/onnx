// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/generator/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    Bernoulli,
    15,
    generated::ai_onnx::Bernoulli_ver15()
        .TypeAndShapeInferenceFunction(defs::generator::utils::BernoulliShapeInference_opset15)
        .SetContextDependentFunctionBodyBuilder(defs::generator::utils::BernoulliFunctionBody_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    Multinomial,
    7,
    generated::ai_onnx::Multinomial_ver7().TypeAndShapeInferenceFunction(
        defs::generator::utils::MultinomialShapeInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormalLike,
    1,
    generated::ai_onnx::RandomNormalLike_ver1().TypeAndShapeInferenceFunction(
        defs::generator::utils::RandomNormalLikeShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniformLike,
    1,
    generated::ai_onnx::RandomUniformLike_ver1().TypeAndShapeInferenceFunction(
        defs::generator::utils::RandomUniformLikeShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormal,
    1,
    generated::ai_onnx::RandomNormal_ver1().TypeAndShapeInferenceFunction(
        defs::generator::utils::RandomNormalShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniform,
    1,
    generated::ai_onnx::RandomUniform_ver1().TypeAndShapeInferenceFunction(
        defs::generator::utils::RandomUniformShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    EyeLike,
    9,
    generated::ai_onnx::EyeLike_ver9().TypeAndShapeInferenceFunction(
        defs::generator::utils::EyeLikeShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    24,
    generated::ai_onnx::Constant_ver24().TypeAndShapeInferenceFunction(defs::generator::utils::ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    23,
    generated::ai_onnx::Constant_ver23().TypeAndShapeInferenceFunction(defs::generator::utils::ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    21,
    generated::ai_onnx::Constant_ver21().TypeAndShapeInferenceFunction(defs::generator::utils::ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    19,
    generated::ai_onnx::Constant_ver19().TypeAndShapeInferenceFunction(defs::generator::utils::ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    13,
    generated::ai_onnx::Constant_ver13().TypeAndShapeInferenceFunction(defs::generator::utils::ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    12,
    generated::ai_onnx::Constant_ver12().TypeAndShapeInferenceFunction(defs::generator::utils::ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    1,
    generated::ai_onnx::Constant_ver1().TypeAndShapeInferenceFunction(
        defs::generator::utils::ConstantShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    9,
    generated::ai_onnx::Constant_ver9().TypeAndShapeInferenceFunction(
        defs::generator::utils::ConstantShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    11,
    generated::ai_onnx::Constant_ver11().TypeAndShapeInferenceFunction(
        defs::generator::utils::ConstantShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    24,
    generated::ai_onnx::ConstantOfShape_ver24().TypeAndShapeInferenceFunction(
        defs::generator::utils::ConstantOfShapeShapeInference_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    23,
    generated::ai_onnx::ConstantOfShape_ver23().TypeAndShapeInferenceFunction(
        defs::generator::utils::ConstantOfShapeShapeInference_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    21,
    generated::ai_onnx::ConstantOfShape_ver21().TypeAndShapeInferenceFunction(
        defs::generator::utils::ConstantOfShapeShapeInference_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    20,
    generated::ai_onnx::ConstantOfShape_ver20().TypeAndShapeInferenceFunction(
        defs::generator::utils::ConstantOfShapeShapeInference_opset20));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    9,
    generated::ai_onnx::ConstantOfShape_ver9().TypeAndShapeInferenceFunction(
        defs::generator::utils::ConstantOfShapeShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    Range,
    11,
    generated::ai_onnx::Range_ver11().TypeAndShapeInferenceFunction(
        defs::generator::utils::RangeShapeInference_opset11));

} // namespace ONNX_NAMESPACE
