// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/math/utils.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static std::function<void(OpSchema&)> MathDocGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::math::utils::MathShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(
    Add,
    14,
    generated::ai_onnx::Add_ver14()
        .FillUsing(MathDocGenerator())
        .PartialDataPropagationFunction(defs::math::utils::AddDataPropagation_opset14));

ONNX_OPERATOR_SET_SCHEMA(
    Sub,
    14,
    generated::ai_onnx::Sub_ver14()
        .FillUsing(MathDocGenerator())
        .PartialDataPropagationFunction(defs::math::utils::SubDataPropagation_opset14));

ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    28,
    generated::ai_onnx::Mod_ver28().TypeAndShapeInferenceFunction(defs::math::utils::ModShapeInference_opset28));

ONNX_OPERATOR_SET_SCHEMA(
    Mul,
    14,
    generated::ai_onnx::Mul_ver14()
        .FillUsing(MathDocGenerator())
        .PartialDataPropagationFunction(defs::math::utils::MulDataPropagation_opset14));

ONNX_OPERATOR_SET_SCHEMA(Div, 14, generated::ai_onnx::Div_ver14().FillUsing(MathDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    13,
    generated::ai_onnx::Neg_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    13,
    generated::ai_onnx::Abs_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    13,
    generated::ai_onnx::Reciprocal_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    13,
    generated::ai_onnx::Floor_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    13,
    generated::ai_onnx::Ceil_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    13,
    generated::ai_onnx::Sqrt_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    14,
    generated::ai_onnx::Relu_ver14().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    16,
    generated::ai_onnx::LeakyRelu_ver16().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    ThresholdedRelu,
    22,
    generated::ai_onnx::ThresholdedRelu_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    22,
    generated::ai_onnx::Selu_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    22,
    generated::ai_onnx::Elu_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Mish,
    22,
    generated::ai_onnx::Mish_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Celu,
    28,
    generated::ai_onnx::Celu_ver28().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Gelu,
    20,
    generated::ai_onnx::Gelu_ver20()
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBodyGelu)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Swish,
    24,
    generated::ai_onnx::Swish_ver24().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    SwiGLU,
    28,
    generated::ai_onnx::SwiGLU_ver28().TypeAndShapeInferenceFunction(defs::math::utils::SwiGLUShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    13,
    generated::ai_onnx::Exp_ver13().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    13,
    generated::ai_onnx::Log_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    13,
    generated::ai_onnx::Tanh_ver13().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    15,
    generated::ai_onnx::Pow_ver15().TypeAndShapeInferenceFunction(defs::math::utils::PowShapeInference_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    16,
    generated::ai_onnx::PRelu_ver16().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    13,
    generated::ai_onnx::Sigmoid_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    22,
    generated::ai_onnx::HardSigmoid_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSwish,
    22,
    generated::ai_onnx::HardSwish_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

// Generate opschema for element-wise ops. Leaves type constraint "T"
// unspecified.
static std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction(defs::math::utils::ElementwiseMultiOpShapeInference);
  };
}

ONNX_OPERATOR_SET_SCHEMA(Max, 13, generated::ai_onnx::Max_ver13().FillUsing(ElementwiseMultiOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Min, 13, generated::ai_onnx::Min_ver13().FillUsing(ElementwiseMultiOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Sum, 13, generated::ai_onnx::Sum_ver13().FillUsing(ElementwiseMultiOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Mean, 13, generated::ai_onnx::Mean_ver13().FillUsing(ElementwiseMultiOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    13,
    generated::ai_onnx::Clip_ver13()
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBodyClip)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    13,
    generated::ai_onnx::Softmax_ver13()
        .TypeAndShapeInferenceFunction(defs::math::utils::SoftmaxShapeInference_opset13)
        // function body builder for opset version 13 (the default opset version is the same
        // as the operator's since_version.
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::SoftmaxFunctionBody_opset13)
        // function body builder for opset version 18.
        // ReduceSum is updated in opset 18 to have axes as the second input.
        // Therefore function body for opset version 18
        // is different than the one defined using opset version 13.
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::SoftmaxFunctionBody_opset18, 18));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    13,
    generated::ai_onnx::LogSoftmax_ver13()
        .TypeAndShapeInferenceFunction(defs::math::utils::LogSoftmaxShapeInference_opset13)
        // Function for opset 13
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::LogSoftmaxFunctionBody_opset13, 13)
        // Function for opset 18
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::LogSoftmaxFunctionBody_opset18, 18));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    13,
    generated::ai_onnx::Hardmax_ver13().TypeAndShapeInferenceFunction(
        defs::math::utils::HardmaxShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Softsign,
    22,
    generated::ai_onnx::Softsign_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Softplus,
    22,
    generated::ai_onnx::Softplus_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    29,
    generated::ai_onnx::Gemm_ver29()
        .TypeAndShapeInferenceFunction(defs::math::utils::GemmShapeInference)
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBodyGemm));

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    13,
    generated::ai_onnx::MatMul_ver13().TypeAndShapeInferenceFunction(defs::math::utils::MatMulShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    GroupedMatMul,
    29,
    generated::ai_onnx::GroupedMatMul_ver29()
        .TypeAndShapeInferenceFunction(defs::math::utils::GroupedMatMulShapeInference)
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBodyGroupedMatMul));

ONNX_OPERATOR_SET_SCHEMA(TopK, 24, generated::ai_onnx::TopK_ver24().FillUsing(defs::math::utils::TopKOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Sin,
    22,
    generated::ai_onnx::Sin_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Cos,
    22,
    generated::ai_onnx::Cos_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Tan,
    22,
    generated::ai_onnx::Tan_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Asin,
    22,
    generated::ai_onnx::Asin_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Acos,
    22,
    generated::ai_onnx::Acos_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Atan,
    22,
    generated::ai_onnx::Atan_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Expand,
    13,
    generated::ai_onnx::Expand_ver13().TypeAndShapeInferenceFunction(defs::math::utils::ExpandShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Sinh,
    22,
    generated::ai_onnx::Sinh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Cosh,
    22,
    generated::ai_onnx::Cosh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Asinh,
    22,
    generated::ai_onnx::Asinh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Acosh,
    22,
    generated::ai_onnx::Acosh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Atanh,
    22,
    generated::ai_onnx::Atanh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Sign,
    13,
    generated::ai_onnx::Sign_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Erf,
    13,
    generated::ai_onnx::Erf_ver13().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    QLinearMatMul,
    21,
    generated::ai_onnx::QLinearMatMul_ver21().TypeAndShapeInferenceFunction(
        defs::math::utils::QLinearMatMulShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    MatMulInteger,
    10,
    generated::ai_onnx::MatMulInteger_ver10().TypeAndShapeInferenceFunction(
        defs::math::utils::MatMulIntegerShapeInference_opset10));

ONNX_OPERATOR_SET_SCHEMA(
    CumProd,
    26,
    generated::ai_onnx::CumProd_ver26().TypeAndShapeInferenceFunction(
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    CumSum,
    14,
    generated::ai_onnx::CumSum_ver14().TypeAndShapeInferenceFunction(
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Round,
    22,
    generated::ai_onnx::Round_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Det,
    22,
    generated::ai_onnx::Det_ver22().TypeAndShapeInferenceFunction(defs::math::utils::DetShapeInference_opset22));

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    22,
    generated::ai_onnx::NegativeLogLikelihoodLoss_ver22()
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBody)
        .TypeAndShapeInferenceFunction(defs::math::utils::NegativeLogLikelihoodLossShapeInference_opset22));

ONNX_OPERATOR_SET_SCHEMA(
    Einsum,
    28,
    generated::ai_onnx::Einsum_ver28().FillUsing(defs::math::utils::EinsumOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    13,
    generated::ai_onnx::SoftmaxCrossEntropyLoss_ver13()
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBodySCE)
        .TypeAndShapeInferenceFunction(defs::math::utils::SoftmaxCrossEntropyLossShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    DFT,
    20,
    generated::ai_onnx::DFT_ver20().TypeAndShapeInferenceFunction(defs::math::utils::DFTShapeInference_opset20));

static std::function<void(OpSchema&)> CosineSumWindowOpDocGenerator() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction(defs::math::utils::CosineSumWindowOpShapeInference);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    HannWindow,
    17,
    generated::ai_onnx::HannWindow_ver17().FillUsing(CosineSumWindowOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    HammingWindow,
    17,
    generated::ai_onnx::HammingWindow_ver17().FillUsing(CosineSumWindowOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    BlackmanWindow,
    17,
    generated::ai_onnx::BlackmanWindow_ver17().FillUsing(CosineSumWindowOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    MelWeightMatrix,
    17,
    generated::ai_onnx::MelWeightMatrix_ver17().TypeAndShapeInferenceFunction(
        defs::math::utils::MelWeightMatrixShapeInference_opset17));

ONNX_OPERATOR_SET_SCHEMA(
    STFT,
    17,
    generated::ai_onnx::STFT_ver17().TypeAndShapeInferenceFunction(defs::math::utils::STFTShapeInference_opset17));
} // namespace ONNX_NAMESPACE
