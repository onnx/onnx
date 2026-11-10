// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/math/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    13,
    generated::ai_onnx::NegativeLogLikelihoodLoss_ver13()
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBody_opset13)
        .TypeAndShapeInferenceFunction(defs::math::utils::NegativeLogLikelihoodLossShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Det,
    11,
    generated::ai_onnx::Det_ver11().TypeAndShapeInferenceFunction(defs::math::utils::DetShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Round,
    11,
    generated::ai_onnx::Round_ver11().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Atanh,
    9,
    generated::ai_onnx::Atanh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Acosh,
    9,
    generated::ai_onnx::Acosh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Asinh,
    9,
    generated::ai_onnx::Asinh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Cosh,
    9,
    generated::ai_onnx::Cosh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Sinh,
    9,
    generated::ai_onnx::Sinh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Atan,
    7,
    generated::ai_onnx::Atan_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Acos,
    7,
    generated::ai_onnx::Acos_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Asin,
    7,
    generated::ai_onnx::Asin_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Tan,
    7,
    generated::ai_onnx::Tan_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Cos,
    7,
    generated::ai_onnx::Cos_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Sin,
    7,
    generated::ai_onnx::Sin_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Softplus,
    1,
    generated::ai_onnx::Softplus_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Softsign,
    1,
    generated::ai_onnx::Softsign_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSwish,
    14,
    generated::ai_onnx::HardSwish_ver14().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    6,
    generated::ai_onnx::HardSigmoid_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Mish,
    18,
    generated::ai_onnx::Mish_ver18().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    6,
    generated::ai_onnx::Elu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    6,
    generated::ai_onnx::Selu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    ThresholdedRelu,
    10,
    generated::ai_onnx::ThresholdedRelu_ver10().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static std::function<void(OpSchema&)> MathDocGenerator_opset13() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::math::utils::MathShapeInference_opset13); };
}

ONNX_OPERATOR_SET_SCHEMA(Add, 13, generated::ai_onnx::Add_ver13().FillUsing(MathDocGenerator_opset13()));

ONNX_OPERATOR_SET_SCHEMA(Sub, 13, generated::ai_onnx::Sub_ver13().FillUsing(MathDocGenerator_opset13()));

ONNX_OPERATOR_SET_SCHEMA(Mul, 13, generated::ai_onnx::Mul_ver13().FillUsing(MathDocGenerator_opset13()));

ONNX_OPERATOR_SET_SCHEMA(Div, 13, generated::ai_onnx::Div_ver13().FillUsing(MathDocGenerator_opset13()));

static std::function<void(OpSchema&)> MathDocGenerator_opset_7() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::math::utils::MathShapeInference_opset_7); };
}

ONNX_OPERATOR_SET_SCHEMA(Add, 7, generated::ai_onnx::Add_ver7().FillUsing(MathDocGenerator_opset_7()));

ONNX_OPERATOR_SET_SCHEMA(Sub, 7, generated::ai_onnx::Sub_ver7().FillUsing(MathDocGenerator_opset_7()));

ONNX_OPERATOR_SET_SCHEMA(Mul, 7, generated::ai_onnx::Mul_ver7().FillUsing(MathDocGenerator_opset_7()));

ONNX_OPERATOR_SET_SCHEMA(Div, 7, generated::ai_onnx::Div_ver7().FillUsing(MathDocGenerator_opset_7()));

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    11,
    generated::ai_onnx::Softmax_ver11().TypeAndShapeInferenceFunction(
        defs::math::utils::SoftmaxShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    11,
    generated::ai_onnx::LogSoftmax_ver11().TypeAndShapeInferenceFunction(
        defs::math::utils::LogSoftmaxShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    11,
    generated::ai_onnx::Hardmax_ver11().TypeAndShapeInferenceFunction(
        defs::math::utils::HardmaxShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    10,
    generated::ai_onnx::Mod_ver10().TypeAndShapeInferenceFunction(defs::math::utils::ModShapeInference_opset10));

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    6,
    generated::ai_onnx::Neg_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    6,
    generated::ai_onnx::Abs_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    6,
    generated::ai_onnx::Reciprocal_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    6,
    generated::ai_onnx::Floor_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    6,
    generated::ai_onnx::Ceil_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    6,
    generated::ai_onnx::Sqrt_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    6,
    generated::ai_onnx::Relu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    13,
    generated::ai_onnx::Relu_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    6,
    generated::ai_onnx::Exp_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    6,
    generated::ai_onnx::Log_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    6,
    generated::ai_onnx::Tanh_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    13,
    generated::ai_onnx::Pow_ver13().TypeAndShapeInferenceFunction(defs::math::utils::PowShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    12,
    generated::ai_onnx::Pow_ver12().TypeAndShapeInferenceFunction(defs::math::utils::PowShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    6,
    generated::ai_onnx::Sigmoid_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

// Generate opschema for element-wise ops. Leaves type constraint "T"
// unspecified.
static std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator_opset8() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction(defs::math::utils::ElementwiseMultiOpShapeInference_opset8);
  };
}

ONNX_OPERATOR_SET_SCHEMA(Max, 12, generated::ai_onnx::Max_ver12().FillUsing(ElementwiseMultiOpDocGenerator_opset8()));

ONNX_OPERATOR_SET_SCHEMA(Min, 12, generated::ai_onnx::Min_ver12().FillUsing(ElementwiseMultiOpDocGenerator_opset8()));

ONNX_OPERATOR_SET_SCHEMA(Sum, 8, generated::ai_onnx::Sum_ver8().FillUsing(ElementwiseMultiOpDocGenerator_opset8()));

ONNX_OPERATOR_SET_SCHEMA(Mean, 8, generated::ai_onnx::Mean_ver8().FillUsing(ElementwiseMultiOpDocGenerator_opset8()));

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    12,
    generated::ai_onnx::Clip_ver12().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    13,
    generated::ai_onnx::Gemm_ver13().TypeAndShapeInferenceFunction(defs::math::utils::GemmShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    11,
    generated::ai_onnx::Gemm_ver11().TypeAndShapeInferenceFunction(defs::math::utils::GemmShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    9,
    generated::ai_onnx::MatMul_ver9().TypeAndShapeInferenceFunction(defs::math::utils::MatMulShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    Expand,
    8,
    generated::ai_onnx::Expand_ver8().TypeAndShapeInferenceFunction(defs::math::utils::ExpandShapeInference_opset8));

ONNX_OPERATOR_SET_SCHEMA(
    Sign,
    9,
    generated::ai_onnx::Sign_ver9().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Erf,
    9,
    generated::ai_onnx::Erf_ver9().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    CumSum,
    11,
    generated::ai_onnx::CumSum_ver11().TypeAndShapeInferenceFunction(
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    12,
    generated::ai_onnx::NegativeLogLikelihoodLoss_ver12()
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBody_opset12)
        .TypeAndShapeInferenceFunction(defs::math::utils::NegativeLogLikelihoodLossShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    12,
    generated::ai_onnx::SoftmaxCrossEntropyLoss_ver12()
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBodySCE_opset12)
        .TypeAndShapeInferenceFunction(defs::math::utils::SoftmaxCrossEntropyLossShapeInference_opset12));

static std::function<void(OpSchema&)> SoftmaxFamilyDocGenerator_opset1() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput); };
}

ONNX_OPERATOR_SET_SCHEMA(Softmax, 1, generated::ai_onnx::Softmax_ver1().FillUsing(SoftmaxFamilyDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    1,
    generated::ai_onnx::LogSoftmax_ver1().FillUsing(SoftmaxFamilyDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(Hardmax, 1, generated::ai_onnx::Hardmax_ver1().FillUsing(SoftmaxFamilyDocGenerator_opset1()));

static std::function<void(OpSchema&)> MathDocGenerator_old_opset6() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput); };
}

ONNX_OPERATOR_SET_SCHEMA(Add, 1, generated::ai_onnx::Add_ver1());

ONNX_OPERATOR_SET_SCHEMA(Sub, 1, generated::ai_onnx::Sub_ver1());

ONNX_OPERATOR_SET_SCHEMA(Mul, 1, generated::ai_onnx::Mul_ver1());

ONNX_OPERATOR_SET_SCHEMA(Div, 1, generated::ai_onnx::Div_ver1());

ONNX_OPERATOR_SET_SCHEMA(Add, 6, generated::ai_onnx::Add_ver6().FillUsing(MathDocGenerator_old_opset6()));

ONNX_OPERATOR_SET_SCHEMA(Sub, 6, generated::ai_onnx::Sub_ver6().FillUsing(MathDocGenerator_old_opset6()));

ONNX_OPERATOR_SET_SCHEMA(Mul, 6, generated::ai_onnx::Mul_ver6().FillUsing(MathDocGenerator_old_opset6()));

ONNX_OPERATOR_SET_SCHEMA(Div, 6, generated::ai_onnx::Div_ver6().FillUsing(MathDocGenerator_old_opset6()));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    1,
    generated::ai_onnx::Pow_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    7,
    generated::ai_onnx::Pow_ver7().TypeAndShapeInferenceFunction(defs::math::utils::PowShapeInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(Neg, 1, generated::ai_onnx::Neg_ver1());

ONNX_OPERATOR_SET_SCHEMA(Abs, 1, generated::ai_onnx::Abs_ver1());

ONNX_OPERATOR_SET_SCHEMA(Reciprocal, 1, generated::ai_onnx::Reciprocal_ver1());

ONNX_OPERATOR_SET_SCHEMA(Floor, 1, generated::ai_onnx::Floor_ver1());

ONNX_OPERATOR_SET_SCHEMA(Ceil, 1, generated::ai_onnx::Ceil_ver1());

ONNX_OPERATOR_SET_SCHEMA(Sqrt, 1, generated::ai_onnx::Sqrt_ver1());

ONNX_OPERATOR_SET_SCHEMA(Relu, 1, generated::ai_onnx::Relu_ver1());

ONNX_OPERATOR_SET_SCHEMA(LeakyRelu, 1, generated::ai_onnx::LeakyRelu_ver1());

ONNX_OPERATOR_SET_SCHEMA(Selu, 1, generated::ai_onnx::Selu_ver1());

ONNX_OPERATOR_SET_SCHEMA(Elu, 1, generated::ai_onnx::Elu_ver1());

ONNX_OPERATOR_SET_SCHEMA(Exp, 1, generated::ai_onnx::Exp_ver1());

ONNX_OPERATOR_SET_SCHEMA(Log, 1, generated::ai_onnx::Log_ver1());

ONNX_OPERATOR_SET_SCHEMA(Tanh, 1, generated::ai_onnx::Tanh_ver1());

ONNX_OPERATOR_SET_SCHEMA(PRelu, 1, generated::ai_onnx::PRelu_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    6,
    generated::ai_onnx::PRelu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    7,
    generated::ai_onnx::PRelu_ver7().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(Sigmoid, 1, generated::ai_onnx::Sigmoid_ver1());

ONNX_OPERATOR_SET_SCHEMA(HardSigmoid, 1, generated::ai_onnx::HardSigmoid_ver1());

ONNX_OPERATOR_SET_SCHEMA(Max, 1, generated::ai_onnx::Max_ver1());

ONNX_OPERATOR_SET_SCHEMA(Min, 1, generated::ai_onnx::Min_ver1());

ONNX_OPERATOR_SET_SCHEMA(Sum, 1, generated::ai_onnx::Sum_ver1());

ONNX_OPERATOR_SET_SCHEMA(Mean, 1, generated::ai_onnx::Mean_ver1());

ONNX_OPERATOR_SET_SCHEMA(Clip, 1, generated::ai_onnx::Clip_ver1());

ONNX_OPERATOR_SET_SCHEMA(Gemm, 1, generated::ai_onnx::Gemm_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    6,
    generated::ai_onnx::Gemm_ver6().TypeAndShapeInferenceFunction(defs::math::utils::GemmShapeInference_opset6));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    7,
    generated::ai_onnx::Gemm_ver7().TypeAndShapeInferenceFunction(defs::math::utils::GemmShapeInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    9,
    generated::ai_onnx::Gemm_ver9().TypeAndShapeInferenceFunction(defs::math::utils::GemmShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    Max,
    6,
    generated::ai_onnx::Max_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    6,
    generated::ai_onnx::Min_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    6,
    generated::ai_onnx::Sum_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    6,
    generated::ai_onnx::Mean_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    1,
    generated::ai_onnx::MatMul_ver1().TypeAndShapeInferenceFunction(defs::math::utils::MatMulShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    1,
    generated::ai_onnx::TopK_ver1().TypeAndShapeInferenceFunction(defs::math::utils::TopKShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    10,
    generated::ai_onnx::TopK_ver10().TypeAndShapeInferenceFunction(defs::math::utils::TopKShapeInference_opset10));

ONNX_OPERATOR_SET_SCHEMA(TopK, 11, generated::ai_onnx::TopK_ver11().FillUsing(defs::math::utils::TopKOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    6,
    generated::ai_onnx::Clip_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    11,
    generated::ai_onnx::Clip_ver11().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator_old() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction(defs::math::utils::ElementwiseMultiOpShapeInference_old);
  };
}

ONNX_OPERATOR_SET_SCHEMA(Max, 8, generated::ai_onnx::Max_ver8().FillUsing(ElementwiseMultiOpDocGenerator_old()));

ONNX_OPERATOR_SET_SCHEMA(Min, 8, generated::ai_onnx::Min_ver8().FillUsing(ElementwiseMultiOpDocGenerator_old()));

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    6,
    generated::ai_onnx::LeakyRelu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    9,
    generated::ai_onnx::PRelu_ver9().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    DFT,
    17,
    generated::ai_onnx::DFT_ver17().TypeAndShapeInferenceFunction(defs::math::utils::DFTShapeInference_opset17));

ONNX_OPERATOR_SET_SCHEMA(
    QLinearMatMul,
    10,
    generated::ai_onnx::QLinearMatMul_ver10().TypeAndShapeInferenceFunction(
        defs::math::utils::QLinearMatMulShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Celu,
    12,
    generated::ai_onnx::Celu_ver12()
        .SetContextDependentFunctionBodyBuilder(defs::math::utils::BuildContextDependentFunctionBodyCelu)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Einsum,
    12,
    generated::ai_onnx::Einsum_ver12().FillUsing(defs::math::utils::EinsumOpGenerator()));
// codeql[cpp/unused-static-variable]: The schema macro defines a debug counter used by static-opset validation.
ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    13,
    generated::ai_onnx::Mod_ver13().TypeAndShapeInferenceFunction(defs::math::utils::ModShapeInference_opset13));

} // namespace ONNX_NAMESPACE
