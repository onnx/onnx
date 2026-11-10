// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE::defs::math::utils {

std::function<void(OpSchema&)> TopKOpGenerator();

std::function<void(OpSchema&)> EinsumOpGenerator();

// Unary elementwise ops on float types: T input -> T output, no attrs, no function body.
std::function<void(OpSchema&)> UnaryFloatMathOpGenerator();

template <typename T>
T GetScalarValueFromTensor(const ONNX_NAMESPACE::TensorProto* t) {
  if (t == nullptr) {
    return T{};
  }

  auto data_type = t->data_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<float>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::DOUBLE:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<double>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::INT32:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int32_t>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::INT64:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int64_t>(t).at(0));
    default:
      fail_shape_inference("Unsupported input data type of ", data_type);
  }
}

void MatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int input1Idx, int input2Idx);

void QLinearMatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx);

int64_t MathOpTwoIntegers(std::string_view op_type, int64_t a, int64_t b);

void TopKShapeInference(InferenceContext& ctx);
void EinsumShapeInference(InferenceContext& ctx);
void MathShapeInference(InferenceContext& ctx);
void AddDataPropagation_opset14(DataPropagationContext& ctx);
void SubDataPropagation_opset14(DataPropagationContext& ctx);
void ModShapeInference_opset28(InferenceContext& ctx);
void MulDataPropagation_opset14(DataPropagationContext& ctx);
bool BuildContextDependentFunctionBodyGelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void SwiGLUShapeInference(InferenceContext& ctx);
void PowShapeInference_opset15(InferenceContext& ctx);
void ElementwiseMultiOpShapeInference(InferenceContext& ctx);
bool BuildContextDependentFunctionBodyClip(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool SoftmaxFunctionBody_opset18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool SoftmaxFunctionBody_opset13(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void SoftmaxShapeInference_opset13(InferenceContext& ctx);
bool LogSoftmaxFunctionBody_opset18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool LogSoftmaxFunctionBody_opset13(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void LogSoftmaxShapeInference_opset13(InferenceContext& ctx);
void HardmaxShapeInference_opset13(InferenceContext& ctx);
void GemmShapeInference(InferenceContext& ctx);
bool BuildContextDependentFunctionBodyGemm(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void MatMulShapeInference_opset13(InferenceContext& ctx);
void GroupedMatMulShapeInference(InferenceContext& ctx);
bool BuildContextDependentFunctionBodyGroupedMatMul(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void ExpandShapeInference_opset13(InferenceContext& ctx);
void MatMulIntegerShapeInference_opset10(ONNX_NAMESPACE::InferenceContext& ctx);
void DetShapeInference_opset22(InferenceContext& ctx);
bool BuildContextDependentFunctionBody(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void NegativeLogLikelihoodLossShapeInference_opset22(InferenceContext& ctx);
bool BuildContextDependentFunctionBodySCE(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void SoftmaxCrossEntropyLossShapeInference_opset13(InferenceContext& ctx);
void DFTShapeInference_opset20(ONNX_NAMESPACE::InferenceContext& ctx);
void CosineSumWindowOpShapeInference(ONNX_NAMESPACE::InferenceContext& ctx);
void MelWeightMatrixShapeInference_opset17(ONNX_NAMESPACE::InferenceContext& ctx);
void STFTShapeInference_opset17(ONNX_NAMESPACE::InferenceContext& ctx);
bool BuildContextDependentFunctionBody_opset13(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void NegativeLogLikelihoodLossShapeInference_opset13(InferenceContext& ctx);
void DetShapeInference_opset11(InferenceContext& ctx);
void MathShapeInference_opset13(InferenceContext& ctx);
void MathShapeInference_opset_7(InferenceContext& ctx);
void SoftmaxShapeInference_opset11(InferenceContext& ctx);
void LogSoftmaxShapeInference_opset11(InferenceContext& ctx);
void HardmaxShapeInference_opset11(InferenceContext& ctx);
void ModShapeInference_opset10(InferenceContext& ctx);
void PowShapeInference_opset13(InferenceContext& ctx);
void PowShapeInference_opset12(InferenceContext& ctx);
void ElementwiseMultiOpShapeInference_opset8(InferenceContext& ctx);
void GemmShapeInference_opset13(InferenceContext& ctx);
void GemmShapeInference_opset11(InferenceContext& ctx);
void MatMulShapeInference_opset9(InferenceContext& ctx);
void ExpandShapeInference_opset8(InferenceContext& ctx);
bool BuildContextDependentFunctionBody_opset12(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void NegativeLogLikelihoodLossShapeInference_opset12(InferenceContext& ctx);
bool BuildContextDependentFunctionBodySCE_opset12(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void SoftmaxCrossEntropyLossShapeInference_opset12(InferenceContext& ctx);
void PowShapeInference_opset7(InferenceContext& ctx);
void GemmShapeInference_opset6(InferenceContext& ctx);
void GemmShapeInference_opset7(InferenceContext& ctx);
void GemmShapeInference_opset9(InferenceContext& ctx);
void MatMulShapeInference_opset1(InferenceContext& ctx);
void TopKShapeInference_opset1(InferenceContext& ctx);
void TopKShapeInference_opset10(InferenceContext& ctx);
void ElementwiseMultiOpShapeInference_old(InferenceContext& ctx);
void DFTShapeInference_opset17(ONNX_NAMESPACE::InferenceContext& ctx);
bool BuildContextDependentFunctionBodyCelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void ModShapeInference_opset13(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::math::utils
