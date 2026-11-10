// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::nn::utils {

/**
 * Reads and validates the 'strides' attribute for Conv/Pool shape inference.
 * Returns the attribute value or a default value if the attribute is not present.
 */
std::vector<int64_t> getConvPoolStrides(InferenceContext& ctx, size_t n_input_dims);

/** Implements shape and type propagation for Attention (23-). */
void AttentionPropagateElemTypeFromInputToOutput(InferenceContext& ctx);

/** Implements CausalMask for Attention. */
bool AttentionAppendFunctionCausalMask(
    const FunctionBodyBuildContext& ctx,
    FunctionBuilder& builder,
    bool padding,
    bool cast_mask_to_bias = false);

void AveragePoolShapeInference_opset22(InferenceContext& ctx);
void MaxPoolShapeInference_opset22(InferenceContext& ctx);
void maxUnpoolShapeInference(InferenceContext& ctx);
void LpPoolShapeInference(InferenceContext& ctx);
void roiPoolTypeShapeInference(InferenceContext& ctx);
void ConvShapeInference(InferenceContext& ctx);
void QLinearConvShapeInference_opset10(InferenceContext& ctx);
void ConvIntegerShapeInference_opset10(InferenceContext& ctx);
void convTransposeShapeInference(InferenceContext& ctx);
void DeformConvShapeInference_opset22(InferenceContext& ctx);
void globalPoolTypeShapeInference(InferenceContext& ctx);
void BatchNormalizationShapeInference_opset15(InferenceContext& ctx);
void DropoutShapeInference_opset22(InferenceContext& ctx);
void FlattenShapeInference_opset25(InferenceContext& ctx);
void TfIdfVectorizerShapeInference_opset9(InferenceContext& ctx);
void col2imShapeInference(InferenceContext& ctx);
bool BuildContextDependentFunctionBodyLayerNormalizationVer17(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool BuildContextDependentFunctionBodyLayerNormalizationVer18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void LayerNormalizationShapeInference_opset17(InferenceContext& ctx);
void GroupNormalizationShapeInference(InferenceContext& ctx);
bool GroupNormalizationFunctionBody_opset21(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool RMSNormalizationFunctionBody_opset23(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void RMSNormalizationShapeInference_opset23(ONNX_NAMESPACE::InferenceContext& ctx);
bool RotaryEmbeddingFunctionBody_opset23(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void RotaryEmbeddingShapeInference_opset23(ONNX_NAMESPACE::InferenceContext& ctx);
void Attention25Inference(InferenceContext& ctx);
bool AttentionFunctionBody_opset25(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool CausalConvWithStateFunctionBody_opset27(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void CausalConvWithStateShapeInference_opset27(InferenceContext& ctx);
void LinearAttentionShapeInference_opset27(ONNX_NAMESPACE::InferenceContext& ctx);
bool LinearAttentionFunctionBody_opset27(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void globalPoolTypeShapeInference_opset2(InferenceContext& ctx);
void DropoutShapeInference_opset13(InferenceContext& ctx);
void maxUnpoolShapeInference_opset11(InferenceContext& ctx);
void globalPoolTypeShapeInference_opset1(InferenceContext& ctx);
void convTransposeShapeInference_opset11(InferenceContext& ctx);
void DeformConvShapeInference_opset19(InferenceContext& ctx);
void ConvShapeInference_opset11(InferenceContext& ctx);
void roiPoolTypeShapeInference_opset1(InferenceContext& ctx);
void LpPoolShapeInference_opset18(InferenceContext& ctx);
void AveragePoolShapeInference_opset19(InferenceContext& ctx);
void MaxPoolShapeInference_opset12(InferenceContext& ctx);
void DropoutShapeInference_opset12(InferenceContext& ctx);
void FlattenShapeInference_opset24(InferenceContext& ctx);
void FlattenShapeInference_opset23(InferenceContext& ctx);
void FlattenShapeInference_opset21(InferenceContext& ctx);
void FlattenShapeInference_opset13(InferenceContext& ctx);
void FlattenShapeInference_opset11(InferenceContext& ctx);
void PoolShapeInference_opset1_to_8(InferenceContext& ctx);
void AveragePoolShapeInference_opset10(InferenceContext& ctx);
void AveragePoolShapeInference_opset11(InferenceContext& ctx);
void MaxPoolShapeInference_opset10(InferenceContext& ctx);
void MaxPoolShapeInference_opset11(InferenceContext& ctx);
void maxUnpoolShapeInference_opset9(InferenceContext& ctx);
void LpPoolShapeInference_opset2(InferenceContext& ctx);
void LpPoolShapeInference_opset11(InferenceContext& ctx);
void ConvShapeInference_opset1(InferenceContext& ctx);
void convTransposeShapeInference_opset1(InferenceContext& ctx);
void BatchNormalizationShapeInference_opset14(InferenceContext& ctx);
void DropoutShapeInference_opset10(InferenceContext& ctx);
void FlattenShapeInference_opset1(InferenceContext& ctx);
void FlattenShapeInference_opset9(InferenceContext& ctx);
bool GroupNormalizationFunctionBody_opset18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool AttentionFunctionBody_opset23(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool AttentionFunctionBody_opset24(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);

} // namespace ONNX_NAMESPACE::defs::nn::utils
