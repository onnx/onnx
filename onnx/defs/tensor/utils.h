// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Shared type/shape inference for the OneHot operator. 'version' is the opset
// version of the calling schema; before opset 11 'indices' were required to be
// non-negative, which is enforced when 'indices' is a constant.
void oneHotShapeInference(InferenceContext& ctx, int version);

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape);

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& sizes_data,
    TensorShapeProto* output_shape);

// Functions called by ops between opset versions in the name inclusively:

void resizeShapeInferenceHelper_opset7_to_10(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape);

enum class KeepAspectRatioPolicy : std::uint8_t {
  STRETCH,
  NOT_LARGER,
  NOT_SMALLER,
};

void KeepAspectRatioHelper(
    KeepAspectRatioPolicy policy,
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& axes,
    std::vector<int64_t>& sizes_data);

} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE::defs::tensor::utils {

void GatherNDShapeInference(InferenceContext& ctx);
void gridSampleShapeInference(InferenceContext& ctx);
void resizeShapeInference_opset18_to_19(InferenceContext& ctx);
void resizeShapeInference_opset13_to_18(InferenceContext& ctx);
void resizeShapeInference_opset11_to_12(InferenceContext& ctx);
void resizeShapeInference_opset7_to_10(InferenceContext& ctx);
void PadShapeInference(InferenceContext& ctx);
void CastDataPropagation_opset28(DataPropagationContext& ctx);
void CastShapeInference_opset28(InferenceContext& ctx);
bool CastLikeFunctionBody_opset25(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void CastLikeShapeInference_opset25(InferenceContext& ctx);
void BitCastShapeInference_opset26(InferenceContext& ctx);
void ReshapeShapeInference_opset25(InferenceContext& ctx);
void ShapeDataPropagation_opset25(DataPropagationContext& ctx);
void ShapeShapeInference_opset25(InferenceContext& ctx);
void SizeDataPropagation_opset25(DataPropagationContext& ctx);
void SizeShapeInference_opset25(InferenceContext& ctx);
void ConcatDataPropagation_opset13(DataPropagationContext& ctx);
void ConcatShapeInference_opset13(InferenceContext& ctx);
void SplitShapeInference_opset18(InferenceContext& ctx);
void SliceDataPropagation_opset13(DataPropagationContext& ctx);
void SliceShapeInference_opset13(InferenceContext& ctx);
void TransposeShapeInference_opset25(InferenceContext& ctx);
void ScatterShapeInference_opset11(InferenceContext& ctx);
void ScatterElementsShapeInference_opset18(InferenceContext& ctx);
void GatherShapeInference_opset13(InferenceContext& ctx);
void GatherElementsShapeInference_opset13(InferenceContext& ctx);
void SqueezeDataPropagation_opset25(DataPropagationContext& ctx);
void SqueezeShapeInference_opset25(InferenceContext& ctx);
void UnsqueezeDataPropagation_opset25(DataPropagationContext& ctx);
void UnsqueezeShapeInference_opset25(InferenceContext& ctx);
bool BuildContextDependentFunctionBodySpaceToDepth(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
bool BuildContextDependentFunctionBodyDepthToSpace(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void SpaceToDepthShapeInference_opset28(InferenceContext& ctx);
void DepthToSpaceShapeInference_opset28(InferenceContext& ctx);
void TileShapeInference_opset13(InferenceContext& ctx);
void AffineGridShapeInference_opset20(InferenceContext& ctx);
void CompressShapeInference_opset28(InferenceContext& ctx);
void OneHotShapeInference_opset28(InferenceContext& ctx);
void IsNaNShapeInference_opset20(InferenceContext& ctx);
void IsInfShapeInference_opset20(InferenceContext& ctx);
void WhereShapeInference_opset16(InferenceContext& ctx);
void NonZeroShapeInference_opset13(InferenceContext& ctx);
void ReverseSequenceShapeInference_opset28(InferenceContext& ctx);
void UniqueShapeInference_opset28(InferenceContext& ctx);
void TriluShapeInference_opset14(InferenceContext& ctx);
bool CenterCropPadFunctionBody_opset18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void CenterCropPadShapeInference_opset18(InferenceContext& ctx);
void TensorScatterShapeInference_opset24(InferenceContext& ctx);
void CastDataPropagation_opset25(DataPropagationContext& ctx);
void CastShapeInference_opset25(InferenceContext& ctx);
void CastDataPropagation_opset24(DataPropagationContext& ctx);
void CastShapeInference_opset24(InferenceContext& ctx);
void CastDataPropagation_opset23(DataPropagationContext& ctx);
void CastShapeInference_opset23(InferenceContext& ctx);
void CastDataPropagation_opset21(DataPropagationContext& ctx);
void CastShapeInference_opset21(InferenceContext& ctx);
void CastDataPropagation_opset19(DataPropagationContext& ctx);
void CastShapeInference_opset19(InferenceContext& ctx);
void CastDataPropagation_opset13(DataPropagationContext& ctx);
void CastShapeInference_opset13(InferenceContext& ctx);
bool CastLikeFunctionBody_opset24(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void CastLikeShapeInference_opset24(InferenceContext& ctx);
bool CastLikeFunctionBody_opset23(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void CastLikeShapeInference_opset23(InferenceContext& ctx);
bool CastLikeFunctionBody_opset21(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void CastLikeShapeInference_opset21(InferenceContext& ctx);
bool CastLikeFunctionBody_opset19(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void CastLikeShapeInference_opset19(InferenceContext& ctx);
bool CastLikeFunctionBody_opset15(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void CastLikeShapeInference_opset15(InferenceContext& ctx);
void CastShapeInference_opset9(InferenceContext& ctx);
void ReshapeShapeInference_opset24(InferenceContext& ctx);
void ReshapeShapeInference_opset23(InferenceContext& ctx);
void ReshapeShapeInference_opset21(InferenceContext& ctx);
void ReshapeShapeInference_opset19(InferenceContext& ctx);
void ReshapeShapeInference_opset13(InferenceContext& ctx);
void ReshapeShapeInference_opset5(InferenceContext& ctx);
void ShapeDataPropagation_opset24(DataPropagationContext& ctx);
void ShapeShapeInference_opset24(InferenceContext& ctx);
void ShapeDataPropagation_opset23(DataPropagationContext& ctx);
void ShapeShapeInference_opset23(InferenceContext& ctx);
void ShapeOp13DataPropagator(DataPropagationContext& ctx);
void ShapeShapeInference_opset13(InferenceContext& ctx);
void ShapeShapeInference_opset1(InferenceContext& ctx);
void SizeDataPropagation_opset24(DataPropagationContext& ctx);
void SizeShapeInference_opset24(InferenceContext& ctx);
void SizeDataPropagation_opset23(DataPropagationContext& ctx);
void SizeShapeInference_opset23(InferenceContext& ctx);
void SizeShapeInference_opset1(InferenceContext& ctx);
void ConcatShapeInference_opset11(InferenceContext& ctx);
void SplitShapeInference_opset11(InferenceContext& ctx);
void SplitShapeInference_opset13(InferenceContext& ctx);
void SliceShapeInference_opset11(InferenceContext& ctx);
void TransposeShapeInference_opset24(InferenceContext& ctx);
void TransposeShapeInference_opset23(InferenceContext& ctx);
void TransposeShapeInference_opset21(InferenceContext& ctx);
void TransposeShapeInference_opset13(InferenceContext& ctx);
void TransposeShapeInference_opset1(InferenceContext& ctx);
void ScatterNDShapeInference_opset16(InferenceContext& ctx);
void ScatterNDShapeInference_opset13(InferenceContext& ctx);
void ScatterNDShapeInference_opset11(InferenceContext& ctx);
void ScatterElementsShapeInference_opset16(InferenceContext& ctx);
void ScatterElementsShapeInference_opset13(InferenceContext& ctx);
void ScatterElementsShapeInference_opset11(InferenceContext& ctx);
void GatherShapeInference_opset11(InferenceContext& ctx);
void GatherElementsShapeInference_opset11(InferenceContext& ctx);
void SqueezeDataPropagation_opset24(DataPropagationContext& ctx);
void SqueezeShapeInference_opset24(InferenceContext& ctx);
void SqueezeDataPropagation_opset23(DataPropagationContext& ctx);
void SqueezeShapeInference_opset23(InferenceContext& ctx);
void SqueezeDataPropagation_opset21(DataPropagationContext& ctx);
void SqueezeShapeInference_opset21(InferenceContext& ctx);
void SqueezeDataPropagation_opset13(DataPropagationContext& ctx);
void SqueezeShapeInference_opset13(InferenceContext& ctx);
void SqueezeShapeInference_opset11(InferenceContext& ctx);
void UnsqueezeDataPropagation_opset24(DataPropagationContext& ctx);
void UnsqueezeShapeInference_opset24(InferenceContext& ctx);
void UnsqueezeDataPropagation_opset23(DataPropagationContext& ctx);
void UnsqueezeShapeInference_opset23(InferenceContext& ctx);
void UnsqueezeDataPropagation_opset21(DataPropagationContext& ctx);
void UnsqueezeShapeInference_opset21(InferenceContext& ctx);
void UnsqueezeDataPropagation_opset13(DataPropagationContext& ctx);
void UnsqueezeShapeInference_opset13(InferenceContext& ctx);
void UnsqueezeShapeInference_opset11(InferenceContext& ctx);
void SpaceToDepthShapeInference_opset1(InferenceContext& ctx);
void SpaceToDepthShapeInference_opset13(InferenceContext& ctx);
void DepthToSpaceShapeInference_opset11(InferenceContext& ctx);
void DepthToSpaceShapeInference_opset13(InferenceContext& ctx);
void TileShapeInference_opset6(InferenceContext& ctx);
void IsNaNShapeInference_opset9(InferenceContext& ctx);
void IsNaNShapeInference_opset13(InferenceContext& ctx);
void IsInfShapeInference_opset10(InferenceContext& ctx);
void NonZeroShapeInference_opset9(InferenceContext& ctx);
void GatherNDShapeInference_opset12(InferenceContext& ctx);
void PadShapeInference_opset11(InferenceContext& ctx);
void CastShapeInference_opset6(InferenceContext& ctx);
void ConcatShapeInference_opset4(InferenceContext& ctx);
void TileShapeInference_opset1(InferenceContext& ctx);
void UpsampleShapeInference_opset7(InferenceContext& ctx);
void SliceShapeInference_opset1(InferenceContext& ctx);
void SliceShapeInference_opset10(InferenceContext& ctx);
void ScatterShapeInference_opset9(InferenceContext& ctx);
void DepthToSpaceShapeInference_opset1(InferenceContext& ctx);
void GatherShapeInference_opset1(InferenceContext& ctx);
void SqueezeShapeInference_opset1(InferenceContext& ctx);
void UnsqueezeShapeInference_opset1(InferenceContext& ctx);
void OneHotShapeInference_opset9(InferenceContext& ctx);
void SplitShapeInference_opset2(InferenceContext& ctx);
void PadShapeInference_opset2(InferenceContext& ctx);
void GatherNDShapeInference_opset11(InferenceContext& ctx);
void WhereShapeInference_opset9(InferenceContext& ctx);
void PadShapeInference_opset13(InferenceContext& ctx);
void ReshapeShapeInference_opset14(InferenceContext& ctx);
void ShapeDataPropagation_opset21(DataPropagationContext& ctx);
void ShapeShapeInference_opset21(InferenceContext& ctx);
void ShapeDataPropagation_opset19(DataPropagationContext& ctx);
void ShapeShapeInference_opset19(InferenceContext& ctx);
void ShapeDataPropagation_opset15(DataPropagationContext& ctx);
void ShapeShapeInference_opset15(InferenceContext& ctx);
void SizeDataPropagation_opset21(DataPropagationContext& ctx);
void SizeShapeInference_opset21(InferenceContext& ctx);
void SizeDataPropagation_opset19(DataPropagationContext& ctx);
void SizeShapeInference_opset19(InferenceContext& ctx);
void SizeDataPropagation_opset13(DataPropagationContext& ctx);
void SizeShapeInference_opset13(InferenceContext& ctx);
void CompressShapeInference_opset11(InferenceContext& ctx);
void OneHotShapeInference_opset11(InferenceContext& ctx);
void ReverseSequenceShapeInference_opset10(InferenceContext& ctx);
void UniqueShapeInference_opset11(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::tensor::utils
