// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/data_propagators.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/tensor/utils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    28,
    generated::ai_onnx::Cast_ver28()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset28)
        .PartialDataPropagationFunction(defs::tensor::utils::CastDataPropagation_opset28));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    25,
    generated::ai_onnx::CastLike_ver25()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastLikeShapeInference_opset25)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::CastLikeFunctionBody_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    BitCast,
    26,
    generated::ai_onnx::BitCast_ver26().TypeAndShapeInferenceFunction(
        defs::tensor::utils::BitCastShapeInference_opset26));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    25,
    generated::ai_onnx::Reshape_ver25().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReshapeShapeInference_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    25,
    generated::ai_onnx::Shape_ver25()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ShapeShapeInference_opset25)
        .PartialDataPropagationFunction(defs::tensor::utils::ShapeDataPropagation_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    25,
    generated::ai_onnx::Size_ver25()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SizeShapeInference_opset25)
        .PartialDataPropagationFunction(defs::tensor::utils::SizeDataPropagation_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    13,
    generated::ai_onnx::Concat_ver13()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ConcatShapeInference_opset13)
        .PartialDataPropagationFunction(defs::tensor::utils::ConcatDataPropagation_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    18,
    generated::ai_onnx::Split_ver18().TypeAndShapeInferenceFunction(defs::tensor::utils::SplitShapeInference_opset18));

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    13,
    generated::ai_onnx::Slice_ver13()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SliceShapeInference_opset13)
        .PartialDataPropagationFunction(defs::tensor::utils::SliceDataPropagation_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    25,
    generated::ai_onnx::Transpose_ver25().TypeAndShapeInferenceFunction(
        defs::tensor::utils::TransposeShapeInference_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    Scatter,
    11,
    generated::ai_onnx::Scatter_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    29,
    generated::ai_onnx::ScatterND_ver29().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    18,
    generated::ai_onnx::ScatterElements_ver18().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterElementsShapeInference_opset18));

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    13,
    generated::ai_onnx::Gather_ver13()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::GatherShapeInference_opset13)
        .PartialDataPropagationFunction(GatherOp13DataPropagator));

ONNX_OPERATOR_SET_SCHEMA(
    GatherElements,
    13,
    generated::ai_onnx::GatherElements_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::GatherElementsShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    25,
    generated::ai_onnx::Squeeze_ver25()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SqueezeShapeInference_opset25)
        .PartialDataPropagationFunction(defs::tensor::utils::SqueezeDataPropagation_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    25,
    generated::ai_onnx::Unsqueeze_ver25()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::UnsqueezeShapeInference_opset25)
        .PartialDataPropagationFunction(defs::tensor::utils::UnsqueezeDataPropagation_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    28,
    generated::ai_onnx::SpaceToDepth_ver28()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SpaceToDepthShapeInference_opset28)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::BuildContextDependentFunctionBodySpaceToDepth));

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    28,
    generated::ai_onnx::DepthToSpace_ver28()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::DepthToSpaceShapeInference_opset28)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::BuildContextDependentFunctionBodyDepthToSpace));

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    13,
    generated::ai_onnx::Tile_ver13().TypeAndShapeInferenceFunction(defs::tensor::utils::TileShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    10,
    generated::ai_onnx::Upsample_ver10().TypeAndShapeInferenceFunction(
        defs::tensor::utils::resizeShapeInference_opset7_to_10));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    19,
    generated::ai_onnx::Resize_ver19().TypeAndShapeInferenceFunction(
        defs::tensor::utils::resizeShapeInference_opset18_to_19));

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    22,
    generated::ai_onnx::GridSample_ver22().TypeAndShapeInferenceFunction(
        defs::tensor::utils::gridSampleShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    AffineGrid,
    20,
    generated::ai_onnx::AffineGrid_ver20().TypeAndShapeInferenceFunction(
        defs::tensor::utils::AffineGridShapeInference_opset20));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    25,
    generated::ai_onnx::Identity_ver25().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    28,
    generated::ai_onnx::Compress_ver28().TypeAndShapeInferenceFunction(
        defs::tensor::utils::CompressShapeInference_opset28));

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    28,
    generated::ai_onnx::OneHot_ver28().TypeAndShapeInferenceFunction(
        defs::tensor::utils::OneHotShapeInference_opset28));

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    20,
    generated::ai_onnx::IsNaN_ver20().TypeAndShapeInferenceFunction(defs::tensor::utils::IsNaNShapeInference_opset20));

ONNX_OPERATOR_SET_SCHEMA(
    IsInf,
    20,
    generated::ai_onnx::IsInf_ver20().TypeAndShapeInferenceFunction(defs::tensor::utils::IsInfShapeInference_opset20));

ONNX_OPERATOR_SET_SCHEMA(
    Where,
    16,
    generated::ai_onnx::Where_ver16().TypeAndShapeInferenceFunction(defs::tensor::utils::WhereShapeInference_opset16));

ONNX_OPERATOR_SET_SCHEMA(
    NonZero,
    13,
    generated::ai_onnx::NonZero_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::NonZeroShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    ReverseSequence,
    28,
    generated::ai_onnx::ReverseSequence_ver28().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReverseSequenceShapeInference_opset28));

ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    28,
    generated::ai_onnx::Unique_ver28().TypeAndShapeInferenceFunction(
        defs::tensor::utils::UniqueShapeInference_opset28));

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    29,
    generated::ai_onnx::GatherND_ver29().TypeAndShapeInferenceFunction(defs::tensor::utils::GatherNDShapeInference));

ONNX_OPERATOR_SET_SCHEMA(Pad, 25, generated::ai_onnx::Pad_ver25().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Trilu,
    14,
    generated::ai_onnx::Trilu_ver14().TypeAndShapeInferenceFunction(defs::tensor::utils::TriluShapeInference_opset14));

ONNX_OPERATOR_SET_SCHEMA(
    CenterCropPad,
    18,
    generated::ai_onnx::CenterCropPad_ver18()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CenterCropPadShapeInference_opset18)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::CenterCropPadFunctionBody_opset18));

ONNX_OPERATOR_SET_SCHEMA(
    TensorScatter,
    24,
    generated::ai_onnx::TensorScatter_ver24().TypeAndShapeInferenceFunction(
        defs::tensor::utils::TensorScatterShapeInference_opset24));

} // namespace ONNX_NAMESPACE
