// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/data_propagators.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/tensor/utils.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    13,
    generated::ai_onnx::GatherND_ver13().TypeAndShapeInferenceFunction(defs::tensor::utils::GatherNDShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    18,
    generated::ai_onnx::ScatterND_ver18().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    25,
    generated::ai_onnx::Cast_ver25()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset25)
        .PartialDataPropagationFunction(defs::tensor::utils::CastDataPropagation_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    20,
    generated::ai_onnx::GridSample_ver20().TypeAndShapeInferenceFunction(
        defs::tensor::utils::gridSampleShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    24,
    generated::ai_onnx::Cast_ver24()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset24)
        .PartialDataPropagationFunction(defs::tensor::utils::CastDataPropagation_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    23,
    generated::ai_onnx::Cast_ver23()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset23)
        .PartialDataPropagationFunction(defs::tensor::utils::CastDataPropagation_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    21,
    generated::ai_onnx::Cast_ver21()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset21)
        .PartialDataPropagationFunction(defs::tensor::utils::CastDataPropagation_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    19,
    generated::ai_onnx::Cast_ver19()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset19)
        .PartialDataPropagationFunction(defs::tensor::utils::CastDataPropagation_opset19));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    13,
    generated::ai_onnx::Cast_ver13()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset13)
        .PartialDataPropagationFunction(defs::tensor::utils::CastDataPropagation_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    24,
    generated::ai_onnx::CastLike_ver24()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastLikeShapeInference_opset24)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::CastLikeFunctionBody_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    23,
    generated::ai_onnx::CastLike_ver23()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastLikeShapeInference_opset23)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::CastLikeFunctionBody_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    21,
    generated::ai_onnx::CastLike_ver21()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastLikeShapeInference_opset21)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::CastLikeFunctionBody_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    19,
    generated::ai_onnx::CastLike_ver19()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastLikeShapeInference_opset19)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::CastLikeFunctionBody_opset19));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    15,
    generated::ai_onnx::CastLike_ver15()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::CastLikeShapeInference_opset15)
        .SetContextDependentFunctionBodyBuilder(defs::tensor::utils::CastLikeFunctionBody_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    9,
    generated::ai_onnx::Cast_ver9().TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    16,
    generated::ai_onnx::GridSample_ver16().TypeAndShapeInferenceFunction(
        defs::tensor::utils::gridSampleShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    24,
    generated::ai_onnx::Reshape_ver24().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReshapeShapeInference_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    23,
    generated::ai_onnx::Reshape_ver23().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReshapeShapeInference_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    21,
    generated::ai_onnx::Reshape_ver21().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReshapeShapeInference_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    19,
    generated::ai_onnx::Reshape_ver19().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReshapeShapeInference_opset19));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    13,
    generated::ai_onnx::Reshape_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReshapeShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    5,
    generated::ai_onnx::Reshape_ver5().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReshapeShapeInference_opset5));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    24,
    generated::ai_onnx::Shape_ver24()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ShapeShapeInference_opset24)
        .PartialDataPropagationFunction(defs::tensor::utils::ShapeDataPropagation_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    23,
    generated::ai_onnx::Shape_ver23()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ShapeShapeInference_opset23)
        .PartialDataPropagationFunction(defs::tensor::utils::ShapeDataPropagation_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    13,
    generated::ai_onnx::Shape_ver13()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ShapeShapeInference_opset13)
        .PartialDataPropagationFunction(defs::tensor::utils::ShapeOp13DataPropagator));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    1,
    generated::ai_onnx::Shape_ver1()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ShapeShapeInference_opset1)
        .PartialDataPropagationFunction(defs::tensor::utils::ShapeOp13DataPropagator));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    24,
    generated::ai_onnx::Size_ver24()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SizeShapeInference_opset24)
        .PartialDataPropagationFunction(defs::tensor::utils::SizeDataPropagation_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    23,
    generated::ai_onnx::Size_ver23()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SizeShapeInference_opset23)
        .PartialDataPropagationFunction(defs::tensor::utils::SizeDataPropagation_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    1,
    generated::ai_onnx::Size_ver1().TypeAndShapeInferenceFunction(defs::tensor::utils::SizeShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    11,
    generated::ai_onnx::Concat_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ConcatShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    11,
    generated::ai_onnx::Split_ver11().TypeAndShapeInferenceFunction(defs::tensor::utils::SplitShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    13,
    generated::ai_onnx::Split_ver13().TypeAndShapeInferenceFunction(defs::tensor::utils::SplitShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    11,
    generated::ai_onnx::Slice_ver11().TypeAndShapeInferenceFunction(defs::tensor::utils::SliceShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    24,
    generated::ai_onnx::Transpose_ver24().TypeAndShapeInferenceFunction(
        defs::tensor::utils::TransposeShapeInference_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    23,
    generated::ai_onnx::Transpose_ver23().TypeAndShapeInferenceFunction(
        defs::tensor::utils::TransposeShapeInference_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    21,
    generated::ai_onnx::Transpose_ver21().TypeAndShapeInferenceFunction(
        defs::tensor::utils::TransposeShapeInference_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    13,
    generated::ai_onnx::Transpose_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::TransposeShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    1,
    generated::ai_onnx::Transpose_ver1().TypeAndShapeInferenceFunction(
        defs::tensor::utils::TransposeShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    16,
    generated::ai_onnx::ScatterND_ver16().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterNDShapeInference_opset16));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    13,
    generated::ai_onnx::ScatterND_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterNDShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    11,
    generated::ai_onnx::ScatterND_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterNDShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    16,
    generated::ai_onnx::ScatterElements_ver16().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterElementsShapeInference_opset16));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    13,
    generated::ai_onnx::ScatterElements_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterElementsShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    11,
    generated::ai_onnx::ScatterElements_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterElementsShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    11,
    generated::ai_onnx::Gather_ver11()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::GatherShapeInference_opset11)
        .PartialDataPropagationFunction(GatherOp13DataPropagator));

ONNX_OPERATOR_SET_SCHEMA(
    GatherElements,
    11,
    generated::ai_onnx::GatherElements_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::GatherElementsShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    24,
    generated::ai_onnx::Squeeze_ver24()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SqueezeShapeInference_opset24)
        .PartialDataPropagationFunction(defs::tensor::utils::SqueezeDataPropagation_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    23,
    generated::ai_onnx::Squeeze_ver23()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SqueezeShapeInference_opset23)
        .PartialDataPropagationFunction(defs::tensor::utils::SqueezeDataPropagation_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    21,
    generated::ai_onnx::Squeeze_ver21()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SqueezeShapeInference_opset21)
        .PartialDataPropagationFunction(defs::tensor::utils::SqueezeDataPropagation_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    13,
    generated::ai_onnx::Squeeze_ver13()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SqueezeShapeInference_opset13)
        .PartialDataPropagationFunction(defs::tensor::utils::SqueezeDataPropagation_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    11,
    generated::ai_onnx::Squeeze_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::SqueezeShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    24,
    generated::ai_onnx::Unsqueeze_ver24()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::UnsqueezeShapeInference_opset24)
        .PartialDataPropagationFunction(defs::tensor::utils::UnsqueezeDataPropagation_opset24));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    23,
    generated::ai_onnx::Unsqueeze_ver23()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::UnsqueezeShapeInference_opset23)
        .PartialDataPropagationFunction(defs::tensor::utils::UnsqueezeDataPropagation_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    21,
    generated::ai_onnx::Unsqueeze_ver21()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::UnsqueezeShapeInference_opset21)
        .PartialDataPropagationFunction(defs::tensor::utils::UnsqueezeDataPropagation_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    13,
    generated::ai_onnx::Unsqueeze_ver13()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::UnsqueezeShapeInference_opset13)
        .PartialDataPropagationFunction(defs::tensor::utils::UnsqueezeDataPropagation_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    11,
    generated::ai_onnx::Unsqueeze_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::UnsqueezeShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    1,
    generated::ai_onnx::SpaceToDepth_ver1().TypeAndShapeInferenceFunction(
        defs::tensor::utils::SpaceToDepthShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    13,
    generated::ai_onnx::SpaceToDepth_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::SpaceToDepthShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    11,
    generated::ai_onnx::DepthToSpace_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::DepthToSpaceShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    13,
    generated::ai_onnx::DepthToSpace_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::DepthToSpaceShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    6,
    generated::ai_onnx::Tile_ver6().TypeAndShapeInferenceFunction(defs::tensor::utils::TileShapeInference_opset6));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    18,
    generated::ai_onnx::Resize_ver18().TypeAndShapeInferenceFunction(
        defs::tensor::utils::resizeShapeInference_opset13_to_18));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    13,
    generated::ai_onnx::Resize_ver13().TypeAndShapeInferenceFunction(
        defs::tensor::utils::resizeShapeInference_opset13_to_18));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    11,
    generated::ai_onnx::Resize_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::resizeShapeInference_opset11_to_12));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    24,
    generated::ai_onnx::Identity_ver24().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    23,
    generated::ai_onnx::Identity_ver23().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    21,
    generated::ai_onnx::Identity_ver21().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    19,
    generated::ai_onnx::Identity_ver19().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    13,
    generated::ai_onnx::Identity_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    1,
    generated::ai_onnx::Identity_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    9,
    generated::ai_onnx::IsNaN_ver9().TypeAndShapeInferenceFunction(defs::tensor::utils::IsNaNShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    13,
    generated::ai_onnx::IsNaN_ver13().TypeAndShapeInferenceFunction(defs::tensor::utils::IsNaNShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    IsInf,
    10,
    generated::ai_onnx::IsInf_ver10().TypeAndShapeInferenceFunction(defs::tensor::utils::IsInfShapeInference_opset10));

ONNX_OPERATOR_SET_SCHEMA(
    NonZero,
    9,
    generated::ai_onnx::NonZero_ver9().TypeAndShapeInferenceFunction(
        defs::tensor::utils::NonZeroShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    12,
    generated::ai_onnx::GatherND_ver12().TypeAndShapeInferenceFunction(
        defs::tensor::utils::GatherNDShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(Pad, 24, generated::ai_onnx::Pad_ver24().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Pad, 23, generated::ai_onnx::Pad_ver23().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Pad, 21, generated::ai_onnx::Pad_ver21().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Pad, 19, generated::ai_onnx::Pad_ver19().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    11,
    generated::ai_onnx::Pad_ver11().TypeAndShapeInferenceFunction(defs::tensor::utils::PadShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(Cast, 1, generated::ai_onnx::Cast_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    6,
    generated::ai_onnx::Cast_ver6().TypeAndShapeInferenceFunction(defs::tensor::utils::CastShapeInference_opset6));

ONNX_OPERATOR_SET_SCHEMA(Concat, 1, generated::ai_onnx::Concat_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    4,
    generated::ai_onnx::Concat_ver4().TypeAndShapeInferenceFunction(defs::tensor::utils::ConcatShapeInference_opset4));

ONNX_OPERATOR_SET_SCHEMA(Split, 1, generated::ai_onnx::Split_ver1());

ONNX_OPERATOR_SET_SCHEMA(Pad, 1, generated::ai_onnx::Pad_ver1());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 1, generated::ai_onnx::Reshape_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    1,
    generated::ai_onnx::Tile_ver1().TypeAndShapeInferenceFunction(defs::tensor::utils::TileShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(Upsample, 1, generated::ai_onnx::Upsample_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    7,
    generated::ai_onnx::Upsample_ver7().TypeAndShapeInferenceFunction(
        defs::tensor::utils::UpsampleShapeInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    9,
    generated::ai_onnx::Upsample_ver9().TypeAndShapeInferenceFunction(
        defs::tensor::utils::resizeShapeInference_opset7_to_10));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    10,
    generated::ai_onnx::Resize_ver10().TypeAndShapeInferenceFunction(
        defs::tensor::utils::resizeShapeInference_opset7_to_10));

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    1,
    generated::ai_onnx::Slice_ver1().TypeAndShapeInferenceFunction(defs::tensor::utils::SliceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    10,
    generated::ai_onnx::Slice_ver10().TypeAndShapeInferenceFunction(defs::tensor::utils::SliceShapeInference_opset10));

ONNX_OPERATOR_SET_SCHEMA(
    Scatter,
    9,
    generated::ai_onnx::Scatter_ver9().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ScatterShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    1,
    generated::ai_onnx::DepthToSpace_ver1().TypeAndShapeInferenceFunction(
        defs::tensor::utils::DepthToSpaceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    1,
    generated::ai_onnx::Gather_ver1()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::GatherShapeInference_opset1)
        .PartialDataPropagationFunction(GatherOp13DataPropagator));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    1,
    generated::ai_onnx::Squeeze_ver1().TypeAndShapeInferenceFunction(
        defs::tensor::utils::SqueezeShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    1,
    generated::ai_onnx::Unsqueeze_ver1().TypeAndShapeInferenceFunction(
        defs::tensor::utils::UnsqueezeShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    9,
    generated::ai_onnx::OneHot_ver9().TypeAndShapeInferenceFunction(defs::tensor::utils::OneHotShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(Compress, 9, generated::ai_onnx::Compress_ver9());

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    2,
    generated::ai_onnx::Split_ver2().TypeAndShapeInferenceFunction(defs::tensor::utils::SplitShapeInference_opset2));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    2,
    generated::ai_onnx::Pad_ver2().TypeAndShapeInferenceFunction(defs::tensor::utils::PadShapeInference_opset2));

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    11,
    generated::ai_onnx::GatherND_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::GatherNDShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    14,
    generated::ai_onnx::Identity_ver14().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Where,
    9,
    generated::ai_onnx::Where_ver9().TypeAndShapeInferenceFunction(defs::tensor::utils::WhereShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    13,
    generated::ai_onnx::Pad_ver13().TypeAndShapeInferenceFunction(defs::tensor::utils::PadShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(Pad, 18, generated::ai_onnx::Pad_ver18().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    16,
    generated::ai_onnx::Identity_ver16().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    14,
    generated::ai_onnx::Reshape_ver14().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReshapeShapeInference_opset14));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    21,
    generated::ai_onnx::Shape_ver21()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ShapeShapeInference_opset21)
        .PartialDataPropagationFunction(defs::tensor::utils::ShapeDataPropagation_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    19,
    generated::ai_onnx::Shape_ver19()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ShapeShapeInference_opset19)
        .PartialDataPropagationFunction(defs::tensor::utils::ShapeDataPropagation_opset19));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    15,
    generated::ai_onnx::Shape_ver15()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::ShapeShapeInference_opset15)
        .PartialDataPropagationFunction(defs::tensor::utils::ShapeDataPropagation_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    21,
    generated::ai_onnx::Size_ver21()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SizeShapeInference_opset21)
        .PartialDataPropagationFunction(defs::tensor::utils::SizeDataPropagation_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    19,
    generated::ai_onnx::Size_ver19()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SizeShapeInference_opset19)
        .PartialDataPropagationFunction(defs::tensor::utils::SizeDataPropagation_opset19));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    13,
    generated::ai_onnx::Size_ver13()
        .TypeAndShapeInferenceFunction(defs::tensor::utils::SizeShapeInference_opset13)
        .PartialDataPropagationFunction(defs::tensor::utils::SizeDataPropagation_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    11,
    generated::ai_onnx::Compress_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::CompressShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    11,
    generated::ai_onnx::OneHot_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::OneHotShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    ReverseSequence,
    10,
    generated::ai_onnx::ReverseSequence_ver10().TypeAndShapeInferenceFunction(
        defs::tensor::utils::ReverseSequenceShapeInference_opset10));

ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    11,
    generated::ai_onnx::Unique_ver11().TypeAndShapeInferenceFunction(
        defs::tensor::utils::UniqueShapeInference_opset11));

} // namespace ONNX_NAMESPACE
