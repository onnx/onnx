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

ONNX_OPERATOR_SET_SCHEMA(GatherND, 13, generated::ai_onnx::GatherND_ver13());
ONNX_OPERATOR_SET_SCHEMA(ScatterND, 18, generated::ai_onnx::ScatterND_ver18());

ONNX_OPERATOR_SET_SCHEMA(Cast, 25, generated::ai_onnx::Cast_ver25());

ONNX_OPERATOR_SET_SCHEMA(GridSample, 20, generated::ai_onnx::GridSample_ver20());

ONNX_OPERATOR_SET_SCHEMA(Cast, 24, generated::ai_onnx::Cast_ver24());

ONNX_OPERATOR_SET_SCHEMA(Cast, 23, generated::ai_onnx::Cast_ver23());

ONNX_OPERATOR_SET_SCHEMA(Cast, 21, generated::ai_onnx::Cast_ver21());

ONNX_OPERATOR_SET_SCHEMA(Cast, 19, generated::ai_onnx::Cast_ver19());

ONNX_OPERATOR_SET_SCHEMA(Cast, 13, generated::ai_onnx::Cast_ver13());

ONNX_OPERATOR_SET_SCHEMA(CastLike, 24, generated::ai_onnx::CastLike_ver24());

ONNX_OPERATOR_SET_SCHEMA(CastLike, 23, generated::ai_onnx::CastLike_ver23());

ONNX_OPERATOR_SET_SCHEMA(CastLike, 21, generated::ai_onnx::CastLike_ver21());

ONNX_OPERATOR_SET_SCHEMA(CastLike, 19, generated::ai_onnx::CastLike_ver19());

ONNX_OPERATOR_SET_SCHEMA(CastLike, 15, generated::ai_onnx::CastLike_ver15());

ONNX_OPERATOR_SET_SCHEMA(Cast, 9, generated::ai_onnx::Cast_ver9());

ONNX_OPERATOR_SET_SCHEMA(GridSample, 16, generated::ai_onnx::GridSample_ver16());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 24, generated::ai_onnx::Reshape_ver24());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 23, generated::ai_onnx::Reshape_ver23());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 21, generated::ai_onnx::Reshape_ver21());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 19, generated::ai_onnx::Reshape_ver19());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 13, generated::ai_onnx::Reshape_ver13());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 5, generated::ai_onnx::Reshape_ver5());

ONNX_OPERATOR_SET_SCHEMA(Shape, 24, generated::ai_onnx::Shape_ver24());

ONNX_OPERATOR_SET_SCHEMA(Shape, 23, generated::ai_onnx::Shape_ver23());

ONNX_OPERATOR_SET_SCHEMA(Shape, 13, generated::ai_onnx::Shape_ver13());

ONNX_OPERATOR_SET_SCHEMA(Shape, 1, generated::ai_onnx::Shape_ver1());

ONNX_OPERATOR_SET_SCHEMA(Size, 24, generated::ai_onnx::Size_ver24());

ONNX_OPERATOR_SET_SCHEMA(Size, 23, generated::ai_onnx::Size_ver23());

ONNX_OPERATOR_SET_SCHEMA(Size, 1, generated::ai_onnx::Size_ver1());

ONNX_OPERATOR_SET_SCHEMA(Concat, 11, generated::ai_onnx::Concat_ver11());

ONNX_OPERATOR_SET_SCHEMA(Split, 11, generated::ai_onnx::Split_ver11());

ONNX_OPERATOR_SET_SCHEMA(Split, 13, generated::ai_onnx::Split_ver13());

ONNX_OPERATOR_SET_SCHEMA(Slice, 11, generated::ai_onnx::Slice_ver11());

ONNX_OPERATOR_SET_SCHEMA(Transpose, 24, generated::ai_onnx::Transpose_ver24());

ONNX_OPERATOR_SET_SCHEMA(Transpose, 23, generated::ai_onnx::Transpose_ver23());

ONNX_OPERATOR_SET_SCHEMA(Transpose, 21, generated::ai_onnx::Transpose_ver21());

ONNX_OPERATOR_SET_SCHEMA(Transpose, 13, generated::ai_onnx::Transpose_ver13());

ONNX_OPERATOR_SET_SCHEMA(Transpose, 1, generated::ai_onnx::Transpose_ver1());

ONNX_OPERATOR_SET_SCHEMA(ScatterND, 16, generated::ai_onnx::ScatterND_ver16());

ONNX_OPERATOR_SET_SCHEMA(ScatterND, 13, generated::ai_onnx::ScatterND_ver13());

ONNX_OPERATOR_SET_SCHEMA(ScatterND, 11, generated::ai_onnx::ScatterND_ver11());

ONNX_OPERATOR_SET_SCHEMA(ScatterElements, 16, generated::ai_onnx::ScatterElements_ver16());

ONNX_OPERATOR_SET_SCHEMA(ScatterElements, 13, generated::ai_onnx::ScatterElements_ver13());

ONNX_OPERATOR_SET_SCHEMA(ScatterElements, 11, generated::ai_onnx::ScatterElements_ver11());

ONNX_OPERATOR_SET_SCHEMA(Gather, 11, generated::ai_onnx::Gather_ver11());

ONNX_OPERATOR_SET_SCHEMA(GatherElements, 11, generated::ai_onnx::GatherElements_ver11());

ONNX_OPERATOR_SET_SCHEMA(Squeeze, 24, generated::ai_onnx::Squeeze_ver24());

ONNX_OPERATOR_SET_SCHEMA(Squeeze, 23, generated::ai_onnx::Squeeze_ver23());

ONNX_OPERATOR_SET_SCHEMA(Squeeze, 21, generated::ai_onnx::Squeeze_ver21());

ONNX_OPERATOR_SET_SCHEMA(Squeeze, 13, generated::ai_onnx::Squeeze_ver13());

ONNX_OPERATOR_SET_SCHEMA(Squeeze, 11, generated::ai_onnx::Squeeze_ver11());

ONNX_OPERATOR_SET_SCHEMA(Unsqueeze, 24, generated::ai_onnx::Unsqueeze_ver24());

ONNX_OPERATOR_SET_SCHEMA(Unsqueeze, 23, generated::ai_onnx::Unsqueeze_ver23());

ONNX_OPERATOR_SET_SCHEMA(Unsqueeze, 21, generated::ai_onnx::Unsqueeze_ver21());

ONNX_OPERATOR_SET_SCHEMA(Unsqueeze, 13, generated::ai_onnx::Unsqueeze_ver13());

ONNX_OPERATOR_SET_SCHEMA(Unsqueeze, 11, generated::ai_onnx::Unsqueeze_ver11());

ONNX_OPERATOR_SET_SCHEMA(SpaceToDepth, 1, generated::ai_onnx::SpaceToDepth_ver1());

ONNX_OPERATOR_SET_SCHEMA(SpaceToDepth, 13, generated::ai_onnx::SpaceToDepth_ver13());

ONNX_OPERATOR_SET_SCHEMA(DepthToSpace, 11, generated::ai_onnx::DepthToSpace_ver11());

ONNX_OPERATOR_SET_SCHEMA(DepthToSpace, 13, generated::ai_onnx::DepthToSpace_ver13());

ONNX_OPERATOR_SET_SCHEMA(Tile, 6, generated::ai_onnx::Tile_ver6());

ONNX_OPERATOR_SET_SCHEMA(Resize, 18, generated::ai_onnx::Resize_ver18());

ONNX_OPERATOR_SET_SCHEMA(Resize, 13, generated::ai_onnx::Resize_ver13());

ONNX_OPERATOR_SET_SCHEMA(Resize, 11, generated::ai_onnx::Resize_ver11());

ONNX_OPERATOR_SET_SCHEMA(Identity, 24, generated::ai_onnx::Identity_ver24());

ONNX_OPERATOR_SET_SCHEMA(Identity, 23, generated::ai_onnx::Identity_ver23());

ONNX_OPERATOR_SET_SCHEMA(Identity, 21, generated::ai_onnx::Identity_ver21());

ONNX_OPERATOR_SET_SCHEMA(Identity, 19, generated::ai_onnx::Identity_ver19());

ONNX_OPERATOR_SET_SCHEMA(Identity, 13, generated::ai_onnx::Identity_ver13());

ONNX_OPERATOR_SET_SCHEMA(Identity, 1, generated::ai_onnx::Identity_ver1());

ONNX_OPERATOR_SET_SCHEMA(IsNaN, 9, generated::ai_onnx::IsNaN_ver9());

ONNX_OPERATOR_SET_SCHEMA(IsNaN, 13, generated::ai_onnx::IsNaN_ver13());

ONNX_OPERATOR_SET_SCHEMA(IsInf, 10, generated::ai_onnx::IsInf_ver10());

ONNX_OPERATOR_SET_SCHEMA(NonZero, 9, generated::ai_onnx::NonZero_ver9());

ONNX_OPERATOR_SET_SCHEMA(GatherND, 12, generated::ai_onnx::GatherND_ver12());

ONNX_OPERATOR_SET_SCHEMA(Pad, 24, generated::ai_onnx::Pad_ver24());

ONNX_OPERATOR_SET_SCHEMA(Pad, 23, generated::ai_onnx::Pad_ver23());

ONNX_OPERATOR_SET_SCHEMA(Pad, 21, generated::ai_onnx::Pad_ver21());

ONNX_OPERATOR_SET_SCHEMA(Pad, 19, generated::ai_onnx::Pad_ver19());

ONNX_OPERATOR_SET_SCHEMA(Pad, 11, generated::ai_onnx::Pad_ver11());

ONNX_OPERATOR_SET_SCHEMA(Cast, 1, generated::ai_onnx::Cast_ver1());

ONNX_OPERATOR_SET_SCHEMA(Cast, 6, generated::ai_onnx::Cast_ver6());

ONNX_OPERATOR_SET_SCHEMA(Concat, 1, generated::ai_onnx::Concat_ver1());

ONNX_OPERATOR_SET_SCHEMA(Concat, 4, generated::ai_onnx::Concat_ver4());

ONNX_OPERATOR_SET_SCHEMA(Split, 1, generated::ai_onnx::Split_ver1());

ONNX_OPERATOR_SET_SCHEMA(Pad, 1, generated::ai_onnx::Pad_ver1());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 1, generated::ai_onnx::Reshape_ver1());

ONNX_OPERATOR_SET_SCHEMA(Tile, 1, generated::ai_onnx::Tile_ver1());

ONNX_OPERATOR_SET_SCHEMA(Upsample, 1, generated::ai_onnx::Upsample_ver1());

ONNX_OPERATOR_SET_SCHEMA(Upsample, 7, generated::ai_onnx::Upsample_ver7());

ONNX_OPERATOR_SET_SCHEMA(Upsample, 9, generated::ai_onnx::Upsample_ver9());

ONNX_OPERATOR_SET_SCHEMA(Resize, 10, generated::ai_onnx::Resize_ver10());

ONNX_OPERATOR_SET_SCHEMA(Slice, 1, generated::ai_onnx::Slice_ver1());

ONNX_OPERATOR_SET_SCHEMA(Slice, 10, generated::ai_onnx::Slice_ver10());

ONNX_OPERATOR_SET_SCHEMA(Scatter, 9, generated::ai_onnx::Scatter_ver9());

ONNX_OPERATOR_SET_SCHEMA(DepthToSpace, 1, generated::ai_onnx::DepthToSpace_ver1());

ONNX_OPERATOR_SET_SCHEMA(Gather, 1, generated::ai_onnx::Gather_ver1());

ONNX_OPERATOR_SET_SCHEMA(Squeeze, 1, generated::ai_onnx::Squeeze_ver1());

ONNX_OPERATOR_SET_SCHEMA(Unsqueeze, 1, generated::ai_onnx::Unsqueeze_ver1());

ONNX_OPERATOR_SET_SCHEMA(OneHot, 9, generated::ai_onnx::OneHot_ver9());

ONNX_OPERATOR_SET_SCHEMA(Compress, 9, generated::ai_onnx::Compress_ver9());

ONNX_OPERATOR_SET_SCHEMA(Split, 2, generated::ai_onnx::Split_ver2());

ONNX_OPERATOR_SET_SCHEMA(Pad, 2, generated::ai_onnx::Pad_ver2());

ONNX_OPERATOR_SET_SCHEMA(GatherND, 11, generated::ai_onnx::GatherND_ver11());

ONNX_OPERATOR_SET_SCHEMA(Identity, 14, generated::ai_onnx::Identity_ver14());

ONNX_OPERATOR_SET_SCHEMA(Where, 9, generated::ai_onnx::Where_ver9());

ONNX_OPERATOR_SET_SCHEMA(Pad, 13, generated::ai_onnx::Pad_ver13());

ONNX_OPERATOR_SET_SCHEMA(Pad, 18, generated::ai_onnx::Pad_ver18());

ONNX_OPERATOR_SET_SCHEMA(Identity, 16, generated::ai_onnx::Identity_ver16());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 14, generated::ai_onnx::Reshape_ver14());

ONNX_OPERATOR_SET_SCHEMA(Shape, 21, generated::ai_onnx::Shape_ver21());

ONNX_OPERATOR_SET_SCHEMA(Shape, 19, generated::ai_onnx::Shape_ver19());

ONNX_OPERATOR_SET_SCHEMA(Shape, 15, generated::ai_onnx::Shape_ver15());

ONNX_OPERATOR_SET_SCHEMA(Size, 21, generated::ai_onnx::Size_ver21());

ONNX_OPERATOR_SET_SCHEMA(Size, 19, generated::ai_onnx::Size_ver19());

ONNX_OPERATOR_SET_SCHEMA(Size, 13, generated::ai_onnx::Size_ver13());

ONNX_OPERATOR_SET_SCHEMA(Compress, 11, generated::ai_onnx::Compress_ver11());

ONNX_OPERATOR_SET_SCHEMA(OneHot, 11, generated::ai_onnx::OneHot_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReverseSequence, 10, generated::ai_onnx::ReverseSequence_ver10());

ONNX_OPERATOR_SET_SCHEMA(Unique, 11, generated::ai_onnx::Unique_ver11());

} // namespace ONNX_NAMESPACE
