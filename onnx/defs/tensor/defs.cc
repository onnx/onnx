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

ONNX_OPERATOR_SET_SCHEMA(Cast, 28, generated::ai_onnx::Cast_ver28());

ONNX_OPERATOR_SET_SCHEMA(CastLike, 25, generated::ai_onnx::CastLike_ver25());

ONNX_OPERATOR_SET_SCHEMA(BitCast, 26, generated::ai_onnx::BitCast_ver26());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 25, generated::ai_onnx::Reshape_ver25());

ONNX_OPERATOR_SET_SCHEMA(Shape, 25, generated::ai_onnx::Shape_ver25());

ONNX_OPERATOR_SET_SCHEMA(Size, 25, generated::ai_onnx::Size_ver25());

ONNX_OPERATOR_SET_SCHEMA(Concat, 13, generated::ai_onnx::Concat_ver13());

ONNX_OPERATOR_SET_SCHEMA(Split, 18, generated::ai_onnx::Split_ver18());

ONNX_OPERATOR_SET_SCHEMA(Slice, 13, generated::ai_onnx::Slice_ver13());

ONNX_OPERATOR_SET_SCHEMA(Transpose, 25, generated::ai_onnx::Transpose_ver25());

ONNX_OPERATOR_SET_SCHEMA(Scatter, 11, generated::ai_onnx::Scatter_ver11());

ONNX_OPERATOR_SET_SCHEMA(ScatterND, 29, generated::ai_onnx::ScatterND_ver29());

ONNX_OPERATOR_SET_SCHEMA(ScatterElements, 18, generated::ai_onnx::ScatterElements_ver18());

ONNX_OPERATOR_SET_SCHEMA(Gather, 13, generated::ai_onnx::Gather_ver13());

ONNX_OPERATOR_SET_SCHEMA(GatherElements, 13, generated::ai_onnx::GatherElements_ver13());

ONNX_OPERATOR_SET_SCHEMA(Squeeze, 25, generated::ai_onnx::Squeeze_ver25());

ONNX_OPERATOR_SET_SCHEMA(Unsqueeze, 25, generated::ai_onnx::Unsqueeze_ver25());

ONNX_OPERATOR_SET_SCHEMA(SpaceToDepth, 28, generated::ai_onnx::SpaceToDepth_ver28());

ONNX_OPERATOR_SET_SCHEMA(DepthToSpace, 28, generated::ai_onnx::DepthToSpace_ver28());

ONNX_OPERATOR_SET_SCHEMA(Tile, 13, generated::ai_onnx::Tile_ver13());

ONNX_OPERATOR_SET_SCHEMA(Upsample, 10, generated::ai_onnx::Upsample_ver10());

ONNX_OPERATOR_SET_SCHEMA(Resize, 19, generated::ai_onnx::Resize_ver19());

ONNX_OPERATOR_SET_SCHEMA(GridSample, 22, generated::ai_onnx::GridSample_ver22());

ONNX_OPERATOR_SET_SCHEMA(AffineGrid, 20, generated::ai_onnx::AffineGrid_ver20());

ONNX_OPERATOR_SET_SCHEMA(Identity, 25, generated::ai_onnx::Identity_ver25());

ONNX_OPERATOR_SET_SCHEMA(Compress, 28, generated::ai_onnx::Compress_ver28());

ONNX_OPERATOR_SET_SCHEMA(OneHot, 28, generated::ai_onnx::OneHot_ver28());

ONNX_OPERATOR_SET_SCHEMA(IsNaN, 20, generated::ai_onnx::IsNaN_ver20());

ONNX_OPERATOR_SET_SCHEMA(IsInf, 20, generated::ai_onnx::IsInf_ver20());

ONNX_OPERATOR_SET_SCHEMA(Where, 16, generated::ai_onnx::Where_ver16());

ONNX_OPERATOR_SET_SCHEMA(NonZero, 13, generated::ai_onnx::NonZero_ver13());

ONNX_OPERATOR_SET_SCHEMA(ReverseSequence, 28, generated::ai_onnx::ReverseSequence_ver28());

ONNX_OPERATOR_SET_SCHEMA(Unique, 28, generated::ai_onnx::Unique_ver28());

ONNX_OPERATOR_SET_SCHEMA(GatherND, 29, generated::ai_onnx::GatherND_ver29());

ONNX_OPERATOR_SET_SCHEMA(Pad, 25, generated::ai_onnx::Pad_ver25());

ONNX_OPERATOR_SET_SCHEMA(Trilu, 14, generated::ai_onnx::Trilu_ver14());

ONNX_OPERATOR_SET_SCHEMA(CenterCropPad, 18, generated::ai_onnx::CenterCropPad_ver18());

ONNX_OPERATOR_SET_SCHEMA(TensorScatter, 24, generated::ai_onnx::TensorScatter_ver24());

} // namespace ONNX_NAMESPACE
