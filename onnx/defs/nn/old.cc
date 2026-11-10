// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/nn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(GlobalLpPool, 2, generated::ai_onnx::GlobalLpPool_ver2());

ONNX_OPERATOR_SET_SCHEMA(Dropout, 13, generated::ai_onnx::Dropout_ver13());

ONNX_OPERATOR_SET_SCHEMA(LpNormalization, 1, generated::ai_onnx::LpNormalization_ver1());

ONNX_OPERATOR_SET_SCHEMA(InstanceNormalization, 6, generated::ai_onnx::InstanceNormalization_ver6());

ONNX_OPERATOR_SET_SCHEMA(GlobalAveragePool, 1, generated::ai_onnx::GlobalAveragePool_ver1());
ONNX_OPERATOR_SET_SCHEMA(GlobalMaxPool, 1, generated::ai_onnx::GlobalMaxPool_ver1());

ONNX_OPERATOR_SET_SCHEMA(DeformConv, 19, generated::ai_onnx::DeformConv_ver19());

ONNX_OPERATOR_SET_SCHEMA(ConvTranspose, 11, generated::ai_onnx::ConvTranspose_ver11());

ONNX_OPERATOR_SET_SCHEMA(Conv, 11, generated::ai_onnx::Conv_ver11());

ONNX_OPERATOR_SET_SCHEMA(MaxRoiPool, 1, generated::ai_onnx::MaxRoiPool_ver1());

ONNX_OPERATOR_SET_SCHEMA(LpPool, 18, generated::ai_onnx::LpPool_ver18());

ONNX_OPERATOR_SET_SCHEMA(MaxUnpool, 11, generated::ai_onnx::MaxUnpool_ver11());

ONNX_OPERATOR_SET_SCHEMA(AveragePool, 19, generated::ai_onnx::AveragePool_ver19());

ONNX_OPERATOR_SET_SCHEMA(MaxPool, 12, generated::ai_onnx::MaxPool_ver12());

ONNX_OPERATOR_SET_SCHEMA(Dropout, 12, generated::ai_onnx::Dropout_ver12());

ONNX_OPERATOR_SET_SCHEMA(Flatten, 24, generated::ai_onnx::Flatten_ver24());

ONNX_OPERATOR_SET_SCHEMA(Flatten, 23, generated::ai_onnx::Flatten_ver23());

ONNX_OPERATOR_SET_SCHEMA(Flatten, 21, generated::ai_onnx::Flatten_ver21());

ONNX_OPERATOR_SET_SCHEMA(Flatten, 13, generated::ai_onnx::Flatten_ver13());

ONNX_OPERATOR_SET_SCHEMA(Flatten, 11, generated::ai_onnx::Flatten_ver11());

ONNX_OPERATOR_SET_SCHEMA(LRN, 1, generated::ai_onnx::LRN_ver1());

ONNX_OPERATOR_SET_SCHEMA(MeanVarianceNormalization, 9, generated::ai_onnx::MeanVarianceNormalization_ver9());

ONNX_OPERATOR_SET_SCHEMA(AveragePool, 1, generated::ai_onnx::AveragePool_ver1());

ONNX_OPERATOR_SET_SCHEMA(AveragePool, 7, generated::ai_onnx::AveragePool_ver7());

ONNX_OPERATOR_SET_SCHEMA(AveragePool, 10, generated::ai_onnx::AveragePool_ver10());

ONNX_OPERATOR_SET_SCHEMA(AveragePool, 11, generated::ai_onnx::AveragePool_ver11());

ONNX_OPERATOR_SET_SCHEMA(MaxPool, 1, generated::ai_onnx::MaxPool_ver1());

ONNX_OPERATOR_SET_SCHEMA(MaxPool, 8, generated::ai_onnx::MaxPool_ver8());

ONNX_OPERATOR_SET_SCHEMA(MaxPool, 10, generated::ai_onnx::MaxPool_ver10());

ONNX_OPERATOR_SET_SCHEMA(MaxPool, 11, generated::ai_onnx::MaxPool_ver11());

ONNX_OPERATOR_SET_SCHEMA(MaxUnpool, 9, generated::ai_onnx::MaxUnpool_ver9());

ONNX_OPERATOR_SET_SCHEMA(LpPool, 1, generated::ai_onnx::LpPool_ver1());

ONNX_OPERATOR_SET_SCHEMA(LpPool, 2, generated::ai_onnx::LpPool_ver2());

ONNX_OPERATOR_SET_SCHEMA(LpPool, 11, generated::ai_onnx::LpPool_ver11());

ONNX_OPERATOR_SET_SCHEMA(Conv, 1, generated::ai_onnx::Conv_ver1());

ONNX_OPERATOR_SET_SCHEMA(ConvTranspose, 1, generated::ai_onnx::ConvTranspose_ver1());

ONNX_OPERATOR_SET_SCHEMA(GlobalLpPool, 1, generated::ai_onnx::GlobalLpPool_ver1());

ONNX_OPERATOR_SET_SCHEMA(BatchNormalization, 1, generated::ai_onnx::BatchNormalization_ver1());

ONNX_OPERATOR_SET_SCHEMA(BatchNormalization, 9, generated::ai_onnx::BatchNormalization_ver9());

ONNX_OPERATOR_SET_SCHEMA(BatchNormalization, 14, generated::ai_onnx::BatchNormalization_ver14());

ONNX_OPERATOR_SET_SCHEMA(InstanceNormalization, 1, generated::ai_onnx::InstanceNormalization_ver1());

ONNX_OPERATOR_SET_SCHEMA(Dropout, 1, generated::ai_onnx::Dropout_ver1());

ONNX_OPERATOR_SET_SCHEMA(Dropout, 6, generated::ai_onnx::Dropout_ver6());

ONNX_OPERATOR_SET_SCHEMA(Dropout, 7, generated::ai_onnx::Dropout_ver7());

ONNX_OPERATOR_SET_SCHEMA(Dropout, 10, generated::ai_onnx::Dropout_ver10());

ONNX_OPERATOR_SET_SCHEMA(BatchNormalization, 6, generated::ai_onnx::BatchNormalization_ver6());

ONNX_OPERATOR_SET_SCHEMA(Flatten, 1, generated::ai_onnx::Flatten_ver1());

ONNX_OPERATOR_SET_SCHEMA(Flatten, 9, generated::ai_onnx::Flatten_ver9());

ONNX_OPERATOR_SET_SCHEMA(BatchNormalization, 7, generated::ai_onnx::BatchNormalization_ver7());

ONNX_OPERATOR_SET_SCHEMA(GroupNormalization, 18, generated::ai_onnx::GroupNormalization_ver18());

ONNX_OPERATOR_SET_SCHEMA(Attention, 23, generated::ai_onnx::Attention_ver23());

ONNX_OPERATOR_SET_SCHEMA(Attention, 24, generated::ai_onnx::Attention_ver24());

} // namespace ONNX_NAMESPACE
