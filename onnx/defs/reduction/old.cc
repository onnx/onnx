// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/reduction/utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 12, generated::ai_onnx::ReduceMax_ver12());

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 12, generated::ai_onnx::ReduceMin_ver12());

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 11, generated::ai_onnx::ReduceSum_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 11, generated::ai_onnx::ReduceSumSquare_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 11, generated::ai_onnx::ReduceMean_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 11, generated::ai_onnx::ReduceProd_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 11, generated::ai_onnx::ReduceLogSum_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSumExp, 11, generated::ai_onnx::ReduceLogSumExp_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 11, generated::ai_onnx::ReduceL1_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 11, generated::ai_onnx::ReduceL2_ver11());

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 12, generated::ai_onnx::ArgMax_ver12());

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 12, generated::ai_onnx::ArgMin_ver12());

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 1, generated::ai_onnx::ReduceMax_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 1, generated::ai_onnx::ReduceMin_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 1, generated::ai_onnx::ReduceSum_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 1, generated::ai_onnx::ReduceSumSquare_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 1, generated::ai_onnx::ReduceMean_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 1, generated::ai_onnx::ReduceProd_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 1, generated::ai_onnx::ReduceLogSum_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSumExp, 1, generated::ai_onnx::ReduceLogSumExp_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 1, generated::ai_onnx::ReduceL1_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 1, generated::ai_onnx::ReduceL2_ver1());

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 11, generated::ai_onnx::ReduceMax_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 11, generated::ai_onnx::ReduceMin_ver11());

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 1, generated::ai_onnx::ArgMax_ver1());

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 1, generated::ai_onnx::ArgMin_ver1());

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 11, generated::ai_onnx::ArgMax_ver11());
ONNX_OPERATOR_SET_SCHEMA(ArgMin, 11, generated::ai_onnx::ArgMin_ver11());

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 13, generated::ai_onnx::ReduceMax_ver13());
ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 13, generated::ai_onnx::ReduceMin_ver13());
ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 13, generated::ai_onnx::ReduceSumSquare_ver13());
ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 13, generated::ai_onnx::ReduceMean_ver13());
ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 13, generated::ai_onnx::ReduceProd_ver13());
ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 13, generated::ai_onnx::ReduceLogSum_ver13());
ONNX_OPERATOR_SET_SCHEMA(ReduceLogSumExp, 13, generated::ai_onnx::ReduceLogSumExp_ver13());
ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 13, generated::ai_onnx::ReduceL1_ver13());
ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 13, generated::ai_onnx::ReduceL2_ver13());

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 18, generated::ai_onnx::ReduceMax_ver18());
ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 18, generated::ai_onnx::ReduceMin_ver18());

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 18, generated::ai_onnx::ReduceLogSum_ver18());

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSumExp, 18, generated::ai_onnx::ReduceLogSumExp_ver18());
} // namespace ONNX_NAMESPACE
