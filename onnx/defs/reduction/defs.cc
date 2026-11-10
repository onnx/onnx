// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <string>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/reduction/utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 20, generated::ai_onnx::ReduceMax_ver20());

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 20, generated::ai_onnx::ReduceMin_ver20());

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 13, generated::ai_onnx::ReduceSum_ver13());

ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 18, generated::ai_onnx::ReduceSumSquare_ver18());

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 18, generated::ai_onnx::ReduceMean_ver18());

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 18, generated::ai_onnx::ReduceProd_ver18());

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 28, generated::ai_onnx::ReduceLogSum_ver28());

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSumExp, 28, generated::ai_onnx::ReduceLogSumExp_ver28());

ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 18, generated::ai_onnx::ReduceL1_ver18());

ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 18, generated::ai_onnx::ReduceL2_ver18());

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 13, generated::ai_onnx::ArgMax_ver13());

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 13, generated::ai_onnx::ArgMin_ver13());

} // namespace ONNX_NAMESPACE
