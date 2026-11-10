// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::logical::utils {

void unaryLogicalOpInference(InferenceContext& ctx);
void binaryLogicalOpInference(InferenceContext& ctx);
void BitShiftShapeInference_opset28(InferenceContext& ctx);
void BinaryBitwiseShapeInference(InferenceContext& ctx);
void BinaryLogicShapeInference_opset12(InferenceContext& ctx);
void logicalOpInference_opset1(InferenceContext& ctx);
void BinaryLogicShapeInference_opset7(InferenceContext& ctx);
void BitShiftShapeInference_opset11(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::logical::utils
