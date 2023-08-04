/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)>
SoftmaxFamilyDocGenerator(const char* name, const char* description, const char* equation);

void matmulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int input1Idx, int input2Idx);

void qlinear_matmul_shape_inference(ONNX_NAMESPACE::InferenceContext& ctx);

const char* qlinear_matmul_doc();

} // namespace ONNX_NAMESPACE
