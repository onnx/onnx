// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

void ClearShape(TypeProto& input_type);

int handle_negative_axis_validate(std::string_view attrib, int axis, int rank);

// Guards the Scan input/output count subtractions against underflow and returns
// the loop state variable count. Shared by the current and opset-9 paths.
size_t ValidateScanCountsAndGetNumLoopStateVars(size_t num_inputs, size_t num_scan_inputs, size_t num_outputs);

} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE::defs::controlflow::utils {

void IfInferenceFunction(InferenceContext& ctx);
void LoopInferenceFunction(InferenceContext& ctx);
void ScanInferenceFunction(InferenceContext& ctx);
void ScanInferenceFunction_opset8(InferenceContext& ctx);
void ScanInferenceFunction_opset9(InferenceContext& ctx);
void LoopInferenceFunction_opset8(InferenceContext& ctx);
void LoopInferenceFunction_opset11(InferenceContext& ctx);
void IfInferenceFunction_opset1(InferenceContext& ctx);
void IfInferenceFunction_opset11(InferenceContext& ctx);
void IfInferenceFunction_opset13(InferenceContext& ctx);
void LoopInferenceFunction_opset13(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::controlflow::utils
