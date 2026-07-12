// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

void ClearShape(TypeProto& input_type);

int handle_negative_axis_validate(const std::string& attrib, int axis, int rank);

// Guards the Scan input/output count subtractions against underflow and returns
// the loop state variable count. Shared by the current and opset-9 paths.
size_t ValidateScanCountsAndGetNumLoopStateVars(size_t num_inputs, size_t num_scan_inputs, size_t num_outputs);

void IfInferenceFunction(InferenceContext& ctx);

void LoopInferenceFunction(InferenceContext& ctx);

void ScanInferenceFunction(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE
