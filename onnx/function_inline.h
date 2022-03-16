/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "onnx/onnx_pb.h"
#include <string>

namespace ONNX_NAMESPACE {
namespace function_inline {

void inline_model_function(ModelProto& model);
void inline_model_function_path(const std::string& model_path, const std::string& target_model_path);

} // namespace checker
} // namespace ONNX_NAMESPACE
