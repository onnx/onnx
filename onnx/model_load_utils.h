/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/common/status.h"
#include "onnx/file_utils.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

using namespace onnx::Common;

Status LoadModel(const std::string& file_path, ModelProto& model_proto);

Status SaveModel(const std::string& file_path, ModelProto& model_proto);

} // namespace ONNX_NAMESPACE
