// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx/common/status.h"
#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

// Helper function for register nodes in
// a FunctionProto. Attributes need to be
// registered separately.
Common::Status BuildNode(
    std::string_view name,
    std::string_view domain,
    std::string_view doc_string,
    std::string_view op_type,
    std::vector<std::string> const& inputs,
    std::vector<std::string> const& outputs,
    /*OUT*/ NodeProto* node);
} // namespace ONNX_NAMESPACE
