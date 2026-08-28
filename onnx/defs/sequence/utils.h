// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE::defs::sequence::utils {

std::function<void(OpSchema&)> SplitToSequenceOpGenerator(
    std::vector<std::string> input_types,
    std::vector<std::string> output_types);

} // namespace ONNX_NAMESPACE::defs::sequence::utils
