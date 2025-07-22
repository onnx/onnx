/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace defs {
namespace sequence {
namespace utils {

std::function<void(OpSchema&)> SplitToSequenceOpGenerator (const std::vector<std::string>& allowed_types);

}
}
}
}