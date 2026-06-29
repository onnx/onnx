// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/assertions.h"

#include <string>

#include "onnx/common/common.h"

namespace ONNX_NAMESPACE {

void throw_assert_error(const std::string& msg) {
  ONNX_THROW_EX(assert_error(msg));
}

void throw_tensor_error(const std::string& msg) {
  ONNX_THROW_EX(tensor_error(msg));
}

} // namespace ONNX_NAMESPACE
