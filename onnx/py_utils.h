// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE {
namespace nb = nanobind;

template <typename Proto>
bool ParseProtoFromPyBytes(Proto* proto, const nb::bytes& bytes) {
  // Get the buffer from Python bytes object
  auto buffer = static_cast<const char*>(bytes.data());
  size_t length = bytes.size();

  return ParseProtoFromBytes(proto, buffer, length);
}
} // namespace ONNX_NAMESPACE
