// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

#include <stdexcept>

#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE {
namespace nb = nanobind;

template <typename Proto>
bool ParseProtoFromPyBytes(Proto* proto, const nb::bytes& bytes) {
  // Get the buffer from Python bytes object
  const auto* buffer = static_cast<const char*>(bytes.data());
  size_t length = bytes.size();

  return ParseProtoFromBytes(proto, buffer, length);
}

// Same as ParseProtoFromPyBytes, but raises a Python-visible exception instead of
// silently leaving `proto` partially populated when `bytes` is malformed, truncated,
// or exceeds the proto size limit.
template <typename Proto>
void ParseProtoFromPyBytesOrThrow(Proto* proto, const nb::bytes& bytes) {
  if (!ParseProtoFromPyBytes(proto, bytes)) {
    throw std::invalid_argument(
        "Unable to parse proto from the given bytes: data is malformed, truncated, or exceeds the size limit.");
  }
}
} // namespace ONNX_NAMESPACE
