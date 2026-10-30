// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

#include <limits>
#include <stdexcept>

#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE {
namespace nb = nanobind;

// Serialize into a newly allocated Python bytes object before exposing it to Python.
inline nb::bytes ProtoToBytes(const google::protobuf::MessageLite& proto) {
  const size_t size = proto.ByteSizeLong();
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Unable to serialize proto: size exceeds the 2 GB protobuf limit.");
  }
  if (size == 0) {
    return nb::bytes("", 0);
  }
  nb::bytes result(nullptr, size);
  if (!proto.SerializeToArray(PyBytes_AsString(result.ptr()), static_cast<int>(size))) {
    throw std::runtime_error("Unable to serialize proto into Python bytes.");
  }
  return result;
}

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
