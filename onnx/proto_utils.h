// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnx/onnx_pb.h"

#ifdef ONNX_USE_LITE_PROTO
#include <google/protobuf/message_lite.h>
#else // ONNX_USE_LITE_PROTO
#include <google/protobuf/message.h>
#endif // !ONNX_USE_LITE_PROTO

namespace ONNX_NAMESPACE {

#ifdef ONNX_USE_LITE_PROTO
inline std::string ProtoDebugString(const ::google::protobuf::MessageLite& proto) {
  // Since the MessageLite interface does not support reflection, there is very
  // little information that this and similar methods can provide.
  // But when using lite proto this is the best we can provide.
  return proto.ShortDebugString();
}
#else
inline std::string ProtoDebugString(const ::google::protobuf::Message& proto) {
  return proto.ShortDebugString();
}
#endif

template <typename Proto>
bool ParseProtoFromBytes(Proto* proto, const char* buffer, size_t length) {
  // Reject inputs larger than the 2 GB proto limit before casting to int.
  // ParseFromArray takes an int size, so passing a truncated value would
  // silently parse fewer bytes than requested.
  constexpr std::int32_t total_bytes_limit = std::numeric_limits<std::int32_t>::max();
  if (length > static_cast<size_t>(total_bytes_limit)) {
    return false;
  }
  // Unlike ParseFromCodedStream, ParseFromArray fails unless the whole buffer
  // is consumed, so data after a 0 or end-group tag is rejected rather than
  // silently dropped.
  return proto->ParseFromArray(buffer, static_cast<int>(length));
}

template <typename T>
inline std::vector<T> RetrieveValues(const AttributeProto& attr);
template <>
inline std::vector<int64_t> RetrieveValues(const AttributeProto& attr) {
  return {attr.ints().begin(), attr.ints().end()};
}

template <>
inline std::vector<std::string> RetrieveValues(const AttributeProto& attr) {
  return {attr.strings().begin(), attr.strings().end()};
}

template <>
inline std::vector<float> RetrieveValues(const AttributeProto& attr) {
  return {attr.floats().begin(), attr.floats().end()};
}

} // namespace ONNX_NAMESPACE
