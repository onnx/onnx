#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

namespace onnx {

template <typename Proto>
bool ParseProtoFromBytes(Proto* proto, const char* buffer, size_t length) {
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  ::google::protobuf::io::CodedInputStream coded_stream(
      new google::protobuf::io::ArrayInputStream(buffer, static_cast<int>(length)));
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}
} // namespace onnx
