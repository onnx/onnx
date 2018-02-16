#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

namespace ONNX_NAMESPACE {

template <typename Proto>
bool ParseProtoFromBytes(Proto* proto, const char* buffer, size_t length) {
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  ::google::protobuf::io::ArrayInputStream input_stream(buffer, static_cast<int>(length));
  ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}
} // namespace ONNX_NAMESPACE
