#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

#include "onnx/onnx_pb.h"

#ifdef ONNX_USE_LITE_PROTO
#include <google/protobuf/message_lite.h>
#else // ONNX_USE_LITE_PROTO
#include <google/protobuf/message.h>
#endif  // !ONNX_USE_LITE_PROTO

namespace ONNX_NAMESPACE {

#ifdef ONNX_USE_LITE_PROTO
using ::google::protobuf::MessageLite;
inline std::string ProtoDebugString(const MessageLite& proto) {
  return proto.SerializeAsString();
}
#else
using ::google::protobuf::Message;
inline std::string ProtoDebugString(const Message& proto) {
  return proto.ShortDebugString();
}
#endif

template <typename Proto>
bool ParseProtoFromBytes(Proto* proto, const char* buffer, size_t length) {
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  ::google::protobuf::io::ArrayInputStream input_stream(buffer, static_cast<int>(length));
  ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  coded_stream.SetTotalBytesLimit((2048LL << 20) - 1, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

template<typename T> inline std::vector<T> RetrieveValues(const AttributeProto& attr);
template<> inline std::vector<int64_t> RetrieveValues(const AttributeProto& attr) {
  return {attr.ints().begin(), attr.ints().end()};
}

template<> inline std::vector<std::string> RetrieveValues(const AttributeProto& attr) {
  return { attr.strings().begin(), attr.strings().end() };
}

template<> inline std::vector<float> RetrieveValues(const AttributeProto& attr) {
  return { attr.floats().begin(), attr.floats().end() };
}

} // namespace ONNX_NAMESPACE
