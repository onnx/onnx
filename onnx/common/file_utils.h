// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <filesystem> // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <string>

#ifdef _WIN32
#include "onnx/common/path.h"
#endif
#include "onnx/checker.h"

namespace ONNX_NAMESPACE {

template <typename T>
void LoadProtoFromPath(const std::string& proto_path, T& proto) {
#ifdef _WIN32
  std::filesystem::path proto_u8_path(utf8str_to_wstring(proto_path));
#else
  std::filesystem::path proto_u8_path(proto_path);
#endif
  std::fstream proto_stream(proto_u8_path, std::ios::in | std::ios::binary);
  if (!proto_stream.good()) {
    fail_check("Unable to open proto file: ", proto_path, ". Please check if it is a valid proto. ");
  }
  constexpr int total_bytes_limit = std::numeric_limits<int>::max();
  std::error_code size_ec;
  const std::uintmax_t file_size = std::filesystem::file_size(proto_u8_path, size_ec);
  if (!size_ec && file_size > static_cast<std::uintmax_t>(total_bytes_limit)) {
    fail_check(
        "Unable to parse proto from file: ", proto_path, ". Please check if it is a valid protobuf file of proto. ");
  }

  google::protobuf::io::IstreamInputStream input(&proto_stream);
  bool parsed = false;
  {
    google::protobuf::io::CodedInputStream coded_input(&input);
    coded_input.SetTotalBytesLimit(total_bytes_limit);
    parsed = proto.ParseFromCodedStream(&coded_input) && coded_input.ConsumedEntireMessage();
  }
  // The coded stream returns any unread bytes to input on destruction. Check
  // for physical EOF too, since reaching its byte limit can end parsing early.
  const void* remaining_data = nullptr;
  int remaining_size = 0;
  const bool has_remaining_data = input.Next(&remaining_data, &remaining_size);
  if (proto_stream.bad() || (proto_stream.fail() && !proto_stream.eof())) {
    fail_check("Unable to read proto file: ", proto_path, ". Please check if it is a valid proto. ");
  }
  if (!parsed || has_remaining_data) {
    fail_check(
        "Unable to parse proto from file: ", proto_path, ". Please check if it is a valid protobuf file of proto. ");
  }
}
} // namespace ONNX_NAMESPACE
