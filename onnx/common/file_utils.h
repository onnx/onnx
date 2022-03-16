/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/checker.h"
#include "onnx/common/path.h"

#include <fstream>

namespace ONNX_NAMESPACE {

template <typename T>
void LoadProtoFromPath(const std::string proto_path, T& proto) {
  std::fstream proto_stream(proto_path, std::ios::in | std::ios::binary);
  if (!proto_stream.good()) {
    fail_check("Unable to open proto file: ", proto_path, ". Please check if it is a valid proto. ");
  }
  std::string data{std::istreambuf_iterator<char>{proto_stream}, std::istreambuf_iterator<char>{}};
  if (!ParseProtoFromBytes(&proto, data.c_str(), data.size())) {
    fail_check(
    "Unable to parse proto from file: ", proto_path, ". Please check if it is a valid protobuf file of proto. ");
  }
}

template <typename Proto>
void SaveProto(Proto* proto, const std::string& file_path) {
  // Use SerializeToString instead of SerializeToOstream due to LITE_PROTO
  std::fstream output(file_path, std::ios::out | std::ios::trunc | std::ios::binary);
  std::string model_string;
  ONNX_TRY {
    proto->SerializeToString(&model_string);
    output << model_string;
  }
  ONNX_CATCH(...) {
    fail_check("Unable to save inferred model to the target path:", file_path);
  }
}


} // namespace ONNX_NAMESPACE