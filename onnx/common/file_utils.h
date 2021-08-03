/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace ONNX_NAMESPACE {

template <typename T>
void LoadProtoFromPath(const std::string proto_path, T& proto) {
  std::fstream tensor_stream(proto_path, std::ios::in | std::ios::binary);
  if (!tensor_stream.good()) {
    fail_check("Unable to open model file:", proto_path, ". Please check if it is a valid file.");
  }
  std::string data{std::istreambuf_iterator<char>{tensor_stream}, std::istreambuf_iterator<char>{}};
  if (!ParseProtoFromBytes(&proto, data.c_str(), data.size())) {
    fail_check(
    "Unable to parse model from file:", proto_path, ". Please check if it is a valid protobuf file of model.");
  }
}

void LoadExternalTensor(const TensorProto& external_tensor,
  TensorProto& loaded_tensor, const std::string model_dir);

} // namespace ONNX_NAMESPACE