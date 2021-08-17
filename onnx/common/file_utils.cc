/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/common/file_utils.h"

namespace ONNX_NAMESPACE {

std::string LoadExternalTensor(const TensorProto& external_tensor, const std::string model_dir) {
  std::string tensor_path;
  int offset = 0;
  int length = 0;
  for (const StringStringEntryProto& entry : external_tensor.external_data()) {
    if (entry.has_value() && entry.key() == "location") {
      tensor_path = path_join(model_dir, entry.value());
    } else if (entry.has_value() && entry.key() == "offset") {
      offset = std::stoi(entry.value());
      if (offset < 0) {
        fail_check("The loaded offset for a external tensor should not be negative. ");
      }
    } else if (entry.has_value() && entry.key() == "length") {
      length = std::stoi(entry.value());
      if (length < 0) {
        fail_check("The loaded length for a external tensor should not be negative. ");
      }
    }
  }
  std::ifstream tensor_stream(tensor_path, std::ios::binary | std::ios::ate);
  if (!tensor_stream.good()) {
    fail_check("Unable to open external tensor: ", tensor_path, ". Please check if it is a valid file. ");
  }

  int remain_length = tensor_stream.tellg();
  std::vector<char> buffer(remain_length);
  tensor_stream.seekg(0, std::ios::beg);
  tensor_stream.read(buffer.data(), remain_length);

  std::vector<char> data = std::vector<char>(buffer.begin() + offset, buffer.begin() + offset + length);
  std::string raw(data.begin(), data.end());
  return raw;
}

} // namespace ONNX_NAMESPACE