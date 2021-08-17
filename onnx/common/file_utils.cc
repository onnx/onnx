/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/common/file_utils.h"

namespace ONNX_NAMESPACE {

void LoadExternalTensor(const TensorProto& external_tensor, std::string& loaded_raw_data,
  const std::string model_dir) {
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
  std::ifstream tensor_stream(tensor_path, std::ios::binary);
  if (!tensor_stream.good()) {
    fail_check("Unable to open external tensor: ", tensor_path, ". Please check if it is a valid file. ");
  }

  std::vector<char> buffer(length);
  tensor_stream.seekg(offset, std::ios::beg);
  size_t total_bytes_read = 0;

  while (total_bytes_read < length) {
    // Reads at most 1GB each time to prevent memory issue
    const size_t max_bytes_to_read = 1 << 30;
    const size_t remain_read = length - total_bytes_read;
    const size_t bytes_read = std::min(remain_read, max_bytes_to_read);
    tensor_stream.read(buffer.data(), bytes_read);
    total_bytes_read += bytes_read;
  }

  std::string char_to_str(buffer.begin(), buffer.end());
  loaded_raw_data = char_to_str;
}

} // namespace ONNX_NAMESPACE