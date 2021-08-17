/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/common/file_utils.h"

namespace ONNX_NAMESPACE {

void LoadExternalTensor(const TensorProto& external_tensor, TensorProto& loaded_tensor,
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
  tensor_stream.seekg(offset, std::ios::beg);
  std::string data(std::istreambuf_iterator<char>(tensor_stream), {});
  loaded_tensor.set_raw_data(data.data(), length);
  loaded_tensor.set_data_location(TensorProto_DataLocation_DEFAULT);
  loaded_tensor.clear_external_data();
}

} // namespace ONNX_NAMESPACE