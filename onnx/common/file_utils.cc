/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/common/file_utils.h"

namespace ONNX_NAMESPACE {

void LoadExternalTensor(const TensorProto& external_tensor,
  TensorProto& loaded_tensor, const std::string model_dir) {
  for (const StringStringEntryProto& entry : external_tensor.external_data()) {
    if (entry.has_key() && entry.has_value() && entry.key() == "location") {
      LoadProtoFromPath(path_join(model_dir, entry.value()), loaded_tensor);
      break;
    }
  }
}

} // namespace ONNX_NAMESPACE