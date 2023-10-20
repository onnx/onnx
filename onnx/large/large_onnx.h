// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <dlpack/dlpack.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace large_onnx {

/** C API to create large onnx models in a single file. */
class LargeModelContainer {
 public:
  LargeModelContainer();
  virtual ~LargeModelContainer();

  void SetModelProto(ModelProto& proto);
  void Append(TensorProto&& large_tensor, DLManagedTensor* dlpack);

  void SerializeToString(std::string& out);
  void Load(const std::string& path, bool load_large_tensors = false);
  void Save(const std::string& path, bool save_large_tensor = true);

 private:
  ModelProto model_proto_;
  std::map<std::string, std::pair<int64_t, TensorProto>> large_tensors_;
};

} // namespace large_onnx
} // namespace  ONNX_NAMESPACE
