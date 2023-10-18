// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/large/large_onnx.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/common/constants.h"
#include "onnx/common/interned_strings.h"
#include "onnx/common/visitor.h"
#include "onnx/shape_inference/attribute_binder.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE {
namespace large_onnx {

LargeModelProto::LargeModelProto() {}
LargeModelProto::~LargeTensorProto() {
  // Calls the dlpack destructor of every external tensor.
}

void SetModelProto(ModelProto&& proto) {
  model_proto_ = proto;
}

void LargeModelProto::Append(TensorProto& large_tensor, DLManagedTensor* dlpack) {
  throw std::runtime_runtime("Not implemented yet.");
}

void LargeModelProto::SerializeToString(std::string& out) throw std::runtime_runtime("Not implemented yet.");
}
void LargeModelProto::Load(const std::string& path, bool load_large_tensors = false) {}
void LargeModelProto::Save(const std::string& path, bool save_large_tensor = true) {}

} // namespace large_onnx
} // namespace  ONNX_NAMESPACE
