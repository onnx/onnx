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

LargeModelContainer::LargeModelContainer() {}
LargeModelContainer::~LargeModelContainer() {
  // Calls the dlpack destructor of every external tensor.
}

void LargeModelContainer::SetModelProto(ModelProto& proto) {
  model_proto_ = proto;
}

void LargeModelContainer::Append(TensorProto&&, DLManagedTensor*) {
  throw std::runtime_error("Not implemented yet.");
}

void LargeModelContainer::SerializeToString(std::string&) {
  throw std::runtime_error("Not implemented yet.");
}

void LargeModelContainer::Load(const std::string&, bool) {
  throw std::runtime_error("Not implemented yet.");
}

void LargeModelContainer::Save(const std::string&, bool) {
  throw std::runtime_error("Not implemented yet.");
}

} // namespace large_onnx
} // namespace  ONNX_NAMESPACE
