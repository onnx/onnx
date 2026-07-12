// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Range in default domain from version 27 to 26

#pragma once

#include <cinttypes>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Range_27_26 final : public Adapter {
 public:
  explicit Range_27_26(const std::vector<TensorProto_DataType>& unallowed_types)
      : Adapter("Range", OpSetID(27), OpSetID(26)), unallowed_types_(unallowed_types) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    // Reject FLOAT16/BFLOAT16 inputs or outputs — not supported by Range v11.
    for (const Value* input : node->inputs()) {
      ONNX_ASSERTM(
          std::find(unallowed_types_.begin(), unallowed_types_.end(), input->elemType()) == unallowed_types_.end(),
          "DataType (",
          input->elemType(),
          ") of input of operator '",
          name(),
          "' is not supported in Opset Version ",
          static_cast<int64_t>(target_version().version()),
          ".");
    }
    for (const Value* output : node->outputs()) {
      ONNX_ASSERTM(
          std::find(unallowed_types_.begin(), unallowed_types_.end(), output->elemType()) == unallowed_types_.end(),
          "DataType (",
          output->elemType(),
          ") of output of operator '",
          name(),
          "' is not supported in Opset Version ",
          static_cast<int64_t>(target_version().version()),
          ".");
    }
    // Remove stash_type — Range v11 has no such attribute.
    if (node->hasAttribute(kstash_type)) {
      node->removeAttribute(kstash_type);
    }
    return node;
  }

 private:
  std::vector<TensorProto_DataType> unallowed_types_;
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
