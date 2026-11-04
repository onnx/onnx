// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for ReduceLogSum and ReduceLogSumExp in default domain from version 27 to 28

#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

// ReduceLogSum/ReduceLogSumExp v28 restrict T to float types. Only the data input (input 0) and
// the output are bound to T. Input 1 is `axes`, which is always tensor(int64) and must not be
// checked, so the generic TypeRestriction adapter cannot be used here.
class ReduceLogSum_27_28 final : public Adapter {
 public:
  explicit ReduceLogSum_27_28(const std::string& op_name, const std::vector<TensorProto_DataType>& unallowed_types)
      : Adapter(op_name, OpSetID(27), OpSetID(28)), unallowed_types_(unallowed_types) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    if (!node->inputs().empty()) {
      assertAllowed(node->inputs()[0], "input");
    }
    for (const Value* output : node->outputs()) {
      assertAllowed(output, "output");
    }
    return node;
  }

 private:
  std::vector<TensorProto_DataType> unallowed_types_;

  void assertAllowed(const Value* val, const char* kind) const {
    ONNX_ASSERTM(
        std::find(unallowed_types_.begin(), unallowed_types_.end(), val->elemType()) == unallowed_types_.end(),
        "DataType (",
        val->elemType(),
        ") of ",
        kind,
        " of operator '",
        name(),
        "' is not supported in Opset Version ",
        static_cast<int64_t>(target_version().version()),
        ".");
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
