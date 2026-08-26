// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/version_converter/adapters/type_restriction.h"

namespace ONNX_NAMESPACE::version_conversion {

class Cast_29_28 final : public TypeRestriction {
 public:
  explicit Cast_29_28(const std::vector<TensorProto_DataType>& unallowed_types)
      : TypeRestriction("Cast", OpSetID(29), OpSetID(28), unallowed_types), unallowed_types_(unallowed_types) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_type_restriction(graph, node);
    const auto to = static_cast<TensorProto_DataType>(node->i(kto));
    ONNX_ASSERTM(
        std::find(unallowed_types_.begin(), unallowed_types_.end(), to) == unallowed_types_.end(),
        "Attribute to of operator 'Cast' uses DataType (",
        to,
        "), which is unallowed for Opset Version ",
        static_cast<int64_t>(target_version().version()),
        ".")
    return node;
  }

 private:
  std::vector<TensorProto_DataType> unallowed_types_;
};

} // namespace ONNX_NAMESPACE::version_conversion
