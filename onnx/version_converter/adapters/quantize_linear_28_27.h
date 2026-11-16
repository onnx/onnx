// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/version_converter/adapters/type_restriction.h"

namespace ONNX_NAMESPACE::version_conversion {

class QuantizeLinear_28_27 final : public TypeRestriction {
 public:
  explicit QuantizeLinear_28_27(const std::vector<TensorProto_DataType>& unallowed_types)
      : TypeRestriction("QuantizeLinear", OpSetID(28), OpSetID(27), unallowed_types),
        unallowed_types_(unallowed_types) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_type_restriction(graph, node);
    if (node->hasAttribute(koutput_dtype)) {
      const auto output_dtype = static_cast<TensorProto_DataType>(node->i(koutput_dtype));
      ONNX_ASSERTM(
          std::find(unallowed_types_.begin(), unallowed_types_.end(), output_dtype) == unallowed_types_.end(),
          "Attribute output_dtype of operator 'QuantizeLinear' uses DataType (",
          output_dtype,
          "), which is unallowed for Opset Version ",
          static_cast<int64_t>(target_version().version()),
          ".")
    }
    return node;
  }

 private:
  std::vector<TensorProto_DataType> unallowed_types_;
};

} // namespace ONNX_NAMESPACE::version_conversion
