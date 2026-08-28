// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Optional* ops in default domain

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {
class OptionalOpsAdapter final : public Adapter {
 public:
  OptionalOpsAdapter(
      const std::string& name,
      const OpSetID& initial_version,
      const OpSetID& target_version,
      const std::vector<TensorProto_DataType>& unallowed_types,
      bool allow_optional_input = true,
      bool allow_nonoptional_input = true,
      bool allow_noinput = false)
      : Adapter(name, initial_version, target_version),
        unallowed_types_(unallowed_types),
        allow_optional_input_(allow_optional_input),
        allow_nonoptional_input_(allow_nonoptional_input),
        allow_noinput_(allow_noinput) {}

  // This adapter checks only input because it's sufficient for existing ops, Optional, OptionalHasElement and
  // OptionalGetElement. Also, assume valid types are optional<seq<tensor<*>>>, optional<tensor<*>>, seq<tensor<*>>, and
  // tensor<*> and uses ONNX_ASSERT to validate type.
  void adapt_optional_ops(const std::shared_ptr<Graph>& /*unused*/, Node* node) const {
    const TypeProto* opt_or_elem_type = nullptr;
    ONNX_ASSERTM(
        allow_noinput_ || (node->inputs().size() > 0),
        "No input to operator '",
        name(),
        "' is unallowed for Opset Version ",
        static_cast<int64_t>(target_version().version()));
    if (allow_noinput_ && node->inputs().empty()) {
      Symbol type = Symbol("type");
      // Node must have "type" attribute.
      ONNX_ASSERT(node->hasAttribute(type));
      opt_or_elem_type = &node->tp(type);
    } else {
      opt_or_elem_type = node->input()->type().get();
    }

    int32_t tensor_elem_type = -1;
    if (opt_or_elem_type == nullptr) {
      ONNX_ASSERTM(
          allow_nonoptional_input_,
          "Non-Optional input to operator '",
          name(),
          "' is unallowed for Opset Version ",
          static_cast<int64_t>(target_version().version()));
      tensor_elem_type = node->input()->elemType();
    } else {
      ONNX_ASSERTM(
          (allow_optional_input_ && opt_or_elem_type->has_optional_type()) ||
              (allow_nonoptional_input_ && !opt_or_elem_type->has_optional_type()),
          "Specified type of Input of operator '",
          name(),
          "' is unallowed for Opset Version ",
          static_cast<int64_t>(target_version().version()));
      const TypeProto& elem_type =
          opt_or_elem_type->has_optional_type() ? opt_or_elem_type->optional_type().elem_type() : *opt_or_elem_type;
      ONNX_ASSERT(elem_type.has_tensor_type() || elem_type.has_sequence_type() || elem_type.has_sparse_tensor_type());
      const TypeProto& tensor_type = elem_type.has_sequence_type() ? elem_type.sequence_type().elem_type() : elem_type;
      ONNX_ASSERT(tensor_type.has_tensor_type() || tensor_type.has_sparse_tensor_type());
      tensor_elem_type = tensor_type.has_tensor_type() ? tensor_type.tensor_type().elem_type()
                                                       : tensor_type.sparse_tensor_type().elem_type();
    }

    ONNX_ASSERTM(
        std::find(unallowed_types_.begin(), unallowed_types_.end(), tensor_elem_type) == unallowed_types_.end(),
        "DataType (",
        tensor_elem_type,
        ") of Input of operator '",
        name(),
        "' is unallowed for Opset Version ",
        static_cast<int64_t>(target_version().version()));
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_optional_ops(graph, node);
    return node;
  }

 private:
  std::vector<TensorProto_DataType> unallowed_types_;
  bool allow_optional_input_;
  bool allow_nonoptional_input_;
  bool allow_noinput_;
};
} // namespace ONNX_NAMESPACE::version_conversion
