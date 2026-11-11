// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class NDIndices_29_28 final : public Adapter {
 public:
  explicit NDIndices_29_28(const std::string& op_name) : Adapter(op_name, OpSetID(29), OpSetID(28)) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    Value* indices = node->inputs()[1];
    if (indices->elemType() != TensorProto_DataType_INT64) {
      // Widening preserves every int32 index, including negative indices.
      // Also cast when type information is absent: both valid input types
      // are losslessly representable as int64.
      Node* cast = graph->create(kCast);
      cast->i_(kto, TensorProto_DataType_INT64);
      cast->addInput(indices);
      cast->insertBefore(node);
      cast->output()->setElemType(TensorProto_DataType_INT64);
      if (indices->has_sizes()) {
        cast->output()->setSizes(indices->sizes());
      }
      node->replaceInput(1, cast->output());
    }
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
