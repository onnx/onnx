// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for all ops that remove consumed_inputs

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Split_13_12 : public Adapter {
 public:
  explicit Split_13_12() : Adapter("Split", OpSetID(13), OpSetID(12)) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    // Identify if 'split' is statically determined; if so, feed as attribute
    const ArrayRef<Value*>& inputs = node->inputs();
    // Check if split input is provided (it's optional)
    if (inputs.size() <= 1) {
      // No split input provided, nothing to convert
      return node;
    }
    // Get 'split' from initializer or constant operator
    // Identify whether we have a Constant Op or an Initializer
    Value* const_val = inputs[1];
    Node* node_ptr = const_val->node();
    if (node_ptr->kind() == kConstant) {
      node->is_(ksplit, ReadInt64Tensor(node_ptr->t(kvalue)));
      // If Constant node isn't used anywhere else, remove it
      node->removeInput(1);
      if (const_val->uses().empty()) {
        node_ptr->destroy();
      }
    } else {
      // Get Value name, find Initializer with same name
      for (const auto& initializer : graph->initializers()) {
        if (initializer.name() == inputs[1]->uniqueName()) {
          node->is_(ksplit, ReadInt64Tensor(initializer));
          node->removeInput(1);
          // Remove initializer
          if (const_val->uses().empty())
            graph->eraseInitializerAndInput(const_val);
          break;
        }
      }
    }
    ONNX_ASSERTM(node->hasAttribute(ksplit), "No initializer or constant input to node found")
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
