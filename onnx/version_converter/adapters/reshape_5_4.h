// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Reshape in default domain from version 5 to 4

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Reshape_5_4 final : public Adapter {
 public:
  explicit Reshape_5_4() : Adapter("Reshape", OpSetID(5), OpSetID(4)) {}

  void adapt_reshape_5_4(const std::shared_ptr<Graph>& graph, Node* node) const {
    // Identify if shape is statically determined; if so, feed as attribute
    const ArrayRef<Value*>& inputs = node->inputs();
    // Check if shape input is provided (it's optional in some contexts)
    if (inputs.size() <= 1) {
      // No shape input provided, nothing to convert
      return;
    }
    // Get shape from initializer or constant operator, not actual shape
    // Identify whether we have a Constant Op or an Initializer
    Value* const_val = inputs[1];
    Node* node_ptr = const_val->node();
    if (node_ptr->kind() == kConstant) {
      node->is_(kshape, ReadInt64Tensor(node_ptr->t(kvalue)));
      // If Constant node isn't used anywhere else, remove it
      node->removeInput(1);
      if (const_val->uses().empty()) {
        node_ptr->destroy();
      }
    } else {
      // Get Value name, find Initializer with same name
      for (const auto& initializer : graph->initializers()) {
        if (initializer.name() == inputs[1]->uniqueName()) {
          node->is_(kshape, ReadInt64Tensor(initializer));
          node->removeInput(1);
          // Remove initializer
          if (const_val->uses().empty())
            graph->eraseInitializerAndInput(const_val);
          break;
        }
      }
    }
    ONNX_ASSERTM(node->hasAttribute(kshape), "No initializer or constant input to Reshape node found")
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_reshape_5_4(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
