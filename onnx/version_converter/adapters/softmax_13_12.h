// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Softmax and LogSoftmax in default domain from version 13 to 12

#pragma once

#include <memory>
#include <string>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Softmax_13_12 final : public Adapter {
 public:
  explicit Softmax_13_12(const std::string& op_name) : Adapter(op_name, OpSetID(13), OpSetID(12)) {}

  void adapt_softmax_13_12(const std::shared_ptr<Graph>& graph, Node* node) const {
    (void)graph; // Suppress unused parameter warning

    int new_axis = node->hasAttribute(kaxis) ? node->i(kaxis) : -1;

    // Handle Softmax on the last axis (default behavior in opset 13)
    if (new_axis == -1) {
      new_axis = node->inputs()[0]->sizes().size() - 1;
      node->i_(kaxis, new_axis);
    } else if (new_axis < 0) {
      // Handle negative axis by converting it to positive
      int input_rank = node->inputs()[0]->sizes().size();
      new_axis = input_rank + new_axis;
      node->i_(kaxis, new_axis);
    }

    // Flatten and Reshape nodes added in opset 13 must be removed
    // Check for Flatten node before Softmax and Reshape node after Softmax
    if (node->inputs()[0]->node()->kind() == kFlatten) {
      Node* flatten = node->inputs()[0]->node();
      const auto flatten_input = flatten->inputs()[0];
      node->replaceInput(0, flatten_input);
      flatten->destroy();
    }

    for (Use u : node->output()->uses()) {
      if (u.user->kind() == kReshape) {
        Node* reshape = u.user;
        const auto reshape_output = reshape->outputs()[0];
        node->output()->replaceAllUsesWith(reshape_output);
        reshape->destroy();
        break;
      }
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_softmax_13_12(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
