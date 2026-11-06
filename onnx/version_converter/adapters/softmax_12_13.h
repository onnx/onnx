// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Softmax, LogSoftmax and Hardmax in default domain from version 12 to 13

#pragma once

#include <memory>
#include <string>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class Softmax_12_13 final : public Adapter {
 public:
  explicit Softmax_12_13(const std::string& op_name) : Adapter(op_name, OpSetID(12), OpSetID(13)) {}

  void adapt_softmax_12_13(const std::shared_ptr<Graph>& graph, Node* node) const {
    ONNX_ASSERTM(node->inputs().size() >= 1, "Softmax node must have at least 1 input")
    int old_axis = node->hasAttribute(kaxis) ? static_cast<int>(node->i(kaxis)) : 1;
    int input_rank = static_cast<int>(node->inputs()[0]->sizes().size());

    if (old_axis < 0)
      old_axis = input_rank + old_axis;

    if (old_axis == input_rank - 1) {
      node->i_(kaxis, -1);
    } else {
      //    -- shape ------------------
      //   /                           |
      // ----- flatten -- softmax -- reshape

      // get original softmax's input shape
      Symbol kShape("Shape");
      Node* shape = graph->create(kShape);
      shape->addInput(node->inputs()[0]);
      shape->insertBefore(node);

      // Insert Flatten node before softmax
      Node* flatten = graph->create(kFlatten);
      flatten->addInput(node->inputs()[0]);
      flatten->insertBefore(node);
      flatten->i_(kaxis, old_axis);
      node->replaceInput(0, flatten->output());

      // Softmax along the last axis of the flattened 2D tensor
      node->i_(kaxis, -1);

      // Insert Reshape node after softmax
      Node* reshape = graph->create(kReshape);
      if (!node->output()->sizes().empty()) {
        reshape->output()->setSizes(node->output()->sizes());
      }
      reshape->output()->setElemType(node->output()->elemType());
      reshape->insertAfter(node);
      ReplaceValueKeepingName(*graph, node->output(), reshape->output());
      reshape->addInput(node->output());
      reshape->addInput(shape->output());
      node->output()->wipeSizes();
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_softmax_12_13(graph, node);
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
