// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Scan in default domain from version 8 to 9

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {
struct Scan_8_9 final : public Adapter {
  explicit Scan_8_9() : Adapter("Scan", OpSetID(8), OpSetID(9)) {}

  void adapt_scan_8_9(const std::shared_ptr<Graph>& /*unused*/, Node* node) const {
    const std::vector<Value*> inputs(node->inputs().vec());
    const std::vector<Value*> outputs(node->outputs().vec());

    // Validate preconditions before mutating the node, so a failed assertion
    // doesn't leave the graph in a partially-converted state.
    ONNX_ASSERTM(!inputs.empty(), "Scan node in opset 8 must have at least 1 input")
    ONNX_ASSERTM(inputs[0]->uniqueName().empty(), "Unsupported conversion to opset 9")

    // Handling Attribute Changes

    Symbol dirs = Symbol("directions");
    if (node->hasAttribute(dirs)) {
      std::vector<int64_t> directions(node->is(dirs));
      node->removeAttribute(dirs);
      node->is_(Symbol("scan_input_directions"), std::move(directions));
    }

    // Handling Input and Output Changes

    node->removeAllInputs();

    // inputs[0] is the optional sequence_lens (asserted to be empty).
    // Scan does not have it as opset 9, so it is intentionally dropped.
    // All other inputs are kept. But opset 9 has no batch dimension, so strip
    // the leading axis when the shape is known, skipping inputs with unknown
    // shape.
    for (size_t i = 1; i < inputs.size(); ++i) {
      Value* input = inputs[i];
      if (!input->sizes().empty()) {
        std::vector<Dimension> new_sizes(input->sizes().begin() + 1, input->sizes().end());
        input->setSizes(new_sizes);
      }
      node->addInput(input);
    }

    for (Value* output : outputs) {
      if (!output->sizes().empty()) {
        std::vector<Dimension> new_sizes(output->sizes().begin() + 1, output->sizes().end());
        output->setSizes(new_sizes);
      }
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_scan_8_9(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
