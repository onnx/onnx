// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Resize in default domain from version 12 to 13

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class Resize_12_13 final : public Adapter {
 public:
  Resize_12_13() : Adapter("Resize", OpSetID(12), OpSetID(13)) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    const Symbol coordinate_transformation_mode("coordinate_transformation_mode");
    if (node->hasAttribute(coordinate_transformation_mode)) {
      ONNX_ASSERTM(
          node->s(coordinate_transformation_mode) != "tf_half_pixel_for_nn",
          "Resize coordinate_transformation_mode='tf_half_pixel_for_nn' is not supported in opset 13.");
    }

    const ArrayRef<Value*>& inputs = node->inputs();
    if (inputs.size() > 3 && inputs[3]->node()->kind() != kUndefined) {
      // Opset 12 requires empty scales when sizes is provided; opset 13 omits scales instead.
      Node* empty_input = graph->create(kUndefined);
      empty_input->insertBefore(node);
      node->replaceInput(2, empty_input->output());
    }
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
