// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Pad in default domain from version 18 to 17

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class Pad_18_17 final : public Adapter {
 public:
  explicit Pad_18_17() : Adapter("Pad", OpSetID(18), OpSetID(17)) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    if (node->inputs().size() == 4) {
      ONNX_ASSERTM(
          node->inputs()[3]->node()->kind() == kUndefined,
          "Pad axes input is not supported when converting from opset 18 "
          "to opset 17.")
      node->removeInput(3);
    }
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
