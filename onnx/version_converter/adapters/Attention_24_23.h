// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Attention in default domain from version 24 to 23

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Attention_24_23 final : public Adapter {
 public:
  explicit Attention_24_23() : Adapter("Attention", OpSetID(24), OpSetID(23)) {}

  void adapt_attention_24_23(const std::shared_ptr<Graph>&, Node* node) const {
    const ArrayRef<Value*>& inputs = node->inputs();

    // Check if nonpad_kv_seqlen input is present (input index 6)
    if (inputs.size() > 6) {
      ONNX_ASSERTM(
          false,
          "%s being converted from %d to %d has nonpad_kv_seqlen input, "
          "which is not supported in opset 23. This conversion cannot be performed.",
          name().c_str(),
          initial_version().version(),
          target_version().version());
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_attention_24_23(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
