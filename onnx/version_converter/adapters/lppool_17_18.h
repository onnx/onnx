/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for MaxPool in default domain from version 8 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"
#include "onnx/version_converter/adapters/transformers.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class LpPool_17_18 final : public Adapter {
 public:
  explicit LpPool_17_18() : Adapter("LpPool", OpSetID(17), OpSetID(18)) {}

  void adapt_lppool_17_18(std::shared_ptr<Graph>, Node* node) const {
    if (!node->hasAttribute(kceil_mode))
      SetAttribute(kceil_mode, 0);
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_lppool_17_18(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
