/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for MaxPool in default domain from version 17 to 18

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class MaxPool_17_18 final : public Adapter {
 public:
  explicit MaxPool_17_18() : Adapter("MaxPool", OpSetID(17), OpSetID(18)) {}

  void adapt_maxpool_17_18(std::shared_ptr<Graph>, Node* node) const {
    if (node->hasAttribute(kstorage_order))
      node->removeAttribute(kstorage_order);
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_maxpool_17_18(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
