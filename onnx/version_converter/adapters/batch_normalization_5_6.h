// Adapter for BatchNormalization in default domain from version 5 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct BatchNormalization_5_6 final : public Adapter {
  explicit BatchNormalization_5_6()
    : Adapter("BatchNormalization", OpSetID(5), OpSetID(6)) {
    }

  void adapt_batch_normalization_5_6(std::shared_ptr<Graph> graph, Node* node) const {
    if (node->hasAttribute(kconsumed_inputs)) node->removeAttribute(kconsumed_inputs);
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_batch_normalization_5_6(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
