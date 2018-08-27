// Adapter for BatchNormalization in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct BatchNormalization_7_6 final : public Adapter {
  explicit BatchNormalization_7_6()
    : Adapter("BatchNormalization", OpSetID(7), OpSetID(6)) {
    }

  void adapt_batch_normalization_7_6(std::shared_ptr<Graph> graph, Node* node) const {
    node->i_(kis_test, 1);
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_batch_normalization_7_6(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
