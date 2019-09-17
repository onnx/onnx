// Adapter for BatchNormalization in default domain from version 8 to 9

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct BatchNormalization_8_9 final : public Adapter {
  explicit BatchNormalization_8_9()
    : Adapter("BatchNormalization", OpSetID(8), OpSetID(9)) {
    }

  void adapt_batch_normalization_8_9(std::shared_ptr<Graph>, Node* node) const {

    Symbol spatial = Symbol("spatial");
      if (node->hasAttribute(spatial)) {
        if (node->i(spatial) == 1) {
          node->removeAttribute(spatial);
            return;
        }
        ONNX_ASSERT("Unsupported conversion due to change in dimensions of inputs when spatial is set to false.");
      }
    }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_batch_normalization_8_9(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
