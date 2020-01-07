// Adapter for Dropout in default domain from version 6 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Dropout_6_7 final : public Adapter {
  public:
    explicit Dropout_6_7(): Adapter("Dropout", OpSetID(6), OpSetID(7)) {}

    void adapt_dropout_6_7(std::shared_ptr<Graph>, Node* node) const {
      if (node->hasAttribute(kis_test)) {
        ONNX_ASSERTM(node->i(kis_test) == 1, "Training is not supported with Dropout Op");
        node->removeAttribute(kis_test);
      }
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_dropout_6_7(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
