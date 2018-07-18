// Adapter for Relu in default domain from version 5 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Relu_5_6 final : public Adapter {
  explicit Relu_5_6()
    : Adapter("Relu", OpSetID(5), OpSetID(6)) {
    }

  void adapt_relu_5_6(std::shared_ptr<Graph> graph, Node* node) const {
    // Remove consumed_inputs (legacy optimization attribute)
    node->removeAttribute(kconsumed_inputs);
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_relu_5_6(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
