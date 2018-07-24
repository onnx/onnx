// Adapter for Reshape in default domain from version 4 to 5

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Reshape_4_5 final : public Adapter {
  public:
    explicit Reshape_4_5()
      : Adapter("Reshape", OpSetID(4), OpSetID(5)) {}

    void adapt_reshape_4_5(std::shared_ptr<Graph> graph, Node* node) const {
      // TODO: Create Input from Attribute

    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_reshape_4_5(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
