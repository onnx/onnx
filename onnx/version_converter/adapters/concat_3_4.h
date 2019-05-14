// Adapter for Concat in default domain from version 3 to 4

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Concat_3_4 final : public Adapter {
  public:
    explicit Concat_3_4()
      : Adapter("Concat", OpSetID(3), OpSetID(4)) {}

    void adapt_concat_3_4(std::shared_ptr<Graph>, Node* node) const {
      // If axis is not present, set to 1
      if(!(node->hasAttribute(kaxis))) node->i_(kaxis, 1);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_concat_3_4(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
