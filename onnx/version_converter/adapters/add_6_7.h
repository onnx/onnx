// Adapter for Add in default domain from version 6 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Add_6_7 final : public Adapter {
  public:
    explicit Add_6_7()
      : Adapter("Add", OpSetID(6), OpSetID(7)) {
      }

    void adapt_add_6_7(std::shared_ptr<Graph> graph, Node* node)
      const {
      // Remove axis and broadcast attributes
      if (node->hasAttribute(kaxis)) node->removeAttribute(kaxis);
      if (node->hasAttribute(kbroadcast)) node->removeAttribute(kbroadcast);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_add_6_7(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
