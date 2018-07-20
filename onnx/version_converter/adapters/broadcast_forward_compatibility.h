// Adapter for broadcasting ops in default domain from version 6 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class BroadcastForwardCompatibility final : public Adapter {
  public:
    explicit BroadcastForwardCompatibility(const std::string& op_name, const OpSetID&
      initial, const OpSetID& target): Adapter(std::move(op_name), std::move(
        initial), std::move(target)) {}

    void adapt_broadcast_forward_compatibility(std::shared_ptr<Graph> graph, Node* node)
      const {
      // Remove axis and broadcast attributes
      if (node->hasAttribute(kaxis)) node->removeAttribute(kaxis);
      if (node->hasAttribute(kbroadcast)) node->removeAttribute(kbroadcast);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_broadcast_forward_compatibility(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
