// Adapter for broadcasting ops in default domain from version 6 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class BroadcastForwardCompatibility final : public Adapter {
  public:
    explicit BroadcastForwardCompatibility(const std::string& op_name, const OpSetID&
      initial, const OpSetID& target): Adapter(op_name, initial, target) {}

    void adapt_broadcast_forward_compatibility(std::shared_ptr<Graph> graph, Node* node)
      const {
      // Remove axis and broadcast attributes
      // Assess whether axis requires reshaping
      if (node->hasAttribute(kbroadcast)) {
        if (node->hasAttribute(kaxis)) {
          const ArrayRef<Value*>& inputs = node->inputs();
          ONNX_ASSERTM(inputs.size() == 2, "Add in opset version 6 can only broadcast"
            " between 2 inputs");
          ONNX_ASSERTM(inputs[0].has_sizes(), "Shape of A must be statically determined.");
          std::vector<Dimension> A_sizes = inputs[0]->sizes();
          ONNX_ASSERTM(inputs[1].has_sizes(), "Shape of B must be statically determined.");
          std::vector<Dimension> B_sizes = inputs[1]->sizes();
          if (node->i(kaxis) != A_sizes - B_sizes) {
            // TODO: Add a Reshape node before input B

          }
        }
        node->removeAttribute(kbroadcast);
      }
      if (node->hasAttribute(kaxis)) node->removeAttribute(kaxis);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_broadcast_forward_compatibility(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
