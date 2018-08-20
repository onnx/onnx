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
        const ArrayRef<Value*>& inputs = node->inputs();
        ONNX_ASSERTM(inputs.size() == 2, "%s in opset version 6 can only broadcast"
          " between 2 inputs", name().c_str());
        ONNX_ASSERTM(inputs[0]->has_sizes(), "Shape of first input not available.");
        std::vector<Dimension> A_sizes = inputs[0]->sizes();
        assertNotParams(A_sizes);
        ONNX_ASSERTM(inputs[1]->has_sizes(), "Shape of second input not available.");
        std::vector<Dimension> B_sizes = inputs[1]->sizes();
        assertNotParams(B_sizes);
        // Also assert that broadcasting syntax are correct if axis is not present
        if (node->hasAttribute(kaxis)) {
          if (!onnx_opset7_broadcastable(node->i(kaxis), A_sizes, B_sizes)) {
            // Add a Reshape node before input B
            Node * n = graph->create(kUnsqueeze);
            n->addInput(inputs[1]);
            std::vector<int64_t> axes;
            for (int i = 0; i < (int) (A_sizes.size() - B_sizes.size()); i++) {
              axes.emplace_back(B_sizes.size() + i);
            }
            n->is_(kaxes, std::forward<const std::vector<int64_t>>(axes));
            // Set 2nd input to node to 1st of n and output of n to 2nd input to node
            node->replaceInput(1, n->output());
            // Move n before node
            node->moveBefore(n);
          } else {
            onnx_opset7_broadcastable(A_sizes.size() - B_sizes.size(), A_sizes,
                B_sizes);
          }
        }
        node->removeAttribute(kbroadcast);
      }
      if (node->hasAttribute(kaxis)) node->removeAttribute(kaxis);
      // Assert multi_broadcastable on inputs
      const ArrayRef<Value*>& inputs = node->inputs();
      numpy_multibroadcastable(inputs[0]->sizes(), inputs[1]->sizes());
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_broadcast_forward_compatibility(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
