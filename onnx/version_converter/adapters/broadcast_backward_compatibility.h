// Adapter for broadcasting ops in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class BroadcastBackwardCompatibility final : public Adapter {
  public:
    explicit BroadcastBackwardCompatibility(const std::string& op_name, const OpSetID&
      initial, const OpSetID& target): Adapter(op_name, initial, target) {}

    void adapt_broadcast_backward_compatibility(std::shared_ptr<Graph> graph, Node* node) const {
      // Verify that broadcasts are allowed in limited spec of opset version 6
      // Multidirectional broadcasting, as defined in Broadcasting.md
      // MathDocGenerator provides differences
      // Main change: encode broadcasting commands as explicit attribute
      const ArrayRef<Value*>& inputs = node->inputs();
      ONNX_ASSERTM(inputs.size() == 2, "%s in opset version 6 can only broadcast"
        " between 2 inputs", name().c_str());
      ONNX_ASSERTM(inputs[0]->has_sizes(), "Shape of first input is not available.");
      std::vector<Dimension> A_sizes = inputs[0]->sizes();
      ONNX_ASSERTM(inputs[1]->has_sizes(), "Shape of second input is not available.");
      std::vector<Dimension> B_sizes = inputs[1]->sizes();
      // Ensure that first input is larger than or equal to the second
      // numpy_unibroadcastable here is considered to be equivalent to opset1_broadcastable
      // This is because backwards conversion does not allow for an axis that is not
      // suffix matching
      if(numpy_unibroadcastable(A_sizes, B_sizes)) {
        // If conditional is not fulfilled, we have a default broadcast
        // Add broadcast attribute
        node->i_(kbroadcast, 1);
      }
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_broadcast_backward_compatibility(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
