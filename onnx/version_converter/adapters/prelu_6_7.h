// Adapter for PRelu in default domain from version 6 to 7

#pragma once

#include "onnx/version_converter/adapters/broadcast_forward_compatibility.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class PRelu_6_7 final : public BroadcastForwardCompatibility {
  public:
    explicit PRelu_6_7(): BroadcastForwardCompatibility("PRelu", OpSetID(6),
        OpSetID(7)) {}

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      const ArrayRef<Value*>& inputs = node->inputs();
      assertInputsAvailable(inputs, name().c_str(), 2);
      ONNX_ASSERTM(inputs[2]->sizes().size() == 1,
          "PRelu-7 only supports 1D slopes.");
      node->i_(kaxis, 1);
      BroadcastForwardCompatibility::adapt(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
