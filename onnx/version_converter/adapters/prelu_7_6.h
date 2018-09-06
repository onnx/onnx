// Adapter for PRelu in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class PRelu_7_6 final : public Adapter {
  public:
    explicit PRelu_7_6(): Adapter("PRelu", OpSetID(7), OpSetID(6)) {}

    void adapt_prelu_7_6(std::shared_ptr<Graph> graph, Node* node) const {
      // Throw an exception if any broadcasting occurs
      const ArrayRef<Value*>& inputs = node->inputs();
      assertInputsAvailable(inputs, name().c_str(), 2);
      std::vector<Dimension> X_sizes = inputs[0]->sizes();
      std::vector<Dimension> slope_sizes = inputs[1]->sizes();
      // TODO: If single element, no conversion necessary
      // TODO: If CHW where HW are all 1's, then squeeze to remove all these dimensions
      // TODO: Assert that axis is 1 in non-1D case
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_prelu_7_6(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
