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
      // If single element in slope, no conversion necessary
      if (slope_sizes.size() != 1 || slope_sizes[0].dim != 1) {
        // TODO: If CHW where HW are all 1's, then squeeze to remove all these dimensions
        for (int i = 1; i < slope_sizes.size(); i++) {
          ONNX_ASSERTM(slope_sizes[i].dim == 1,
              "All trailing dimensions of slope input into PRelu must be 1");
        }
        // TODO: Add Squeeze op

      }
      // TODO: Assert that axis is 1 in non-1D case
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_prelu_7_6(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
