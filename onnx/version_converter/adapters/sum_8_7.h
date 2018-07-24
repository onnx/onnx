// Adapter for Sum in default domain from version 8 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Sum_8_7 final : public Adapter {
  public:
    explicit Sum_8_7(): Adapter("Sum", OpSetID(8), OpSetID(7)) {}

    void adapt_sum_8_7(std::shared_ptr<Graph> graph, Node* node) const {
      // TODO: Throw an exception if any broadcasting occurs
      const ArrayRef<Value*>& inputs = node->inputs();
      // Determine if inputs are of different sizes
      for (int i = 1; i < (int) inputs.size(); i++) {
        std::vector<Dimension> A_sizes = inputs[i - 1]->sizes();
        std::vector<Dimension> B_sizes = inputs[i]->sizes();
        std::string error = "Sum in OpSet Version 6 does not support broadcasting";
        ONNX_ASSERT(A_sizes.size() == B_sizes.size());
        for (int j = 1; j < A_sizes.size(); j++) {
          ONNX_ASSERT(A_sizes[j - 1].dim == B_sizes[j].dim);
        }
      }
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_sum_8_7(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
