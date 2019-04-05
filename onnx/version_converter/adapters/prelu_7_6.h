// Adapter for PRelu in default domain from version 8 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class PRelu_7_6 final : public Adapter {
  public:
    explicit PRelu_7_6(): Adapter("PRelu", OpSetID(7), OpSetID(6)) {}

    void adapt_prelu_7_6(std::shared_ptr<Graph> graph, Node* node) const {
      // Throw an exception if any broadcasting occurs
      const ArrayRef<Value*>& inputs = node->inputs();
      std::vector<Dimension> X_sizes = inputs[0]->sizes();
      std::vector<Dimension> slope_sizes = inputs[1]->sizes();
      ONNX_ASSERTM(slope_sizes.size() == 1 || check_numpy_unibroadcastable_and_require_broadcast(X_sizes,
            slope_sizes) == 0,
          "OpSet Version 6 of PRelu does not support broadcasting.");
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_prelu_7_6(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
