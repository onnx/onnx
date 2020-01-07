// Adapter for BatchNormalization in default domain from version 6 to 5

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class BatchNormalization_6_5 final : public Adapter {
  public:
    explicit BatchNormalization_6_5()
      : Adapter("BatchNormalization", OpSetID(6), OpSetID(5)) {
      }

    void adapt_batch_normalization_6_5(std::shared_ptr<Graph>, Node* node) const {
      node->is_(kconsumed_inputs, {0, 0});
    }

     void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_batch_normalization_6_5(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
