// Adapter for AveragePool in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class AveragePool_7_6 final : public Adapter {
  public:
    explicit AveragePool_7_6(): Adapter("AveragePool", OpSetID(7), OpSetID(6)) {}

    void adapt_averagepool_7_6(std::shared_ptr<Graph>, Node* node) const {
      if (node->hasAttribute(kcount_include_pad))
        ONNX_ASSERTM(node->i(kcount_include_pad) == 0, "AveragePool in Opset "
            "Version 6 does not support including pad");
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_averagepool_7_6(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
