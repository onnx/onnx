// Adapter for Gemm in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Gemm_7_6 final : public Adapter {
  public:
    explicit Gemm_7_6(): Adapter("Gemm", OpSetID(7), OpSetID(6)) {}

    void adapt_gemm_7_6(std::shared_ptr<Graph> graph, Node* node) const {
      const ArrayRef<Value*>& inputs = node->inputs();
      ONNX_ASSERTM(inputs.size() == 3, "3 Inputs must be provided to Gemm");
      // Determine if C is broadcastable
      ONNX_ASSERTM(inputs[2]->has_sizes(), "Insufficient shape information for 3rd input");
      const auto& C_shape = inputs[2]->sizes();
      if (numpy_unibroadcastable(C_shape)) {
        node->i_(kbroadcast, 1);
      }
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_gemm_7_6(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
