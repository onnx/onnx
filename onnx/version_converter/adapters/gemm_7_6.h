// Adapter for Gemm in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Gemm_7_6 final : public Adapter {
  public:
    explicit Gemm_7_6(): Adapter("Gemm", OpSetID(7), OpSetID(6)) {}

    void adapt_gemm_7_6(std::shared_ptr<Graph> graph, Node* node) const {
      const ArrayRef<Value*>& inputs = node->inputs();
      // Determine if C is broadcastable
      // Get M and N
      int M = inputs[0]->sizes()[0].dim;
      if (node->hasAttribute(ktransA) && node->i(ktransA) != 0)
        M = inputs[0]->sizes()[1].dim;
      int N = inputs[1]->sizes()[1].dim;
      if (node->hasAttribute(ktransA) && node->i(ktransA) != 0)
        N = inputs[0]->sizes()[0].dim;
      const auto& C_shape = inputs[2]->sizes();
      int C_M = C_shape[0].dim;
      std::string error = "C not unidirectionally broadcastable to (M, N)";
      if (C_shape.size() == 2) {
        int C_N = C_shape[1].dim;
        ONNX_ASSERT((C_M == M || C_M == 1) && (C_N == N || C_N == 1));
      } else {
        ONNX_ASSERT(C_M == N || C_M == 1);
      }
      node->i_(kbroadcast, 1);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_gemm_7_6(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
