// Adapter for Gemm in default domain from version 6 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Gemm_6_7 final : public Adapter {
  public:
    explicit Gemm_6_7(): Adapter("Gemm", OpSetID(6), OpSetID(7)) {}

    void adapt_gemm_6_7(std::shared_ptr<Graph> graph, Node* node) const {
      const ArrayRef<Value*>& inputs = node->inputs();
      ONNX_ASSERTM(inputs.size() == 3, "3 Inputs must be provided to Gemm");
      ONNX_ASSERTM(inputs[0]->has_sizes(), "Insufficient shape information for 1st input");
      const auto& A_shape = inputs[0]->sizes();
      assertNotParams(A_shape);
      ONNX_ASSERTM(inputs[1]->has_sizes(), "Insufficient shape information for 2nd input");
      const auto& B_shape = inputs[1]->sizes();
      assertNotParams(B_shape);
      // Determine if C is broadcastable
      ONNX_ASSERTM(inputs[2]->has_sizes(), "Insufficient shape information for 3rd input");
      const auto& C_shape = inputs[2]->sizes();
      assertNotParams(C_shape);
      // Create (M, N) to input to numpy_unibroadcastable
      // TODO: Reconcile fact that shapes aren't determined for 1st 2 inputs
      std::vector<Dimension> MN;
      if (node->hasAttribute(ktransA) && node->i(ktransA) == 1) {
        MN.emplace_back(A_shape[1]);
      } else {
        MN.emplace_back(A_shape[0]);
      }
      if (node->hasAttribute(ktransB) && node->i(ktransB) == 1) {
        MN.emplace_back(B_shape[0]);
      } else {
        MN.emplace_back(B_shape[1]);
      }
      numpy_unibroadcastable(MN, C_shape);
      if (node->hasAttribute(kbroadcast)) node->removeAttribute(kbroadcast);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_gemm_6_7(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
