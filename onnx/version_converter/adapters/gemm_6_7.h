// Adapter for Gemm in default domain from version 6 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Gemm_6_7 final : public Adapter {
  public:
    explicit Gemm_6_7(): Adapter("Gemm", OpSetID(6), OpSetID(7)) {}

    void adapt_gemm_6_7(std::shared_ptr<Graph> graph, Node* node) const {
      const ArrayRef<Value*>& inputs = node->inputs();
      assertInputsAvailable(inputs, name().c_str(), 3);
      const auto& A_shape = inputs[0]->sizes();
      const auto& B_shape = inputs[1]->sizes();
      // Determine if C is broadcastable
      const auto& C_shape = inputs[2]->sizes();
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
      assert_numpy_unibroadcastable_and_require_broadcast(MN, C_shape);
      if (node->hasAttribute(kbroadcast)) node->removeAttribute(kbroadcast);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_gemm_6_7(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
