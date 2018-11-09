// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopPad final : public PredicateBasedPass {
  explicit EliminateNopPad()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_pad";
  }
  static bool is_nop_pad(const std::vector<int64_t>& pads) {
    for (size_t i = 0; i < pads.size(); i++)
      if (pads[i] > 0)
        return false;
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    return (node->kind() == kPad && node->hasAttribute(kpads)) &&
        is_nop_pad(node->is(kpads));
  }
  bool runTransform(Node* node, Graph&, NodeDestroyType& destroy_current)
      override {
    node->output()->replaceAllUsesWith(node->input());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
