// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopDropout final : public PredicateBasedPass {
  explicit EliminateNopDropout()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_dropout";
  }

  bool patternMatchPredicate(Node* node) override {
    return (node->kind() == kDropout && node->hasAttribute(kratio)) &&
        node->f(kratio) == 0.0;
  }

  bool runTransform(Node* node, Graph&, NodeDestroyType& destroy_current)
      override {
    // Don't assume that theres only one output.
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      node->outputs()[i]->replaceAllUsesWith(node->input());
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
