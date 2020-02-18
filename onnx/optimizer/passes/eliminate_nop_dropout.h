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
    // in opset 12, ratio is an input of Dropout rather than an attribute,
    // however we don't want to to remove Dropout fro opset 12+, since it
    // supports training-friendly models, for which the Dropout ops are required
    return (node->kind() == kDropout && node->hasAttribute(kratio)) &&
        node->f(kratio) == 0.0;
  }

  bool runTransform(Node* node, Graph&, NodeDestroyType& destroy_current)
      override {
    // Don't assume that theres only one output.
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      node->outputs()[i]->replaceAllUsesWith(node->input());
    }
    if (node->outputs()[0]->has_sizes()) {
        node->input()->setSizes(node->outputs()[0]->sizes());
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
