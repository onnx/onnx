// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveTransposes final : public PredicateBasedPass {
  explicit FuseConsecutiveTransposes()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_transposes";
  }

  // returns a vector `ret` such that transposing by `ret` is equivalent
  // to transposing by `t1` and then by `t2`
  std::vector<int64_t> compose_transposes(
      const std::vector<int64_t>& t1,
      const std::vector<int64_t>& t2) {
    ONNX_ASSERT(t1.size() == t2.size());
    std::vector<int64_t> ret;
    ret.reserve(t1.size());
    for (size_t i = 0; i < t1.size(); i++) {
      ONNX_ASSERT(t1[i] < static_cast<int64_t>(t2.size()));
      ONNX_ASSERT(
          t2[static_cast<size_t>(t1[i])] < static_cast<int64_t>(t2.size()));
      ret.push_back(t2[static_cast<size_t>(t1[i])]);
    }
    return ret;
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kTranspose &&
        node->input()->node()->kind() == kTranspose;
  }

  bool runTransform(Node* n, Graph&, NodeDestroyType& destroy_current)
      override {
    auto origInput = n->input();
    if (!n->hasAttribute(kperm) && !origInput->node()->hasAttribute(kperm)) {
      // One special case (two consecutive transposes with no perm,
      // since we do not have the shape information here, we have
      // to eliminate two transpose together.
      n->replaceAllUsesWith(origInput->node()->input()->node());
      destroy_current = NodeDestroyType::DestroyTwo;
      return true;
    }
    if (!n->hasAttribute(kperm) || !origInput->node()->hasAttribute(kperm)) {
      destroy_current = NodeDestroyType::DestroyZero;
      return false;
    }
    n->is_(
        kperm, compose_transposes(origInput->node()->is(kperm), n->is(kperm)));
    n->replaceInput(0, origInput->node()->input());
    if (origInput->uses().size() == 0) {
      origInput->node()->destroy();
    }
    destroy_current = NodeDestroyType::DestroyZero;
    return false;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
