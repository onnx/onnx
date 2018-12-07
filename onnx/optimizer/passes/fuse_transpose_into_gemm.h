// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseTransposeIntoGemm final : public PredicateBasedPass {
  explicit FuseTransposeIntoGemm()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_transpose_into_gemm";
  }
  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kGemm;
  }
  bool runTransform(Node* n, Graph&, NodeDestroyType& destroy_current)
      override {
    const std::vector<int64_t> simple_trans_perm({1, 0});
    destroy_current = NodeDestroyType::DestroyZero;
    bool ret_val = false;
    for (size_t i : {0, 1}) {
      auto inp = n->inputs()[i];
      auto trans = i == 0 ? ktransA : ktransB;
      if (inp->node()->kind() == kTranspose &&
          inp->node()->is(kperm) == simple_trans_perm) {
        n->replaceInput(i, inp->node()->input());
        n->i_(trans, n->hasAttribute(trans) ? !n->i(trans) : 1);
        if (inp->uses().size() == 0) {
          inp->node()->destroy();
          ret_val = true;
        }
      }
    }
    return ret_val;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
