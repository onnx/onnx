// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

const std::unordered_set<NodeKind> reduction_operators{kReduceL1,
                                                       kReduceL2,
                                                       kReduceLogSum,
                                                       kReduceLogSumExp,
                                                       kReduceMax,
                                                       kReduceMean,
                                                       kReduceMin,
                                                       kReduceProd,
                                                       kReduceSum,
                                                       kReduceSumSquare};

struct FuseConsecutiveReduceUnsqueeze final : public PredicateBasedPass {
  explicit FuseConsecutiveReduceUnsqueeze()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_reduce_unsqueeze";
  }
  bool patternMatchPredicate(Node* node) override {
    // check that the current node is of type Unsqueeze and has defined axes
    bool cur_node_check =
        node->kind() == kUnsqueeze && node->hasAttribute(kaxes);
    if (cur_node_check) {
      Node* prev_node = node->input()->node();
      // check that the previous node a reduction operator and has defined
      // axes/keepdims
      bool reduction_node_check = reduction_operators.find(prev_node->kind()) !=
              reduction_operators.end() &&
          prev_node->hasAttribute(kaxes) && prev_node->hasAttribute(kkeepdims);
      if (reduction_node_check) {
        // insure that keepdims is set to false currently
        return prev_node->i(kkeepdims) == 0 && node->is(kaxes) == prev_node->is(kaxes);
      }
    }
    return false;
  }
  bool runTransform(Node* node, Graph&, NodeDestroyType& destroy_current)
      override {
    Node* reduction_op = node->input()->node();
    // set keepdims flag to be true
    reduction_op->i_(kkeepdims, 1);
    // remove unnecessary unsqueeze
    reduction_op->output()->setSizes(node->output()->sizes());
    reduction_op->output()->setElemType(node->output()->elemType());
    node->output()->replaceAllUsesWith(node->input());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
