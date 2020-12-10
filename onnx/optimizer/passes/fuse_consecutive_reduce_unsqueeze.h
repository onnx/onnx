// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

const std::unordered_set<NodeKind> reduction_operators{
    kReduceL1,
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
    bool cur_node_check = node->kind() == kUnsqueeze;
    if (cur_node_check) {
      Node* prev_node = node->inputs()[0]->node();
      // check that the previous node a reduction operator and has defined
      // axes/keepdims
      bool reduction_node_check = reduction_operators.find(prev_node->kind()) !=
              reduction_operators.end() &&
          prev_node->hasAttribute(kkeepdims);
      if (reduction_node_check) {
        // insure that keepdims is set to false currently
        return prev_node->i(kkeepdims) == 0;
      }
    }
    return false;
  }
  bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current)
      override {
    // get unsqueeze axes
    auto unsqueeze_axes_input = node->inputs()[1];
    auto unsqueeze_axes_initializer =
        graph.getInitializer(unsqueeze_axes_input->uniqueName());
    if (unsqueeze_axes_initializer == graph.initializers().end())
      return false;
    const auto& unsqueeze_axes =
        ParseData<int64_t>(&*unsqueeze_axes_initializer);

    Node* reduction_op = node->inputs()[0]->node();
    std::vector<int64_t> reduction_axes;
    if (reduction_op->hasAttribute(kaxes)) {
      // get reduction axes from attribute value
      reduction_axes = reduction_op->is(kaxes);
    } else {
      auto reduction_axes_input = reduction_op->inputs()[1];
      auto reduction_axes_initializer =
          graph.getInitializer(reduction_axes_input->uniqueName());
      if (reduction_axes_initializer == graph.initializers().end())
        return false;
      reduction_axes = ParseData<int64_t>(&*reduction_axes_initializer);
    }
    if (unsqueeze_axes != reduction_axes)
      return false;

    // set keepdims flag to be true
    reduction_op->i_(kkeepdims, 1);
    // remove unnecessary unsqueeze
    reduction_op->output()->setSizes(node->output()->sizes());
    reduction_op->output()->setElemType(node->output()->elemType());
    node->output()->replaceAllUsesWith(node->inputs()[0]);
    if (unsqueeze_axes_input->uses().size() == 1) {
      node->removeInput(1);
      graph.eraseInitializerAndInput(unsqueeze_axes_input);
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
