// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveLogSoftmax final : public PredicateBasedPass {
  explicit FuseConsecutiveLogSoftmax()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_log_softmax";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kLog && node->input()->node()->kind() == kSoftmax &&
        node->input()->uses().size() == 1;
  }
  bool runTransform(
      Node* log_node,
      Graph& graph,
      NodeDestroyType& destroy_current) override {
    Value* log_node_output = log_node->output();
    Node* softmax_node = log_node->inputs()[0]->node();
    Node* log_softmax_node = graph.create(kLogSoftmax, 1);

    // log_softmax_node construction
    log_softmax_node->i_(kaxis, softmax_node->i(kaxis));
    log_softmax_node->addInput(softmax_node->input());
    log_softmax_node->insertBefore(softmax_node);
    log_softmax_node->output()->setSizes(log_node_output->sizes());
    log_softmax_node->output()->setElemType(log_node_output->elemType());

    log_node->replaceAllUsesWith(log_softmax_node);
    log_node->removeAllInputs();
    destroy_current = NodeDestroyType::DestroyTwo;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
