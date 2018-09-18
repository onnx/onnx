// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveLogSoftmax final : public OptimizePass {
  explicit FuseConsecutiveLogSoftmax()
      : OptimizePass("fuse_consecutive_log_softmax", API_TYPE::IR) {}

  void fuse_consecutive_log_softmax(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(
          n, [this](Graph& g) { fuse_consecutive_log_softmax(g); });
      if (n->kind() == kLog && n->input()->node()->kind() == kSoftmax &&
          n->input()->uses().size() == 1) {
        Node* log_node = n;
        Value* log_node_output = log_node->output();
        Node* softmax_node = log_node->inputs()[0]->node();
        auto orig_input = log_node->input();
        Node* log_softmax_node = graph.create(kLogSoftmax, 1);

        // log_softmax_node construction
        log_softmax_node->i_(kaxis, softmax_node->i(kaxis));
        log_softmax_node->addInput(softmax_node->input());
        log_softmax_node->insertBefore(softmax_node);
        log_softmax_node->output()->setSizes(log_node_output->sizes());
        log_softmax_node->output()->setElemType(log_node_output->elemType());

        log_node->replaceAllUsesWith(log_softmax_node);
        log_node->removeAllInputs();
        it.destroyCurrent();
        it.destroyCurrent();
      }
    }
  }

  void optimize(Graph& graph) override {
    fuse_consecutive_log_softmax(graph);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
