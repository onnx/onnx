
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.
#pragma once
#include "onnx/optimizer/pass.h"
namespace ONNX_NAMESPACE {
namespace optimization {
struct EliminateDeadEnd final : public FullGraphBasedPass {
  explicit EliminateDeadEnd()
      : FullGraphBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "eliminate_deadend";
  }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::CountBased;
  }
  unsigned int EliminateDead(Graph& graph) {
    unsigned int nodes_removed = 0;
    auto nodes = graph.nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      if (!node->hasUses()) {
        nodes_removed++;
        it.destroyCurrent();
      }
    }
    return nodes_removed;
  }
  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    auto nodes_removed = this->EliminateDead(graph);
    return std::shared_ptr<PostPassAnalysis>(
        new CountBasedPassAnalysis(this, nodes_removed, false, false));
  }
};
} // namespace optimization
} // namespace ONNX_NAMESPACE