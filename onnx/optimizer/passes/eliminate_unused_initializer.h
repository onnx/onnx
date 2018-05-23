// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   A = Const
//   D = Add(B, C)
// After:
//   D = Add(B, C)
//
// this pass can handle the case satisfy all following conditions:
//   condition 1: A is not used as any node's input
//   condition 2: A is not an output


#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateUnusedInitializer final : public OptimizePass {
  explicit EliminateUnusedInitializer()
      : OptimizePass("eliminate_unused_initializer", API_TYPE::IR) {}

  void erase_used_initializers(
      Graph& g,
      std::unordered_set<std::string> * initializer_names) {
    for (auto output : g.outputs()) {
      initializer_names->erase(output->uniqueName());
    }
    for (auto it = g.begin(); it != g.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(
          n,
          [this, initializer_names](Graph& g) {
            erase_used_initializers(g, initializer_names);
          });
      for (auto* input : n->inputs()) {
        initializer_names->erase(input->uniqueName());
      }
    }
  }

  void eliminate_unused_initializer(Graph& graph) {
    std::unordered_set<std::string> initializer_names(
        graph.initializer_names().begin(), graph.initializer_names().end());
    erase_used_initializers(graph, &initializer_names);

    // remove initializer and input if need
    for (std::string name : initializer_names) {
      graph.eraseInitializer(name);
      auto iter = std::find_if(
          graph.inputs().begin(), graph.inputs().end(), [&name](Value* input) {
            return input->uniqueName() == name;
          });
      if (iter != graph.inputs().end()) {
        graph.eraseInput(std::distance(graph.inputs().begin(), iter));
      }
    }
  }

  void optimize(Graph& graph) override {
    eliminate_unused_initializer(graph);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
