// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct EliminateIdentity final : public OptimizePass {
  explicit EliminateIdentity()
    : OptimizePass("eliminate_identity", API_TYPE::IR) {
  }

  void eliminate_identity(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){eliminate_identity(g);});
      if (n->kind() == kIdentity) {
        n->output()->replaceAllUsesWith(n->input());
        it.destroyCurrent();
      }
    }
  }

  void optimize(Graph& graph) override {
    eliminate_identity(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
