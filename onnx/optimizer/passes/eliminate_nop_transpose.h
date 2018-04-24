// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct EliminateNopTranspose final : public OptimizePass {
  explicit EliminateNopTranspose()
    : OptimizePass("eliminate_nop_transpose", API_TYPE::IR) {
  }

  static bool is_nop_transpose(const std::vector<int64_t> & perm) {
    for (size_t i = 0; i < perm.size(); i++)
      if (perm[i] != (int)i)
        return false;
    return true;
  }

  void eliminate_nop_transpose(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){eliminate_nop_transpose(g);});
      if (n->kind() == kTranspose && n->hasAttribute(kperm)) {
        if (is_nop_transpose(n->is(kperm))) {
          n->output()->replaceAllUsesWith(n->input());
          it.destroyCurrent();
          continue;
        }
      }
    }
  }

  void optimize(Graph& graph) override {
    eliminate_nop_transpose(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
