// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct EliminateNopPad final : public OptimizePass {
  explicit EliminateNopPad()
    : OptimizePass("eliminate_nop_pad", API_TYPE::IR) {
  }

  static bool is_nop_pad(const std::vector<int64_t> & pads) {
    for (size_t i = 0; i < pads.size(); i++)
      if (pads[i] > 0)
        return false;
    return true;
  }

  void eliminate_nop_pad(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){eliminate_nop_pad(g);});
      if (n->kind() == kPad && n->hasAttribute(kpads)) {
        if (is_nop_pad(n->is(kpads))) {
          n->output()->replaceAllUsesWith(n->input());
          it.destroyCurrent();
          continue;
        }
      }
    }
  }

  void optimize(Graph& graph) override {
    eliminate_nop_pad(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
