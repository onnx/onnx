// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct FuseConsecutiveTransposes : public OptimizePass {
  explicit FuseConsecutiveTransposes()
    : OptimizePass("fuse_consecutive_transposes", API_TYPE::IR) {
  }

  // returns a vector `ret` such that transposing by `ret` is equivalent
  // to transposing by `t1` and then by `t2`
  std::vector<int64_t> compose_transposes(const std::vector<int64_t> & t1,
      const std::vector<int64_t> & t2) {
    ONNX_ASSERT(t1.size() == t2.size());
    std::vector<int64_t> ret;
    for (size_t i = 0; i < t1.size(); i++) {
      ONNX_ASSERT(   t1[i]  < (int)t2.size());
      ONNX_ASSERT(t2[t1[i]] < (int)t2.size());
      ret.push_back(t2[t1[i]]);
    }
    return ret;
  }

  void fuse_consecutive_transposes(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      if (n->kind() == kTranspose && n->input()->node()->kind() == kTranspose) {
        auto origInput = n->input();
        if (!n->hasAttribute(kperm) && !origInput->node()->hasAttribute(kperm)) {
          // One special case (two consecutive transposes with no perm,
          // since we do not have the shape information here, we have
          // to eliminate two transpose together.
          n->replaceAllUsesWith(origInput->node()->input()->node());
          it.destroyCurrent();
          it.destroyCurrent();
          continue;
        }
        if (!n->hasAttribute(kperm) || !origInput->node()->hasAttribute(kperm)) {
          continue;
        }
        n->is_(kperm, compose_transposes(origInput->node()->is(kperm), n->is(kperm)));
        n->replaceInput(0, origInput->node()->input());
        if (origInput->uses().size() == 0) {
          origInput->node()->destroy();
        }
        continue;
      }
    }
  }

  virtual void optimize(Graph& graph) {
    fuse_consecutive_transposes(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
