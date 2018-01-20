#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace onnx { namespace optimization {

struct EliminateNopTranspose : public OptimizePass {
  explicit EliminateNopTranspose()
    : OptimizePass("eliminate_nop_transpose", API_TYPE::ir) {
  }

  static bool is_nop_transpose(const std::vector<int64_t> & perm) {
    for (size_t i = 0; i < perm.size(); i++)
      if (perm[i] != (int)i)
        return false;
    return true;
  }

  void eliminate_nop_transpose(std::shared_ptr<Graph>& graph) {
    for (auto it = graph->begin(); it != graph->end(); ++it) {
      auto* n = *it;

      if (n->kind() == kTranspose) {
        if (is_nop_transpose(n->is(kperm))) {
          n->replaceAllUsesWith(n->input()->node());
          it.destroyCurrent();
          continue;
        }
      }
    }
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

  void fuse_consecutive_transposes(std::shared_ptr<Graph>& graph) {
    for (auto it = graph->begin(); it != graph->end(); ++it) {
      auto* n = *it;

      if (n->kind() == kTranspose && n->input()->node()->kind() == kTranspose) {
        auto origInput = n->input();
        n->is_(kperm, compose_transposes(origInput->node()->is(kperm), n->is(kperm)));
        n->replaceInput(0, origInput->node()->input());
        if (origInput->uses().size() == 0) {
          origInput->node()->destroy();
        }
        continue;
      }
    }
  }

  virtual void optimize(std::shared_ptr<Graph>& graph) {
    fuse_consecutive_transposes(graph);
  }
};

}} // namespace onnx::optimization
