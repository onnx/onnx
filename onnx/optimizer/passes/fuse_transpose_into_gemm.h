// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct FuseTransposeIntoGemm : public OptimizePass {
  explicit FuseTransposeIntoGemm()
    : OptimizePass("fuse_transpose_into_gemm", API_TYPE::IR) {
  }

  void fuse_transpose_into_gemm(Graph& graph) {
    static const std::vector<int64_t> simple_trans_perm({1,0});

    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;

      if (n->kind() == kGemm) {
        for (size_t i : {0,1}) {
          auto inp = n->inputs()[i];
          auto trans = i == 0 ? ktransA : ktransB;
          if (inp->node()->kind() == kTranspose && inp->node()->is(kperm) == simple_trans_perm) {
            n->replaceInput(i, inp->node()->input());
            n->i_(trans, n->hasAttribute(trans) ? !n->i(trans) : 1);
            if (inp->uses().size() == 0) {
              inp->node()->destroy();
            }
          }
        }
      }
    }
  }

  virtual void optimize(Graph& graph) {
    fuse_transpose_into_gemm(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
