// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct FuseConvAddIntoBias final : public OptimizePass {
  explicit FuseConvAddIntoBias()
    : OptimizePass("fuse_conv_add_into_bias", API_TYPE::IR) {
  }

  static int idx_of_conv(const onnx::ArrayRef<onnx::Value *> & values) {
    for (size_t i = 0; i < values.size(); i++)
      if (values[i]->node()->kind() == kConv)
        return i;
    return -1;
  }

  void fuse_conv_add_into_bias(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      if (n->kind() == kAdd) {
        int idx = idx_of_conv(n->inputs());
        if (idx != -1 && n->inputs()[idx]->node()->inputs().size() <= 2) {
          auto origConv = n->inputs()[idx];
          auto origBias = n->inputs()[1 - idx];
          auto broadcast = kbroadcast;
          auto axis = kaxis;
          auto size = ksize;
          if (n->hasAttribute(broadcast) && n->i(broadcast) == 1
                  && n->hasAttribute(axis) && n->i(axis) == 1) {
            origConv->node()->addInput(origBias);
            n->replaceAllUsesWith(origConv->node());
            it.destroyCurrent();
          }
        }
      }
    }
  }

  void optimize(Graph& graph) override {
    fuse_conv_add_into_bias(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
