// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct FuseAddBiasIntoConv final : public OptimizePass {
  explicit FuseAddBiasIntoConv()
    : OptimizePass("fuse_add_bias_into_conv", API_TYPE::IR) {
  }

  static int idx_of_conv(const ArrayRef<Value *> & values) {
    for (size_t i = 0; i < values.size(); i++)
      if (values[i]->node()->kind() == kConv)
        return i;
    return -1;
  }

  void fuse_add_bias_into_conv(Graph& graph) {
    int size_lack_count = 0;
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      if (n->kind() == kAdd) {
        int idx = idx_of_conv(n->inputs());
        if (idx != -1 && n->inputs()[idx]->node()->inputs().size() == 2) {
          auto orig_conv = n->inputs()[idx];
          auto orig_bias = n->inputs()[1 - idx];
          auto conv_shape = orig_conv->sizes();
          auto bias_shape = orig_bias->sizes();
          auto weight_shape = orig_conv->node()->inputs()[1]->sizes();
          if (bias_shape.size() == 0
              || conv_shape.size() == 0
                  || weight_shape.size() == 0) {
            size_lack_count += 1;
            continue;
          }
          if (bias_shape.size() != 1
              || bias_shape[0].dim != conv_shape[1].dim
                  || bias_shape[0].dim != weight_shape[0].dim) {
            continue;
          }
          if (n->hasAttribute(kbroadcast) && n->i(kbroadcast) == 1
                  && n->hasAttribute(kaxis) &&
              (n->i(kaxis) == 1 || n->i(kaxis) == 1 - conv_shape.size())) {
            orig_conv->node()->addInput(orig_bias);
            n->replaceAllUsesWith(orig_conv->node());
            it.destroyCurrent();
          }
        }
      }
    }
    if (size_lack_count != 0) {
      std::cout <<
                "Warning: failed to fuse Add into Conv bias due to lack of size information."
                << std::endl;
    }
  }

  void optimize(Graph& graph) override {
    fuse_add_bias_into_conv(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
