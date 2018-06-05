
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"
#include <math.h>

namespace ONNX_NAMESPACE { namespace optimization {

struct FuseBNIntoConv final : public OptimizePass {
  explicit FuseBNIntoConv()
    : OptimizePass("fuse_consecutive_transposes", API_TYPE::IR) {
  }

  bool check_initializers(Node* conv, Node* bn) {
    auto bn_inputs = bn->inputs();
    auto conv_inputs = conv->inputs();
    for (int i = 1; i <= 4; i++)  {
      if (get_initializer_index(bn_inputs[i]) == -1)  {
        return false;
      }
    }
    if (get_initializer_index(conv_inputs[1]) == -1)  {
      return false;
    }
    if (conv_inputs.size() == 3 && get_initializer_index(conv_inputs[1]) == -1) {
      return false;
    }
    return true;
  }

  void modify_conv(Node* conv, Node* bn)  {
    auto bn_inputs = bn->inputs();
    auto conv_inputs = conv->inputs();
    auto bn_shape = bn->sizes();
    auto conv_shape = conv->sizes();
    auto s = initializers()[get_initializer_index(bn_inputs[1])];
    auto bbn = initializers()[get_initializer_index(bn_inputs[2])];
    auto m = initializers()[get_initializer_index(bn_inputs[3])];
    auto var = initializers()[get_initializer_index(bn_inputs[4])];
    auto epsilon = bn->f(kepsilon);
    auto frac = s / sqrt(var + epsilon);
    conv_inputs[1].scale(frac);
    if (conv_inputs.size() == 2)  {
      Value* bc = new Value(, 2);
      conv_inputs.addInput(bc);
    }
    (conv_inputs[2].add(m.scale(-1))).scale(frac)).add(bbn);
  }

  void fuse_bn_into_conv(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){fuse_bn_into_conv(g);});
      if (n->kind() == kBatchNormalization && n->input()->node()->kind() == kConv) {
        auto origInput = n->input();
        if (origInput->uses().size() > 1 ||
            n->outputs().size() > 1 ||
            !check_initializers(n->input()->node(), n)) {
          continue;
        }
        n->output().replaceAllUsesWith(origInput);
        modify_conv(n->input()->node(), n);
        it.destroyCurrent();
      }
    }
  }

  void optimize(Graph& graph) override {
    fuse_bn_into_conv(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
