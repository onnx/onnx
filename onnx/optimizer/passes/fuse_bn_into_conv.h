
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

  void modify_conv(Node* conv, Node* bn)  {
    auto bn_inputs = bn.inputs();
    auto conv_inputs = conv.inputs();
    auto s = bn_inputs[1];
    auto bbn = bn_inputs[2];
    auto m = bn_inputs[3];
    auto var = bn_inputs[4];
    auto epsilon = bn->f(kepsilon);
    auto frac = s / sqrt(var + epsilon);
    conv_inputs[1] *= frac;
    Value* bc;
    if (conv_inputs.size() == 2)  {
      bc = 0;
      conv_inputs.addInput(bc);
    } else {
      bc = conv_inputs[2];
    }
    conv_inputs[2] = (bc - m) * frac + bbn;
  }

  void fuse_bn_into_conv(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){fuse_bn_into_conv(g);});
      if (n->kind() == kBatchNormalization && n->input()->node()->kind() == kConv) {
        auto origInput = n->input();
        if (origInput->uses().size() > 1 || n->outputs().size() > 1) {
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
