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
          // check if bias is Const or in graph's initializers
          if (orig_bias->node()->kind() != kConstant
              && orig_bias->node()->kind() != kParam) {
            continue;
          }
          // check if conv is only used by Add
          if (std::find_if(orig_conv->uses().begin(),
                           orig_conv->uses().end(),
                           [n](Use u){ return u.user != n; })
              != orig_conv->uses().end()) {
            continue;
          }
          auto conv_shape = orig_conv->sizes();
          auto bias_shape = orig_bias->sizes();
          auto weight_shape = orig_conv->node()->inputs()[1]->sizes();
          int64_t M = -1;
          // check if has int conv_shape
          // get feature M
          if (conv_shape.size() > 1 && conv_shape[1].is_int) {
            M = conv_shape[1].dim;
          }
          // check if has int weight_shape
          // get feature M
          if (weight_shape.size() > 0 && weight_shape[0].is_int) {
            ONNX_ASSERT(M == -1 || M == weight_shape[0].dim);
            M = weight_shape[0].dim;
          }
          // check if there is enough information
          if (M == -1 || bias_shape.size() == 0 || !bias_shape[0].is_int) {
            size_lack_count += 1;
            continue;
          }
          // check attributes
          std::vector<Dimension> new_bias_shape;
          if (bias_shape.size() < conv_shape.size()) {
            if (!n->hasAttribute(kbroadcast) || !n->hasAttribute(kaxis)) {
              continue;
            }
            if (n->i(kbroadcast) != 1 || (n->i(kaxis) != 1 && n->i(kaxis) != 1 - conv_shape.size())) {
              continue;
            }
          }
          // check if need squeeze and make squeeze axes
          std::vector<int64_t> squeeze_axes;
          int axis = 0;
          for (auto d : bias_shape) {
            if (d.is_int && d.dim != 1) {
              new_bias_shape.push_back(d);
            } else {
              squeeze_axes.push_back(axis);
            }
            axis++;
          }
          if (new_bias_shape.size() != 1) {
            continue;
          }
          // check if mismatch M
          if (new_bias_shape[0].dim != M) {
            continue;
          }
          // insert Squeeze node if necessary
          if (squeeze_axes.size() > 0) {
            Node *squeeze = graph.create(kSqueeze, orig_bias);
            squeeze->is_(kaxes, std::move(squeeze_axes));
            squeeze->insertBefore(orig_conv->node());
            orig_bias = squeeze->output();
          }
          // add bias as 3rd input
          orig_conv->node()->addInput(orig_bias);
          // replace the use of n with conv
          n->replaceAllUsesWith(orig_conv->node());
          it.destroyCurrent();
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
